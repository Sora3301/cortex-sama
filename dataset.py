"""
dataset.py —— 数据扫描、JSONL 触控解析、视频帧读取与 PyTorch Dataset

本文件需要实现的内容：
1. parse_finger_jsonl(jsonl_path, n_slots=4, total_frames=None) —— 解析单首谱面的 fingers-*.jsonl：
   - 读取每行 JSON 的 frame（帧号）、fingers（当前在屏手指 [{id, x, y}]）、events（当前帧事件 [{id, phase, x, y}]）
   - 按触控 id 跨帧持久化绑定到固定槽位 (0 .. n_slots-1)：
     a) 释放离开屏幕的 id：若某槽位原有的 id 不在当前帧 fingers 中，清空该槽位 (None)
     b) 更新持续按住的 id：仍在屏的 id 保持原槽位不变，坐标裁剪到 [-1, 1]，类别设为 TYPE_HOLD(2)
     c) 分配新按下的 id：找第一个空闲槽位绑定，若 phase=="flick" 设为 TYPE_FLICK(3)，否则设为 TYPE_DOWN(1)
     d) 补齐跳帧：对 JSONL 中跳过的重复帧，延续上一帧已按下槽位的 TYPE_HOLD(2) 状态与坐标
   - 返回:
     type_labels:  (num_frames, n_slots) int64
     coord_labels: (num_frames, n_slots, 2) float32

2. VideoFrameReader —— 视频区间帧读取器：
   - 给定视频路径 mp4_path，提供 get_range(start_frame, end_frame) 方法
   - 读取 [start_frame, end_frame) 区间的连续帧（可用 decord 或 ffmpeg 管道），返回 (N, H, W, 3) uint8 数组

3. discover_recordings(data_root, max_videos=None) —— 扫描数据集目录：
   - 遍历 ~/dev/map-set-train 下的子文件夹，按时间戳配对 recording-<ts>.mp4 与 fingers-<ts>.jsonl
   - 返回 [(mp4_path, jsonl_path), ...] 列表

4. build_dataset_index(pairs, n_slots=4, T=16, stride=16, keep_idle_prob=0.1) —— 构建滑窗索引：
   - 对每个配对调用 parse_finger_jsonl 得到全曲标签
   - 按步长 stride 生成滑窗起点 (video_idx, start_frame)，按一定概率过滤全空置片段
   - 返回 (paths, type_labels_list, coord_labels_list, starts)

5. VideoChunkDataset(Dataset) —— 滑窗视频块数据集：
   - __len__(): 返回滑窗样本总数 len(starts)
   - __getitem__(index):
     a) 根据 starts[index] 取出 (video_idx, start_frame)
     b) 调用 VideoFrameReader 读取 [start_frame - warmup_len, start_frame + T) 连续帧并归一化到 [0, 1]
     c) 拆分为 warmup_frames (W, 3, H, W) 和 target_frames (T, 3, H, W)（开头不足重复首帧，结尾不足补零）
     d) 切片取出对应的 type_label (T, 4) 和 coord_label (T, 4, 2)
     e) 返回 (warmup_frames, target_frames, type_label, coord_label)
"""
