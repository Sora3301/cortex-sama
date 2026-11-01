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

from torch.utils.data import DataLoader, Dataset
import json
from dataclasses import dataclass, field
from config import Action_type as Action_type
import config as C

@dataclass
class Coordf_with_type:
    type:Action_type = Action_type.TYPE_IDLE
    x:float = 0.0
    y:float = 0.0

@dataclass
class Frame_teacher:
    slot: list[Coordf_with_type]= field(default_factory=list)
    def isOverSlot(self):
        if len(self.slot) > C.n_slot: return True
        else: return False

def jsonPhase_to_ActionType(str):
    if str=="down": return Action_type.TYPE_DOWN
    elif str=="hold": return Action_type.TYPE_HOLD
    elif str=="flick": return Action_type.TYPE_FLICK
    else: return Action_type.TYPE_IDLE

def parse_finger_jsonl(jsonl_path):
    record = {}
    frame_num = 0
    with open(jsonl_path, encoding="utf-8") as F:
        for line in F:
            line = line.strip()
            if line:
                obj = json.loads(line)
                frame_id = obj["frame"]
                record[frame_id] = obj
                frame_num = frame_id
    frame_label_list: list[Frame_teacher] = []
    for i in range(frame_num + 1):
        frame_label = Frame_teacher()
        if i in record:
            events = record[i]["events"]
            for e in events:
                if e.get("bad"): continue
                slot_label = Coordf_with_type(jsonPhase_to_ActionType(e["phase"]), float(e["x"]), float(e["y"]))
                if abs(slot_label.x) > 1.0 or abs(slot_label.y) > 1.0: continue
                frame_label.slot.append(slot_label)
            #排序
            frame_label.slot.sort(
                key=lambda s: (s.x, s.y, s.type.value)
            )
        frame_label_list.append(frame_label)
    return frame_label_list





        


    

    

class VideoChunkDataset(Dataset):
    def __init__(self, vedio, teaching_data, start, T, warmup):
        pass


      
