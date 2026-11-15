"""
dataset.py —— 数据扫描、事件流解析、视频帧读取与 PyTorch Dataset

本文件需要实现的内容：
1. Coordf_with_type / Frame_teacher —— 单帧标签结构：
     - Coordf_with_type: 一个槽位的动作类型 + 归一化坐标
     - Frame_teacher:   一帧的槽位列表（活跃槽位已按 (x, y, type) 规范排序）
2. jsonPhase_to_ActionType(phase) —— 把事件流里的 down/hold/flick 映射成 Action_type
3. parse_finger_jsonl(jsonl_path, total_frames=None) —— 解析预处理后的事件流 JSONL：
     - 每行 = 一个帧对象 {"frame": int, "events": [{"slot", "phase", "x", "y", ...}]}
     - 返回 list[Frame_teacher]，下标 i 严格对应视频第 i 帧
     - 事件流只保留有事件的帧，所以要用 total_frames（视频总帧数）补齐到全 idle
     - 帧内按 (x, y, type) 排序，保证同一幅画面的标签唯一确定
4. VideoChunkDataset(Dataset) —— 滑窗视频块数据集
"""
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
from config import Action_type as Action_type




@dataclass
class Coordf_with_type:
    type: Action_type = Action_type.TYPE_IDLE
    x: float = 0.0
    y: float = 0.0


@dataclass
class Frame_teacher:
    slot: list[Coordf_with_type] = field(default_factory=list)

    def isOverSlot(self) -> bool:
        return len(self.slot) > C.n_slot

    def getSlot(self) -> int:
        return len(self.slot)


def jsonPhase_to_ActionType(phase: str) -> Action_type:
    if phase == "down":
        return Action_type.TYPE_DOWN
    elif phase == "hold":
        return Action_type.TYPE_HOLD
    elif phase == "flick":
        return Action_type.TYPE_FLICK
    else:
        return Action_type.TYPE_IDLE

def count_max_slot(
    label:list[Frame_teacher],
    start:int,
    end:int = -1
):
    if end < 0: 
        end = start + C.seq_len
    max_slot = 0
    end = min(end, len(label))
    for i in range(start, end):
        max_slot = max(max_slot, label[i].getSlot())
    return max_slot


def parse_finger_jsonl(
    jsonl_path: str,
    total_frames: int | None = None,
) -> list[Frame_teacher]:
    record: dict[int, dict[str, Any]] = {}
    frame_num: int = -1
    with open(jsonl_path, encoding="utf-8") as F:
        for line in F:
            line = line.strip()
            if line:
                obj = json.loads(line)
                frame_id = int(obj["frame"])
                record[frame_id] = obj
                frame_num = max(frame_num, frame_id)

    n_frames = frame_num + 1
    if total_frames is not None:
        n_frames = max(n_frames, total_frames)

    frame_label_list: list[Frame_teacher] = []
    for i in range(n_frames):
        frame_label = Frame_teacher()
        if i in record:
            events: list[dict[str, Any]] = record[i].get("events", [])
            for e in events:
                if e.get("bad"):
                    continue
                x = float(e["x"])
                y = float(e["y"])
                slot_label = Coordf_with_type(jsonPhase_to_ActionType(e["phase"]), x, y)
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                if abs(x) > 1.0 or abs(y) > 1.0:
                    continue
                frame_label.slot.append(slot_label)
            # 逐帧规范排序：同一幅画面的标签唯一确定
            frame_label.slot.sort(
                key=lambda s: (s.x, s.y, s.type.value)
            )
        frame_label_list.append(frame_label)
    return frame_label_list


class VideoChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """滑窗视频块数据集。

    __getitem__ 返回 (warmup, x, type_label, coord_label)：
      warmup:      (warmup, 3, H, W) float32 ∈ [0,1]（normalize=False 时为 uint8）
      x:           (T, 3, H, W)      float32 ∈ [0,1]（normalize=False 时为 uint8）
      type_label:  (T, n_slots) int64                 每槽位动作类型
      coord_label: (T, n_slots, 2) float32 ∈ [-1,1]   每槽位归一化坐标

    约定：
      - video[i] 可为文件名（相对 video_dir）或已存在的视频路径，teaching_data[i] 是该视频按帧下标的标签列表，
        start[i] 是目标块第一帧的绝对帧号（warmup 帧取 [start-warmup, start)）
      - 槽位编号 = 该帧活跃槽位按 (x, y, type) 排序后的位置，不足 n_slots 的填 TYPE_IDLE

    实现说明：使用 PyAV (libav) 在 worker 进程内直接 seek 并解码目标帧区间，避免每 chunk
    启停 ffmpeg 子进程的开销；build_chunks 保证 [start-warmup, start+T) 都在视频范围内，
    解码帧数不足时直接报错。
    """

    def __init__(
        self,
        video: list[str],
        teaching_data: list[list[Frame_teacher]],
        start: list[int],
        T: int,  # chunk大小
        warmup: int,  # warmup大小
        video_dir: str | Path = C.video_path,
        normalize: bool = True,
    ) -> None:
        if not (len(video) == len(teaching_data) == len(start)):
            raise ValueError("video / teaching_data / start 长度必须一致")
        if T <= 0:
            raise ValueError("T 必须大于 0")
        if warmup < 0:
            raise ValueError("warmup 不能为负数")
        if any(s < warmup for s in start):
            raise ValueError("start 中的帧号必须 >= warmup")
        self.video = video
        self.teaching_data = teaching_data
        self.start = start
        self.T = T
        self.warmup = warmup
        self.video_dir = Path(video_dir)
        self.normalize = normalize
        self.n_slots = C.n_slot
        self._resolved_paths: dict[str, str] = {}

    def _resolve_path(self, name: str) -> str:
        cached = self._resolved_paths.get(name)
        if cached is not None:
            return cached
        p = Path(name)
        resolved = str(p) if p.is_file() else str(self.video_dir / name)
        self._resolved_paths[name] = resolved
        return resolved

    def _read_frames(self, name: str, start_frame: int, count: int) -> torch.Tensor:
        """解码 [start_frame, start_frame+count) 帧，返回 (count, 3, H, W) uint8 张量"""
        if start_frame < 0:
            raise ValueError(f"start_frame 不能为负数: {start_frame}")
        path = self._resolve_path(name)
        with av.open(path) as container:
            if not container.streams.video:
                raise RuntimeError(f"视频文件不含视频流: {path}")
            stream = container.streams.video[0]
            stream.thread_count = 1
            rate = stream.average_rate or stream.guessed_rate
            tb = stream.time_base
            if rate is None or rate <= 0 or tb is None:
                raise RuntimeError(f"无法读取视频帧率或 time_base: {path}")

            start_pts = stream.start_time if stream.start_time is not None else 0
            step_pts = Fraction(1) / (rate * tb)
            target_pts = start_pts + round(start_frame * step_pts)
            min_pts = target_pts - step_pts / 2

            if start_frame > 0:
                container.seek(target_pts, backward=True, any_frame=False, stream=stream)

            buf: np.ndarray | None = None
            got = 0
            for frame in container.decode(stream):
                pts = frame.pts
                if pts is None or pts < min_pts:
                    continue
                if got == 0 and pts > target_pts + step_pts / 2:
                    raise RuntimeError(
                        f"视频 {name} seek 到第 {start_frame} 帧失败（首帧 pts={pts} 超过目标 {target_pts}）"
                    )
                if buf is None:
                    buf = np.empty((count, frame.height, frame.width, 3), dtype=np.uint8)
                buf[got] = frame.to_ndarray(format="rgb24")
                got += 1
                if got == count:
                    break

        if buf is None or got != count:
            raise RuntimeError(
                f"视频 {name} 从第 {start_frame} 帧起只解码出 {got}/{count} 帧，"
                "说明 start 越界或视频/manifest 帧数不一致"
            )
        return torch.from_numpy(buf).permute(0, 3, 1, 2)

    def __len__(self) -> int:
        return len(self.start)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self.video[idx]
        s = self.start[idx]
        labels = self.teaching_data[idx]

        frames = self._read_frames(name, s - self.warmup, self.warmup + self.T)
        warmup = frames[: self.warmup].contiguous()
        x = frames[self.warmup :].contiguous()
        if self.normalize:
            warmup = warmup.float().div_(255.0)
            x = x.float().div_(255.0)

        type_label = torch.zeros(self.T, self.n_slots, dtype=torch.long)
        coord_label = torch.zeros(self.T, self.n_slots, 2, dtype=torch.float32)
        for t in range(self.T):
            pos = s + t
            frame_label = labels[pos] if 0 <= pos < len(labels) else None
            if frame_label is None:
                continue
            for i, slot in enumerate(frame_label.slot[: self.n_slots]):
                type_label[t, i] = slot.type.value
                coord_label[t, i, 0] = slot.x
                coord_label[t, i, 1] = slot.y
        return warmup, x, type_label, coord_label
