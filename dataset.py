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
4. VideoChunkDataset(Dataset) —— 滑窗视频块数据集（待实现）
"""
from typing import Literal
import os
import json
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

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

def parse_map_info(
    vedio_name: str # 带扩展名
) -> tuple[Literal[-1], Literal[-1]] | tuple[Any, Any]:
    with open(C.label_path + "/manifast.json", encoding="utf-8") as F:
        data = json.load(F)
        target = next((obj for obj in data if obj.get("folder") == os.path.splitext(vedio_name)), None)
    
    if target is not None:
        return int(target["num_frames"]), int(target["max_slots"])
    else:
        print("没找到", vedio_name, "喵")
        return -1, -1

def count_max_slot(
    label:list[Frame_teacher],
    start:int,
    end:int = -1
):
    if end < 0: end = start + C.seq_len
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
    def __init__(
        self,
        vedio: list[str],
        teaching_data: list[list[Frame_teacher]],
        start: list[tuple[int, int]],
        T: int,
        warmup: int,
    ) -> None:
        pass
