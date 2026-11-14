"""
dataset.py —— 数据扫描、事件流解析、视频帧读取与 PyTorch Dataset
"""
import json
import math
import subprocess
import os
from dataclasses import dataclass, field
from typing import Any

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


def parse_finger_jsonl(jsonl_path: str, total_frames: int | None = None) -> list[Frame_teacher]:
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
            frame_label.slot.sort(key=lambda s: (s.x, s.y, s.type.value))
        frame_label_list.append(frame_label)
    return frame_label_list


class VideoChunkDataset(Dataset):
    def __init__(
        self,
        video: list[str],
        teaching_data: list[list[Frame_teacher]],
        start: list[int],
        T: int,
        warmup: int,
        fps: float = 60.0,
    ) -> None:
        self.video = video
        self.teaching_data = teaching_data
        self.start = start
        self.T = T
        self.warmup = warmup
        self.fps = fps

    def __len__(self):
        return len(self.start)

    def read_rgb(self, video_file, start_frame, num_frames):
        start_time = start_frame / self.fps
        command = [
            'ffmpeg', 
            '-ss', str(start_time),
            '-i', video_file,
            '-frames:v', str(num_frames),
            '-vf', 'scale=320:180',
            '-f', 'rawvideo', 
            '-pix_fmt', 'rgb24',
            '-loglevel', 'error', 
            'pipe:1'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        
        video_array = np.frombuffer(out, dtype=np.uint8)
        actual_frames = len(video_array) // (180 * 320 * 3)
        
        if actual_frames == 0:
            return np.zeros((num_frames, 180, 320, 3), dtype=np.uint8)
            
        video_array = video_array[:actual_frames * 180 * 320 * 3].reshape(-1, 180, 320, 3)
        if actual_frames < num_frames:
            pad_frames = np.zeros((num_frames - actual_frames, 180, 320, 3), dtype=np.uint8)
            video_array = np.concatenate([video_array, pad_frames], axis=0)
            
        video_array = video_array.transpose(0, 3, 1, 2)
        return video_array

    def __getitem__(self, index):
        vid_path = self.video[index]
        start_idx = self.start[index]
        labels = self.teaching_data[index]
        
        total_frames = self.warmup + self.T
        read_start = start_idx - self.warmup
        
        video_np = self.read_rgb(vid_path, read_start, total_frames)
        
        warmup_x = torch.from_numpy(video_np[:self.warmup]).float() / 255.0
        target_x = torch.from_numpy(video_np[self.warmup:]).float() / 255.0
        
        max_slots = C.n_slot
        chunk_len = self.T
        
        type_label = torch.zeros((chunk_len, max_slots), dtype=torch.long)
        coord_label = torch.zeros((chunk_len, max_slots, 2), dtype=torch.float32)
        
        for t in range(chunk_len):
            current_frame = start_idx + t
            if current_frame < len(labels):
                frame_label = labels[current_frame]
                for i, slot in enumerate(frame_label.slot):
                    if i >= max_slots:
                        break
                    type_label[t, i] = slot.type.value
                    coord_label[t, i, 0] = slot.x
                    coord_label[t, i, 1] = slot.y
                    
        return warmup_x, target_x, type_label, coord_label

if __name__ == "__main__":
    # 记得换成测试的路径
    data_dir = "C:/Users/HUAWEI/OneDrive/Desktop/Mapset/map-set-train/map-set-train.dir"
    
    videos = []
    all_teaching_data = []
    starts = []
    
    print("正在扫描文件夹并解析...")
    for paths, dirs, files in os.walk(data_dir):
        video_path = None
        jsonl_path = None  
        json_path = None
        
        for file_name in files:
            if file_name.endswith(".mp4"):
                video_path = os.path.join(paths, file_name)
            elif file_name.endswith(".jsonl"):
                jsonl_path = os.path.join(paths, file_name)
            elif file_name.endswith(".json"): # 兼容你原来的
                json_path = os.path.join(paths, file_name)
                
        target_chart = jsonl_path if jsonl_path else json_path
        
        if video_path and target_chart:
            if target_chart.endswith(".jsonl"):
                frame_labels = parse_finger_jsonl(target_chart)
            else:
                continue 
                
            videos.append(video_path)
            all_teaching_data.append(frame_labels)
            starts.append(100)  
            
            if len(videos) >= 1: #
                break

    if len(videos) > 0:
        print(f"成功解析 {len(videos)} 组数据")
        
        dataset = VideoChunkDataset(
            video=videos, 
            teaching_data=all_teaching_data, 
            start=starts, 
            T=16, 
            warmup=4
        )
        
        warmup_x, target_x, type_label, coord_label = dataset[0]
        
        print("\n读取成功！")
        print("Warmup 图像形状:", warmup_x.shape)    # [4, 3, 180, 320]
        print("Target 图像形状:", target_x.shape)    # [16, 3, 180, 320]
        print("动作类型形状:", type_label.shape)     # [16, 4]
        print("动作坐标形状:", coord_label.shape)    # [16, 4, 2]
    else:
        print("Notfound")