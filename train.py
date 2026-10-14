import argparse
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from decord import VideoReader as DecordReader, cpu as decord_cpu
except ImportError:
    DecordReader = None
    decord_cpu = None


# 槽位动作类别定义 (n_types = 4)
TYPE_IDLE = 0   # 空置
TYPE_DOWN = 1   # 新按下 (click / drag / hold 起始帧)
TYPE_HOLD = 2   # 持续按住或移动 (hold / drag 后续帧)
TYPE_FLICK = 3  # 划动音符起始帧 (flick)


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f5 = F.relu(self.conv5(f4))
        return f5


class FingerHead(nn.Module):
    def __init__(self, in_ch=256, hidden=128, gru_hidden=256, n_slots=4, n_types=4):
        super().__init__()
        self.n_slots = n_slots
        self.n_types = n_types
        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=gru_hidden,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        self.trunk = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.type_head = nn.Linear(hidden, n_slots * n_types)
        self.coord_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )
        self.type_emb = nn.Embedding(n_types, hidden)
        self.slot_emb = nn.Embedding(n_slots, hidden)

    def _predict(self, h, device):
        B = h.size(0)
        logits = self.type_head(h).view(B, self.n_slots, self.n_types)  # (B, n_slots, n_types)

        # 软 embedding：用 type 概率加权 embedding 表
        w = logits.softmax(-1)                                          # (B, n_slots, n_types)
        slot = torch.arange(self.n_slots, device=device).expand(B, -1)
        cond = w @ self.type_emb.weight + self.slot_emb(slot)           # (B, n_slots, hidden)

        feat = h.unsqueeze(1).expand(-1, self.n_slots, -1)              # (B, n_slots, hidden)
        coords = self.coord_head(torch.cat([feat, cond], dim=-1))       # (B, n_slots, 2)
        return logits, coords

    def forward(self, x, B, T, h=None):
        """x: (B*T, 256, H, W), h: (num_layers, B, gru_hidden)"""
        g = F.adaptive_max_pool2d(x, 1).flatten(1)                      # (B*T, 256)
        g = g.reshape(B, T, -1)                                         # (B, T, 256)
        out, h = self.gru(g, h)                                         # (B, T, gru_hidden)
        f = self.trunk(out.reshape(B * T, -1))                          # (B*T, hidden)
        logits, coords = self._predict(f, x.device)                     # (B*T, n_slots, n_types), (B*T, n_slots, 2)
        logits = logits.view(B, T, self.n_slots, self.n_types)
        coords = coords.view(B, T, self.n_slots, 2)
        return logits, coords, h

    def step(self, x, h=None):
        """单帧流式推理: x 为 (B, 256, H, W)"""
        B = x.size(0)
        if h is None:
            h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=x.device, dtype=x.dtype)
        g = F.adaptive_max_pool2d(x, 1).flatten(1)                      # (B, 256)
        out, h = self.gru(g.unsqueeze(1), h)                            # (B, 1, gru_hidden)
        logits, coords = self._predict(self.trunk(out[:, 0]), x.device)
        return logits, coords, h


class Model(nn.Module):
    def __init__(self, n_slots=4, n_types=4):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = FingerHead(n_slots=n_slots, n_types=n_types)

    def forward(self, x, h=None):
        """x: (B, T, 3, H, W)"""
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))                              # (B*T, 256, h, w)
        return self.head(f, B, T, h)

    def step(self, frame, h=None):
        """单帧推理: frame 为 (B, 3, H, W)"""
        f = self.backbone(frame)
        return self.head.step(f, h)


def parse_finger_jsonl(jsonl_path, n_slots=4, total_frames=None):
    """解析 fingers-*.jsonl，按触控 id 持久化绑定槽位，生成逐帧标签。

    返回:
        type_labels:  (N, n_slots) int64，取值 0=idle, 1=down, 2=hold, 3=flick
        coord_labels: (N, n_slots, 2) float32，范围 [-1.0, 1.0]
    """
    records = {}
    max_frame = -1
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fidx = int(obj["frame"])
            records[fidx] = obj
            if fidx > max_frame:
                max_frame = fidx

    num_frames = max(total_frames or 0, max_frame + 1)
    if num_frames <= 0:
        return (
            np.zeros((0, n_slots), dtype=np.int64),
            np.zeros((0, n_slots, 2), dtype=np.float32),
        )

    type_labels = np.zeros((num_frames, n_slots), dtype=np.int64)
    coord_labels = np.zeros((num_frames, n_slots, 2), dtype=np.float32)

    slot_ids = [None] * n_slots
    slot_coords = np.zeros((n_slots, 2), dtype=np.float32)

    for f in range(num_frames):
        rec = records.get(f)
        if rec is None:
            # 未记录帧（音频时间戳未推进的重复帧）：已按下的手指维持 hold 状态与上一帧坐标
            for s in range(n_slots):
                if slot_ids[s] is not None:
                    type_labels[f, s] = TYPE_HOLD
                    coord_labels[f, s] = slot_coords[s]
            continue

        fingers_list = rec.get("fingers", [])
        active_map = {fg["id"]: (float(fg["x"]), float(fg["y"])) for fg in fingers_list}
        event_phases = {
            ev["id"]: ev["phase"]
            for ev in rec.get("events", [])
            if ev.get("phase") in ("down", "flick")
        }

        # 1. 释放已经离开屏幕的触控 id
        for s in range(n_slots):
            fid = slot_ids[s]
            if fid is not None and fid not in active_map:
                slot_ids[s] = None
                slot_coords[s] = 0.0

        # 2. 更新已在槽位中持续按住的触控 id
        bound_ids = set()
        for s in range(n_slots):
            fid = slot_ids[s]
            if fid is not None:
                bound_ids.add(fid)
                x, y = active_map[fid]
                slot_coords[s, 0] = np.clip(x, -1.0, 1.0)
                slot_coords[s, 1] = np.clip(y, -1.0, 1.0)
                phase = event_phases.get(fid)
                if phase == "flick":
                    type_labels[f, s] = TYPE_FLICK
                elif phase == "down":
                    type_labels[f, s] = TYPE_DOWN
                else:
                    type_labels[f, s] = TYPE_HOLD
                coord_labels[f, s] = slot_coords[s]

        # 3. 为新出现的触控 id 分配空闲槽位
        for fg in fingers_list:
            fid = fg["id"]
            if fid in bound_ids:
                continue
            free_slot = None
            for s in range(n_slots):
                if slot_ids[s] is None:
                    free_slot = s
                    break
            if free_slot is None:
                # 超过最大槽位数时忽略多余手指
                break
            slot_ids[free_slot] = fid
            bound_ids.add(fid)
            slot_coords[free_slot, 0] = np.clip(float(fg["x"]), -1.0, 1.0)
            slot_coords[free_slot, 1] = np.clip(float(fg["y"]), -1.0, 1.0)
            phase = event_phases.get(fid, "down")
            type_labels[f, free_slot] = TYPE_FLICK if phase == "flick" else TYPE_DOWN
            coord_labels[f, free_slot] = slot_coords[free_slot]

    return type_labels, coord_labels


class VideoFrameReader:
    """视频帧读取器：优先尝试 decord，若环境 FFmpeg 不兼容则自动回退到 ffmpeg 管道读取。"""

    _use_decord = DecordReader is not None
    _probed = False
    _ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    @classmethod
    def probe_backend(cls, sample_path):
        if cls._probed or not cls._use_decord:
            return
        cls._probed = True
        try:
            r = DecordReader(str(sample_path), ctx=decord_cpu(0))
            _ = r.get_batch([0]).asnumpy()
        except Exception:
            cls._use_decord = False

    def __init__(self, path, num_frames, width=320, height=180, fps=60.0):
        self.path = str(path)
        self.num_frames = int(num_frames)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._decord_reader = None

    def __len__(self):
        return self.num_frames

    def get_range(self, start_frame, end_frame):
        """读取 [start_frame, end_frame) 区间帧，返回 (N, H, W, 3) uint8 numpy 数组。"""
        count = max(0, end_frame - start_frame)
        if count == 0:
            return np.zeros((0, self.height, self.width, 3), dtype=np.uint8)

        if VideoFrameReader._use_decord:
            try:
                if self._decord_reader is None:
                    self._decord_reader = DecordReader(self.path, ctx=decord_cpu(0))
                total = len(self._decord_reader)
                indices = [min(max(0, i), total - 1) for i in range(start_frame, end_frame)]
                return self._decord_reader.get_batch(indices).asnumpy()
            except Exception:
                VideoFrameReader._use_decord = False
                self._decord_reader = None

        start_sec = max(0.0, start_frame / self.fps)
        cmd = [
            VideoFrameReader._ffmpeg_bin,
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.6f}",
            "-i",
            self.path,
            "-frames:v",
            str(count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        raw = subprocess.check_output(cmd)
        frame_bytes = self.height * self.width * 3
        actual = len(raw) // frame_bytes
        if actual <= 0:
            return np.zeros((count, self.height, self.width, 3), dtype=np.uint8)
        frames = np.frombuffer(bytearray(raw[: actual * frame_bytes]), dtype=np.uint8).reshape(
            actual, self.height, self.width, 3
        )
        if actual < count:
            pad = np.repeat(frames[-1:], count - actual, axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        return frames


def discover_recordings(data_root, max_videos=None):
    """扫描 ~/dev/map-set-train 下所有谱面文件夹，配对 recording-*.mp4 与 fingers-*.jsonl。"""
    root = Path(os.path.expanduser(data_root))
    if not root.is_dir():
        raise FileNotFoundError(f"训练数据目录不存在: {root}")

    pairs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        mp4_map = {}
        jsonl_map = {}
        for f in entry.iterdir():
            name = f.name
            if name.startswith("recording-") and name.endswith(".mp4"):
                ts = name[len("recording-") : -len(".mp4")]
                mp4_map[ts] = f
            elif name.startswith("fingers-") and name.endswith(".jsonl"):
                ts = name[len("fingers-") : -len(".jsonl")]
                jsonl_map[ts] = f
        for ts in sorted(set(mp4_map) & set(jsonl_map)):
            pairs.append((mp4_map[ts], jsonl_map[ts]))
            if max_videos is not None and len(pairs) >= max_videos:
                return pairs
    return pairs


def build_dataset_index(
    pairs,
    n_slots=4,
    T=16,
    stride=16,
    keep_idle_prob=0.15,
    seed=42,
):
    """解析谱面标签并构建滑窗起点索引。"""
    rng = random.Random(seed)
    paths = []
    type_labels_list = []
    coord_labels_list = []
    starts = []

    for video_idx, (mp4_path, jsonl_path) in enumerate(pairs):
        type_lbl, coord_lbl = parse_finger_jsonl(jsonl_path, n_slots=n_slots)
        num_frames = len(type_lbl)
        if num_frames < T:
            continue

        curr_idx = len(paths)
        paths.append(str(mp4_path))
        type_labels_list.append(type_lbl)
        coord_labels_list.append(coord_lbl)

        for start_frame in range(0, num_frames - T + 1, stride):
            chunk_types = type_lbl[start_frame : start_frame + T]
            has_action = bool((chunk_types > 0).any())
            if has_action or rng.random() < keep_idle_prob:
                starts.append((curr_idx, start_frame))

    return paths, type_labels_list, coord_labels_list, starts


class VideoChunkDataset(Dataset):
    """滑窗样本：每条 = warmup 帧(拿状态,不算 loss) + target T 帧(算 loss)。"""

    def __init__(
        self,
        paths,
        type_labels,
        coord_labels,
        starts,
        T=16,
        warmup=16,
        width=320,
        height=180,
        fps=60.0,
    ):
        self.paths = paths
        self.type_labels = type_labels
        self.coord_labels = coord_labels
        self.starts = starts
        self.target_len = T
        self.warmup_len = warmup
        self.width = width
        self.height = height
        self.fps = fps
        self._reader_cache = {}

    def __len__(self):
        return len(self.starts)

    def _reader(self, video_index):
        path = self.paths[video_index]
        reader = self._reader_cache.get(path)
        if reader is None:
            num_frames = len(self.type_labels[video_index])
            reader = VideoFrameReader(
                path,
                num_frames=num_frames,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )
            self._reader_cache = {path: reader}
        return reader

    def __getitem__(self, index):
        video_index, start_frame = self.starts[index]
        reader = self._reader(video_index)
        total_frames = len(reader)

        warmup_start = max(0, start_frame - self.warmup_len)
        target_end = min(start_frame + self.target_len, total_frames)

        clip_np = reader.get_range(warmup_start, target_end)
        clip = torch.from_numpy(clip_np).permute(0, 3, 1, 2).float().div_(255.0)

        warmup_count = start_frame - warmup_start
        warmup_frames = clip[:warmup_count]
        target_frames = clip[warmup_count:]

        if warmup_frames.size(0) < self.warmup_len:
            first_frame = warmup_frames[:1] if warmup_frames.size(0) > 0 else target_frames[:1]
            padding = first_frame.expand(self.warmup_len - warmup_frames.size(0), -1, -1, -1)
            warmup_frames = torch.cat([padding, warmup_frames], dim=0)

        if target_frames.size(0) < self.target_len:
            height, width = target_frames.shape[-2], target_frames.shape[-1]
            blank = target_frames.new_zeros(
                self.target_len - target_frames.size(0), 3, height, width
            )
            target_frames = torch.cat([target_frames, blank], dim=0)

        type_slice = self.type_labels[video_index][start_frame : start_frame + self.target_len]
        coord_slice = self.coord_labels[video_index][start_frame : start_frame + self.target_len]

        type_label = torch.as_tensor(type_slice, dtype=torch.long)
        coord_label = torch.as_tensor(coord_slice, dtype=torch.float32)

        if type_label.size(0) < self.target_len:
            pad_len = self.target_len - type_label.size(0)
            n_slots = type_label.size(1)
            type_label = torch.cat(
                [type_label, torch.zeros(pad_len, n_slots, dtype=torch.long)], dim=0
            )
            coord_label = torch.cat(
                [coord_label, torch.zeros(pad_len, n_slots, 2, dtype=torch.float32)], dim=0
            )

        return warmup_frames, target_frames, type_label, coord_label


def compute_loss(logits, coords, type_label, coord_label, coord_w=1.0, class_weights=None):
    """计算分类交叉熵损失与活跃槽位坐标 MSE 损失。"""
    n_types = logits.size(-1)
    loss_type = F.cross_entropy(
        logits.reshape(-1, n_types),
        type_label.reshape(-1),
        weight=class_weights,
    )

    # 仅在非空置槽位 (type_label > 0) 上监督坐标，避免空置槽位把预测拉向 (0, 0)
    active_mask = type_label > TYPE_IDLE
    if active_mask.any():
        loss_coord = F.mse_loss(coords[active_mask], coord_label[active_mask])
    else:
        loss_coord = coords.sum() * 0.0

    loss = loss_type + coord_w * loss_coord
    return loss, loss_type, loss_coord


@torch.no_grad()
def evaluate(model, loader, coord_w=1.0, class_weights=None, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_type_loss = 0.0
    total_coord_loss = 0.0
    total_samples = 0
    correct_types = 0
    total_slots = 0
    active_coord_mae = 0.0
    active_slots = 0

    for warmup, x, type_label, coord_label in loader:
        warmup = warmup.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        type_label = type_label.to(device, non_blocking=True)
        coord_label = coord_label.to(device, non_blocking=True)

        h = None
        if warmup.size(1) > 0:
            _, _, h = model(warmup)

        logits, coords, _ = model(x, h)
        loss, loss_type, loss_coord = compute_loss(
            logits, coords, type_label, coord_label, coord_w=coord_w, class_weights=class_weights
        )

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_type_loss += loss_type.item() * bs
        total_coord_loss += loss_coord.item() * bs
        total_samples += bs

        preds = logits.argmax(dim=-1)
        correct_types += (preds == type_label).sum().item()
        total_slots += type_label.numel()

        active_mask = type_label > TYPE_IDLE
        n_active = int(active_mask.sum().item())
        if n_active > 0:
            active_coord_mae += F.l1_loss(coords[active_mask], coord_label[active_mask], reduction="sum").item() / 2.0
            active_slots += n_active

    denom = max(1, total_samples)
    return {
        "loss": total_loss / denom,
        "type_loss": total_type_loss / denom,
        "coord_loss": total_coord_loss / denom,
        "type_acc": correct_types / max(1, total_slots),
        "coord_mae": active_coord_mae / max(1, active_slots),
    }


def train_chunks(
    model,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    coord_w=2.0,
    device="cpu",
    save_dir="checkpoints",
    log_interval=20,
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    # 类别权重：抑制 idle 占绝大多数导致的类别不平衡 (0=idle, 1=down, 2=hold, 3=flick)
    class_weights = torch.tensor([0.25, 3.0, 1.0, 4.0], dtype=torch.float32, device=device)
    use_amp = str(device).startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_type_loss = 0.0
        total_coord_loss = 0.0
        total_samples = 0
        t0 = time.time()

        for step_idx, (warmup, x, type_label, coord_label) in enumerate(train_loader, start=1):
            warmup = warmup.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            type_label = type_label.to(device, non_blocking=True)
            coord_label = coord_label.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                h = None
                if warmup.size(1) > 0:
                    with torch.no_grad():
                        _, _, h = model(warmup)

                logits, coords, _ = model(x, h)
                loss, loss_type, loss_coord = compute_loss(
                    logits,
                    coords,
                    type_label,
                    coord_label,
                    coord_w=coord_w,
                    class_weights=class_weights,
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_type_loss += loss_type.item() * bs
            total_coord_loss += loss_coord.item() * bs
            total_samples += bs

            if log_interval > 0 and step_idx % log_interval == 0:
                avg_l = total_loss / max(1, total_samples)
                avg_t = total_type_loss / max(1, total_samples)
                avg_c = total_coord_loss / max(1, total_samples)
                print(
                    f"  [Epoch {epoch:03d} | Step {step_idx:04d}/{len(train_loader):04d}] "
                    f"loss={avg_l:.4f} (type={avg_t:.4f}, coord={avg_c:.4f})"
                )

        scheduler.step()
        elapsed = time.time() - t0
        train_loss = total_loss / max(1, total_samples)
        train_type = total_type_loss / max(1, total_samples)
        train_coord = total_coord_loss / max(1, total_samples)

        msg = (
            f"Epoch {epoch:03d}/{epochs:03d} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} (type={train_type:.4f}, coord={train_coord:.4f})"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "train_loss": train_loss,
        }

        if val_loader is not None and len(val_loader) > 0:
            val_metrics = evaluate(
                model, val_loader, coord_w=coord_w, class_weights=class_weights, device=device
            )
            ckpt["val_metrics"] = val_metrics
            msg += (
                f" | val_loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['type_acc']*100:.2f}% "
                f"coord_mae={val_metrics['coord_mae']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(ckpt, os.path.join(save_dir, "best.pt"))
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                torch.save(ckpt, os.path.join(save_dir, "best.pt"))

        torch.save(ckpt, os.path.join(save_dir, "last.pt"))
        print(msg)


@torch.no_grad()
def stream_infer(model, chunks, device="cpu"):
    """流式推理：h 跨块传递，每帧都出预测。chunks 为按序切好的 (B, T, 3, H, W)。"""
    model.eval()
    h = None
    type_out, coord_out = [], []
    for x in chunks:
        logits, coords, h = model(x.to(device), h)
        type_out.append(logits.argmax(-1).cpu())
        coord_out.append(coords.cpu())
    return torch.cat(type_out, dim=1), torch.cat(coord_out, dim=1)


def main():
    parser = argparse.ArgumentParser(description="Phigros 视觉模型训练脚本")
    parser.add_argument("--data-dir", type=str, default="~/dev/map-set-train", help="训练数据集根目录")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="模型权重保存目录")
    parser.add_argument("--max-videos", type=int, default=None, help="最多加载多少个谱面视频（默认全部）")
    parser.add_argument("--seq-len", type=int, default=16, help="目标预测块帧数 T")
    parser.add_argument("--warmup-len", type=int, default=16, help="GRU 状态热身帧数")
    parser.add_argument("--stride", type=int, default=32, help="滑窗采样步长（帧）")
    parser.add_argument("--keep-idle-prob", type=float, default=0.1, help="纯空置片段保留概率")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集视频比例")
    parser.add_argument("--batch-size", type=int, default=8, help="训练批大小")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 进程数")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--coord-weight", type=float, default=2.0, help="坐标损失权重")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None, help="每轮最大训练步数（用于快速验证）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"正在扫描数据集: {os.path.expanduser(args.data_dir)} ...")
    pairs = discover_recordings(args.data_dir, max_videos=args.max_videos)
    if not pairs:
        raise RuntimeError(f"未在 {args.data_dir} 中找到有效的 (recording-*.mp4, fingers-*.jsonl) 数据对")

    VideoFrameReader.probe_backend(pairs[0][0])

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    val_count = int(len(pairs) * args.val_ratio) if len(pairs) > 1 else 0
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    print(f"共找到 {len(pairs)} 个谱面录像 | 训练集: {len(train_pairs)} | 验证集: {len(val_pairs)}")

    train_paths, train_types, train_coords, train_starts = build_dataset_index(
        train_pairs,
        n_slots=4,
        T=args.seq_len,
        stride=args.stride,
        keep_idle_prob=args.keep_idle_prob,
        seed=args.seed,
    )
    if args.max_steps_per_epoch is not None:
        max_samples = args.max_steps_per_epoch * args.batch_size
        rng.shuffle(train_starts)
        train_starts = train_starts[:max_samples]

    train_dataset = VideoChunkDataset(
        train_paths,
        train_types,
        train_coords,
        train_starts,
        T=args.seq_len,
        warmup=args.warmup_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    val_loader = None
    if val_pairs:
        val_paths, val_types, val_coords, val_starts = build_dataset_index(
            val_pairs,
            n_slots=4,
            T=args.seq_len,
            stride=args.stride * 2,
            keep_idle_prob=args.keep_idle_prob,
            seed=args.seed + 1,
        )
        if args.max_steps_per_epoch is not None:
            val_starts = val_starts[: max(1, args.max_steps_per_epoch // 2) * args.batch_size]
        if val_starts:
            val_dataset = VideoChunkDataset(
                val_paths,
                val_types,
                val_coords,
                val_starts,
                T=args.seq_len,
                warmup=args.warmup_len,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.device.startswith("cuda"),
            )

    print(
        f"训练样本块数: {len(train_dataset)} "
        f"| 验证样本块数: {len(val_loader.dataset) if val_loader else 0} "
        f"| 设备: {args.device}"
    )

    model = Model(n_slots=4, n_types=4)
    train_chunks(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        coord_w=args.coord_weight,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
