"""infer_testing.py —— 加载视频、跑模型，并按 Phira 规则给预测结果打分。

用法：
    python infer_testing.py
    python infer_testing.py --max-frames 900 --device cuda --out pred.jsonl

视频和 label 路径目前写在 main() 里；--out 为模型输出 JSONL 路径。
每帧一行，格式与 infer_onnx.py 一致，只写非 idle 的槽位。

计分约定：
    - 以标签里的 down（Tap/Hold 起点）和 flick 作为音符；
      连续的 hold 帧不重复计为音符。
    - 模型预测中，类型从 idle 进入 down/hold/flick 视为一个预测音符起点。
    - 每个标签音符在 ±220ms 内寻找 x 坐标接近的预测音符；
      找到后用时间差调用 Phira 的 Tap 判定窗口。
    - Flick 只给 Perfect/Miss；Hold 头部只给 Perfect/Good/Miss；
      找不到预测音符则为 Miss。
    - 多出来的预测音符不扣分（和 Phira 的额外点击行为一致）。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import config as C
import dataset as D
from config import Action_type
from model import Model
from phira_score import Judgement, PhiraScore, judge_tap

WIDTH = 320
HEIGHT = 180
FRAME_BYTES = WIDTH * HEIGHT * 3

MATCH_WINDOW = 0.22       # 秒，Phira 的 Bad 窗口
MATCH_X = 0.25            # 归一化 x 坐标容差
MATCH_TIME_WEIGHT = 0.05  # 匹配时 x 距离的权重

TYPE_NAME = {
    Action_type.TYPE_DOWN: "down",
    Action_type.TYPE_HOLD: "hold",
    Action_type.TYPE_FLICK: "flick",
}


def probe_fps(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    raw = json.loads(out.stdout)["streams"][0]["r_frame_rate"]
    num, den = (int(x) for x in raw.split("/"))
    return num / den


class VideoLoader:
    """用 ffmpeg 把整个视频解码到内存（320x180 RGB），按帧顺序输出。

    内部用 uint8 保存整段视频以节省内存，取单帧时才转成 float32。
    """

    def __init__(self, path: str | Path, max_frames: int = 0) -> None:
        self.fps = probe_fps(path)

        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an"]
        if max_frames > 0:
            cmd += ["-frames:v", str(max_frames)]
        cmd += [
            "-vf", f"scale={WIDTH}:{HEIGHT}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 解码失败: {proc.stderr.decode(errors='replace').strip()}"
            )

        n_frames = len(proc.stdout) // FRAME_BYTES
        if n_frames == 0:
            raise RuntimeError(f"视频没有解码到帧: {path}")

        buf = np.frombuffer(proc.stdout, dtype=np.uint8)
        # (T, H, W, 3) uint8 -> (T, 3, H, W) uint8
        frames = buf[: n_frames * FRAME_BYTES].copy().reshape(
            n_frames, HEIGHT, WIDTH, 3
        )
        self.frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()

    def __len__(self) -> int:
        return self.frames.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """返回第 index 帧：(3, 180, 320) float32 ∈ [0,1]。"""
        return self.frames[index].float().div_(255.0)

    def iter_frames(
        self,
        start: int = 0,
        count: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """从 start 开始顺序输出 count 帧；count=None 表示到视频结尾。"""
        end = len(self) if count is None else min(len(self), start + count)
        for i in range(start, end):
            yield self[i]

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self.iter_frames()


@dataclass(frozen=True)
class NoteEvent:
    frame: int
    kind: str          # tap / hold / flick
    x: float
    y: float
    score: float = 1.0


@dataclass(frozen=True)
class InferenceTiming:
    total_ms: float
    per_frame_ms: float
    min_frame_ms: float
    max_frame_ms: float
    frames: int
    chunks: int

    @property
    def fps(self) -> float:
        return 1000.0 / self.per_frame_ms if self.per_frame_ms > 0 else 0.0


@dataclass(frozen=True)
class VideoPrediction:
    notes: list[NoteEvent]
    types: torch.Tensor
    coords: torch.Tensor
    probs: torch.Tensor
    timing: InferenceTiming


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(ckpt: str, device: torch.device) -> Model:
    model = Model()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def collect_gt_notes(labels: list[D.Frame_teacher]) -> list[NoteEvent]:
    """从逐帧标签中提取音符起点。"""
    notes: list[NoteEvent] = []
    for t, frame in enumerate(labels):
        for slot, item in enumerate(frame.slot):
            if item.type == Action_type.TYPE_FLICK:
                kind = "flick"
            elif item.type == Action_type.TYPE_DOWN:
                kind = "hold" if _has_hold_after(labels, t, item) else "tap"
            else:
                continue
            notes.append(NoteEvent(t, kind, item.x, item.y))
    return notes


def _has_hold_after(
    labels: list[D.Frame_teacher],
    frame: int,
    note: D.Coordf_with_type,
) -> bool:
    if frame + 1 >= len(labels):
        return False
    # 槽位会按坐标重排，所以按 x 坐标找下一帧的 hold
    return any(
        item.type == Action_type.TYPE_HOLD and abs(item.x - note.x) <= MATCH_X
        for item in labels[frame + 1].slot
    )


@torch.inference_mode()
def predict_notes(
    model: Model,
    video: VideoLoader,
    seq: int,
    device: torch.device,
) -> VideoPrediction:
    """逐块跑模型，把逐帧类型序列转换成预测音符起点，并统计耗时。"""
    all_types: list[torch.Tensor] = []
    all_coords: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    chunk_frame_ms: list[float] = []
    total_ms = 0.0
    total_frames = 0

    h: torch.Tensor | None = None
    for start in range(0, len(video), seq):
        end = min(start + seq, len(video))
        n_frames = end - start

        # CUDA 是异步执行，必须同步后再计时
        sync_device(device)
        t0 = time.perf_counter()

        chunk = video.frames[start:end].to(device).float().div_(255.0)
        logits, coords, h = model(chunk.unsqueeze(0), h=h)
        types = logits[0].argmax(-1).cpu()
        coords_cpu = coords[0].cpu()
        probs = logits[0].softmax(-1).cpu()

        sync_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        total_ms += elapsed_ms
        total_frames += n_frames
        chunk_frame_ms.append(elapsed_ms / n_frames)
        all_types.append(types)
        all_coords.append(coords_cpu)
        all_probs.append(probs)

    timing = InferenceTiming(
        total_ms=total_ms,
        per_frame_ms=total_ms / total_frames if total_frames else 0.0,
        min_frame_ms=min(chunk_frame_ms) if chunk_frame_ms else 0.0,
        max_frame_ms=max(chunk_frame_ms) if chunk_frame_ms else 0.0,
        frames=total_frames,
        chunks=len(chunk_frame_ms),
    )

    types = torch.cat(all_types)       # (T, n_slots)
    coords = torch.cat(all_coords)     # (T, n_slots, 2)
    probs = torch.cat(all_probs)       # (T, n_slots, n_types)

    notes: list[NoteEvent] = []
    prev = [int(Action_type.TYPE_IDLE)] * types.shape[1]
    for t in range(types.shape[0]):
        for slot in range(types.shape[1]):
            cur = int(types[t, slot])
            old = prev[slot]

            # 一个音符的起点：idle -> 任意动作，或 hold -> down 这种换音符
            if cur == Action_type.TYPE_DOWN:
                onset = old not in (Action_type.TYPE_DOWN, Action_type.TYPE_HOLD)
                kind = "tap"
            elif cur == Action_type.TYPE_HOLD:
                onset = old not in (Action_type.TYPE_HOLD, Action_type.TYPE_DOWN)
                kind = "hold"
            elif cur == Action_type.TYPE_FLICK:
                onset = old != Action_type.TYPE_FLICK
                kind = "flick"
            else:
                onset = False
                kind = ""

            if onset:
                notes.append(NoteEvent(
                    t, kind,
                    float(coords[t, slot, 0]),
                    float(coords[t, slot, 1]),
                ))
            prev[slot] = cur
    return VideoPrediction(notes, types, coords, probs, timing)


def write_predictions(path: str | Path, prediction: VideoPrediction) -> None:
    """把逐帧预测写成 JSONL，格式与 infer_onnx.py 一致。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for t in range(prediction.types.shape[0]):
            events = []
            for slot in range(prediction.types.shape[1]):
                type_id = int(prediction.types[t, slot])
                if type_id == Action_type.TYPE_IDLE:
                    continue

                x = float(prediction.coords[t, slot, 0])
                y = float(prediction.coords[t, slot, 1])
                events.append({
                    "slot": slot,
                    "phase": TYPE_NAME[type_id],
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "px": round((x + 1) / 2 * WIDTH, 1),
                    "py": round((y + 1) / 2 * HEIGHT, 1),
                    "score": round(float(prediction.probs[t, slot, type_id]), 4),
                })

            events.sort(key=lambda e: (e["x"], e["y"], e["phase"]))
            out.write(json.dumps(
                {"frame": t, "events": events},
                ensure_ascii=False,
            ) + "\n")


def match_notes(
    gt_notes: list[NoteEvent],
    pred_notes: list[NoteEvent],
    fps: float,
) -> list[tuple[NoteEvent, NoteEvent | None]]:
    """按时间和 x 坐标把预测音符匹配到标签音符。"""
    max_delta = math.ceil(MATCH_WINDOW * fps)
    by_frame: dict[int, list[int]] = {}
    for i, pred in enumerate(pred_notes):
        by_frame.setdefault(pred.frame, []).append(i)

    used = [False] * len(pred_notes)
    matched: list[tuple[NoteEvent, NoteEvent | None]] = []

    for gt in gt_notes:
        best: tuple[float, int] | None = None
        for frame in range(gt.frame - max_delta, gt.frame + max_delta + 1):
            for i in by_frame.get(frame, []):
                if used[i]:
                    continue
                pred = pred_notes[i]
                if gt.kind == "flick":
                    if pred.kind != "flick":
                        continue
                elif pred.kind == "flick":
                    continue
                dx = abs(pred.x - gt.x)
                if dx > MATCH_X:
                    continue
                dt = abs(pred.frame - gt.frame) / fps
                key = dt + MATCH_TIME_WEIGHT * dx
                if best is None or key < best[0]:
                    best = (key, i)

        if best is None:
            matched.append((gt, None))
        else:
            used[best[1]] = True
            matched.append((gt, pred_notes[best[1]]))
    return matched


def score_matches(
    matches: list[tuple[NoteEvent, NoteEvent | None]],
    fps: float,
) -> dict[str, object]:
    scorer = PhiraScore(num_notes=len(matches))

    for gt, pred in matches:
        if pred is None:
            scorer.commit(Judgement.MISS)
            continue

        offset = (pred.frame - gt.frame) / fps
        judgement = judge_tap(offset)

        if gt.kind == "flick":
            # Flick 只有 Perfect / Miss
            if judgement in (Judgement.PERFECT, Judgement.GOOD):
                scorer.commit(Judgement.PERFECT)
            else:
                scorer.commit(Judgement.MISS)
        elif gt.kind == "hold":
            # Hold 头部只有 Perfect / Good / Miss
            if judgement is Judgement.PERFECT:
                scorer.commit(Judgement.PERFECT)
            elif judgement is Judgement.GOOD:
                scorer.commit(Judgement.GOOD)
            else:
                scorer.commit(Judgement.MISS)
        else:
            scorer.commit(judgement)

    return scorer.result()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="视频推理 + Phira 计分")
    p.add_argument("--ckpt", default="checkpoints/best.pt", help=".pt 权重路径")
    p.add_argument("--seq", type=int, default=C.seq_len, help="每块帧数")
    p.add_argument("--max-frames", type=int, default=0,
                   help="最多读取多少帧，0 表示整段视频")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out", default="pred.jsonl",
                   help="模型输出 JSONL 路径，传空字符串表示不保存")
    return p.parse_args()


def pick_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            print("warning: CUDA 不可用，回退到 CPU", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.seq <= 0:
        raise SystemExit("--seq 必须大于 0")

    total_start = time.perf_counter()

    device = pick_device(args.device)

    t0 = time.perf_counter()
    video = VideoLoader("../map-set-video/钢琴-アイデン貞貞メルトダウン [hd Lv.8] [157].mp4", args.max_frames)
    video_load_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    labels = D.parse_finger_jsonl("../map-set-labels/钢琴-アイデン貞貞メルトダウン_[hd_Lv.8]_[157]-7dc672f7.jsonl", total_frames=len(video))
    labels = labels[:len(video)]
    label_parse_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    model = load_model(args.ckpt, device)
    model_load_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    gt_notes = collect_gt_notes(labels)
    gt_notes_ms = (time.perf_counter() - t0) * 1000.0

    prediction = predict_notes(model, video, args.seq, device)
    pred_notes = prediction.notes
    infer_timing = prediction.timing

    t0 = time.perf_counter()
    matches = match_notes(gt_notes, pred_notes, video.fps)
    match_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    result = score_matches(matches, video.fps)
    score_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    if args.out:
        write_predictions(args.out, prediction)
    save_ms = (time.perf_counter() - t0) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    result["timing"] = {
        "video_load_ms": round(video_load_ms, 2),
        "label_parse_ms": round(label_parse_ms, 2),
        "model_load_ms": round(model_load_ms, 2),
        "gt_notes_ms": round(gt_notes_ms, 2),
        "infer_total_ms": round(infer_timing.total_ms, 2),
        "infer_per_frame_ms": round(infer_timing.per_frame_ms, 3),
        "infer_min_frame_ms": round(infer_timing.min_frame_ms, 3),
        "infer_max_frame_ms": round(infer_timing.max_frame_ms, 3),
        "infer_fps": round(infer_timing.fps, 2),
        "infer_frames": infer_timing.frames,
        "infer_chunks": infer_timing.chunks,
        "match_ms": round(match_ms, 2),
        "score_ms": round(score_ms, 2),
        "save_ms": round(save_ms, 2),
        "total_ms": round(total_ms, 2),
    }

    print(
        f"video={len(video)} frames @ {video.fps:.3f}fps "
        f"gt_notes={len(gt_notes)} pred_notes={len(pred_notes)} "
        f"device={device}",
        file=sys.stderr,
    )
    print(
        f"timing: video_load={video_load_ms:.1f}ms "
        f"label_parse={label_parse_ms:.1f}ms "
        f"model_load={model_load_ms:.1f}ms "
        f"infer={infer_timing.total_ms:.1f}ms "
        f"({infer_timing.per_frame_ms:.3f}ms/frame, "
        f"{infer_timing.fps:.1f}fps, "
        f"min={infer_timing.min_frame_ms:.3f}, "
        f"max={infer_timing.max_frame_ms:.3f}) "
        f"match={match_ms:.1f}ms score={score_ms:.1f}ms "
        f"save={save_ms:.1f}ms total={total_ms:.1f}ms",
        file=sys.stderr,
    )
    if args.out:
        print(f"saved predictions to {args.out}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
