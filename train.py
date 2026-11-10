"""
train.py —— 损失函数、训练/验证循环与程序入口

本文件需要实现的内容：
1. compute_loss(logits, coords, type_label, coord_label, coord_w=2.0, class_weights=None) —— 计算混合损失：
    - 动作分类损失 type_loss:
        使用 F.cross_entropy(logits.reshape(-1, n_types), type_label.reshape(-1), weight=class_weights)
    - 坐标回归损失 coord_loss:
        构造活跃掩码 active_mask = (type_label > 0)，仅对非空置槽位计算 masked MSE；
        若没有活跃槽位，则 coord_loss 置为 0
    - 返回总损失 loss = type_loss + coord_w * coord_loss（以及分项 loss 便于打印日志）

2. evaluate(model, val_loader, coord_w=2.0, class_weights=None, device="cpu") —— 验证集评估（可选但推荐）：
    - model.eval() + @torch.no_grad()
    - 遍历 val_loader，统计验证集平均 loss、分类准确率 (logits.argmax(-1) == type_label) 和活跃槽位坐标误差
    - 返回验证指标字典

3. train_chunks(model, train_loader, val_loader=None, ...) —— 核心训练循环：
    - 将 model 和 class_weights 移至 device，初始化 AdamW 优化器（及可选的学习率调度器 / GradScaler）
    - 外层循环遍历 range(1, epochs + 1)：
        a) model.train()
        b) 内层循环遍历 train_loader 取出 (warmup, x, type_label, coord_label) 并移至 device
        c) 前向传播: logits, coords, _ = model(x, warmup=warmup)
        d) 计算损失: loss = compute_loss(...)
        e) 反向传播与参数更新: opt.zero_grad() -> loss.backward() -> opt.step()
        f) 在 GPU 上累加 total_type_loss / total_coord_loss，每隔 log_interval 步再转 Python 打印进度
        g) 每轮结束后（可选跑 evaluate 验证）保存模型权重 torch.save(..., "last.pt" / "best.pt")

4. main() —— 组装与启动入口：
    - 调用 dataset.py 中的 discover_recordings 和 build_dataset_index 准备训练/验证数据
    - 实例化 VideoChunkDataset 并用 DataLoader 包装成 train_loader 和 val_loader
    - 实例化 model.py 中的 Model()，调用 train_chunks 开始训练
"""

import torch
import torch.nn.functional as F
import torch.optim, torch.cuda
from torch.utils.data import DataLoader

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config as C
import model as M
import dataset as D

from config import Action_type as Action_type

# 损失函数
# 如果id持续绑定槽位，可能需要匹配手指的编号（未实现）
def compute_loss(
    logits: torch.Tensor,
    coords: torch.Tensor,
    type_label: torch.Tensor,
    coord_label: torch.Tensor,
    coord_w: float = C.coord_w,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_types = logits.size(-1)
    type_loss = F.cross_entropy(
        logits.reshape(-1, n_types),
        type_label.reshape(-1),
        weight = class_weights
    )
    active_mask = type_label > Action_type.TYPE_IDLE.value
    mask = active_mask.unsqueeze(-1)
    diff = torch.where(mask, coords - coord_label, torch.zeros_like(coords))
    coord_num = active_mask.sum()
    coord_loss = diff.pow(2).sum() / (2 * coord_num).clamp(min=1)

    loss = type_loss + coord_w * coord_loss
    return loss, type_loss, coord_loss, coord_num

@torch.no_grad()
def evaluate(
    model: M.Model,
    val_loader: DataLoader[Any],
    coord_w: float = 2.0,
    device: str = C.device,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    if class_weights is None:
        class_weights = torch.tensor(
            C.class_weight, dtype=torch.float32, device=device
        )
    model.eval()
    total_type_loss = torch.zeros((), device=device)
    total_coord_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), dtype=torch.long, device=device)
    total_slots = 0
    total_coord_err = torch.zeros((), device=device)
    total_active_coords = torch.zeros((), dtype=torch.long, device=device)
    epoch_size = 0
    total_coord_num = torch.zeros((), dtype=torch.long, device=device)
    for warmup, x, type_label, coord_label in val_loader:
        warmup = warmup.to(device, non_blocking = True)
        x = x.to(device, non_blocking = True)
        type_label = type_label.to(device, non_blocking = True)
        coord_label = coord_label.to(device, non_blocking = True)

        logits, coords, h = model(x, warmup)
        loss, type_loss, coord_loss, coord_num = compute_loss(
            logits, coords, type_label, coord_label, coord_w, class_weights
        )

        bs = x.size(0) # batch_size可能不同，每个样本帧数必定相同
        total_type_loss += type_loss.detach() * bs
        total_coord_loss += coord_loss.detach() * coord_num
        epoch_size += bs
        total_coord_num += coord_num

        total_correct += (logits.argmax(-1) == type_label).sum()
        total_slots += type_label.numel()

        active_mask = type_label > Action_type.TYPE_IDLE.value
        mask = active_mask.unsqueeze(-1)
        coord_err = torch.where(
            mask, (coords - coord_label).abs(), torch.zeros_like(coords)
        )
        total_coord_err += coord_err.sum()
        total_active_coords += coord_num * coords.size(-1)

    type_loss_t = total_type_loss / max(1, epoch_size)
    coord_loss_t = total_coord_loss / total_coord_num.clamp(min=1)
    loss_t = type_loss_t + coord_w * coord_loss_t
    acc_t = total_correct / total_slots if total_slots else total_correct * 0.0
    coord_mae_t = total_coord_err / total_active_coords.clamp(min=1)
    loss, type_loss, coord_loss, acc, coord_mae = torch.stack(
        [loss_t, type_loss_t, coord_loss_t, acc_t, coord_mae_t]
    ).tolist()
    metrics = {
        "loss": loss,
        "type_loss": type_loss,
        "coord_loss": coord_loss,
        "acc": acc,
        "coord_mae": coord_mae,
    }
    print(
        f" evaluate:"
        f" average_loss = {metrics['loss']:.4f}"
        f" type_loss={metrics['type_loss']:.4f}"
        f" coord_loss={metrics['coord_loss']:.4f}"
        f" acc={metrics['acc']:.4f}"
        f" coord_mae={metrics['coord_mae']:.4f}"
    )
    return metrics


def train_chunks(
    model: M.Model,
    train_loader: DataLoader[Any],       # 训练集数据加载器
    val_loader: DataLoader[Any] | None = None,  # 验证集数据加载器（可选）
    epochs: int = 20,
    lr: float = 1e-3,
    lr_min_ratio: float = 0.05,   # 余弦退火终点 lr = lr * lr_min_ratio
    coord_w: float = C.coord_w,
    device: str = C.device,
    save_dir: str = C.save_dir,
    log_interval: int = 20,
) -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 余弦退火：lr 随 epoch 从 lr 平滑衰减到 lr * lr_min_ratio
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * lr_min_ratio
    )
    class_weights = torch.tensor(
        C.class_weight, dtype=torch.float32, device=device
    )
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_file = save_path / "train_log.csv"
    log_header = [
        "epoch", "train_loss", "train_type_loss", "train_coord_loss",
        "val_loss", "val_type_loss", "val_coord_loss", "val_acc", "val_coord_mae",
        "is_best", "lr",
    ]
    log_is_new = not log_file.exists() or log_file.stat().st_size == 0
    log_f = log_file.open("a", encoding="utf-8", newline="")
    writer = csv.writer(log_f)
    if log_is_new:
        writer.writerow(log_header)
    print(f"  training log: {log_file}")
    best_val_loss = float("inf")
    for i in range(1, epochs + 1):
        model.train()
        lr_now = opt.param_groups[0]["lr"]   # 本轮实际使用的学习率
        step_idx = 0
        total_type_loss = torch.zeros((), device=device)
        total_coord_loss = torch.zeros((), device=device)
        epoch_size = 0
        total_coord_num = torch.zeros((), dtype=torch.long, device=device)
        for warmup, x, type_label, coord_label in train_loader:
            step_idx += 1
            warmup = warmup.to(device, non_blocking = True)
            x = x.to(device, non_blocking = True)
            type_label = type_label.to(device, non_blocking = True)
            coord_label = coord_label.to(device, non_blocking = True)

            logits, coords, h = model(x, warmup)
            loss, type_loss, coord_loss, coord_num = compute_loss(logits, coords, type_label, coord_label, coord_w, class_weights)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0) # batch_size可能不同，每个样本帧数必定相同
            total_type_loss += type_loss.detach() * bs
            total_coord_loss += coord_loss.detach() * coord_num
            total_coord_num += coord_num
            epoch_size += bs

            if log_interval > 0 and step_idx % log_interval == 0:
                avg_type_loss, avg_coord_loss = torch.stack([
                    total_type_loss / max(1, epoch_size),
                    total_coord_loss / total_coord_num.clamp(min=1),
                ]).tolist()
                avg_loss = avg_type_loss + coord_w * avg_coord_loss
                print(
                    f"  [Epoch {i:03d} | Step {step_idx:04d}/{len(train_loader):04d}] "
                    f"  loss={avg_loss:.4f} type_loss={avg_type_loss:.4f} coord_loss={avg_coord_loss:.4f}"
                )
        avg_type_loss, avg_coord_loss = torch.stack([
            total_type_loss / max(1, epoch_size),
            total_coord_loss / total_coord_num.clamp(min=1),
        ]).tolist()
        avg_loss = avg_type_loss + coord_w * avg_coord_loss
        print(
            f" [Epoch {i:03d} / {epochs:03d}]"
            f" average_loss = {avg_loss:.4f} type_loss={avg_type_loss:.4f} coord_loss={avg_coord_loss:.4f}"
            f" lr={lr_now:.2e}"
        )
        scheduler.step()   # 本轮 opt.step 全部完成后更新学习率

        # 每轮保存权重快照（按 epoch 命名），last.pt 指向最新一轮；有验证集时另存 best.pt
        torch.save(model.state_dict(), save_path / f"epoch_{i:03d}.pt")
        torch.save(model.state_dict(), save_path / "last.pt")
        print(f"  saved epoch_{i:03d}.pt")
        metrics: dict[str, float] | None = None
        is_best = False
        if val_loader is not None:
            metrics = evaluate(
                model, val_loader, coord_w, device, class_weights
            )
            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
                is_best = True
                torch.save(model.state_dict(), save_path / "best.pt")
                print(
                    f"  saved best: {save_path / 'best.pt'}"
                    f" (val_loss={best_val_loss:.4f})"
                )
            model.train()   # evaluate 内部切到了 eval，这里切回训练模式

        # 每轮结束把 train/val 指标写入 CSV 日志
        row: list[object] = [i, avg_loss, avg_type_loss, avg_coord_loss]
        if metrics is not None:
            row += [
                metrics["loss"], metrics["type_loss"], metrics["coord_loss"],
                metrics["acc"], metrics["coord_mae"],
            ]
        else:
            row += ["", "", "", "", ""]
        row.append(int(is_best))
        row.append(lr_now)
        writer.writerow(row)
        log_f.flush()   # 每轮落盘，中断也不丢历史

    log_f.close()

@dataclass
class DataPair:
    label_name: str | None = None
    video_name: str | None = None
    num_frames: int | None = None
    max_slots: int | None = None


ChunkSet = tuple[list[str], list[int], list[list[D.Frame_teacher]]]


def find_maps() -> dict[int, DataPair]:
    dataPairList: dict[int, DataPair] = {}

    with (Path(C.label_path) / "manifest.json").open(encoding="utf-8") as F:
        manifest = json.load(F)
        info = {}
        for obj in manifest:
            info[obj["folder"]] = obj

    for p in Path(C.label_path).rglob("*.jsonl"):
        name = p.name
        i = name.rindex("_[") + len("_[")   
        j = name.index("]-", i)              
        pair = dataPairList.setdefault(int(name[i:j]), DataPair())
        pair.label_name = name

    for p in Path(C.video_path).rglob("*.mp4"):
        name = p.name
        i = name.rfind("[") + 1
        j = name.rfind("]")
        pair = dataPairList.setdefault(int(name[i:j]), DataPair())
        pair.video_name = name
        meta = info.get(Path(name).stem)
        if meta is not None:
            pair.num_frames = int(meta["num_frames"])
            pair.max_slots = int(meta["max_slots"])
    # 检查
    result = {
        key: pair
        for key, pair in dataPairList.items()
        if pair.label_name and 
        pair.video_name and pair.num_frames
    }
    print(f"找到 {len(result)} 个谱面")
    print(f"跳过 {len(dataPairList) - len(result)} 个缺少 label/video/manifest 的谱面")
    return result


def build_chunks(
    dataPairs: dict[int, DataPair],
    seed: int,
    m_train: int,
    n_val: int,
) -> tuple[ChunkSet, ChunkSet]:
    """遍历所有谱面：前 m_train 个进训练集，最后 n_val 个进验证集，其余跳过。"""
    items = sorted(dataPairs.items())
    total = len(items)

    train: ChunkSet = ([], [], [])
    val: ChunkSet = ([], [], [])
    rng_train = random.Random(seed)

    for idx, (_, item) in enumerate(items):
        is_train = idx < m_train
        if is_train:
            names, starts, labels = train
        elif idx >= total - n_val:
            names, starts, labels = val
        else:
            continue

        if item.video_name is None or item.label_name is None:
            continue
        frame_num = item.num_frames
        if frame_num is None or frame_num <= 0:
            print(f"跳过 {item.video_name}: manifest 缺少 num_frames")
            continue

        frame_label: list[D.Frame_teacher] = D.parse_finger_jsonl(
            str(Path(C.label_path) / item.label_name),
            frame_num,
        )

        for frame in range(C.warmup_len, frame_num - C.seq_len + 1, C.seq_len):
            chunk_max_slot = D.count_max_slot(frame_label, frame)
            if chunk_max_slot > C.n_slot:
                continue
            if is_train and chunk_max_slot == 0 and rng_train.random() >= C.p_empty_chunk:
                continue
            names.append(item.video_name)
            starts.append(frame)
            labels.append(frame_label)
    return train, val

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(C.seed)
    dataPairs: dict[int, DataPair] = find_maps()

    (train_names, 
    train_starts, 
    train_labels
    ), (val_names, 
    val_starts, 
    val_labels,
    ) = build_chunks(
        dataPairs,
        seed=C.seed,
        m_train=C.n_train,
        n_val=C.n_val,
    )
    print(
        f"训练谱面/样本: {len(set(train_names))}/{len(train_starts)} | "
        f"验证谱面/样本: {len(set(val_names))}/{len(val_starts)}"
    )

    train_dataset = D.VideoChunkDataset(
        train_names,
        train_labels,
        train_starts,
        C.seq_len,
        C.warmup_len,
    )
    val_dataset = D.VideoChunkDataset(
        val_names,
        val_labels,
        val_starts,
        C.seq_len,
        C.warmup_len,
    )

    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=C.batch_size,
        shuffle=True,
        num_workers=C.num_threads,
        pin_memory=C.pin_memory

    )
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset,
        batch_size=C.batch_size,
        shuffle=False,
        num_workers=C.num_threads,
        pin_memory=C.pin_memory
    )
    # val 只用于训练期选择 best.pt；最终测试直接用真实场景的连续视频推理，
    # 不复用这里的 val 结果作为最终指标。
    model = M.Model()
    train_chunks(
        model,
        train_loader,
        val_loader,
        epochs = C.epochs,
        lr = C.lr,
        coord_w = C.coord_w,
        device = C.device,
        save_dir=C.save_dir,
        log_interval=C.log_interval,
    )
    return

if __name__ == "__main__":
    main()
