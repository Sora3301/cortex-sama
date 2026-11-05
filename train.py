"""
train.py —— 损失函数、训练/验证循环与程序入口

本文件需要实现的内容：
1. compute_loss(logits, coords, type_label, coord_label, coord_w=2.0, class_weights=None) —— 计算混合损失：
    - 动作分类损失 type_loss:
        使用 F.cross_entropy(logits.reshape(-1, n_types), type_label.reshape(-1), weight=class_weights)
    - 坐标回归损失 coord_loss:
        构造活跃掩码 active_mask = (type_label > 0)，仅对非空置槽位计算 F.mse_loss(coords[active_mask], coord_label[active_mask])；
        若 active_mask.any() 为 False，则 coord_loss 置为 0
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
        f) 累加 total_loss += loss.item() * bs 与样本数 epoch_size += bs，每隔 log_interval 步打印进度
        g) 每轮结束后（可选跑 evaluate 验证）保存模型权重 torch.save(..., "last.pt" / "best.pt")

4. main() —— 组装与启动入口：
    - 调用 dataset.py 中的 discover_recordings 和 build_dataset_index 准备训练/验证数据
    - 实例化 VideoChunkDataset 并用 DataLoader 包装成 train_loader 和 val_loader
    - 实例化 model.py 中的 Model()，调用 train_trunks 开始训练
"""

import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

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
    coord_w: float = 1.0,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_types = logits.size(-1)
    type_loss = F.cross_entropy(
        logits.reshape(-1, n_types),
        type_label.reshape(-1),
        weight = class_weights
    )
    active_mask = type_label > Action_type.TYPE_IDLE.value
    if active_mask.any():
        coord_loss = F.mse_loss(coords[active_mask], coord_label[active_mask])
    else:
        coord_loss = coords.sum() * 0.0
    
    loss = type_loss + coord_w * coord_loss
    return loss, type_loss, coord_loss

def evaluate(
    model: M.Model,
    val_loader: DataLoader[Any],
    coord_w: float = 2.0,
    device: str = C.device,
) -> None:
    val_loss = 0.0
    total_loss = 0.0
    total_type_loss = 0.0
    total_coord_loss = 0.0
    epoch_size = 0
    class_weights = torch.tensor(
        C.class_weight, dtype=torch.float32, device=device
    )
    for warmup, x, type_label, coord_label in val_loader:
        warmup = warmup.to(device, non_blocking = True)
        x = x.to(device, non_blocking = True)
        type_label = type_label.to(device, non_blocking = True)
        coord_label = coord_label.to(device, non_blocking = True)

        logits, coords, h = model(x, warmup)
        loss, _, _ = compute_loss(logits, coords, type_label, coord_label, coord_w, class_weights)

        bs = x.size(0) # batch_size可能不同，每个样本帧数必定相同
        total_loss += loss.item() * bs
        epoch_size += bs

    val_loss = total_loss / epoch_size
    avg_type_loss = total_type_loss / epoch_size
    avg_coord_loss = total_coord_loss / epoch_size
    print(
        f" evaluate:"
        f" average_loss = {val_loss:.4f} type_loss={avg_type_loss:.4f} coord_loss={avg_coord_loss:.4f}"
    )


def train_chunks(
    model: M.Model,
    train_loader: DataLoader[Any],       # 训练集数据加载器
    epochs: int = 20,
    lr: float = 1e-3,
    coord_w: float = C.coord_w,
    device: str = C.device,
    save_dir: str = C.save_dir,
    log_interval: int = 20,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    class_weights = torch.tensor(
        C.class_weight, dtype=torch.float32, device=device
    )
    for i in range(1, epochs + 1):
        step_idx = 0
        total_loss = 0.0
        total_type_loss = 0.0
        total_coord_loss = 0.0
        epoch_size = 0
        for warmup, x, type_label, coord_label in train_loader:
            step_idx += 1
            warmup = warmup.to(device, non_blocking = True)
            x = x.to(device, non_blocking = True)
            type_label = type_label.to(device, non_blocking = True)
            coord_label = coord_label.to(device, non_blocking = True)

            logits, coords, h = model(x, warmup)
            loss, type_loss, coord_loss = compute_loss(logits, coords, type_label, coord_label, coord_w, class_weights)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0) # batch_size可能不同，每个样本帧数必定相同
            total_loss += loss.item() * bs
            total_type_loss += type_loss.item() * bs
            total_coord_loss += coord_loss.item() * bs
            epoch_size += bs

            if log_interval > 0 and step_idx % log_interval == 0:
                avg_loss = total_loss / epoch_size
                avg_type_loss = total_type_loss / epoch_size
                avg_coord_loss = total_coord_loss / epoch_size
                print(
                    f"  [Epoch {i:03d} | Step {step_idx:04d}/{len(train_loader):04d}] "
                    f"  loss={avg_loss:.4f} type_loss={avg_type_loss:.4f} coord_loss={avg_coord_loss:.4f}"
                )
        avg_loss = total_loss / epoch_size
        avg_type_loss = total_type_loss / epoch_size
        avg_coord_loss = total_coord_loss / epoch_size
        print(
            f" [Epoch {i:03d} / {epochs:03d}]"
            f" average_loss = {avg_loss:.4f} type_loss={avg_type_loss:.4f} coord_loss={avg_coord_loss:.4f}"
        )

@dataclass
class DataPair:
    label_name: str | None = None
    vedio_name: str | None = None
    num_frames: int | None = None
    max_slots: int | None = None


ChunkSet = tuple[list[str], list[int], list[list[D.Frame_teacher]]]


def find_maps() -> dict[int, DataPair]:
    dataPairList: dict[int, DataPair] = {}

    # manifest 只读一次，补齐帧数和最大槽位数
    with (Path(C.label_path) / "manifest.json").open(encoding="utf-8") as F:
        manifest = json.load(F)
        info = {}
        for obj in manifest:
            info[obj["folder"]] = obj

    for p in Path(C.label_path).rglob("*.jsonl"):
        name = p.name
        i = name.rindex("_[") + len("_[")   # 跳过起始标记
        j = name.index("]-", i)              # 从 i 之后找结束标记
        pair = dataPairList.setdefault(int(name[i:j]), DataPair())
        pair.label_name = name

    for p in Path(C.vedio_path).rglob("*.mp4"):
        name = p.name
        i = name.rfind("[") + 1
        j = name.rfind("]")
        pair = dataPairList.setdefault(int(name[i:j]), DataPair())
        pair.vedio_name = name
        meta = info.get(Path(name).stem)
        if meta is not None:
            pair.num_frames = int(meta["num_frames"])
            pair.max_slots = int(meta["max_slots"])
    # 检查
    result = {
        key: pair
        for key, pair in dataPairList.items()
        if pair.label_name and 
        pair.vedio_name and pair.num_frames
    }
    print(f"找到 {len(result)} 个谱面")
    print(f"跳过 {len(dataPairList) - len(result)} 个缺少 label/video/manifest 的谱面")
    return result


def build_chunks(
    dataPairs: dict[int, DataPair],
    seed: int,
    m_train: int,
    n_test: int,
) -> tuple[ChunkSet, ChunkSet]:
    """遍历所有谱面：前 m_train 个进训练集，最后 n_test 个进测试集，其余跳过。"""
    items = sorted(dataPairs.items())
    total = len(items)

    train: ChunkSet = ([], [], [])
    test: ChunkSet = ([], [], [])
    rng_train = random.Random(seed)
    rng_test = random.Random(seed + 1)

    for idx, (_, item) in enumerate(items):
        if idx < m_train:
            names, starts, labels = train
            rng = rng_train
        elif idx >= total - n_test:
            names, starts, labels = test
            rng = rng_test
        else:
            continue

        if item.vedio_name is None or item.label_name is None:
            continue
        frame_num = item.num_frames
        if frame_num is None or frame_num <= 0:
            print(f"跳过 {item.vedio_name}: manifest 缺少 num_frames")
            continue

        frame_label: list[D.Frame_teacher] = D.parse_finger_jsonl(
            str(Path(C.label_path) / item.label_name),
            frame_num,
        )

        for frame in range(C.warmup_len, frame_num - C.seq_len + 1, C.seq_len):
            chunk_max_slot = D.count_max_slot(frame_label, frame)
            if chunk_max_slot > C.n_slot:
                continue
            if chunk_max_slot == 0 and rng.random() >= C.p_empty_chunk:
                continue
            names.append(item.vedio_name)
            starts.append(frame)
            labels.append(frame_label)
    return train, test


def main() -> None:
    dataPairs: dict[int, DataPair] = find_maps()

    # 第一版：取 id 最小的 80 个谱面训练，id 最大的 20 个谱面测试
    (train_names, 
    train_starts, 
    train_labels
    ), (val_names, 
    val_starts, 
    val_labels,
    ) = build_chunks(
        dataPairs,
        seed=20261104,
        m_train=C.n_train,
        n_test=C.n_test,
    )
    print(
        f"训练谱面/样本: {len(set(train_names))}/{len(train_starts)} | "
        f"测试谱面/样本: {len(set(val_names))}/{len(val_starts)}"
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
    model = M.Model()
    train_chunks(
        model,
        train_loader,
        epochs = 20,
        lr = 1e-3,
        coord_w = C.coord_w,
        device = C.device,
        save_dir="checkpoints",
        log_interval=20,
    )
    evaluate(
        model,
        val_loader,
        coord_w=C.coord_w,
        device=C.device,
    )
    return

if __name__ == "__main__":
    main()
