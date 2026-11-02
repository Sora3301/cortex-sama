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

from typing import Any

import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

import config as C
import model as M

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
    device: str = "cpu",
) -> None:
    val_loss = 0.0
    total_loss = 0.0
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
    print(
        f" evaluate:"
        f" val_loss = {val_loss:.4f}"
    )


def train_chunks(
    model: M.Model,
    train_loader: DataLoader[Any],       # 训练集数据加载器
    epochs: int = 20,
    lr: float = 1e-3,
    coord_w: float = C.coord_w,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    log_interval: int = 20,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    class_weights = torch.tensor(
        C.class_weight, dtype=torch.float32, device=device
    )
    for i in range(1, epochs + 1):
        step_idx = 0
        total_loss = 0.0
        epoch_size = 0
        for warmup, x, type_label, coord_label in train_loader:
            step_idx += 1
            warmup = warmup.to(device, non_blocking = True)
            x = x.to(device, non_blocking = True)
            type_label = type_label.to(device, non_blocking = True)
            coord_label = coord_label.to(device, non_blocking = True)

            logits, coords, h = model(x, warmup)
            loss, _, _ = compute_loss(logits, coords, type_label, coord_label, coord_w, class_weights)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0) # batch_size可能不同，每个样本帧数必定相同
            total_loss += loss.item() * bs
            epoch_size += bs

            if log_interval > 0 and step_idx % log_interval == 0:
                avg_loss = total_loss / epoch_size
                print(
                    f"  [Epoch {i:03d} | Step {step_idx:04d}/{len(train_loader):04d}] "
                    f"  loss={avg_loss:.4f} "
                )
        avg_loss = total_loss / epoch_size
        print(
            f" [Epoch {i:03d} / {epochs:03d}]"
            f" average_loss = {avg_loss:.4f}"
        )




def main() -> None:
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
        shuffle=True,
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
        device = 'cuda',
        save_dir="checkpoints",
        log_interval=20,
    )
    evaluate(
        model,
        val_loader,
        coord_w=C.coord_w,
        device='cuda',
    )
    return

if __name__ == "__main__":
    main()
