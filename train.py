import torch
import torch.optim


def compute_loss(logits, coords, type_label, coord_label):
    pass

def train_trunks(
    model,
    train_loader,       # 训练集数据加载器
    val_loader=None,    # 验证集数据加载器
    epochs=20,
    lr=1e-3,
    coord_w=2.0,
    device="cpu",
    save_dir="checkpoints",
    log_interval=20,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for i in (0, epochs):
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
            loss = compute_loss(logits, coords, type_label, coord_label)

            opt.zero_grad()
            # pyrefly: ignore[missing-attribute]
            loss.backward()
            opt.step()

            bs = x.size(0) ##batch_size可能不同，每个样本帧数必定相同
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




def main():
    return

if __name__ == "__main__":
    main()
