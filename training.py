import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from decord import VideoReader, cpu


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
    def __init__(self, in_ch=256, hidden=128, gru_hidden=256, n_slots=4, n_types=3):
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
        self.type_head = nn.Linear(hidden, n_slots * n_types)  # 每槽位 type logits
        self.coord_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),nn.Tanh()
        )
        self.type_emb = nn.Embedding(n_types, hidden)   # type 种类 -> 向量
        self.slot_emb = nn.Embedding(n_slots, hidden)   # 槽位身份 -> 向量

    def _predict(self, h, device):
        B = h.size(0)
        logits = self.type_head(h).view(B, self.n_slots, self.n_types)  # (B,4,n_types)

        # 软 embedding：用 type 概率加权 embedding 表
        w = logits.softmax(-1)                                        # (B,4,n_types)
        slot = torch.arange(self.n_slots, device=device).expand(B, -1)
        cond = w @ self.type_emb.weight + self.slot_emb(slot)         # (B,4,hidden)

        feat = h.unsqueeze(1).expand(-1, self.n_slots, -1)            # (B,4,hidden)
        coords = self.coord_head(torch.cat([feat, cond], dim=-1))  # (B,4,2)
        return logits, coords

    def forward(self, x, B, T, h=None):                 # x: (B*T, 256, H, W), h: (num_layers,B,gru_hidden)
        g = F.adaptive_max_pool2d(x, 1).flatten(1)      # (B*T, 256) 全局 map
        g = g.reshape(B, T, -1)                         # (B, T, 256) 按时间排好
        out, h = self.gru(g, h)                         # (B, T, gru_hidden)，h 传入传出实现跨块续接
        f = self.trunk(out.reshape(B * T, -1))          # (B*T, hidden) 每帧都过 trunk
        logits, coords = self._predict(f, x.device)     # (B*T,4,n_types), (B*T,4,2)
        logits = logits.view(B, T, self.n_slots, self.n_types)   # (B,T,4,n_types)
        coords = coords.view(B, T, self.n_slots, 2)              # (B,T,4,2)
        return logits, coords, h
    '''
    def step(self, x, h=None):                          # x: (B, 256, H, W) 单帧
        B = x.size(0)
        if h is None:                                   # 序列开头, h 置零
            h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=x.device)
        g = F.adaptive_max_pool2d(x, 1).flatten(1)      # (B, 256)
        out, h = self.gru(g.unsqueeze(1), h)            # (B,1,gru_hidden), 新 h
        logits, coords = self._predict(self.trunk(out[:, 0]), x.device)
        return logits, coords, h                         # 把新 h 传给下一步
    '''


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = FingerHead()

    def forward(self, x, h=None):               # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))      # (B*T, 256, h, w)
        return self.head(f, B, T, h)


def train(model, loader, epochs=50, lr=1e-3, coord_w=1.0, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_types = model.head.n_types

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, type_label, coord_label in loader:
            x = x.to(device)                        # (B, T, 3, H, W)
            type_label = type_label.to(device)      # (B, T, 4) long，每帧每槽位
            coord_label = coord_label.to(device)    # (B, T, 4, 2) float, 归一化到 [-1,1]

            logits, coords, _ = model(x)            # (B,T,4,n_types), (B,T,4,2)
            loss_type = F.cross_entropy(
                logits.reshape(-1, n_types), type_label.reshape(-1)
            )
            loss_coord = F.mse_loss(coords, coord_label)
            loss = loss_type + coord_w * loss_coord

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)

        print(f"epoch {epoch:3d}  loss {total / len(loader.dataset):.4f}")

class MapDataset(Dataset):
    def __init__(self, x, type_label, coord_label):
        self.x = x
        self.type_label = type_label
        self.coord_label = coord_label

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.type_label[index], self.coord_label[index]
    
class MapDataloader:
    @staticmethod
    def loadVedio(path, ctx=None):
        vr = VideoReader(path, ctx=ctx or cpu(0))
        return vr[:].asnumpy()                  # (N,H,W,3) uint8 RGB，整片进内存需谨慎

class VideoChunkDataset(Dataset):
    """滑窗样本：每条 = warmup 帧(拿状态,不算 loss) + target T 帧(算 loss)。

    paths:        视频路径列表
    type_labels:  每个视频的逐帧 type 标签列表，元素形状 (Nv, n_slots) long
    coord_labels: 每个视频的逐帧 coord 标签列表，元素形状 (Nv, n_slots, 2) float，[-1,1]
    starts:       [(video_idx, start_frame), ...] 目标块起点
    """

    def __init__(self, paths, type_labels, coord_labels, starts, T=16, warmup=16, ctx=None):
        self.paths = paths
        self.type_labels = type_labels
        self.coord_labels = coord_labels
        self.starts = starts
        self.target_len = T
        self.warmup_len = warmup
        self.ctx = ctx
        self._reader_cache = {}

    def __len__(self):
        return len(self.starts)

    def _reader(self, path):
        reader = self._reader_cache.get(path)
        if reader is None:
            reader = VideoReader(path, ctx=self.ctx or cpu(0))
            self._reader_cache = {path: reader}   # 只缓存最近一个，控内存
        return reader

    def __getitem__(self, index):
        video_index, start_frame = self.starts[index]
        reader = self._reader(self.paths[video_index])
        total_frames = len(reader)

        warmup_start = max(0, start_frame - self.warmup_len)
        target_end = min(start_frame + self.target_len, total_frames)

        clip = reader.get_batch(list(range(warmup_start, target_end))).asnumpy()
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float().div_(255)

        warmup_frames = clip[: start_frame - warmup_start]
        target_frames = clip[start_frame - warmup_start :]

        if warmup_frames.size(0) < self.warmup_len:      # 视频开头：重复首帧补足 warmup
            first_frame = warmup_frames[:1] if warmup_frames.size(0) > 0 else target_frames[:1]
            padding = first_frame.expand(self.warmup_len - warmup_frames.size(0), -1, -1, -1)
            warmup_frames = torch.cat([padding, warmup_frames])

        if target_frames.size(0) < self.target_len:      # 视频结尾：空白帧补足 T
            height, width = target_frames.shape[-2], target_frames.shape[-1]
            blank = target_frames.new_zeros(
                self.target_len - target_frames.size(0), 3, height, width
            )
            target_frames = torch.cat([target_frames, blank])

        type_label = torch.as_tensor(
            self.type_labels[video_index][start_frame:start_frame + self.target_len]
        ).long()
        coord_label = torch.as_tensor(
            self.coord_labels[video_index][start_frame:start_frame + self.target_len],
            dtype=torch.float32,
        )
        return warmup_frames, target_frames, type_label, coord_label

def train_chunks(model, loader, epochs=50, lr=1e-3, coord_w=1.0, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_types = model.head.n_types

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for warmup, x, type_label, coord_label in loader:
            warmup = warmup.to(device)              # (B, W, 3, H, W)
            x = x.to(device)                        # (B, T, 3, H, W)
            type_label = type_label.to(device)      # (B, T, 4) long
            coord_label = coord_label.to(device)    # (B, T, 4, 2) float

            h = None
            if warmup.size(1) > 0:                  # 热身只为拿状态, 不建图
                with torch.no_grad():
                    _, _, h = model(warmup)

            logits, coords, _ = model(x, h)         # 从 warmup 状态出发
            loss_type = F.cross_entropy(
                logits.reshape(-1, n_types), type_label.reshape(-1)
            )
            loss_coord = F.mse_loss(coords, coord_label)
            loss = loss_type + coord_w * loss_coord

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)

        print(f"epoch {epoch:3d}  loss {total / len(loader.dataset):.4f}")

@torch.no_grad()
def stream_infer(model, chunks, device="cpu"):
    """流式推理：h 跨块传递，每帧都出预测。chunks 为按序切好的 (B,T,3,H,W)。"""
    model.eval()
    h = None
    type_out, coord_out = [], []
    for x in chunks:
        logits, coords, h = model(x.to(device), h)
        type_out.append(logits.argmax(-1).cpu())
        coord_out.append(coords.cpu())
    return torch.cat(type_out, dim=1), torch.cat(coord_out, dim=1)

if __name__ == "__main__":
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(1, 12, 3, 64, 64)               # 完整序列

    with torch.no_grad():
        full_logits, full_coords, _ = model(x)      # 整段一次算
        warm, tgt = x[:, :4], x[:, 4:8]             # warmup 4 帧, target 4 帧
        _, _, h = model(warm)                       # 拿状态
        seg_logits, seg_coords, _ = model(tgt, h)   # 从状态续接

    print("coord 状态续接==整段:", torch.allclose(seg_coords, full_coords[:, 4:8], atol=1e-5))
    print("type  状态续接==整段:", (seg_logits.argmax(-1) == full_logits[:, 4:8].argmax(-1)).all().item())

    