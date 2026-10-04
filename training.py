import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(hidden, 2),
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
        coords = torch.tanh(self.coord_head(torch.cat([feat, cond], dim=-1)))  # (B,4,2)
        return logits, coords

    def forward(self, x, B, T):                         # x: (B*T, 256, H, W)
        g = F.adaptive_max_pool2d(x, 1).flatten(1)      # (B*T, 256) 全局 map
        g = g.view(B, T, -1)                            # (B, T, 256) 按时间排好
        out, _ = self.gru(g)                            # (B, T, gru_hidden) 每帧都有输出
        h = self.trunk(out.reshape(B * T, -1))          # (B*T, hidden) 每帧都过 trunk
        logits, coords = self._predict(h, x.device)     # (B*T,4,n_types), (B*T,4,2)
        logits = logits.view(B, T, self.n_slots, self.n_types)   # (B,T,4,n_types)
        coords = coords.view(B, T, self.n_slots, 2)              # (B,T,4,2)
        return logits, coords
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

    def forward(self, x):                       # x: (B, T, 3, H, W)
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))      # (B*T, 256, h, w)
        return self.head(f, B, T)


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

            logits, coords = model(x)               # (B,T,4,n_types), (B,T,4,2)
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