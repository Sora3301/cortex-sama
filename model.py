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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = FingerHead()

    def _forward(self, x, h=None):               # x: (B, T, 3, H, W)
        if x.dim() == 4:                        # 若传入单帧 (B, 3, H, W)，自动扩成 T=1
            x = x.unsqueeze(1)
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))      # (B*T, 256, h, w)
        return self.head(f, B, T, h)

    def forward(self, x, warmup=None, h=None):
        if h is not None:
            return self._forward(x, h) 
        if warmup is not None and warmup.size(1) > 0:
            with torch.no_grad():               # 免去 warmup 阶段 CNN/GRU 的计算图显存
                _, _, h = self._forward(warmup) # 不要传 self
        return self._forward(x, h) 


