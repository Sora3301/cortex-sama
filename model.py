"""
model.py —— 视觉特征提取与多指时序预测模型

本文件需要实现的内容：
1. 动作类型常量定义（可选，也可放在公共处）：
   - TYPE_IDLE  = 0  # 空置
   - TYPE_DOWN  = 1  # 新按下 (click / drag / hold 起始帧)
   - TYPE_HOLD  = 2  # 持续按住或移动 (hold / drag 后续帧)
   - TYPE_FLICK = 3  # 划动音符起始帧 (flick)

2. CNNBackbone(nn.Module) —— 单帧视觉特征提取主干：
   - 输入: (B*T, 3, H, W) 归一化到 [0, 1] 的 RGB 图像张量
   - 结构: 5 层 stride=2 的 Conv2d + ReLU (3 -> 32 -> 64 -> 128 -> 128 -> 256)
   - 输出: (B*T, 256, h, w) 特征图

3. FingerHead(nn.Module) —— GRU 时序记忆 + 多指槽位预测头：
   - 输入: CNN 特征 x (B*T, 256, h, w)、批大小 B、时间步 T、可选隐藏状态 h (1, B, gru_hidden)
   - 处理流程:
     a) 空间池化压平为 (B, T, 256)，送入单向 GRU 得到时序特征 out 和新隐藏状态 h
     b) 经过 trunk (MLP) 提取每帧隐藏表示 f (B*T, hidden)
     c) type_head 预测每个槽位的动作类别打分 logits: (B, T, n_slots, n_types)
     d) 用 logits.softmax(-1) 加权 type_emb，加上 slot_emb，与 f 拼接后经 coord_head + Tanh
        预测每个槽位的归一化坐标 coords: (B, T, n_slots, 2) ∈ [-1, 1]
   - 输出: (logits, coords, h)

4. Model(nn.Module) —— 完整模型封装：
   - 组合 CNNBackbone 与 FingerHead
   - forward(x, warmup=None, h=None):
     a) 若传入 h，直接接着 h 跑 _forward(x, h)（用于流式推理）
     b) 若传入 warmup，先在 torch.no_grad() 下跑 _forward(warmup) 拿到预热状态 h，再跑 _forward(x, h)
     c) 支持 5 维序列输入 (B, T, 3, H, W) 及 4 维单帧输入 (B, 3, H, W)
"""

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


