"""model.py —— 视觉特征提取、时序建模与多指预测模型。

结构：
    F5 -> TypeHead -> type logits + 全局时序上下文 ctx
    F4 + F5 -> FPN -> CausalTCN -> FiLM(ctx) -> HeatmapHead -> soft-argmax -> coords

说明：
    - type 分支使用语义更强的 F5，并由 GRU 提供时序上下文
    - coord 分支使用 FPN 融合后的空间特征，由 CausalTCN 提供时序信息
    - heatmap 输出每个 slot 的空间概率图，soft-argmax 得到归一化坐标
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class CNNBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (f4, f5)，形状分别为 (N,128,12,20) 和 (N,256,6,10)。"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        f4 = F.relu(self.conv4(x))
        f5 = F.relu(self.conv5(f4))
        return f4, f5


class FPNFusion(nn.Module):
    """把 F4 的细节和 F5 的语义融合成 12x20 的空间特征图。"""

    def __init__(self, c4: int = 128, c5: int = 256, out_ch: int = 256) -> None:
        super().__init__()
        self.lat4 = nn.Conv2d(c4, out_ch, kernel_size=1)
        self.lat5 = nn.Conv2d(c5, out_ch, kernel_size=1)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, f4: torch.Tensor, f5: torch.Tensor) -> torch.Tensor:
        p4 = self.lat4(f4)
        p5 = self.lat5(f5)
        p5 = F.interpolate(
            p5,
            size=p4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.smooth(p4 + p5)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation：用全局上下文调制空间特征。"""

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, feat_dim * 2)

        # 初始化为 identity：gamma=0, beta=0
        weight = self.to_gamma_beta.weight
        bias = self.to_gamma_beta.bias
        if weight is not None:
            nn.init.zeros_(weight)
        if bias is not None:
            nn.init.zeros_(bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1 + gamma) + beta


class CausalBlock(nn.Module):
    """沿时间维的因果卷积残差块。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T)
        pad = (self.kernel_size - 1) * self.dilation
        x_pad = F.pad(x, (pad, 0))
        return x + F.relu(self.conv(x_pad))


class CausalTCN(nn.Module):
    """因果时间卷积网络，输入/输出都是 (B,T,C,H,W)。

    每个空间位置单独沿时间卷积，channel 混合由 Conv1d 负责。
    """

    def __init__(
        self,
        channels: int,
        hidden_size: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(channels, hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalBlock(hidden_size, kernel_size, dilation)
            for dilation in dilations
        ])
        self.out_proj = nn.Conv1d(hidden_size, channels, kernel_size=1)
        # 因果感受野需要的历史帧数：RF - 1
        self.cache_len = sum((kernel_size - 1) * d for d in dilations)

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        if self.cache_len > 0 and cache is not None:
            cache = cache[:, -self.cache_len:]
            x_all = torch.cat([cache, x], dim=1)
        else:
            x_all = x

        T_all = x_all.shape[1]

        # (B,T,C,H,W) -> (B*H*W, C, T)
        h = x_all.permute(0, 3, 4, 2, 1).reshape(B * H * W, C, T_all)
        h = self.in_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_proj(h)

        # (B*H*W,C,T) -> (B,T,C,H,W)
        h = h.reshape(B, H, W, C, T_all).permute(0, 4, 3, 1, 2).contiguous()

        # 只返回当前 x 对应的输出，cache 部分丢弃
        out = h[:, -T:]
        new_cache = x_all[:, -self.cache_len:] if self.cache_len > 0 else None
        return out, new_cache


class HeatmapHead(nn.Module):
    """每个 slot 输出一张 heatmap。"""

    def __init__(
        self,
        in_ch: int = 256,
        n_slots: int = C.n_slot,
        out_size: tuple[int, int] = (24, 40),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=out_size, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_slots, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def heatmap_to_coords(heat: torch.Tensor) -> torch.Tensor:
    """把 (N,S,H,W) heatmap 转成 (N,S,2) 归一化坐标。"""
    N, S, H, W = heat.shape
    device = heat.device
    dtype = heat.dtype

    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W * 2 - 1
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H * 2 - 1
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(1, 1, -1)
    grid_y = grid_y.reshape(1, 1, -1)

    prob = heat.flatten(2).softmax(dim=-1)  # (N,S,H*W)
    x = (prob * grid_x).sum(dim=-1)
    y = (prob * grid_y).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


class TypeHead(nn.Module):
    """用 F5 的全局语义 + GRU 预测每个 slot 的动作类型。"""

    def __init__(
        self,
        in_ch: int = 256,
        hidden: int = 128,
        gru_hidden: int = 256,
        n_slots: int = C.n_slot,
        n_types: int = 4,
    ) -> None:
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
        self.chunk = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.type_head = nn.Linear(hidden, n_slots * n_types)

    def forward(
        self,
        x: torch.Tensor,
        B: int,
        T: int,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B*T, 256, H, W)
        g = F.adaptive_max_pool2d(x, 1).flatten(1)      # (B*T, 256)
        g = g.reshape(B, T, -1)                         # (B, T, 256)
        out, h = self.gru(g, h)                         # (B, T, gru_hidden)
        ctx = self.chunk(out.reshape(B * T, -1))        # (B*T, hidden)
        logits = self.type_head(ctx).view(B, T, self.n_slots, self.n_types)
        return logits, ctx.view(B, T, -1), h


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = TypeHead()
        self.fpn = FPNFusion()
        self.tcn = CausalTCN(channels=256, hidden_size=128)
        self.film = FiLM(cond_dim=128, feat_dim=256)
        self.heatmap_head = HeatmapHead(
            in_ch=256,
            n_slots=C.n_slot,
            out_size=(24, 40),
        )

    def _extract_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """返回类型分支的 f5、坐标分支的 fused，以及 B/T。"""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T = x.shape[:2]

        f4, f5 = self.backbone(x.flatten(0, 1))
        fused = self.fpn(f4, f5)
        fused = fused.reshape(B, T, *fused.shape[1:])
        return f5, fused, B, T

    def _forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        tcn_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, 3, H, W)
        f5, fused, B, T = self._extract_features(x)
        logits, ctx, h = self.head(f5, B, T, h)

        local, _ = self.tcn(fused, tcn_cache)       # (B,T,256,12,20)
        local = local.reshape(B * T, *local.shape[2:])  # (B*T,256,12,20)

        local = self.film(local, ctx.reshape(B * T, -1))
        heat = self.heatmap_head(local)             # (B*T,S,24,40)
        coords = heatmap_to_coords(heat).view(B, T, self.head.n_slots, 2)

        return logits, coords, h

    def forward(
        self,
        x: torch.Tensor,
        warmup: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h is not None:
            return self._forward(x, h)

        if warmup is not None:
            if warmup.dim() == 4:
                warmup = warmup.unsqueeze(1)
            if warmup.size(1) > 0:
                with torch.no_grad():
                    f5_w, fused_w, B_w, T_w = self._extract_features(warmup)
                    _, _, h = self.head(f5_w, B_w, T_w)
                    cache = fused_w[:, -self.tcn.cache_len:]
                return self._forward(x, h, cache)

        return self._forward(x, h)
