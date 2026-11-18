"""model.py —— 视觉特征提取、时序建模与多指预测模型。

结构：
    F4 + F5 -> FPN(+CoordConv) -> CausalTCN -> local
    local -> HeatmapHead -> heat -> soft-argmax -> coords + prob
    F5 -> GRU -> 全局时序上下文 ctx
    [slot_feat(local 按 prob 池化), coords, slot_emb, ctx] -> CrossSlotBlock -> type logits

说明：
    - 坐标分支由 FPN + CoordConv + CausalTCN 定位出每个 slot 的高斯热力图并计算 coords
    - 分类分支用热力图概率 prob 从含时序的 local 特征图加权提取各槽位局部特征，
      再与 F5+GRU 全局上下文及槽位间 Self-Attention 融合预测动作类型
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
    """把 F4 的细节和 F5 的语义融合成 12x20 的空间特征图，并注入归一化坐标通道。"""

    def __init__(self, c4: int = 128, c5: int = 256, out_ch: int = 256) -> None:
        super().__init__()
        self.lat4 = nn.Conv2d(c4, out_ch, kernel_size=1)
        self.lat5 = nn.Conv2d(c5, out_ch, kernel_size=1)
        self.out_ch = out_ch
        self.smooth = nn.Sequential(
            nn.Conv2d(out_ch + 2, out_ch, kernel_size=3, padding=1),
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
        x = p4 + p5
        N, _, H, W = x.shape
        xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W * 2 - 1
        ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H * 2 - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords_map = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(N, -1, -1, -1)
        return self.smooth(torch.cat([x, coords_map], dim=1))


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
        B, T, ch, H, W = x.shape

        if self.cache_len > 0 and cache is not None:
            cache = cache[:, -self.cache_len:]
            x_all = torch.cat([cache, x], dim=1)
        else:
            x_all = x

        T_all = x_all.shape[1]

        # (B,T,C,H,W) -> (B*H*W, C, T)
        h = x_all.permute(0, 3, 4, 2, 1).reshape(B * H * W, ch, T_all)
        h = self.in_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_proj(h)

        # (B*H*W,C,T) -> (B,T,C,H,W)
        h = h.reshape(B, H, W, ch, T_all).permute(0, 4, 3, 1, 2).contiguous()

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


def heatmap_to_coords_and_prob(heat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 (N,S,H,W) heatmap 转成 (N,S,2) 归一化坐标和 (N,S,H,W) 概率图。"""
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
    return torch.stack([x, y], dim=-1), prob.view(N, S, H, W)


def heatmap_to_coords(heat: torch.Tensor) -> torch.Tensor:
    """把 (N,S,H,W) heatmap 转成 (N,S,2) 归一化坐标。"""
    coords, _ = heatmap_to_coords_and_prob(heat)
    return coords


class CrossSlotBlock(nn.Module):
    """槽位间轻量 Self-Attention + FFN，用于共享信息并抑制空闲槽位重复预测。"""

    def __init__(self, dim: int = 128, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, S, D)
        N, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(N, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N, S, D)
        x = x + self.drop1(self.proj(out))
        x = x + self.ffn(self.norm2(x))
        return x


class TypeHead(nn.Module):
    """结合 F5+GRU 全局上下文与按热力图池化的局部时空特征预测每个 slot 的动作类型。"""

    def __init__(
        self,
        in_ch: int = 256,
        local_ch: int = 256,
        slot_dim: int = 64,
        hidden: int = 128,
        gru_hidden: int = 256,
        slot_emb_dim: int = 16,
        n_slots: int = C.n_slot,
        n_types: int = 4,
        dropout: float = 0.1,
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
        self.global_proj = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.local_proj = nn.Sequential(
            nn.Conv2d(local_ch, slot_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.slot_emb = nn.Embedding(n_slots, slot_emb_dim)
        self.slot_in = nn.Sequential(
            nn.Linear(slot_dim + 2 + slot_emb_dim + hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cross_slot = CrossSlotBlock(dim=hidden, n_heads=4, dropout=dropout)
        self.type_head = nn.Linear(hidden, n_types)

    def forward(
        self,
        f5: torch.Tensor,
        local: torch.Tensor,
        prob: torch.Tensor,
        coords: torch.Tensor,
        B: int,
        T: int,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # f5: (B*T, 256, 6, 10), local: (B*T, 256, 12, 20)
        # prob: (B*T, S, 24, 40), coords: (B*T, S, 2)
        g = F.adaptive_max_pool2d(f5, 1).flatten(1).reshape(B, T, -1)
        out, h = self.gru(g, h)
        ctx = self.global_proj(out.reshape(B * T, -1))  # (B*T, hidden)

        feat_map = self.local_proj(local)               # (B*T, slot_dim, 12, 20)
        prob_low = F.adaptive_avg_pool2d(prob.detach(), feat_map.shape[-2:])
        prob_low = prob_low / prob_low.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        slot_feat = torch.einsum("nchw,nshw->nsc", feat_map, prob_low)  # (B*T, S, slot_dim)

        slot_ids = torch.arange(self.n_slots, device=local.device)
        slot_emb = self.slot_emb(slot_ids).unsqueeze(0).expand(B * T, -1, -1)
        ctx_exp = ctx.unsqueeze(1).expand(-1, self.n_slots, -1)

        token = torch.cat([slot_feat, coords.detach(), slot_emb, ctx_exp], dim=-1)
        token = self.slot_in(token)                     # (B*T, S, hidden)
        token = self.cross_slot(token)                  # (B*T, S, hidden)
        logits = self.type_head(token).view(B, T, self.n_slots, self.n_types)
        return logits, h


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = CNNBackbone()
        self.fpn = FPNFusion()
        self.tcn = CausalTCN(channels=256, hidden_size=128)
        self.heatmap_head = HeatmapHead(
            in_ch=256,
            n_slots=C.n_slot,
            out_size=(24, 40),
        )
        self.head = TypeHead()

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        # x: (B, T, 3, H, W)
        f5, fused, B, T = self._extract_features(x)

        local, tcn_cache = self.tcn(fused, tcn_cache)       # (B,T,256,12,20)
        local_flat = local.reshape(B * T, *local.shape[2:])  # (B*T,256,12,20)

        heat = self.heatmap_head(local_flat)                 # (B*T,S,24,40)
        coords_flat, prob = heatmap_to_coords_and_prob(heat)

        logits, h = self.head(f5, local_flat, prob, coords_flat, B, T, h)
        coords = coords_flat.view(B, T, self.head.n_slots, 2)
        heat_out = heat.view(B, T, self.head.n_slots, *heat.shape[2:])
        return logits, coords, h, tcn_cache, heat_out

    def forward(
        self,
        x: torch.Tensor,
        warmup: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        tcn_cache: torch.Tensor | None = None,
        return_heat: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if h is not None or tcn_cache is not None:
            out = self._forward(x, h, tcn_cache)
            return out if return_heat else out[:4]

        if warmup is not None:
            if warmup.dim() == 4:
                warmup = warmup.unsqueeze(1)
            if warmup.size(1) > 0:
                with torch.no_grad():
                    _, _, h, cache, _ = self._forward(warmup)
                out = self._forward(x, h, cache)
                return out if return_heat else out[:4]

        out = self._forward(x, h, tcn_cache)
        return out if return_heat else out[:4]
