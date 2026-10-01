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
        return f4, f5

class CNNHeatmap(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 4, 1, 1, 0)

    @staticmethod
    def soft_argmax_2d(heatmap, beta=1.0):
        # heatmap: (N, K, H, W)
        N, K, H, W = heatmap.shape
        p = F.softmax((heatmap * beta).view(N, K, -1), dim=-1).view(N, K, H, W)

        ys = torch.arange(H, dtype=heatmap.dtype, device=heatmap.device).view(1, 1, H, 1)
        xs = torch.arange(W, dtype=heatmap.dtype, device=heatmap.device).view(1, 1, 1, W)

        x = (p * xs).sum(dim=(2, 3))
        y = (p * ys).sum(dim=(2, 3))
        return torch.stack([x, y], dim=-1)  # (N, K, 2)

    def forward(self, x):
        x = self.conv(x)
        return self.soft_argmax_2d(x)

class FingerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            bias=True,
            batch_first=True,    # 推荐设为 True
            dropout=0.3,         # 层间 dropout
            bidirectional=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()


        )
    
    def forward(self, x):
        x = F.adaptive_max_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.gru(x)

        return x

    




