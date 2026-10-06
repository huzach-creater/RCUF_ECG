import torch
from torch import nn


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.proj = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + self.proj(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, model_size="small", embedding_dim=256, base_channels=None, dropout=0.2):
        super().__init__()
        if base_channels is None:
            base_channels = {"tiny": 16, "small": 32, "standard": 64}.get(model_size, 32)
        if model_size == "tiny" and embedding_dim == 256:
            embedding_dim = 128
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers = []
        in_ch = channels[0]
        for i, ch in enumerate(channels):
            stride = 1 if i == 0 else 2
            layers.append(BasicBlock1D(in_ch, ch, stride=stride))
            layers.append(BasicBlock1D(ch, ch, stride=1))
            in_ch = ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels[-1], embedding_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        h = self.pool(self.backbone(self.stem(x))).squeeze(-1)
        z = self.embedding(h)
        logits = self.classifier(z)
        if return_embedding:
            return logits, z
        return logits
