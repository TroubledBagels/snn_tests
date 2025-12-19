import torch
import torch.nn as nn
import torch.nn.functional as F

class hard_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
            return x
        else:
            return torch.clamp((x + 3.0), 0.0, 6.0) / 6.0

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = channels // reduction
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion=6):
        super().__init__()

        hidden_dim = in_channels * expansion
        use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # 1. Expansion (skip if expansion == 1)
        if expansion != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ]

        # 2. Depthwise convolution
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ]

        # 3. Squeeze & Excitation
        layers.append(SEBlock(hidden_dim))

        # 4. Projection
        layers += [
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]

        self.block = nn.Sequential(*layers)
        self.use_residual = use_residual

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            MBConv(32, 16, kernel_size=3, stride=1, expansion=1),
            MBConv(16, 24, kernel_size=3, stride=2),
            MBConv(24, 24, kernel_size=3, stride=1),
            MBConv(24, 40, kernel_size=5, stride=2),
            MBConv(40, 40, kernel_size=5, stride=1),
            MBConv(40, 80, kernel_size=3, stride=2),
            MBConv(80, 80, kernel_size=3, stride=1),
            MBConv(80, 112, kernel_size=5, stride=1),
            MBConv(112, 112, kernel_size=5, stride=1),
            MBConv(112, 192, kernel_size=5, stride=2),
            MBConv(192, 192, kernel_size=5, stride=1),
            MBConv(192, 320, kernel_size=3, stride=1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = EfficientNetB0(num_classes=10)
    print(model)
    sample_input = torch.randn(1, 3, 32, 32)
    output = model(sample_input)
    print(output.shape)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
