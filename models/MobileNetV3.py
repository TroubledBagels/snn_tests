import torch
import torch.nn as nn

class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, c):
        return c * nn.ReLU6(inplace=self.inplace)(c + 3) / 6

class hard_sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(hard_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, c):
        return nn.ReLU6(inplace=self.inplace)(c + 3) / 6

class SEModule(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden_dim = channels // reduction_ratio

        self.fc1 = nn.Linear(channels, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, channels, bias=True)
        self.hsig = hard_sigmoid(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = nn.ReLU(inplace=True)(y)
        y = self.fc2(y)
        y = (self.hsig(y)).view(b, c, 1, 1)
        y = y.view(b, c, 1, 1)
        return x * y

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, kernel_size, expand_ratio, use_se=False, activation_func=nn.ReLU6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.use_se = use_se

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(activation_func(inplace=True))
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(activation_func(inplace=True))

        self.pre_se = nn.Sequential(*layers)

        if use_se:
            self.se = SEModule(hidden_dim)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.pre_se(x)

        if self.use_se:
            out = self.se(out)

        out = self.project(out)

        if self.use_res_connect:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        block = InvertedResidual
        input_channel = 16
        last_channel = 1280

        features = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            h_swish(inplace=True)
        ]

        cfg = [
            # t, c, n, s, k, se, nl
            [1, 16, 1, 1, 3, True, nn.ReLU],
            [4.5, 24, 1, 1, 3, False, nn.ReLU],
            [3.67, 24, 1, 1, 3, False, nn.ReLU],
            [4, 40, 1, 2, 5, True, h_swish],
            [6, 40, 2, 1, 5, True, h_swish],
            [3, 48, 2, 1, 5, True, h_swish]
        ]

        for t, c, n, s, k, se, nl in cfg:
            output_channel = int(round(c * width_mult))
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, kernel_size=k, expand_ratio=t, use_se=se,  activation_func=nl))
                input_channel = output_channel

        features += [
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ]

        self.features = nn.Sequential(*features)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = MobileNetV3(num_classes=10)
    print(model)
    sample_input = torch.randn(1, 3, 32, 32)
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
