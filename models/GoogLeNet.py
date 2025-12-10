import torch
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        self.branch1 = ConvModule(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvModule(in_channels, red_3x3, kernel_size=1),
            ConvModule(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvModule(in_channels, red_5x5, kernel_size=1),
            ConvModule(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvModule(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        if not x.is_cuda:
            # On CPU, just do sequential
            b1 = self.branch1(x)
            b2 = self.branch2(x)
            b3 = self.branch3(x)
            b4 = self.branch4(x)
            return torch.cat([b1, b2, b3, b4], dim=1)

        # Create streams
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        s3 = torch.cuda.Stream()
        s4 = torch.cuda.Stream()

        # Launch branches in different streams
        with torch.cuda.stream(s1):
            out1 = self.branch1(x)
        with torch.cuda.stream(s2):
            out2 = self.branch2(x)
        with torch.cuda.stream(s3):
            out3 = self.branch3(x)
        with torch.cuda.stream(s4):
            out4 = self.branch4(x)

        # Sync all streams with default before concat
        torch.cuda.synchronize()

        return torch.cat([out1, out2, out3, out4], dim=1)

class DownsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleModule, self).__init__()
        self.conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(x)
        return torch.cat([conv_out, pool_out], 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            ConvModule(3, 96, kernel_size=3, stride=1, padding=1)
        )

        self.block_group_1 = nn.Sequential(
            InceptionModule(96, 32, 32 ,32 ,32, 32, 32),
            InceptionModule(128, 32, 48,  48, 48, 48, 32),
            DownsampleModule(160, 80)
        )

        self.block_group_2 = nn.Sequential(
            InceptionModule(240, 112, 48, 48, 32, 32, 48),
            InceptionModule(240, 96, 64, 64, 32, 32, 32),
            InceptionModule(224, 80, 80, 80, 32, 32, 32),
            InceptionModule(224, 48, 96, 96, 32, 32, 32),
            InceptionModule(208, 112, 48, 48, 32, 32, 48),
            DownsampleModule(240, 96)
        )

        self.block_group_3 = nn.Sequential(
            InceptionModule(336, 176, 160, 160, 96, 96, 96),
            InceptionModule(528, 176, 160, 160, 96, 96, 96)
        )

        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(528, num_classes)
        )

    def forward(self, x):
        x = self.pre_layers(x)
        print(x.shape)
        x = self.block_group_1(x)
        print(x.shape)
        x = self.block_group_2(x)
        print(x.shape)
        x = self.block_group_3(x)
        print(x.shape)
        x = self.classifier_head(x)
        return x

if __name__ == '__main__':
    model = GoogLeNet(num_classes=10)
    print(model)
    sample_input = torch.randn(1, 3, 32, 32)
    output = model(sample_input)
    print(output.shape)