import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate

class SimpleMelModel(nn.Module):
    def __init__(self, out_c):
        super(SimpleMelModel, self).__init__()
        self.spike_grad = snn.surrogate.atan()
        self.conv1 = nn.Conv1d(1, 32, 7)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.conv3 = nn.Conv1d(64, 64, 3)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20 * 64, 128)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(128, out_c)
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)

    def forward(self, x):
        B, C, T = x.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk_rec = []

        for t in range(T):
            xt = x[:, :, t]
            xt = self.conv1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.conv2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            xt = self.conv3(xt)
            xt, mem3 = self.lif3(xt, mem3)
            xt = xt.view(B, -1)
            xt = self.fc1(xt)
            xt, mem4 = self.lif4(xt, mem4)
            xt = self.fc2(xt)
            xt, mem5 = self.lif5(xt, mem5)
            spk_rec.append(mem5)

        out = torch.stack(spk_rec).mean(dim=0)

        return out, (mem1, mem2, mem3, mem4)

class SimpleMFCCModel(nn.Module):
    def __init__(self, out_c):
        super(SimpleMFCCModel, self).__init__()
        self.spike_grad = snn.surrogate.atan()
        self.conv1 = nn.Conv1d(1, 32, 7)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.conv3 = nn.Conv1d(64, 64, 3)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(18 * 64, 128)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(128, out_c)
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)

    def forward(self, x):
        B, C, T = x.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spk_rec = []

        for t in range(T):
            xt = x[:, :, t]
            xt = self.conv1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.conv2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            xt = self.conv3(xt)
            xt, mem3 = self.lif3(xt, mem3)
            xt = xt.view(B, -1)
            xt = self.fc1(xt)
            xt, mem4 = self.lif4(xt, mem4)
            xt = self.fc2(xt)
            xt, mem5 = self.lif5(xt, mem5)
            spk_rec.append(mem5)

        out = torch.stack(spk_rec).mean(dim=0)

        return out, (mem1, mem2, mem3, mem4)

class RegMFCCModel(nn.Module):
    def __init__(self, out_c):
        super(RegMFCCModel, self).__init__()
        # 2D Convolutional version without spiking
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1))
        self.aap = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, out_c)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = RegMFCCModel(out_c=10)
    sample_input = torch.randn(1, 32, 24)
    output = model(sample_input)
    print(output[0].shape)
    print("Learnable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))