import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate

class SimpleConvModel(nn.Module):
    def __init__(self, in_c, out_c):
        super(SimpleConvModel, self).__init__()
        self.spike_grad = snn.surrogate.atan()
        self.conv1 = nn.Conv2d(in_c, 16, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 64 * 64, 4096)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(4096, 256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.fc3 = nn.Linear(256, out_c)
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)

    def forward(self, x):
        B, T, C, H, W = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        spk_rec = []

        for t in range(T):
            xt = x[:, t, :, :, :]  # Shape: (B, C, H, W)
            xt = self.conv1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.conv2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            xt = self.flatten(xt)
            xt = self.fc1(xt)
            xt = self.dropout1(xt)
            xt, mem3 = self.lif3(xt, mem3)
            xt = self.fc2(xt)
            xt = self.dropout2(xt)
            xt, mem4 = self.lif4(xt, mem4)
            xt = self.fc3(xt)
            xt, mem5 = self.lif5(xt, mem5)
            # spk_rec.append(xt)
            spk_rec.append(mem4)

        # out = torch.stack(spk_rec)
        out = torch.stack(spk_rec).mean(dim=0)

        return out, (mem1, mem2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleConvModel(1, 5).to(device)
    sample = torch.randn(32, 100, 1, 64, 64).to(device)
    output = model(sample)
    print(output[0].shape)  # Should print torch.Size([1, 10])
