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
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 64 * 64, out_c)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)

    def forward(self, x):
        B, T, C, H, W = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []

        for t in range(T):
            xt = x[:, t, :, :, :]  # Shape: (B, C, H, W)
            xt = self.conv1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.flatten(xt)
            xt = self.fc1(xt)
            xt, mem2 = self.lif2(xt, mem2)
            spk_rec.append(xt)

        return torch.stack(spk_rec), (mem1, mem2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleConvModel().to(device)
    sample = torch.randn(1, 100, 64, 64).to(device)
    output = model(sample)
    print(output[0].shape)  # Should print torch.Size([1, 10])
