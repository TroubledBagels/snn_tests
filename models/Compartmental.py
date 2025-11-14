import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class VowelConsClassifier(nn.Module):
    def __init__(self):
        super(VowelConsClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 20, 2)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())

    def forward(self, x):
        B, C, T = x.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []

        for t in range(T):
            xt = x[:, :, t]
            xt = self.conv1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.conv2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            xt = xt.view(B, -1)
            xt = self.fc1(xt)
            xt, mem3 = self.lif3(xt, mem3)
            spk_rec.append(mem3)

        out = torch.stack(spk_rec).mean(dim=0)

        return out

class CompartmentalModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

if __name__ == "__main__":
    model = VowelConsClassifier()
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))