import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate

class SimpleRecModel(nn.Module):
    def __init__(self, in_c, out_c):
        super(SimpleRecModel, self).__init__()
        self.spike_grad = snn.surrogate.atan()
        self.conv1 = nn.Conv2d(in_c, 16, kernel_size=7, stride=2, padding=1)
        self.lif1 = snn.Leaky(beta=0.94, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 20 * 20, 2048)
        self.dropout1 = nn.Dropout(p=0.35)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(2048, 256)
        self.rlif = snn.RLeaky(beta=0.5, spike_grad=self.spike_grad, linear_features=256)
        self.fc4 = nn.Linear(256, out_c)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)

    def forward(self, x):
        B, T, C, H, W = x.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_r, mem_r = self.rlif.init_rleaky()
        mem4 = self.lif4.init_leaky()

        spk_rec = []

        for step in range(T):
            xt = x[:, step, :, :, :]  # Shape: (B, C, H, W)
            xt = self.conv1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.conv2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            xt = self.flatten(xt)
            xt = self.fc1(xt)
            xt = self.dropout1(xt)
            xt, mem3 = self.lif3(xt, mem3)
            xt = self.fc2(xt)
            xt, mem_r = self.rlif(xt, spk_r, mem_r)
            xt = self.fc4(mem_r)
            xt, mem4 = self.lif4(xt, mem4)
            spk_rec.append(mem4)

        out = torch.stack(spk_rec).mean(dim=0)
        return out, (mem1, mem2, mem_r)

if __name__ == "__main__":
    model = SimpleRecModel(in_c=1, out_c=75)
    x = torch.randn(4, 800, 1, 88, 88)  # Batch size of 4, 800 time steps, 1 channel, 88x88 spatial dimensions
    out, mems = model(x)
    print(out.shape)  # Should output: torch.Size([4, 75])
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

