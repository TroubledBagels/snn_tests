import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate

class BaseModel(nn.Module):
    def __init__(self, out_c):
        super(BaseModel, self).__init__()
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


class EnsembleAudioModel(nn.Module):
    def __init__(self, model_list):
        super(EnsembleAudioModel, self).__init__()
        self.models = nn.ModuleList(model_list)

    def forward(self, x):
        outputs = []
        for model in self.models:
            out, _ = model(x)
            outputs.append(out)
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output