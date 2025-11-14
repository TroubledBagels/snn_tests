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
        self.fc1 = nn.Linear(18 * 64, out_c)
        # self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
        # self.fc2 = nn.Linear(128, out_c)
        # self.lif5 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)

    def forward(self, x, inference=False):
        B, C, T = x.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        # mem4 = self.lif4.init_leaky()
        # mem5 = self.lif5.init_leaky()

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
            spk_rec.append(xt)
        xt = torch.stack(spk_rec).mean(dim=0)
        xt = self.fc1(xt)
        out = xt
        # xt, mem4 = self.lif4(xt, mem4)
        # xt = self.fc2(xt)
        # xt, mem5 = self.lif5(xt, mem5)
        # spk_rec.append(mem5)

        # if inference:
        #     out = torch.stack(spk_rec)
        # else:
        #     out = torch.stack(spk_rec).mean(dim=0)

        return out, (mem1, mem2, mem3)

class EnsembleMeanModel(nn.Module):
    def __init__(self, model_list):
        super(EnsembleMeanModel, self).__init__()
        self.models = nn.ModuleList(model_list)

    def forward(self, x):
        outputs = []
        for model in self.models:
            out, _ = model(x, True) # B x T x Out_C
            outputs.append(out)
        # mean outputs to make a B x T x Out_C tensor
        out = torch.stack(outputs).mean(dim=0)

        return out, None

    def lock_subnets(self):
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False


class EnsembleAudioModel(nn.Module):
    def __init__(self, model_list):
        super(EnsembleAudioModel, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.fc1 = nn.Linear(38 * len(model_list), 128)
        # self.lif = snn.Leaky(beta=0.9, spike_grad=snn.surrogate.atan())
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 38)
        # self.lif2 = snn.Leaky(beta=0.9, spike_grad=snn.surrogate.atan())

    def forward(self, x):
        outputs = []
        for model in self.models:
            out, _ = model(x, True) # B x Out_C
            outputs.append(out)
        # concat outputs to make a B x Out_C * Num_Models tensor
        concat_outputs = torch.cat(outputs, dim=1) # B x (Out_C * Num_Models)
        # pass through fc layers

        xt = self.fc1(concat_outputs)
        xt = self.relu(xt)
        xt = self.fc2(xt)
        out = xt

        return out, (None)

    def lock_subnets(self):
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

if __name__ == '__main__':
    model = BaseModel(out_c=38)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.randn(1, 30, 12)
    out, _ = model(x)
    print(out.shape)