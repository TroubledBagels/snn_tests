import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate

class SimpleConvModel(nn.Module):
    def __init__(self, inp_channels: int = 1, num_classes: int = 10):
        super(SimpleConvModel, self).__init__()
        self.spike_grad = snn.surrogate.atan()
        self.conv = nn.Conv2d(inp_channels, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.ln1 = nn.LayerNorm([8, 32, 32])
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.dropout = nn.Dropout(0.2)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.ln2 = nn.LayerNorm([32, 32, 32])
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.ln3 = nn.LayerNorm([32, 32, 32])
        self.lif3_5 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        self.aap = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*3*3, 64)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(64, num_classes)
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad, output=True, learn_beta=True, learn_threshold=True)

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.view(B, T, 1, H, W)  # Add channel dimension
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem3_5 = self.lif3_5.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        spike_rec = []

        for t in range(T):
            xt = x[:, t, :, :, :]  # Shape: (T, C, H, W)
            xt = self.conv(xt)
            xt = self.ln1(xt)
            #xt, mem1 = self.lif1(xt, mem1)
            xt = self.conv2(xt)
            xt = self.dropout(xt)
            #xt, mem2 = self.lif2(xt, mem2)
            # res = xt.clone()  # Residual connection
            xt = self.conv3(xt)
            xt = self.ln2(xt)
            #xt, mem3 = self.lif3(xt, mem3)
            # xt = xt + res  # Add residual
            #res = xt.clone()
            xt = self.conv4(xt)
            xt = self.ln3(xt)
            #xt, mem3_5 = self.lif3_5(xt, mem3_5)
            #xt = xt + res
            xt = self.aap(xt)
            xt = self.flatten(xt)
            xt = self.fc1(xt)
            #xt, mem4 = self.lif4(xt, mem4)
            xt = self.fc2(xt)
            #xt, mem5 = self.lif5(xt, mem5)
            spike_rec.append(xt)

        out = torch.stack(spike_rec).mean(dim=0)
        #out = torch.stack(spike_rec)
        return out, (mem1, mem2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleConvModel().to(device)
    sample = torch.randn(1, 100, 64, 64).to(device)
    output = model(sample)
    print(output[0].shape)  # Should print torch.Size([1, 10])
