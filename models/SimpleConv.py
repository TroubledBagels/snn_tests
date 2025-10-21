import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate

# class SimpleConvModel(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(SimpleConvModel, self).__init__()
#         self.spike_grad = snn.surrogate.atan()
#         self.conv1 = nn.Conv2d(in_c, 16, kernel_size=5, stride=2, padding=1)
#         self.lif1 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)
#         self.lif2 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(16 * 21 * 21, 2048)
#         self.dropout1 = nn.Dropout(p=0.35)
#         self.lif3 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
#         self.fc2 = nn.Linear(2048, 256)
#         self.dropout2 = nn.Dropout(p=0.35)
#         self.lif4 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
#         self.fc3 = nn.Linear(256, out_c)
#         self.lif5 = snn.Leaky(beta=0.9, spike_grad=self.spike_grad)
#
#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         mem1 = self.lif1.init_leaky()
#         mem2 = self.lif2.init_leaky()
#         mem3 = self.lif3.init_leaky()
#         mem4 = self.lif4.init_leaky()
#         mem5 = self.lif5.init_leaky()
#         spk_rec = []
#
#         for t in range(T):
#             xt = x[:, t, :, :, :]  # Shape: (B, C, H, W)
#             xt = self.conv1(xt)
#             xt, mem1 = self.lif1(xt, mem1)
#             xt = self.conv2(xt)
#             xt, mem2 = self.lif2(xt, mem2)
#             xt = self.flatten(xt)
#             xt = self.fc1(xt)
#             xt = self.dropout1(xt)
#             xt, mem3 = self.lif3(xt, mem3)
#             xt = self.fc2(xt)
#             xt = self.dropout2(xt)
#             xt, mem4 = self.lif4(xt, mem4)
#             xt = self.fc3(xt)
#             xt, mem5 = self.lif5(xt, mem5)
#             # spk_rec.append(xt)
#             spk_rec.append(mem5)
#
#         # out = torch.stack(spk_rec)
#         out = torch.stack(spk_rec).mean(dim=0)
#
#         return out, (mem1, mem2)

class SimpleConvModel(nn.Module):
    def __init__(self, in_c, out_c):
        super(SimpleConvModel, self).__init__()
        self.beta1 = 0.94
        self.beta2 = 0.9
        self.beta3 = 0.5
        self.spike_grad = snn.surrogate.atan()
        # Starts 1 x 88 x 88
        self.conv1 = nn.Conv2d(in_c, 16, kernel_size=7, stride=2, padding=1)  # Now 16 x 42 x 42
        self.lif1 = snn.Leaky(beta=self.beta1, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)  # Now 32 x 21 x 21
        self.ln1 = nn.LayerNorm([32, 21, 21])
        self.lif2 = snn.Leaky(beta=self.beta1, spike_grad=self.spike_grad)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)  # Now 64 x 9 x 9
        self.lif3 = snn.Leaky(beta=self.beta2, spike_grad=self.spike_grad)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1)  # Now 64 x 4 x 4
        self.ln2 = nn.LayerNorm([64, 4, 4])
        self.lif4 = snn.Leaky(beta=self.beta2, spike_grad=self.spike_grad)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4*4*64, out_c)
        self.dropout1 = nn.Dropout(p=0.35)
        self.lif5 = snn.Leaky(beta=self.beta3, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(2048, 256)
        self.dropout2 = nn.Dropout(p=0.35)
        self.lif6 = snn.Leaky(beta=self.beta3, spike_grad=self.spike_grad)
        self.fc3 = nn.Linear(256, out_c)
        self.lif7 = snn.Leaky(beta=self.beta3, spike_grad=self.spike_grad)


    def forward(self, x):
        B, T, C, H, W = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        mem7 = self.lif7.init_leaky()
        spk_rec = []

        for t in range(T):
            xt = x[:, t, :, :, :]  # Shape: (B, C, H, W)
            xt = self.conv1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.conv2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            xt = self.conv3(xt)
            xt, mem3 = self.lif3(xt, mem3)
            xt = self.conv4(xt)
            xt, mem4 = self.lif4(xt, mem4)
            xt = self.flatten(xt)
            xt = self.fc1(xt)
            # xt = self.dropout1(xt)
            xt, mem5 = self.lif5(xt, mem5)
            # xt = self.fc2(xt)
            # xt = self.dropout2(xt)
            # xt, mem6 = self.lif6(xt, mem6)
            # xt = self.fc3(xt)
            # xt, mem7 = self.lif7(xt, mem7)
            # spk_rec.append(xt)
            spk_rec.append(mem5)

        # out = torch.stack(spk_rec)
        out = torch.stack(spk_rec).mean(dim=0)

        return out, (mem1, mem2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleConvModel(1, 75).to(device)
    sample = torch.randn(32, 100, 1, 88, 88).to(device)
    output = model(sample)
    print(output[0].shape)  # Should print torch.Size([1, 5])
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
