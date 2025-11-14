import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from utils import fft_dataset
import matplotlib.pyplot as plt

class BasicClassifier(nn.Module):
    def __init__(self, num_c):
        super(BasicClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 18, 256)
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(256, num_c)
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())

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
            spk_rec.append(xt)

        out = torch.stack(spk_rec)

        return out

    def lock_weights(self):
        for param in self.parameters():
            param.requires_grad = False

class IncrementalModel(nn.Module):
    def __init__(self, num_c, training=True, num_classifiers=None):
        super(IncrementalModel, self).__init__()
        self.num_c = num_c
        self.training = training
        self.classifiers = nn.ModuleList()

        if not training:
            for i in range(self.num_c):
                classifier = BasicClassifier(num_classifiers)
                self.classifiers.append(classifier)

        self.fc1 = nn.Linear(1, 256)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(256, num_c)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())

    def expand_network(self):
        if len(self.classifiers) > 0:
            self.classifiers[-1].lock_weights()
        classifier = BasicClassifier(self.num_c)
        self.classifiers.append(classifier)
        temp = self.fc1
        if len(self.classifiers) > 1:
            self.fc1 = nn.Linear(temp.in_features + self.num_c, 256)
        else:
            self.fc1 = nn.Linear(self.num_c, 256)
        self.saved_weights_fc1 = temp.weight.data.clone()
        self.saved_bias_fc1 = temp.bias.data.clone()
        self.fc1.weight.data[:,:-self.num_c] = self.saved_weights_fc1
        self.fc1.bias.data = self.saved_bias_fc1

    def forward(self, x):
        out_list = []
        for classifier in self.classifiers:
            out = classifier(x) # Shape: (T, B, num_c)
            B, _, T = out.shape
            out = out.permute(1, 0, 2)  # Shape: (B, T, num_c)
            out_list.append(out)

        # Shape: (B, T, num_c * num_classifiers)
        combined_out = torch.cat(out_list, dim=2)
        B, T, C = combined_out.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []

        self.fc1.weight.data[:,:-self.num_c] = self.saved_weights_fc1

        for t in range(T):
            xt = combined_out[:, t, :].view(B, -1)
            xt = self.fc1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.fc2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            spk_rec.append(mem2)

        out = torch.stack(spk_rec).mean(dim=0).unsqueeze(0)  # Shape: (T, B, num_c)

        return out

if __name__ == "__main__":
    num_c = 38
    num_classifiers = 10
    epochs = 3
    model = IncrementalModel(num_c)

    ds = fft_dataset.MFCCDataset(type="all")
    train_set, test_set = torch.utils.data.random_split(ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_acc = []
    for c in range(num_classifiers):
        print(f"Training with classifier {c+1}/{num_classifiers}")
        model.expand_network()
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm.tqdm(train_loader)
            for i, (data, labels) in enumerate(pbar):
                data, labels = data.to(device), labels.to(device)
                optimiser.zero_grad()
                outputs = model(data)
                outputs = outputs.mean(dim=0)  # Average over time steps
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(i+1):.4f}")

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm.tqdm(test_loader)
            for data, labels in pbar:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                outputs = outputs.mean(dim=0)  # Average over time steps
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_description(f"Testing Accuracy: {100 * correct / total:.2f}%")
        print(f'Accuracy after adding classifier {c+1}: {100 * correct / total:.2f}%')
        test_acc.append(100 * correct / total)
        plt.plot(test_acc)
        plt.title("Accuracy per Number of Classifiers")
        plt.xlabel("Number of Classifiers")
        plt.ylabel("Accuracy (%)")
        plt.show()

    torch.save(model.state_dict(), f"models/{c+1}_{num_classifiers}.pth")
