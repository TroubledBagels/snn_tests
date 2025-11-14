import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from utils import fft_dataset
import numpy as np
from torch.utils.data import DataLoader
import tqdm
import utils.heatmap
import cv2
import random

class BinarySquareClassifier(nn.Module):
    def __init__(self, c_1, c_2):
        super(BinarySquareClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 20, 3)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())

        self.c_1 = c_1
        self.c_2 = c_2

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
            spk_rec.append(xt)

        out = torch.stack(spk_rec).sum(dim=0)

        return out


class BinarySquareModel(nn.Module):
    def __init__(self, num_c, train=True):
        super(BinarySquareModel, self).__init__()
        self.bc_list = nn.ModuleList()
        self.num_c = num_c

        if not train:
            for i in range(num_c):
                for j in range(i + 1, num_c):
                    if i != j:
                        bclass = BinarySquareClassifier(i, j)
                        self.bc_list.append(bclass)
            return
        train_size = int(0.8 * len(ds))
        test_size = len(ds) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
        unique_pairs = []
        for i in range(num_c):
            for j in range(i + 1, num_c):
                if i != j:
                    unique_pairs.append((i, j))
        unique_pairs = sorted(list(set(unique_pairs)))
        self.acc_dict = {}
        for (i, j) in unique_pairs:
            print(f"Training ternary classifier for classes {i}, {j}, and else")
            bclass = BinarySquareClassifier(i, j)
            optimiser = torch.optim.Adam(bclass.parameters(), lr=0.001)

            bclass, acc = train_bclass(bclass, self.train_dataset, self.test_dataset, nn.CrossEntropyLoss(), optimiser, device)
            self.bc_list.append(bclass)
            self.acc_dict[(i, j)] = acc - 0.5  # Center accuracy around 0.5

        print(self.acc_dict)

    def get_model_by_classes(self, c_1, c_2):
        for model in self.bc_list:
            if (model.c_1 == c_1 and model.c_2 == c_2) or (model.c_1 == c_2 and model.c_2 == c_1):
                return model
        return None

    def forward(self, x):
        out_dict = {}
        heat_out = {}
        class_heat_out = {}
        for model in self.bc_list:
            out = model(x)
            out_dict[model.c_1] = out_dict.get(model.c_1, 0) + out[0, 0].item()
            out_dict[model.c_2] = out_dict.get(model.c_2, 0) + out[0, 1].item()
            heat_out[(model.c_1, model.c_2)] = out[0, 0].item() - out[0, 1].item()
            class_heat_out[(model.c_1, model.c_2)] = 0.99 if out[0, 0].item() > out[0, 1].item() else 0.01
        for i in range(self.num_c):
            if i not in out_dict:
                out_dict[i] = 0
        out = torch.zeros((1, len(out_dict)))
        for k, v in out_dict.items():
            out[0, k] = v

        # Normalise heat_out to between 0 and 1
        max_heat = max(heat_out.values())
        min_heat = min(heat_out.values())
        for k in heat_out.keys():
            heat_out[k] = (heat_out[k] - min_heat) / (max_heat - min_heat + 1e-8)

        # If top two are equal, check binary classifier between them
        sorted_counts = sorted(out_dict.items(), key=lambda item: item[1], reverse=True)
        if sorted_counts[0][1] == sorted_counts[1][1]:
            c_1 = sorted_counts[0][0]
            c_2 = sorted_counts[1][0]
            model = self.get_model_by_classes(c_1, c_2)
            if model is not None:
                out_bi = model(x)
                if out_bi[0, 0] > out_bi[0, 1]:
                    out = torch.zeros((1, len(out_dict)))
                    out[0, c_1] = 1
                else:
                    out = torch.zeros((1, len(out_dict)))
                    out[0, c_2] = 1

        return out, (heat_out, class_heat_out)


def train_bclass(model, tr_ds, te_ds, criterion, optimiser, device):
    req_idx = [model.c_1, model.c_2]
    train_dataset = fft_dataset.get_by_labels(tr_ds, req_idx)
    test_dataset = fft_dataset.get_by_labels(te_ds, req_idx)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    epochs = 1
    model.to(device)
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Classes: ({req_idx[0]}, {req_idx[1]}), Epoch {epoch+1}/{epochs}")
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs  = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            pbar.set_postfix({"Loss": train_loss / (pbar.n + 1)})
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(test_loader, desc="Testing")
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    return model, accuracy

def test_bcs(model):
    ds = fft_dataset.MFCCDataset(type="all")
    train_ds, test_ds = torch.utils.data.random_split(ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))])
    acc_dict = {}
    for bc in model.bc_list:
        bc.eval()
        correct = 0
        total = 0
        req_idx = [bc.c_1, bc.c_2]
        test_dataset = fft_dataset.get_by_labels(test_ds, req_idx)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            pbar = tqdm.tqdm(test_loader, desc=f"Testing Ternary Classifier for classes {bc.c_1} and {bc.c_2}")
            for data, labels in pbar:
                data, labels = data.to(device), labels.to(device)
                outputs = bc(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        acc_dict[(bc.c_1, bc.c_2)] = accuracy - 0.5 # Center accuracy around 0.5
    return acc_dict

def run_sample_and_visualise(model, ds, sample_idx: int | None = None):
    if sample_idx is None:
        sample_idx = random.randint(0, len(ds) - 1)
    data, label = ds[sample_idx]
    data = data.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs, hm_dicts = model(data)
        _, predicted = torch.max(outputs.data, 1)
    print(f"True Label: {ds.str_labels[sample_idx]}, Predicted Label: {ds.sorted_unique[predicted.item()]}")
    heatmap_img, classes = utils.heatmap.generate_heatmap(hm_dicts[0], ds.sorted_unique, use_acc=False)
    cv2.imshow(f"Ternary Square Classifier Heatmap {sample_idx}", heatmap_img)
    cv2.imwrite(f"./imgs/binary_square/{sample_idx}_vote_heatmap.png", heatmap_img)
    bin_heatmap_img, classes = utils.heatmap.generate_heatmap(hm_dicts[1], ds.sorted_unique, use_acc=False)
    cv2.imshow(f"Ternary Square Classifier Binary Heatmap {sample_idx}", bin_heatmap_img)
    cv2.imwrite(f"./imgs/binary_square/{sample_idx}_class_heatmap.png", bin_heatmap_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = fft_dataset.MFCCDataset(type="all")

    c_num = 38

    model = BinarySquareModel(c_num, True).to(device)

    torch.save(model.state_dict(), f"tsquare_{c_num}.pth")
    # model.load_state_dict(torch.load(f"tsquare_{c_num}.pth", map_location=device))
    print(f"num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # run_sample_and_visualise(model, ds, 28384)

    # for i in range(4):
    #     run_sample_and_visualise(model, ds, None)

    # acc_d = test_bcs(model)
    # model.acc_dict = acc_d

    heatmap_img, classes = utils.heatmap.generate_heatmap(model.acc_dict, ds.sorted_unique)
    cv2.imwrite("binary_square_heatmap.png", heatmap_img)
    # Display the heatmap
    cv2.imshow("Binary Square Classifier Heatmap", heatmap_img)
    cv2.waitKey(0)
    # train_ds, test_ds = torch.utils.data.random_split(ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))])
    # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # correct = 0
    # total = 0
    # model.eval()
    # with torch.no_grad():
    #     pbar = tqdm.tqdm(test_loader, desc="Testing Binary Square Model")
    #     for data, labels in pbar:
    #         data, labels = data.to(device), labels.to(device)
    #         outputs, _ = model(data)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # accuracy = correct / total
    # print(f'Overall Test Accuracy: {accuracy*100:.2f}%')