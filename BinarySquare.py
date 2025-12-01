import torch
import torch.nn as nn
import snntorch as snn
from sklearn import metrics
from snntorch import surrogate
from utils import fft_dataset
import numpy as np
from torch.utils.data import DataLoader
import tqdm
import utils.heatmap
import cv2
import random
import utils.general as g

class BinarySquareClassifierConventional(nn.Module):
    def __init__(self, c_1, c_2):
        super(BinarySquareClassifierConventional, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(32, 2)

        self.c_1 = c_1
        self.c_2 = c_2

    def forward(self, x):
        B, C, T = x.shape

        out = torch.zeros((B, 2)).to(x.device)

        for t in range(T):
            xt = x[:, :, t]
            xt = self.fc1(xt)
            xt = self.relu1(xt)
            xt = self.fc2(xt)
            xt = self.relu2(xt)
            xt = self.fc3(xt)
            out += xt

        return out

class BinarySquareModelConventional(nn.Module):
    def __init__(self, num_c, tr_ds, te_ds, train=True):
        super(BinarySquareModelConventional, self).__init__()
        self.bc_list = nn.ModuleList()
        self.num_c = num_c

        if not train:
            for i in range(num_c):
                for j in range(i + 1, num_c):
                    if i != j:
                        bclass = BinarySquareClassifierConventional(i, j)
                        self.bc_list.append(bclass)
            return
        self.train_dataset = tr_ds
        self.test_dataset = te_ds
        unique_pairs = []
        for i in range(num_c):
            for j in range(i + 1, num_c):
                if i != j:
                    unique_pairs.append((i, j))
        unique_pairs = sorted(list(set(unique_pairs)))
        self.acc_dict = {}
        for (i, j) in unique_pairs:
            print(f"Training binary classifier for classes {i} and {j}")
            bclass = BinarySquareClassifierConventional(i, j)
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

class BinarySquareClassifier(nn.Module):
    def __init__(self, c_1, c_2):
        super(BinarySquareClassifier, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(32, 2)
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
            xt = self.fc1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.fc2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            xt = self.fc3(xt)
            xt, mem3 = self.lif3(xt, mem3)
            spk_rec.append(xt)

        out = torch.stack(spk_rec).sum(dim=0)

        return out

class BinarySquareModel(nn.Module):
    def __init__(self, num_c, tr_ds, te_ds, train=True):
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
        self.train_dataset = tr_ds
        self.test_dataset = te_ds
        unique_pairs = []
        for i in range(num_c):
            for j in range(i + 1, num_c):
                if i != j:
                    unique_pairs.append((i, j))
        unique_pairs = sorted(list(set(unique_pairs)))
        self.acc_dict = {}
        for (i, j) in unique_pairs:
            print(f"Training binary classifier for classes {i} and {j}")
            bclass = BinarySquareClassifier(i, j)
            optimiser = torch.optim.Adam(bclass.parameters(), lr=0.001)

            bclass, acc = train_bclass(bclass, self.train_dataset, self.test_dataset, None, optimiser, device)
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

class BinarySquareModelCombined(nn.Module):
    def __init__(self, num_c, net_output=False):
        super(BinarySquareModelCombined, self).__init__()
        self.num_classifiers = (num_c * (num_c - 1)) // 2
        self.num_classes = num_c
        self.fc1 = nn.Linear(30, 64 * self.num_classifiers)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(64 * self.num_classifiers, 32 * self.num_classifiers)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.fc3 = nn.Linear(32 * self.num_classifiers, 2 * self.num_classifiers)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        if net_output:
            self.output_layer = nn.Linear(2 * self.num_classifiers, num_c)
        self.net_output = net_output

    def forward(self, x):
        B, C, T = x.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec = []

        for t in range(T):
            xt = x[:, :, t]
            xt = self.fc1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = self.fc2(xt)
            xt, mem2 = self.lif2(xt, mem2)
            xt = self.fc3(xt)
            xt, mem3 = self.lif3(xt, mem3)
            spk_rec.append(xt)

        out = torch.stack(spk_rec).sum(dim=0)

        # Ratio of votes needed for each classifier to count
        ratio = 0.4 * T

        if self.net_output:
            out_final = self.output_layer(out)
        else:
            out_dict = {}
            for i in range(self.num_classes):
                if i not in out_dict:
                    out_dict[i] = 0
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    idx = ((i * (self.num_classes - 1)) - ((i * (i - 1)) // 2) + (j - i - 1)) * 2
                    if i != j:
                        if out[0, idx].item() - out[0, idx+1].item() > ratio or out[0, idx+1].item() - out[0, idx].item() > ratio:
                            out_dict[i] += out[0, idx].item()
                            out_dict[j] += out[0, idx+1].item()

            out_final = torch.zeros((1, len(out_dict)))
            for k, v in out_dict.items():
                out_final[0, k] = v

        return out_final, (mem1, mem2, mem3)

    def load_from_ensemble(self, ensemble_model):
        with torch.no_grad():
            fc1_weight = torch.Tensor()
            fc1_bias = torch.Tensor()
            fc2_weight = torch.zeros(self.fc2.weight.shape)
            fc2_bias = torch.Tensor()
            fc3_weight = torch.zeros(self.fc3.weight.shape)
            fc3_bias = torch.Tensor()
            print(self.fc1.weight.shape)
            print(self.fc1.bias.shape)
            print(self.fc2.weight.shape)
            print(self.fc2.bias.shape)
            print(self.fc3.weight.shape)
            print(self.fc3.bias.shape)
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    if i != j:
                        bclass = ensemble_model.get_model_by_classes(i, j)
                        if bclass is not None:
                            # Load weights into corresponding slice of fc layers
                            if i == 0 and j == 1:
                                fc1_weight = bclass.fc1.weight.data.clone()
                                fc1_bias = bclass.fc1.bias.data.clone()
                                fc2_weight[0:32, 0:64] = bclass.fc2.weight.data.clone()
                                fc2_bias = bclass.fc2.bias.data.clone()
                                fc3_weight[0:2, 0:32] = bclass.fc3.weight.data.clone()
                                fc3_bias = bclass.fc3.bias.data.clone()
                            else:
                                idx = ((i * (self.num_classes - 1)) - ((i * (i - 1)) // 2) + (j - i - 1))
                                fc1_weight = torch.cat((fc1_weight, bclass.fc1.weight.data.clone()), dim=0)
                                fc1_bias = torch.cat((fc1_bias, bclass.fc1.bias.data.clone()), dim=0)
                                fc2_weight[32 * idx:32 * (idx+1), 64 * idx:64 * (idx+1)] = bclass.fc2.weight.data.clone()
                                fc2_bias = torch.cat((fc2_bias, bclass.fc2.bias.data.clone()), dim=0)
                                fc3_weight[2 * idx:2 * (idx+1), 32 * idx:32 * (idx+1)] = bclass.fc3.weight.data.clone()
                                fc3_bias = torch.cat((fc3_bias, bclass.fc3.bias.data.clone()), dim=0)
            self.fc1.weight.data = fc1_weight
            self.fc1.bias.data = fc1_bias
            self.fc2.weight.data = fc2_weight
            self.fc2.bias.data = fc2_bias
            self.fc3.weight.data = fc3_weight
            self.fc3.bias.data = fc3_bias
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False
            self.fc3.weight.requires_grad = False
            self.fc3.bias.requires_grad = False
            print("Loaded weights from ensemble model.")
            print(self.fc1.weight.shape)
            print(self.fc1.bias.shape)
            print(self.fc2.weight.shape)
            print(self.fc2.bias.shape)
            print(self.fc3.weight.shape)
            print(self.fc3.bias.shape)

    def train_output_layer(self, tr_ds, te_ds):
        train_dataset = tr_ds
        test_dataset = te_ds

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        optimiser = torch.optim.Adam(self.output_layer.parameters(), lr=0.001)
        # count class occurrences for loss weighting
        class_counts = [0] * self.num_classes
        for _, label in train_dataset:
            class_counts[int(label.item())] += 1
        total_counts = sum(class_counts)
        class_weights = [total_counts / (self.num_classes * class_counts[i]) for i in range(self.num_classes)]
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        epochs = 5
        self.to(device)
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            pbar = tqdm.tqdm(train_loader, desc=f"Training Output Layer, Epoch {epoch+1}/{epochs}")
            for data, labels in pbar:
                data, labels = data.to(device), labels.to(device)
                optimiser.zero_grad()
                outputs, _  = self(data)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimiser.step()
                train_loss += loss.item()
                pbar.set_postfix({"Loss": train_loss / (pbar.n + 1)})

            self.eval()
            pbar = tqdm.tqdm(test_loader, desc="Testing Output Layer")
            correct = 0
            total = 0
            with torch.no_grad():
                for data, labels in pbar:
                    data, labels = data.to(device), labels.to(device)
                    outputs, _ = self(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f'Test Accuracy after Epoch {epoch+1}: {accuracy*100:.2f}%')

def train_bclass(model, tr_ds, te_ds, criterion, optimiser, device):
    req_idx = [model.c_1, model.c_2]
    train_dataset = fft_dataset.get_by_labels(tr_ds, req_idx, False, with_balance=1)
    test_dataset = fft_dataset.get_by_labels(te_ds, req_idx, False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if criterion is None:
        # Calculate class weights
        class_counts = [0, 0]
        for _, label in train_dataset:
            class_counts[label] += 1
        total_counts = sum(class_counts)
        class_weights = [total_counts / (2 * class_counts[i]) for i in range(2)]
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


    epochs = 3
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
    tps = {}
    fps = {}
    fns = {}
    y_true = []
    y_pred = []
    with torch.no_grad():
        pbar = tqdm.tqdm(test_loader, desc="Testing")
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.append(labels.item())
            y_pred.append(predicted.item())
            if predicted.item() == labels.item():
                tps[labels.item()] = tps.get(labels.item(), 0) + 1
            else:
                fps[predicted.item()] = fps.get(predicted.item(), 0) + 1
                fns[labels.item()] = fns.get(labels.item(), 0) + 1
    accuracy = correct / total
    for i in range(2):
        prec = tps.get(i, 0) / (tps.get(i, 0) + fps.get(i, 0) + 1e-8)
        rec = tps.get(i, 0) / (tps.get(i, 0) + fns.get(i, 0) + 1e-8)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        print(f"Class {req_idx[i]} - Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    cm = metrics.confusion_matrix(y_true, y_pred)
    g.plot_confusion_matrix(cm, [g.CLASS_NAMES[req_idx[0]], g.CLASS_NAMES[req_idx[1]]])
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

def run_sample_and_visualise(model, ds, sample_idx = None):
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
    cv2.imshow(f"Binary Square Classifier Heatmap {sample_idx}", heatmap_img)
    cv2.imwrite(f"./imgs/binary_square/{sample_idx}_vote_heatmap.png", heatmap_img)
    bin_heatmap_img, classes = utils.heatmap.generate_heatmap(hm_dicts[1], ds.sorted_unique, use_acc=False)
    cv2.imshow(f"Binary Square Classifier Binary Heatmap {sample_idx}", bin_heatmap_img)
    cv2.imwrite(f"./imgs/binary_square/{sample_idx}_class_heatmap.png", bin_heatmap_img)
    cv2.waitKey(0)

def generate_confusion_matrix(tp_dict, fp_dict, fn_dict, num_c):
    confusion_matrix = np.zeros((num_c, num_c), dtype=int)
    for true_label in range(num_c):
        for pred_label in range(num_c):
            if true_label == pred_label:
                confusion_matrix[true_label][pred_label] = tp_dict.get(true_label, 0)
            else:
                confusion_matrix[true_label][pred_label] = fp_dict.get(pred_label, 0) if pred_label in fp_dict else 0
    return confusion_matrix

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = fft_dataset.MFCCDataset(type="all")
    train_ds, test_ds = torch.utils.data.random_split(ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))])

    # train_ds = fft_dataset.MFCCDataset(type="all", source="/data/d-phonemes/train")
    # test_ds = fft_dataset.MFCCDataset(type="all", source="/data/timit-phonemes/test")

    c_num = 38

    model = BinarySquareModel(c_num, train_ds, test_ds, False).to(device)
    comb_model = BinarySquareModelCombined(c_num, True).to(device)
    print(len(model.bc_list))

    # torch.save(model.state_dict(), f"./bsquares/bsquare_{c_num}_fc_bal.pth")
    model.load_state_dict(torch.load(f"./bsquares/bsquare_{c_num}_fc_bal.pth", map_location=device))
    print(f"total num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"class num params: {sum(p.numel() for p in model.bc_list[0].parameters() if p.requires_grad)}")
    # run_sample_and_visualise(model, ds, 28384)
    print(f"Loading from ensemble...")
    # comb_model.load_from_ensemble(model)
    # comb_model.train_output_layer(train_ds, test_ds)
    comb_model_weights = torch.load(f"./bsquares/bsquare_combined_{c_num}_net_bal.pth", map_location=device)
    comb_model.load_state_dict(comb_model_weights)
    print(comb_model)
    print(f"combined total num params: {sum(p.numel() for p in comb_model.parameters() if p.requires_grad)}")
    # for i in range(4):
    #     run_sample_and_visualise(model, ds, None)

    # acc_d = test_bcs(model)
    # model.acc_dict = acc_d
    #
    # heatmap_img, classes = utils.heatmap.generate_heatmap(model.acc_dict, ds.sorted_unique)
    # cv2.imwrite(f"tsquare_heatmap_{c_num}.png", heatmap_img)
    # # # Display the heatmap
    # cv2.imshow("Binary Square Classifier Heatmap", heatmap_img)
    # cv2.waitKey(0)
    # train_ds, test_ds = torch.utils.data.random_split(ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # sample, label = next(iter(test_loader))
    # model_out, _ = model(sample.to(device))
    # comb_model_out, _ = comb_model(sample.to(device))
    # model_count = 0
    # comb_model_count = 0
    # print(model_out.shape, comb_model_out.shape)
    # for i in range(len(model_out[0])):
    #     model_count += model_out[0][i]
    #     comb_model_count += comb_model_out[0][i]
    #     print(f"output {i}: {model_out[0][i]} {comb_model_out[0][i]} {'MATCH' if (model_out[0][i] == comb_model_out[0][i]) else 'DIFF'} {model_count} {comb_model_count}")

    correct = 0
    total = 0
    comb_model.eval()
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}
    y_true = []
    y_pred = []
    with torch.no_grad():
        pbar = tqdm.tqdm(test_loader, desc="Testing Binary Square Model")
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            outputs, _ = comb_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                if true_label == pred_label:
                    tp_dict[true_label] = tp_dict.get(true_label, 0) + 1
                else:
                    fp_dict[pred_label] = fp_dict.get(pred_label, 0) + 1
                    fn_dict[true_label] = fn_dict.get(true_label, 0) + 1
    accuracy = correct / total
    prec_dict = {}
    rec_dict = {}
    f1_dict = {}
    for c in range(c_num):
        tp = tp_dict.get(c, 0)
        fp = fp_dict.get(c, 0)
        fn = fn_dict.get(c, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        prec_dict[c] = precision
        rec_dict[c] = recall
        f1_dict[c] = f1_score
    classes_per_line = 10
    print("Class-wise F1 Scores:")
    for i in range(0, len(ds.sorted_unique), classes_per_line):
        line_classes = [j for j in range(i, min(i + classes_per_line, len(ds.sorted_unique)))]
        line_f1s = [f"{f1_dict[c]:.4f}" for c in line_classes]
        print("Label:", " ".join(f"{ds.sorted_unique[i:i+classes_per_line][j]:>6}" for j in range(len(line_classes))))
        print("Classes:", " ".join(f"{c:>6}" for c in line_classes))
        print("F1 Scores:", " ".join(f"{f:>5}" for f in line_f1s))
        print("Precision:", " ".join(f"{prec_dict[c]:>5.4f}" for c in line_classes))
        print("Recall:   ", " ".join(f"{rec_dict[c]:>5.4f}" for c in line_classes))
    print(f"Mean F1 Score: {sum(f1_dict.values())/len(f1_dict):.4f}")
    print(f'Overall Test Accuracy: {accuracy*100:.2f}%')

    cm = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    np.savetxt(f"./confusion_matrices/binary_square_confusion_matrix_{c_num}_net_bal.csv", cm, delimiter=",", fmt='%d')
    torch.save(comb_model.state_dict(), f"./bsquares/bsquare_combined_{c_num}_net_bal.pth")