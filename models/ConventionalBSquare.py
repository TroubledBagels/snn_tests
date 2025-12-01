import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import tqdm
import tonic
from tonic.transforms import ToFrame
from torch.utils.data import DataLoader, Dataset
from random import shuffle
import utils.general as g
import cv2

class ListDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (events, label)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        events, label = self.samples[idx]
        if self.transform is not None:
            events = self.transform(events)  # ToFrame applied per sample
        return events, label

class BClassifier(nn.Module):
    def __init__(self, c_1, c_2, input_size, hidden_size, output_size, num_layers=2):
        super(BClassifier, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        for i in range(num_layers - 1):
            setattr(self, f'fc{i+2}', nn.Linear(hidden_size, hidden_size))
        self.fc_out = nn.Linear(hidden_size, output_size)
        print(f"Initialized BClassifier for classes {c_1} and {c_2} with {num_layers} layers.")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        for i in range(self.num_layers - 1):
            x = getattr(self, f'fc{i+2}')(x)
            x = torch.relu(x)
        x = self.fc_out(x)
        return x, None

    def get_hidden_weights(self):
        weights = []
        biases = []
        for i in range(self.num_layers - 1):
            weights.append(getattr(self, f'fc{i+2}').weight.data.clone())
            biases.append(getattr(self, f'fc{i+2}').bias.data.clone())
        return weights, biases

class BConvClassifier(nn.Module):
    def __init__(self, c_1, c_2, input_channels, hidden_channels, output_size, num_layers=2):
        super(BConvClassifier, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=5, stride=2, padding=2)
        for i in range(num_layers - 1):
            setattr(self, f'conv{i+2}', nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            # setattr(self, f'pool{i+2}', nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc_out = nn.Linear(hidden_channels * 16 * 16, output_size)
        print(f"Initialized BConvClassifier for classes {c_1} and {c_2} with {num_layers} conv layers.")

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        for i in range(self.num_layers - 1):
            x = getattr(self, f'conv{i+2}')(x)
            x = torch.relu(x)
            # x = getattr(self, f'pool{i+2}')(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x, None

    def get_hidden_weights(self):
        weights = []
        biases = []
        for i in range(self.num_layers - 1):
            weights.append(getattr(self, f'conv{i+2}').weight.data.clone())
            biases.append(getattr(self, f'conv{i+2}').bias.data.clone())
        return weights, biases

class TinyCNN(nn.Module):
    def __init__(self, c_1, c_2, hid, inp, out, num_layers=2):
        super(TinyCNN, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.do = nn.Dropout(0.2)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.do(x)
        # x = self.conv2(x)
        # x = torch.relu(x)
        # x = self.bn2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.bn3(x)
        x = self.do(x)
        # x = self.conv4(x)
        # x = torch.relu(x)
        x = self.gap(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x, None

    def get_hidden_weights(self):
        weights = [self.conv2.weight.data.clone(), self.conv3.weight.data.clone(), self.fc1.weight.data.clone()]
        biases = [self.conv2.bias.data.clone(), self.conv3.bias.data.clone(), self.fc1.bias.data.clone()]
        return weights, biases

class BSquareModel(nn.Module):
    def __init__(self, num_classes: int, input_size=30, hidden_size=32, num_layers=2, binary_voting=False, bclass=BClassifier, net_out=False):
        super(BSquareModel, self).__init__()
        self.num_classes = num_classes
        self.binary_voting = binary_voting
        self.net_out = net_out
        self.classifiers = nn.ModuleList()
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                self.classifiers.append(bclass(i, j, input_size, hidden_size, 2, num_layers))
        if net_out:
            self.out_layer = nn.Sequential(
                nn.Linear(len(self.classifiers) * 2, 64),
                nn.Linear(64, num_classes)
            )
            print("Initialized ANN output layer for BSquareModel.")

    def forward(self, x) -> torch.Tensor:
        B = x.size(0)
        votes = torch.zeros(B, self.num_classes, device=x.device)
        if self.net_out:
            out_list = []
            for classifier in self.classifiers:
                out, _ = classifier(x)
                out_list.append(out)
            out_tensor = torch.cat(out_list, dim=1)
            ann_out = self.out_layer(out_tensor)
            return ann_out
        else:
            for classifier in self.classifiers:
                out, _ = classifier(x)
                c_1, c_2 = classifier.c_1, classifier.c_2
                if self.binary_voting:
                    preds = out.argmax(dim=1)
                    for b in range(B):
                        if preds[b] == 0:
                            votes[b, c_1] += 1
                        else:
                            votes[b, c_2] += 1
                else:
                    # add individual spikes
                    for b in range(B):
                        if abs(votes[b, c_1] - votes[b, c_2]) > 0.1:
                            votes[b, c_1] += out[b, 0]
                            votes[b, c_2] += out[b, 1]
                    # votes[:, c_1] += out[:, 0]
                    # votes[:, c_2] += out[:, 1]
        return votes

    def train_classifiers(self, train_ds, test_ds, epochs=3, lr=1e-3, device='cpu'):
        print("Training classifiers...")
        criterion = nn.CrossEntropyLoss()
        print("Initialising optimisers...")
        optimizers = [torch.optim.Adam(classifier.parameters(), lr=lr) for classifier in self.classifiers]
        tr_ds_dict = {}
        tl_ds_dict = {}
        for i in range(self.num_classes):
            tr_ds_dict[i] = []
            tl_ds_dict[i] = []
        print("Separating training set by class...")
        for i in range(len(train_ds)):
            data, label = train_ds[i]
            tr_ds_dict[label].append((data, label))
            if (i+1) % 100 == 0:
                print(f"\rProcessed {i+1}/{len(train_ds)} training samples", end="")
        print("")
        print("Separating test set by class...")
        for i in range(len(test_ds)):
            data, label = test_ds[i]
            tl_ds_dict[label].append((data, label))
            if (i+1) % 100 == 0:
                print(f"\rProcessed {i+1}/{len(test_ds)} test samples", end="")
        print("")

        print("Training...")
        acc_dict = {}
        test_loss_dict = {}
        for idx, classifier in enumerate(self.classifiers):
            cl_tr_ds = tr_ds_dict[classifier.c_1] + tr_ds_dict[classifier.c_2]
            shuffle(cl_tr_ds)
            cl_tr_dataset = ListDataset(cl_tr_ds, transform=None)
            train_dl = DataLoader(cl_tr_dataset, batch_size=64, shuffle=True)
            cur_best = None
            cur_best_acc = 0.0
            best_loss = 0.0

            cl_te_ds = tl_ds_dict[classifier.c_1] + tl_ds_dict[classifier.c_2]
            cl_te_dataset = ListDataset(cl_te_ds, transform=None)
            test_dl = DataLoader(cl_te_dataset, batch_size=64, shuffle=False)

            classifier.train()
            for epoch in range(epochs):
                pbar = tqdm.tqdm(train_dl)
                mean_loss = 0.0
                for i, (data, target) in enumerate(pbar):
                    data = data.float()
                    data, target = data.to(device), target.to(device)
                    target_binary = (target == classifier.c_2).long()
                    optimizers[idx].zero_grad()
                    output, _ = classifier(data)
                    loss = criterion(output, target_binary)
                    loss.backward()
                    mean_loss += loss.item()
                    optimizers[idx].step()
                    pbar.set_description(f"Classifier {classifier.c_1} vs {classifier.c_2} Epoch {epoch+1}/{epochs} Loss: {mean_loss/(i+1):.4f}")

                correct = 0
                total = 0
                te_loss = 0.0
                classifier.eval()
                with torch.no_grad():
                    pbar = tqdm.tqdm(test_dl)
                    for data, target in pbar:
                        data = data.float()
                        data, target = data.to(device), target.to(device)
                        target_binary = (target == classifier.c_2).long()
                        te_loss += criterion(output, target_binary).item()
                        output, _ = classifier(data)
                        preds = output.argmax(dim=1)
                        correct += (preds == target_binary).sum().item()
                        total += target_binary.size(0)
                        pbar.set_description(f"Evaluating Classifier {classifier.c_1} vs {classifier.c_2}")
                accuracy = 100 * correct / total
                print(f"Classifier {classifier.c_1} vs {classifier.c_2} Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%")
                if cur_best is None or accuracy > cur_best_acc:
                    cur_best = classifier.state_dict()
                    cur_best_acc = accuracy
                    best_loss = te_loss / len(test_dl)
            print(f"Best accuracy for Classifier {classifier.c_1} vs {classifier.c_2}: {cur_best_acc:.2f}%")
            classifier.load_state_dict(cur_best)
            acc_dict[(classifier.c_1, classifier.c_2)] = cur_best_acc
            test_loss_dict[(classifier.c_1, classifier.c_2)] = best_loss
        print("Finished training all classifiers.")
        print("Classifier accuracies:")
        for key in acc_dict:
            print(f" Classes {key[0]} vs {key[1]}: {acc_dict[key]:.2f}% with loss: {test_loss_dict[key]:.4f}")
        return acc_dict

    def get_model_by_classes(self, c_1, c_2):
        for classifier in self.classifiers:
            if (classifier.c_1 == c_1 and classifier.c_2 == c_2) or (classifier.c_1 == c_2 and classifier.c_2 == c_1):
                return classifier
        return None

    def train_output_layer(self, tr_ds, te_ds, epochs=3, lr=1e-3, device='cpu'):
        print("Training ANN output layer...")
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.out_layer.parameters(), lr=lr)

        train_dl = DataLoader(tr_ds, batch_size=100, shuffle=True)
        test_dl = DataLoader(te_ds, batch_size=100, shuffle=False)

        for epoch in range(epochs):
            self.out_layer.train()
            pbar = tqdm.tqdm(train_dl)
            mean_loss = 0.0
            for i, (data, target) in enumerate(pbar):
                data = data.float()
                data, target = data.to(device), target.to(device)
                optimiser.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)
                loss.backward()
                mean_loss += loss.item()
                optimiser.step()
                pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {mean_loss/(i+1):.4f}")

            correct = 0
            total = 0
            self.out_layer.eval()
            with torch.no_grad():
                pbar = tqdm.tqdm(test_dl)
                for data, target in pbar:
                    data = data.float()
                    data, target = data.to(device), target.to(device)
                    output = self.forward(data)
                    preds = output.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
                    pbar.set_description(f"Evaluating ANN Output Layer")
            accuracy = 100 * correct / total
            print(f"ANN Output Layer Test Accuracy: {accuracy:.2f}%")

    def load_from_no_net(self, no_net_model):
        with torch.no_grad():
            for i, classifier in enumerate(self.classifiers):
                no_net_classifier = no_net_model.get_model_by_classes(classifier.c_1, classifier.c_2)
                if no_net_classifier is not None:
                    print(f"Loading weights for classes {classifier.c_1} and {classifier.c_2}...")
                    classifier.load_state_dict(no_net_classifier.state_dict())
            print("Loaded weights from no-net model.")


class BSquareModelCombined(nn.Module):
    def __init__(self, num_classes, input_size=30, hidden_size=32, num_layers=2, binary_voting=False, net_output=False):
        super(BSquareModelCombined, self).__init__()
        self.binary_voting = binary_voting
        self.net_output = net_output
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        tri_num = (num_classes * (num_classes - 1)) // 2
        total_hidden_size = hidden_size * tri_num
        self.fc1 = nn.Linear(input_size, total_hidden_size)
        for i in range(num_layers - 1):
            setattr(self, f'fc{i+2}', nn.Linear(total_hidden_size, total_hidden_size))
        self.fc_out = nn.Linear(total_hidden_size, 2 * tri_num)
        print(f"Initialized BSquareModelCombined with {num_layers} layers.")
        if net_output:
            self.ann_out = nn.Linear(tri_num * 2, num_classes)

    def forward(self, x, ratio=0.0) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, -1)
        x = self.fc1(x)
        x = torch.relu(x)
        for i in range(self.num_layers - 1):
            x = getattr(self, f'fc{i+2}')(x)
            x = torch.relu(x)
        out = self.fc_out(x)

        votes = torch.zeros(B, self.num_classes, device=x.device)
        if self.net_output:
            ann_out = self.ann_out(out)
            return ann_out
        else:
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    if not self.binary_voting:
                        idx = ((i * (self.num_classes - 1)) - ((i * (i - 1)) // 2) + (j - i - 1)) * 2
                        for b in range(B):
                            if out[b, idx] - out[b, idx + 1] > ratio or out[b, idx + 1] - out[b, idx] >= ratio:
                                votes[b, i] += out[b, idx]
                                votes[b, j] += out[b, idx + 1]
                    else:
                        idx = ((i * (self.num_classes - 1)) - ((i * (i - 1)) // 2) + (j - i - 1)) * 2
                        preds = out[:, idx:idx+2].argmax(dim=1)
                        for b in range(B):
                            if preds[b] == 0:
                                votes[b, i] += 1
                            else:
                                votes[b, j] += 1

        return votes

    def load_ensemble(self, ensemble_model):
        with torch.no_grad():
            fc1_weight = torch.Tensor()
            fc1_bias = torch.Tensor()
            fc_out_weight = torch.zeros(self.fc_out.weight.shape)
            fc_out_bias = torch.Tensor()
            other_fc_weights = [torch.zeros(getattr(self, f'fc{l+2}').weight.shape) for l in range(self.num_layers - 1)]
            other_fc_biases = [torch.Tensor() for _ in range(self.num_layers - 1)]
            for i in range(self.num_classes):
                for j in range(i+1, self.num_classes):
                    if i != j:
                        bclass = ensemble_model.get_model_by_classes(i, j)
                        print(f"Loading weights for classes {i} and {j}...")
                        if bclass is not None:
                            # Load weights into corresponding slice of fc layers
                            if i == 0 and j == 1:
                                fc1_weight = bclass.fc1.weight.data.clone()
                                fc1_bias = bclass.fc1.bias.data.clone()
                                fc_out_weight[0:2, 0:self.hidden_size] = bclass.fc_out.weight.data.clone()
                                fc_out_bias = bclass.fc_out.bias.data.clone()
                                hidden_weights, hidden_biases = bclass.get_hidden_weights()
                                for l in range(len(other_fc_weights)):
                                    other_fc_weights[l][0:self.hidden_size, 0:self.hidden_size] = hidden_weights[l]
                                    other_fc_biases[l] = hidden_biases[l]
                            else:
                                idx = ((i * (self.num_classes - 1)) - ((i * (i - 1)) // 2) + (j - i - 1))
                                fc1_weight = torch.cat((fc1_weight, bclass.fc1.weight.data.clone()), dim=0)
                                fc1_bias = torch.cat((fc1_bias, bclass.fc1.bias.data.clone()), dim=0)
                                hidden_weights, hidden_biases = bclass.get_hidden_weights()
                                for l in range(self.num_layers - 1):
                                    print(self.hidden_size * idx, self.hidden_size * (idx+1))

                                    other_fc_weights[l][self.hidden_size * idx:self.hidden_size * (idx+1), self.hidden_size * idx:self.hidden_size * (idx+1)] = hidden_weights[l]
                                    other_fc_biases[l] = torch.cat((other_fc_biases[l], hidden_biases[l]), dim=0)
                                fc_out_weight[2*idx:2*(idx+1), self.hidden_size*idx:self.hidden_size*(idx+1)] = bclass.fc_out.weight.data.clone()
                                fc_out_bias = torch.cat((fc_out_bias, bclass.fc_out.bias.data.clone()), dim=0)
            self.fc1.weight.data = fc1_weight
            self.fc1.bias.data = fc1_bias
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False
            self.fc_out.weight.data = fc_out_weight
            self.fc_out.bias.data = fc_out_bias
            self.fc_out.weight.requires_grad = False
            self.fc_out.bias.requires_grad = False
            print(f"fc1 weight: {self.fc1.weight.shape}")
            for l in range(self.num_layers - 1):
                getattr(self, f'fc{l+2}').weight.data = other_fc_weights[l]
                getattr(self, f'fc{l+2}').bias.data = other_fc_biases[l]
                getattr(self, f'fc{l+2}').weight.requires_grad = False
                getattr(self, f'fc{l+2}').bias.requires_grad = False
                print(f"fc{l+2} weight: {getattr(self, f'fc{l+2}').weight.shape}")
            print(f"fc_out weight: {self.fc_out.weight.shape}")
            print("Loaded weights from ensemble model.")

    def train_ann_out(self, tr_ds, te_ds, epochs=3, lr=1e-3):
        print("Training ANN output layer...")
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.ann_out.parameters(), lr=lr)

        train_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
        test_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

        for epoch in range(epochs):
            self.ann_out.train()
            pbar = tqdm.tqdm(train_dl)
            mean_loss = 0.0
            for i, (data, target) in enumerate(pbar):
                data = data.float()
                optimiser.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)
                loss.backward()
                mean_loss += loss.item()
                optimiser.step()
                pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {mean_loss/(i+1):.4f}")

            correct = 0
            total = 0
            self.ann_out.eval()
            with torch.no_grad():
                pbar = tqdm.tqdm(test_dl)
                for data, target in pbar:
                    data = data.float()
                    output = self.forward(data)
                    preds = output.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
                    pbar.set_description(f"Evaluating ANN Output Layer")
            accuracy = 100 * correct / total
            print(f"ANN Output Layer Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    model = BSquareModel(num_classes=4, input_size=34*34*2, hidden_size=16, num_layers=2, binary_voting=False)
    comb_model = BSquareModelCombined(num_classes=4, input_size=34*34*2, hidden_size=16, num_layers=2, binary_voting=False, net_output=False)
    print(model)
    print(comb_model)
    comb_model.load_ensemble(model)
    fc2 = getattr(model.classifiers[2], 'fc2')
    comb_fc2 = getattr(comb_model, 'fc2')
    print(comb_fc2.weight.data[32:48, 32:48].shape)
    print(fc2.weight.data.shape)
    print((comb_fc2.weight.data[32:48, 32:48] == fc2.weight.data).sum())
    print(comb_model)
    x = torch.randn(1, 2, 34, 34)  # Example input
    votes = comb_model(x)
    votes_m = model(x)
    print(votes == votes_m)
    print("Votes:", votes, votes_m)
