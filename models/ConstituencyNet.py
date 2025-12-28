import torch
import torch.nn as nn
from torch.utils.data import Dataset

import tqdm

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

class SmallConstituency(nn.Module):
    def __init__(self, class_list):
        super(SmallConstituency, self).__init__()
        self.class_list = class_list
        self.num_outputs = len(class_list)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.do = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(2)
        self.fc1 = nn.Linear(64 * 2 * 2, self.num_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.do(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.do(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.gap(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        return x

    def get_hidden_weights(self):
        weights = [self.conv2.weight.data.clone(), self.conv3.weight.data.clone(), self.conv4.weight.data.clone(), self.fc1.weight.data.clone()]
        biases = [self.conv2.bias.data.clone(), self.conv3.bias.data.clone(), self.conv4.bias.data.clone(), self.fc1.bias.data.clone()]
        return weights, biases

class ConstituencyNet(nn.Module):
    def __init__(self, constituency_structures, out_type='sum', num_classes=10):
        # out_type is 'rp' for ranked pairs, 'bin' for binary, 'ann' for ann, 'sum' for sum
        super(ConstituencyNet, self).__init__()
        self.num_constituencies = len(constituency_structures)
        self.num_classes = num_classes
        self.rp = out_type == 'rp'
        self.bin = out_type == 'bin'
        self.ann = out_type == 'ann'
        self.sum = out_type == 'sum'
        for i in range(len(constituency_structures)):
            setattr(self, f"constituency_{i}", SmallConstituency(constituency_structures[i]))
        self.classifiers = [getattr(self, f"constituency_{i}") for i in range(len(constituency_structures))]

        if self.ann:
            total_num_outputs = 0
            for classifier in self.classifiers:
                total_num_outputs += classifier.num_outputs
            self.ann_layer = nn.Linear(total_num_outputs, self.num_classes)

    def forward(self, x):
        if not x.is_cuda:
            out_list = []
            for classifier in self.classifiers:
                classifier.eval()
                out = classifier(x)
                out_list.append(nn.Softmax(dim=1)(out))
        else:
            stream_list = []
            for classifier in self.classifiers:
                classifier.eval()
                stream = torch.cuda.Stream()
                stream_list.append(stream)
            out_list = []
            for i, classifier in enumerate(self.classifiers):
                with torch.cuda.stream(stream_list[i]):
                    out = classifier(x)
                    if self.ann:
                        out_list.append(out)
                    else:
                        out_list.append(nn.Softmax(dim=1)(out))
            torch.cuda.synchronize()

        if self.sum:
            final_out = torch.zeros(x.size(0), self.num_classes).to(x.device)
            for i, classifier in enumerate(self.classifiers):
                for j, class_idx in enumerate(classifier.class_list):
                    final_out[:, class_idx] += out_list[i][:, j]
        elif self.rp:
            # Build pairwise comparison matrix
            final_out = torch.zeros(x.size(0), self.num_classes, self.num_classes).to(x.device)
            for i, classifier in enumerate(self.classifiers):
                subset = classifier.class_list
                scores = out_list[i]

                order = torch.argsort(scores, dim=1, descending=True)

                for r1 in range(len(subset)):
                    a = torch.tensor([subset[idx] for idx in order[:, r1].tolist()], device=x.device)
                    for r2 in range(r1 + 1, len(subset)):
                        b = torch.tensor([subset[idx] for idx in order[:, r2].tolist()], device=x.device)
                        final_out[torch.arange(x.size(0)), a, b] += 1
            # Count wins
            final_out = torch.sum(final_out > (len(self.classifiers) / 2), dim=2)
        elif self.ann:
            # Concatenate all classifier outputs
            concat_out = torch.cat(out_list, dim=1)
            # Creates B x 50 tensor if 10 constituencies with 5 outputs each
            concat_out = concat_out - concat_out.mean(dim=1, keepdim=True)
            concat_out = concat_out / (concat_out.std(dim=1, keepdim=True) + 1e-6)
            final_out = self.ann_layer(concat_out)

        return final_out

    def train_classifiers(self, tr_ds, te_ds, epochs=3, lr=1e-3, device='cpu'):
        print("Training ConstituencyNet Classifiers")
        # criterion = nn.KLDivLoss(reduction='batchmean')
        criterion = nn.CrossEntropyLoss()
        optimisers = [torch.optim.Adam(classifier.parameters(), lr=lr) for classifier in self.classifiers]

        for i, classifier in enumerate(self.classifiers):
            classifier.to(device)
            classifier.train()

            train_ds_cl = []
            test_ds_cl = []
            for data, label in tr_ds:
                if label in classifier.class_list:
                    train_ds_cl.append((data, classifier.class_list.index(label)))
            for data, label in te_ds:
                if label in classifier.class_list:
                    test_ds_cl.append((data, classifier.class_list.index(label)))

            tr_dataset = ListDataset(train_ds_cl)
            te_dataset = ListDataset(test_ds_cl)

            tr_dl = torch.utils.data.DataLoader(tr_dataset, batch_size=64, shuffle=True)
            te_dl = torch.utils.data.DataLoader(te_dataset, batch_size=64, shuffle=False)

            for epoch in range(epochs):
                pbar = tqdm.tqdm(tr_dl)
                mean_loss = 0.0
                for j, (data, labels) in enumerate(pbar):
                    data, labels = data.to(device), labels.to(device)
                    optimisers[i].zero_grad()
                    outputs = classifier(data)
                    loss = criterion(outputs, labels)
                    mean_loss += loss.item()
                    loss.backward()
                    optimisers[i].step()
                    classifier.zero_grad()
                    pbar.set_description(f"Classifier {i+1}/{self.num_constituencies} Epoch {epoch+1} Loss: {mean_loss/(j+1):.4f}")

                qbar = tqdm.tqdm(te_dl)
                correct = 0
                total = 0
                top2 = 0
                top3 = 0
                for j, (data, labels) in enumerate(qbar):
                    data, labels = data.to(device), labels.to(device)
                    outputs = classifier(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    top2 += (labels.unsqueeze(1) == torch.topk(outputs, 2, dim=1).indices).any(dim=1).sum().item()
                    top3 += (labels.unsqueeze(1) == torch.topk(outputs, 3, dim=1).indices).any(dim=1).sum().item()
                    qbar.set_description(f"Classifier {i+1}/{self.num_constituencies} Epoch {epoch+1} Test Accuracy: {100 * correct / total:.2f}%")
                print(f"Classifier {i+1}/{self.num_constituencies} Epoch {epoch+1} Test Accuracy: {100 * correct / total:.2f}%")
                print(f"Classifier {i+1}/{self.num_constituencies} Epoch {epoch+1} Test Top-2 Accuracy: {100 * top2 / total:.2f}%")
                print(f"Classifier {i+1}/{self.num_constituencies} Epoch {epoch+1} Test Top-3 Accuracy: {100 * top3 / total:.2f}%")
            classifier.to('cpu')
        print("Training complete.")

    def train_ann(self, tr_ds, te_ds, epochs=3, lr=1e-3, device='cpu'):
        if not self.ann:
            print("ANN layer not defined in this model.")
            return

        print("Training ANN Layer")
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.ann_layer.parameters(), lr=lr)

        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=64, shuffle=True)
        te_dl = torch.utils.data.DataLoader(te_ds, batch_size=64, shuffle=False)

        self.ann_layer.to(device)

        for epoch in range(epochs):
            self.ann_layer.train()
            pbar = tqdm.tqdm(tr_dl)
            mean_loss = 0.0
            for j, (data, labels) in enumerate(pbar):
                data, labels = data.to(device), labels.to(device)
                optimiser.zero_grad()
                outputs = self.forward(data)
                loss = criterion(outputs, labels)
                mean_loss += loss.item()
                loss.backward()
                optimiser.step()
                # self.ann_layer.zero_grad()
                pbar.set_description(f"ANN Epoch {epoch+1} Loss: {mean_loss/(j+1):.4f}")

            self.ann_layer.eval()
            with torch.no_grad():
                qbar = tqdm.tqdm(te_dl)
                correct = 0
                total = 0
                top2 = 0
                top3 = 0
                for j, (data, labels) in enumerate(qbar):
                    data, labels = data.to(device), labels.to(device)
                    outputs = self.forward(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    top2 += (labels.unsqueeze(1) == torch.topk(outputs, 2, dim=1).indices).any(dim=1).sum().item()
                    top3 += (labels.unsqueeze(1) == torch.topk(outputs, 3, dim=1).indices).any(dim=1).sum().item()
                    qbar.set_description(f"ANN Epoch {epoch+1} Test Accuracy: {100 * correct / total:.2f}%")
                print(f"ANN Epoch {epoch+1} Test Accuracy: {100 * correct / total:.2f}%")
                print(f"ANN Epoch {epoch+1} Test Top-2 Accuracy: {100 * top2 / total:.2f}%")
                print(f"ANN Epoch {epoch+1} Test Top-3 Accuracy: {100 * top3 / total:.2f}%")

    def load_from_no_net(self, no_net_model):
        with torch.no_grad():
            for i, classifier in enumerate(self.classifiers):
                no_net_classifers = no_net_model.classifiers
                classifier.load_state_dict(no_net_classifers[i].state_dict())
                for param in classifier.parameters():
                    param.requires_grad = False
            print("Loaded weights from no-net model.")

if __name__ == "__main__":
    constituencies = [
        [0, 1, 6, 7, 8],
        [0, 1, 2, 5, 9],
        [1, 2, 3, 4, 8],
        [1, 3, 6, 8, 9],
        [0, 1, 4, 5, 9],
        [1, 2, 4, 5, 8],
        [1, 3, 6, 7, 8],
        [0, 1, 5, 6, 7],
        [1, 2, 6, 7, 8],
        [2, 3, 4, 8, 9]
    ]
    model = ConstituencyNet([constituencies], out_type='ann', num_classes=10)
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    print(output)
    print(output.shape)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")