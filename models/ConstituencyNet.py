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
    def __init__(self, constituency_structures, out_type='rp', num_classes=10):
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
                    out_list.append(nn.Softmax(dim=1)(out))
            torch.cuda.synchronize()

        final_out = torch.zeros_like(out_list[0])
        for out in out_list:
            final_out += out
        return final_out

    def train_classifiers(self, tr_ds, te_ds, epochs=3, lr=1e-3, device='cpu'):
        print("Training ConstituencyNet Classifiers")
        # criterion = nn.KLDivLoss(reduction='batchmean')
        criterion = nn.CrossEntropyLoss()
        optimisers = [torch.optim.Adam(classifier.parameters(), lr=lr) for classifier in self.classifiers]

        tr_ds_dict = {}
        tl_ds_dict = {}
        for i in range(self.num_classes):
            tr_ds_dict[i] = []
            tl_ds_dict[i] = []
        print("Separating training set by class...")
        for i in range(len(tr_ds)):
            data, label = tr_ds[i]
            tr_ds_dict[label].append((data, label))
            if (i + 1) % 100 == 0:
                print(f"\rProcessed {i + 1}/{len(tr_ds)} training samples", end="")
        print("")
        print("Separating test set by class...")
        for i in range(len(te_ds)):
            data, label = te_ds[i]
            tl_ds_dict[label].append((data, label))
            if (i + 1) % 100 == 0:
                print(f"\rProcessed {i + 1}/{len(te_ds)} test samples", end="")
        print("")


        acc_dict = {}
        test_loss_dict = {}
        for idx, classifier in enumerate(self.classifiers):
            print(f"Training Classifier {idx+1}/{self.num_constituencies} with {len(classifier.class_list)} classes")
            classifier.to(device)

            cur_best_acc = 0.0
            best_loss = float('inf')
            best_model = None

            classifier.train()

            important_classes = classifier.class_list
            cl_tr_ds = []
            cl_te_ds = []
            for cls in important_classes:
                cl_tr_ds.extend(tr_ds_dict[cls])
                cl_te_ds.extend(tl_ds_dict[cls])

            cl_tr_ds_relative = []

            for i, (data, label) in enumerate(cl_tr_ds):
                cl_tr_ds_relative.append((data, important_classes.index(label)))
            cl_tr_ds = cl_tr_ds_relative

            cl_te_ds_relative = []
            for i, (data, label) in enumerate(cl_te_ds):
                cl_te_ds_relative.append((data, important_classes.index(label)))
            cl_te_ds = cl_te_ds_relative

            cl_tr_dataset = ListDataset(cl_tr_ds, transform=None)
            cl_te_dataset = ListDataset(cl_te_ds, transform=None)

            cl_tr_dl = torch.utils.data.DataLoader(cl_tr_dataset, batch_size=64, shuffle=True)
            cl_te_dl = torch.utils.data.DataLoader(cl_te_dataset, batch_size=64, shuffle=False)

            for epoch in range(epochs):
                pbar = tqdm.tqdm(cl_tr_dl)
                mean_loss = 0.0
                for i, (data, target) in enumerate(pbar):
                    data = data.float()
                    data, target = data.to(device), target.to(device)

                    # Create 1-hot encoding for target if it is in classifer.class_list
                    # If not, set softmax target to all zeros
                    # target_binary = torch.zeros(len(target), classifier.num_outputs, device=device)
                    # for j, t in enumerate(target):
                    #     if t.item() in classifier.class_list:
                    #         class_idx = classifier.class_list.index(t.item())
                    #         target_binary[j][class_idx] = 1.0
                    # target_binary = target_binary / target_binary.sum(dim=1, keepdim=True).clamp(min=1e-6)

                    optimisers[idx].zero_grad()
                    output = classifier(data)
                    # log_output = nn.LogSoftmax(dim=1)(output)
                    # loss = criterion(log_output, target_binary)
                    loss = criterion(output, target)
                    loss.backward()
                    mean_loss += loss.item()
                    optimisers[idx].step()
                    pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {mean_loss/(i+1):.4f}")

                correct = 0
                qbar = tqdm.tqdm(cl_te_dl)
                for i, (data, target) in enumerate(qbar):
                    data = data.float()
                    data, target = data.to(device), target.to(device)
                    with torch.no_grad():
                        output = classifier(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    acc = correct / len(cl_te_ds)
                    qbar.set_description(f"Test Accuracy: {acc*100:.2f}%")
                acc = correct / len(cl_te_ds)
                print(f"Classifier {idx} Test Accuracy: {acc*100:.2f}%")
                if acc > cur_best_acc:
                    cur_best_acc = acc
                    best_model = classifier.state_dict()
            classifier.load_state_dict(best_model)
            acc_dict[f"classifier_{idx}_accuracy"] = cur_best_acc
        return acc_dict


if __name__ == "__main__":
    model = ConstituencyNet([[0, 1, 2, 3, 4]])
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    print(output.shape)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")