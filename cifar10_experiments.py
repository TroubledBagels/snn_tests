import torch
import torch.nn as nn

import torchvision

import tqdm
import pathlib

import sys

class BClassModel(nn.Module):
    def __init__(self):
        super(BClassModel, self).__init__()
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
        # self.gap = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 2 * 2, 2)
        # self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.do(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x += res
        x = self.pool(x)
        x = self.conv3(x)
        res = x
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.do(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x += res
        x = self.gap(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        # x = torch.relu(x)
        # x = self.fc2(x)
        return x

def get_classes(ds, c_1, c_2):
    samples = []
    for img, label in ds:
        if label == c_1:
            samples.append((img, 0))
        elif label == c_2:
            samples.append((img, 1))
    return torch.utils.data.TensorDataset(torch.stack([s[0] for s in samples]), torch.tensor([s[1] for s in samples]))

if __name__ == "__main__":
    torch.manual_seed(42)
    home_dir = pathlib.Path.home()
    save_dir = home_dir / "data" / "cifar10"

    augmentation_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    tr_ds = torchvision.datasets.CIFAR10(root=save_dir, train=True, transform=augmentation_transform, download=True)
    te_ds = torchvision.datasets.CIFAR10(root=save_dir, train=False, transform=test_transform, download=True)
    print(f"Train size: {len(tr_ds)}, Test size: {len(te_ds)}")

    class_pair = (3, 5)
    tr_subset = get_classes(tr_ds, *class_pair)
    te_subset = get_classes(te_ds, *class_pair)
    print(f"Subset Train size: {len(tr_subset)}, Subset Test size: {len(te_subset)}")

    model = BClassModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CIFAR10_CLASSSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # print(f"Training binary classifier for classes: {CIFAR10_CLASSSES[class_pair[0]]} vs {CIFAR10_CLASSSES[class_pair[1]]}")

    model.to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    cur_best = None
    cur_best_acc = 0.0
    for epoch in range(200):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm.tqdm(torch.utils.data.DataLoader(tr_subset, batch_size=100, shuffle=True)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        tps = {}
        fps = {}
        fns = {}
        with torch.no_grad():
            for images, labels in tqdm.tqdm(torch.utils.data.DataLoader(te_subset, batch_size=100, shuffle=False)):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Add to tp, fp, fn
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    if true_label == pred_label:
                        tps[true_label] = tps.get(true_label, 0) + 1
                    if true_label != pred_label:
                        fps[pred_label] = fps.get(pred_label, 0) + 1
                        fns[true_label] = fns.get(true_label, 0) + 1
        print()
        print(f'Test Accuracy of the model at epoch {epoch} on the test images: {100 * correct / total:.2f}%, training loss: {running_loss/len(tr_ds):.7f}')
        print(f"Classwise F1:")
        for cls in [0, 1]:
            tp = tps.get(cls, 0)
            fp = fps.get(cls, 0)
            fn = fns.get(cls, 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f" Class {cls}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
        if cur_best is None or (correct / total > cur_best_acc):
            cur_best = model.state_dict()
            cur_best_acc = correct / total
    print(f"Best accuracy: {cur_best_acc*100}%")

