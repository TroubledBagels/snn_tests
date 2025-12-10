import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pathlib

import tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.MobileNetV2 import MobileNetV2
from models.MobileNetV3 import MobileNetV3
from models.WideResNet import WideResNet
from models.ResNet18 import ResNet18
from models.AlexNet import AlexNet
from models.LeNet import LeNet
from models.VGG19 import VGG19, VGG11
from models.EfficientNetB0 import EfficientNetB0
from models.GoogLeNet import GoogLeNet

import argparse

class SmallCNN(nn.Module):
    def __init__(self, c_1, c_2, hid, inp, out, num_layers=2):
        super(SmallCNN, self).__init__()
        self.c_1 = c_1
        self.c_2 = c_2
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
        # self.fc2 = nn.Linear(256, 2)

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
        # x = torch.relu(x)
        # x = self.fc2(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='mnv2', help='Model to use: mnv2, mnv3, wrn, rn18, an, le, vgg19, vgg11, enb0, gn, scnn')
    return parser.parse_args()

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

if __name__ == "__main__":
    # tr_ds = torchvision.datasets.CIFAR10(
    #     root=pathlib.Path.home() / 'data' / 'cifar10',
    #     train=True,
    #     download=True,
    #     transform=transforms.ToTensor()
    # )
    #
    # te_ds = torchvision.datasets.CIFAR10(
    #     root=pathlib.Path.home() / 'data' / 'cifar10',
    #     train=False,
    #     download=True,
    #     transform=transforms.ToTensor()
    # )
    #
    # tr_list = []
    # for i in range(len(tr_ds)):
    #     tr_list.append((tr_ds[i][0], tr_ds[i][1]))
    #
    # te_list = []
    # for i in range(len(te_ds)):
    #     te_list.append((te_ds[i][0], te_ds[i][1]))
    #
    # tr_list_ds = ListDataset(tr_list, transform=None)
    # te_list_ds = ListDataset(te_list, transform=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #
    # print(f"Dataset size: {len(tr_ds)}")
    #
    # tr_dl = DataLoader(tr_list_ds, batch_size=64, shuffle=True)
    # te_dl = DataLoader(te_list_ds, batch_size=1, shuffle=False)

    model_name = parse_args().m.lower()
    if model_name == 'mnv2':
        model = MobileNetV2(num_classes=10).to(device)
    elif model_name == 'mnv3':
        model = MobileNetV3(num_classes=10).to(device)
    elif model_name == 'wrn':
        model = WideResNet(depth=28, width=10, num_classes=10).to(device)
    elif model_name == 'rn18':
        model = ResNet18(num_classes=10).to(device)
    elif model_name == 'an':
        model = AlexNet(num_classes=10).to(device)
    elif model_name == 'le':
        model = LeNet(num_classes=10).to(device)
    elif model_name == 'vgg19':
        model = VGG19(num_classes=10).to(device)
    elif model_name == 'vgg11':
        model = VGG11(num_classes=10).to(device)
    elif model_name == 'enb0':
        model = EfficientNetB0(num_classes=10).to(device)
    elif model_name == 'gn':
        model = GoogLeNet(num_classes=10).to(device)
    elif model_name == 'scnn':
        model = SmallCNN(1, 2, 1, 1, 2).to(device)
    else:
        print(f"Unknown model name: {model_name}. Using SmallCNN as default.")
        model = SmallCNN(1, 2, 1, 1, 2).to(device)

    print(f"Using model: {model_name}.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    #
    # # model = GoogLeNet(num_classes=10).to(device)
    # # model = SmallCNN(1, 2, 1, 1, 2).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # for epoch in range(3):
    #     # pbar = tqdm.tqdm(tr_dl)
    #     # running_loss = 0.0
    #     # for images, labels in pbar:
    #     #     optimiser.zero_grad()
    #     #     outputs = model(images)
    #     #     loss = criterion(outputs, labels)
    #     #     loss.backward()
    #     #     optimiser.step()
    #     #
    #     #     running_loss += loss.item()
    #     #     pbar.set_description(f"Epoch {epoch+1}, Loss: {running_loss / (pbar.n + 1):.4f}")
    #
    #     correct = 0
    #     total = 0
    #     qbar = tqdm.tqdm(te_dl)
    #     with torch.no_grad():
    #         for images, labels in qbar:
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #             qbar.set_description(f"Epoch {epoch+1}, Test Accuracy: {100 * correct / total:.2f}%")

    print("Training complete.")
