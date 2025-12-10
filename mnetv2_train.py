import torch
import torchvision
import torchvision.transforms as transforms
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
from models.ConventionalBSquare import SmallCNN

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='mnv2', help='Model to use: mnv2, mnv3, wrn, rn18, an, le, vgg19, vgg11, enb0, gn, scnn')
    return parser.parse_args()

if __name__ == "__main__":
    tr_ds = torchvision.datasets.CIFAR10(
        root=pathlib.Path.home() / 'data' / 'cifar10',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    te_ds = torchvision.datasets.CIFAR10(
        root=pathlib.Path.home() / 'data' / 'cifar10',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Dataset size: {len(tr_ds)}")

    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

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
        model = SmallCNN(1, 2, 1, 1, 2).to(device)\

    print(f"Using model: {model_name}.")

    # model = GoogLeNet(num_classes=10).to(device)
    # model = SmallCNN(1, 2, 1, 1, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        # pbar = tqdm.tqdm(tr_dl)
        # running_loss = 0.0
        # for images, labels in pbar:
        #     optimiser.zero_grad()
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimiser.step()
        #
        #     running_loss += loss.item()
        #     pbar.set_description(f"Epoch {epoch+1}, Loss: {running_loss / (pbar.n + 1):.4f}")

        correct = 0
        total = 0
        qbar = tqdm.tqdm(te_dl)
        with torch.no_grad():
            for images, labels in qbar:
                images, labels = images.to(device), labels.to(device)
                if isinstance(model, SmallCNN):
                    outputs, _ = model(images)
                else:
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                qbar.set_description(f"Epoch {epoch+1}, Test Accuracy: {100 * correct / total:.2f}%")

    print("Training complete.")
