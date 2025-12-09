import torch
import torchvision
import torchvision.transforms as transforms
import pathlib

import tqdm
from torch import nn
from torch.utils.data import DataLoader

from models.MobileNetV2 import MobileNetV2
from models.WideResNet import WideResNet

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

    print(f"Dataset size: {len(tr_ds)}")

    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False)

    model = WideResNet(num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        pbar = tqdm.tqdm(tr_dl)
        running_loss = 0.0
        for images, labels in pbar:
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}, Loss: {running_loss / (pbar.n + 1):.4f}")

        correct = 0
        total = 0
        qbar = tqdm.tqdm(te_dl)
        with torch.no_grad():
            for images, labels in qbar:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                qbar.set_description(f"Epoch {epoch+1}, Test Accuracy: {100 * correct / total:.2f}%")

    print("Training complete.")
