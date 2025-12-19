import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import pathlib

import models.ConstituencyNet as CN

if __name__ == '__main__':
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

    home_dir = pathlib.Path.home()
    save_dir = home_dir / "data" / "cifar10"
    te_ds = torchvision.datasets.CIFAR10(root=save_dir, train=False, transform=test_transform, download=True)
    tr_ds = torchvision.datasets.CIFAR10(root=save_dir, train=True, transform=augmentation_transform, download=True)
    print(f"Train size: {len(tr_ds)}, Test size