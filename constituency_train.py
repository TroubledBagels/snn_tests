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
    print(f"Train size: {len(tr_ds)}, Test size: {len(te_ds)}")

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

    model = CN.ConstituencyNet(constituencies)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)

    model.train_classifiers(tr_ds, te_ds, epochs=5, lr=1e-3, device=device)
    torch.save(model.state_dict(), "constituency_net_cifar10.pth")