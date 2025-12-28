import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import pathlib
import tqdm

import models.ConstituencyNet as CN

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train ConstituencyNet on CIFAR-10")
    parser.add_argument('-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-bs', type=int, default=64, help='Batch size for training')
    parser.add_argument('-i', action='store_true', default=False, help='Run inference only if set')
    parser.add_argument('-m', type=str, default="", help='Path to load model from for inference only and save model to')
    parser.add_argument('-o', type=str, default="sum", help='Output type: sum, bin, ann, or rp')
    return parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(42)
    args = parse_args()
    epochs = args.e
    lr = args.lr
    batch_size = args.bs
    inference_only = args.i
    model_path = args.m
    out_type = args.o
    print(f"Parameters: epochs: {epochs}, lr: {lr}, batch_size: {batch_size}, inference_only: {inference_only}")
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

    constituencies_sequential = []
    for i in range(10):
        constituencies_sequential += constituencies[i]
    print(f"constituencies_sequential: {constituencies_sequential}")

    # Create a vector for a linear layer weight that simply sums the outputs of all constituencies into a 10-dimensional output based on their assigned classes
    weight_vector = torch.zeros(10, 50)
    for i in range(10):
        for j in range(len(constituencies_sequential)):
            if constituencies_sequential[j] == i:
                weight_vector[i, j] = 1.0
    print(f"Weight vector sum: {weight_vector.sum()}")

    model = CN.ConstituencyNet(
        constituencies,
        out_type=out_type,
        num_classes=10
    )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)

    if not inference_only:
        if out_type == "ann":
            print(f"Please train a non-ANN model first before training ANN output type. Exiting.")
            exit(1)
        model.train_classifiers(tr_ds, te_ds, epochs=100, lr=1e-3, device=device)
    else:
        if not out_type == "ann":
            if model_path != "":
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded model from {model_path}")
            else:
                print("Model path not provided for inference only mode. Exiting.")
                exit(1)
        else:
            temp_model = CN.ConstituencyNet(
                constituencies
            )
            temp_model.load_state_dict(torch.load(model_path, map_location=device))
            model.load_from_no_net(temp_model)
            # model.ann_layer.weight = nn.Parameter(weight_vector)
            # model.ann_layer.bias = nn.Parameter(torch.zeros(10))
            model.train_ann(tr_ds, te_ds, epochs=epochs, lr=lr,
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    te_dl = torch.utils.data.DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    correct = 0
    model.eval()
    top2 = 0
    top3 = 0
    pbar = tqdm.tqdm(te_dl)
    for i, (data, labels) in enumerate(pbar):
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        top2 += (labels.unsqueeze(1) == torch.topk(outputs, 2, dim=1).indices).any(dim=1).sum().item()
        top3 += (labels.unsqueeze(1) == torch.topk(outputs, 3, dim=1).indices).any(dim=1).sum().item()
        pbar.set_description(f"Accuracy: {100 * correct / len(te_ds):.2f}%")
    print(f"Final Test Accuracy: {100 * correct / len(te_ds):.2f}%")

    torch.save(model.state_dict(), model_path)