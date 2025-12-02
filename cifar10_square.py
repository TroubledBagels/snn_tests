import  torch
import torch.nn as nn

from models import ConventionalBSquare as CBS
import torchvision
import torchvision.transforms as transforms
import pathlib
import tqdm
import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils.heatmap as hm

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = "./bsquares/cifar10_bal.pth"

    home_dir = pathlib.Path.home()
    save_dir = home_dir / "data" / "cifar10"
    te_ds = torchvision.datasets.CIFAR10(root=save_dir, train=False, transform=transforms.ToTensor(), download=True)
    tr_ds = torchvision.datasets.CIFAR10(root=save_dir, train=True, transform=transforms.ToTensor(), download=True)
    print(f"Train size: {len(tr_ds)}, Test size: {len(te_ds)}")
    print(tr_ds[0][0].shape)

    model = CBS.BSquareModel(
        num_classes=10,
        input_size=3,
        hidden_size=16,
        num_layers=3,
        binary_voting=False,
        bclass=CBS.TinyCNN,
        net_out=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(model_dir, map_location=device))
    # accuracy_dict = model.train_classifiers(tr_ds, te_ds, device=device, epochs=20)

    # saved_weights = torch.load(model_dir, map_location=device)
    # no_net_model = CBS.BSquareModel(
    #     num_classes=10,
    #     input_size=3,
    #     hidden_size=16,
    #     num_layers=3,
    #     binary_voting=False,
    #     bclass=CBS.TinyCNN,
    #     net_out=False
    # )
    # no_net_model.to(device)
    # no_net_model.load_state_dict(saved_weights)
    # model.load_from_no_net(no_net_model)
    # model.train_output_layer(tr_ds, te_ds, epochs=10, lr=1e-3, device=device)

    model.eval()
    correct = 0
    total = 0
    tps = {}
    fps = {}
    fns = {}
    with torch.no_grad():
        for images, labels in tqdm.tqdm(torch.utils.data.DataLoader(te_ds, batch_size=100, shuffle=False)):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                if true_label == pred_label:
                    tps[true_label] = tps.get(true_label, 0) + 1
                else:
                    fps[pred_label] = fps.get(pred_label, 0) + 1
                    fns[true_label] = fns.get(true_label, 0) + 1
    print()
    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total} %')
    torch.save(model.state_dict(), "./bsquares/cifar10_bal_1fc.pth")
    print(f"F1 Scores:")
    for cls in range(10):
        tp = tps.get(cls, 0)
        fp = fps.get(cls, 0)
        fn = fns.get(cls, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f" Class {cls}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

    random_idx = random.randint(0, len(te_ds)-1)
    print(f"Random test sample index: {random_idx}")
    sample, label = te_ds[random_idx]
    label = torch.tensor(label).long()
    sample, label = sample.to(device), label.to(device)
    output, vote_dict = model(sample.unsqueeze(0))
    print(f"Sample output shape: {output.shape}")
    print(f"Label: {label}")
    print(f"Predicted class: {torch.argmax(output, dim=1).item()}")
    print(f"Vote dict: {vote_dict}")

    num_to_str_label = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9"
    }

    heatmap_img, classes = hm.generate_heatmap(vote_dict, num_to_str_label, use_acc=False, tuple_based=True)
    cv2.imwrite("heatmap.png", heatmap_img)
    # cv2.imshow("heatmap", heatmap_img)
    # cv2.waitKey(0)