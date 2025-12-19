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
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 B-Square Training and Evaluation")
    parser.add_argument('-m', type=str, default="./bsquares/cifar10_bal_4conv_1fc_ac_full.pth", help='Path to the model directory')
    parser.add_argument('-t', type=float, default=0.0, help='Threshold for B-Square model')
    parser.add_argument('-i', action='store_true', default=False, help='Run only inference if set')
    parser.add_argument('-b', action='store_true', default=False, help='Use binary voting if set')
    parser.add_argument('-s', action='store_true', default=False, help='Use similarity weighting if set')
    parser.add_argument('-ns', action='store_true', default=False, help='Do not use softmax if set')
    parser.add_argument('-g', action='store_true', default=False, help='Use graph-based read-out if set')
    parser.add_argument('-l', action='store_true', default=False, help='Use linear-based read-out if set, overrides all else')
    return parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    args = parse_args()
    model_dir = args.m
    threshold = args.t
    inference_only = args.i
    binary_voting = args.b
    similarity_weighting = args.s
    no_softmax = args.ns
    graph_readout = args.g
    linear_readout = args.l
    print(f"Parameters: m: {model_dir}, t: {threshold}, i: {inference_only}, b: {binary_voting}, s: {similarity_weighting}, ns: {no_softmax}, g: {graph_readout}")

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
    print(tr_ds[0][0].shape)

    model = CBS.BSquareModel(
        num_classes=10,
        input_size=3,
        hidden_size=16,
        num_layers=3,
        binary_voting=binary_voting,
        bclass=CBS.SmallCNN,
        net_out=linear_readout,
        threshold=threshold,
        sim_weighted=similarity_weighting,
        no_soft=no_softmax,
        graph_based=graph_readout
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    if inference_only and not linear_readout:
        model.load_state_dict(torch.load(model_dir, map_location=device))
        model.threshold = threshold
        print("Model loaded for inference only.")
    elif linear_readout and inference_only:
        temp_model = CBS.BSquareModel(
            num_classes=10,
            input_size=3,
            hidden_size=16,
            num_layers=3,
            binary_voting=binary_voting,
            bclass=CBS.SmallCNN,
            net_out=False,
            threshold=threshold,
            sim_weighted=similarity_weighting,
            no_soft=no_softmax,
            graph_based=graph_readout
        )
        temp_model.to(device)
        temp_model.load_state_dict(torch.load(model_dir, map_location=device))
        print(temp_model)
        model.load_from_no_net(temp_model)
        print("Model backbone loaded for linear readout training.")
    else:
        accuracy_dict = model.train_classifiers(
            train_ds=tr_ds,
            test_ds=te_ds,
            device=device,
            epochs=100,
            training_type='noise'
        )

    if linear_readout:
        model.train_output_layer(
            tr_ds, te_ds, epochs=10, lr=1e-3, device=device
        )

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
    if not inference_only:
        torch.save(model.state_dict(), model_dir)
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