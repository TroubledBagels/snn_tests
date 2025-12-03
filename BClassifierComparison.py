import models.ConventionalBSquare as CBS
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pathlib

import pandas as pd

import tqdm

from torch.utils.data import DataLoader

if __name__ == '__main__':
    data_dir = pathlib.Path.home() / "data" / "cifar10"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result_df = pd.DataFrame(columns=["Class_Pair", "TinyCNN", "SmallCNN", "SeparableSmallCNN", "MediumCNN", "SeparableMediumCNN"])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    tr_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    te_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)

    tr_data_dict = {}
    for i in range(len(tr_ds)):
        img, label = tr_ds[i]
        if label not in tr_data_dict:
            tr_data_dict[label] = []
        tr_data_dict[label].append((img, label))

    te_data_dict = {}
    for i in range(len(te_ds)):
        img, label = te_ds[i]
        if label not in te_data_dict:
            te_data_dict[label] = []
        te_data_dict[label].append((img, label))

    for i in range(10):
        for j in range(i+1, 10):
            tiny_model = CBS.TinyCNN(i, j, 0, 0, 0, 0)
            small_model = CBS.SmallCNN(i, j, 0, 0, 0, 0)
            small_sep_model = CBS.SeparableSmallCNN(i, j, 0, 0, 0, 0)
            medium_model = CBS.MediumCNN(i, j, 0, 0, 0, 0)
            medium_sep_model = CBS.SeparableMediumCNN(i, j, 0, 0, 0, 0)

            models = {
                "TinyCNN": tiny_model,
                "SmallCNN": small_model,
                "SeparableSmallCNN": small_sep_model,
                "MediumCNN": medium_model,
                "SeparableMediumCNN": medium_sep_model
            }

            tr_subset = torch.utils.data.TensorDataset(torch.stack([item[0] for item in tr_data_dict[i] + tr_data_dict[j]]), torch.tensor([0]*len(tr_data_dict[i]) + [1]*len(tr_data_dict[j])))
            te_subset = torch.utils.data.TensorDataset(torch.stack([item[0] for item in te_data_dict[i] + te_data_dict[j]]), torch.tensor([0]*len(te_data_dict[i]) + [1]*len(te_data_dict[j])))

            tr_dl = DataLoader(tr_subset, batch_size=100, shuffle=True)
            te_dl = DataLoader(te_subset, batch_size=100, shuffle=False)

            df_record = {"Class_Pair": f"{i} vs {j}"}

            for model_name, model in models.items():
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                loss_fn = nn.CrossEntropyLoss()

                cur_best = 0.0

                for epoch in range(50):
                    model.train()

                    pbar = tqdm.tqdm(tr_dl)
                    running_loss = 0.0
                    for k, (images, labels) in enumerate(pbar):
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs, _ = model(images)
                        loss = loss_fn(outputs, labels)
                        running_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        pbar.set_description(f"Model: {model_name} for classes {i} vs {j} Epoch {epoch} Loss: {running_loss / (k+1):.4f}")

                    model.eval()
                    qbar = tqdm.tqdm(te_dl)
                    correct = 0
                    for k, (images, labels) in enumerate(qbar):
                        images, labels = images.to(device), labels.to(device)
                        outputs, _ = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total = labels.size(0)
                        correct += (predicted == labels).sum().item()
                        qbar.set_description(f"Evaluating Model: {model_name} for classes {i} vs {j} Epoch {epoch} Acc: {correct/(k+1):.2f}% Best: {cur_best * 100:.2f}%")

                    acc = correct / len(te_dl) / 100.0
                    if acc > cur_best:
                        cur_best = acc

                df_record[model_name] = cur_best
            result_df.loc[len(result_df)] = df_record
            print(result_df)
    print(result_df)
    result_df.to_csv("bclassifier_comparison_results.csv", index=False)
