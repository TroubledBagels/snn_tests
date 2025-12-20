import torch
from models.ConstituencyNet import SmallConstituency
import torchvision
import pathlib
import tqdm

class ListDataset(torch.utils.data.Dataset):
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
    home = pathlib.Path.home()
    save_dir = home / "data" / "cifar10"
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    te_ds = torchvision.datasets.CIFAR10(root=save_dir, train=False, transform=test_transform, download=True)
    tr_ds = torchvision.datasets.CIFAR10(root=save_dir, train=True, transform=test_transform, download=True)

    constituency = [0, 1, 6, 7, 8]
    model = SmallConstituency(constituency)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(model)

    train_list = []
    test_list = []

    for i, (data, label) in enumerate(tr_ds):
        if label in constituency:
            train_list.append((data, constituency.index(label)))

    for i, (data, label) in enumerate(te_ds):
        if label in constituency:
            test_list.append((data, constituency.index(label)))

    tr_dataset = ListDataset(train_list)
    te_dataset = ListDataset(test_list)

    tr_dl = torch.utils.data.DataLoader(tr_dataset, batch_size=64, shuffle=True)
    te_dl = torch.utils.data.DataLoader(te_dataset, batch_size=64, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        pbar = tqdm.tqdm(tr_dl)
        mean_loss = 0.0
        for i, (data, labels) in enumerate(pbar):
            data, labels = data.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            mean_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1} Loss: {mean_loss/(i+1):.4f}")
            loss.backward()
            optimiser.step()
            model.zero_grad()

        qbar = tqdm.tqdm(te_dl)
        correct = 0
        top2 = 0
        top3 = 0
        total = 0
        for i, (data, labels) in enumerate(qbar):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top2 += (labels.unsqueeze(1) == torch.topk(outputs, 2, dim=1).indices).any(dim=1).sum().item()
            top3 += (labels.unsqueeze(1) == torch.topk(outputs, 3, dim=1).indices).any(dim=1).sum().item()
            qbar.set_description(f"Test Accuracy: {100 * correct / total:.2f}%")

        print(f"Epoch {epoch+1} Test Accuracy: {100 * correct / total:.2f}%")

    print("Training complete.")
