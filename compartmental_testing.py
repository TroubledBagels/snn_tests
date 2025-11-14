import torch
import numpy as np
from models import Compartmental as comp
import utils.fft_dataset as fd
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm

def test(model, test_dl, device, loss_fn):
    model.eval()
    correct = 0
    top3 = 0
    top5 = 0
    top10 = 0
    loss_re = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())
            loss_re += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    loss_re = loss_re / total
    return  accuracy, loss_re

def train(epochs, model, train_dl, device, loss_fn, optimiser, test_dl):
    test_acc_rec = []
    test_loss_rec = []
    train_acc_rec = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        pbar = tqdm.tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", unit="sample")
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        test_acc, test_loss = test(model, test_dl, device, loss_fn)
        test_acc_rec.append(test_acc)
        test_loss_rec.append(test_loss)
        print(f"Train Acc: {100. * correct / len(train_dl.dataset):.2f}%, Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")
        train_acc_rec.append(100. * correct / len(train_dl.dataset))
    return model, (test_acc_rec, train_acc_rec, test_loss_rec)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ds = fd.MFCCDataset(type="v_or_c")
    print(f"Dataset size: {len(ds)}, Number of classes: {ds.max_c}")

    train_ds, test_ds = torch.utils.data.random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    batch_size = 1
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = comp.VowelConsClassifier().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    model, (te_acc, tr_acc, te_loss) = train(epochs, model, train_dl, device, loss_fn, optimizer, test_dl)
    torch.save(model.state_dict(), "compartmental_model.pth")

    plt.figure()
    plt.title("Compartmental Model Test Accuracy")
    plt.plot(range(1, epochs+1), tr_acc, label="Train Accuracy")
    plt.plot(range(1, epochs+1), te_acc, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.grid()
    plt.savefig("compartmental_test_accuracy.png")
    plt.show()

