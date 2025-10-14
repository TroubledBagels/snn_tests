import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
import snntorch as snn


def plot_loss(loss_rec, test_loss_rec: list[list], save_name="loss.png"):
    plt.figure(figsize=(8, 8))
    print(type(loss_rec), type(test_loss_rec))
    # Remove those more than 3 std dev away from mean
    loss_rec = np.array(loss_rec)
    loss_rec = loss_rec[np.abs(loss_rec - np.mean(loss_rec)) < 3 * np.std(loss_rec)]
    # Scatter plot of loss
    plt.scatter([i for i in range(len(loss_rec))], loss_rec, marker='x', s=1, color='c', label='Train Loss')
    # Lines of best fit
    plt.plot(np.convolve(loss_rec, np.ones(100)/100, mode='valid'), color='blue', label='Train Loss (smoothed)')
    plt.scatter([(i+1)*len(loss_rec)//len(test_loss_rec) for i in range(len(test_loss_rec))], [test_loss_rec[i] for i in range(len(test_loss_rec))], color='orange', label='Test Loss')
    points = [(i+1)*len(loss_rec)//len(test_loss_rec) for i in range(len(test_loss_rec))]
    a, b, c = np.polyfit(points, test_loss_rec, 2)
    plt.plot(points, [a*(x**2) + b*x + c for x in points], color='red', label='Test Loss (trend)')
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    # plt.ylim(0, 2)
    plt.title("Loss Over Time")
    plt.show()
    plt.savefig(save_name)

def plot_acc(acc_rec, test_acc_rec: list[list], save_name="acc.png"):
    plt.figure(figsize=(8,8))
    plt.scatter([i for i in range(len(acc_rec))], acc_rec, marker='x', s=1, color='c', label='Train Acc')
    plt.plot(np.convolve(acc_rec, np.ones(100)/100, mode='valid'), color='blue', label='Train Acc (smoothed)')
    plt.scatter([(i+1)*len(acc_rec)//len(test_acc_rec) for i in range(len(test_acc_rec))], [test_acc_rec[i] for i in range(len(test_acc_rec))], color='orange', label='Test Acc')
    points = [(i+1)*len(acc_rec)//len(test_acc_rec) for i in range(len(test_acc_rec))]
    a, b, c = np.polyfit(points, test_acc_rec, 2)
    plt.plot(points, [a*(x**2) + b*x + c for x in points], color='red', label='Test Acc (trend)')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.show()
    plt.savefig(save_name)

def train(model, train_dl, test_dl, device, loss_fn=nn.CrossEntropyLoss(), lr=1e-4, epochs=10, weight_decay=0, save_name=None):
    print("Training model...")
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Using optimiser: {optimiser}")
    model.to(device)
    print(f"Model training on {device}...")
    model.train()

    loss_rec = []
    test_loss_rec = []
    acc_rec = []
    test_acc_rec = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        pbar = tqdm.tqdm(train_dl)
        for i, (inputs, labels) in enumerate(pbar):
            labels = labels.to(device)
            inputs = inputs.to(device)

            optimiser.zero_grad()

            spikes, _ = model(inputs)
            # spikes = model(inputs)
            loss = loss_fn(spikes, labels.long())
            loss_rec.append(loss.item())

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            loss.backward()
            optimiser.step()
            
            # preds = spikes.sum(dim=0).argmax(dim=1)
            preds = spikes.argmax(dim=1)
            acc = (preds == labels).sum().item() / labels.size(0)
            total_acc += acc

            avg_acc = total_acc / (i + 1)
            acc_rec.append(avg_acc)

            pbar.set_description(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")
        pbar.close()
        test_acc, test_loss = test(model, test_dl, device, loss_fn)
        test_acc_rec.append(test_acc)
        test_loss_rec.append(np.mean(test_loss))
    print("Training complete.")
    
    if save_name is None:
        plot_loss(loss_rec, test_loss_rec)
        plot_acc(acc_rec, test_acc_rec)
    else:
        plot_loss(loss_rec, test_loss_rec, save_name=save_name+"_loss.png")
        plot_acc(acc_rec, test_acc_rec, save_name=save_name+"_acc.png")

    return model

def test(model, test_dl, device, loss_fn):
    model.to(device)
    model.eval()

    pbar = tqdm.tqdm(test_dl)
    correct = 0
    total = 0
    test_loss = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(pbar):
            labels = labels.to(device).long()
            inputs = inputs.to(device)
            spikes, mems = model(inputs)
            # spikes = model(inputs)
            # one-hot encode labels
            # labels_onehot = torch.zeros(labels.size(0), 2).to(device)
            loss = loss_fn(spikes, labels.long())
            test_loss.append(loss.item())
            # preds = spikes.sum(dim=0).argmax(dim=1)
            preds = spikes.argmax(dim=1)
            correct += (preds.squeeze() == labels).sum().item()
            total += labels.size(0)
            pbar.set_description(f"Testing Batch {i+1}, Loss: {sum(test_loss)/len(test_loss):.4f},  Accuracy: {correct/total:.4f}")
    accuracy = correct / total
    return accuracy, test_loss

def add_event_fade(frames: list[torch.Tensor], decay: float = 0.9):
    fade_map = torch.zeros(frames[0].shape[1:4]) # H x W
    faded_frames = []
    for f in range(len(frames)):
        faded_frames.append([])
        for frame in frames[f]:
            fade_map = fade_map * decay + frame
            mask = (fade_map > -0.05) & (fade_map < 0.05)
            fade_map[mask] = 0  # Deadzone
            faded_frames[f].append(fade_map.clone())
        faded_frames[f] = torch.stack(faded_frames[f])
    return faded_frames

if __name__ == "__main__":
    # Test event_fade
    rdm = torch.randn(100, 1, 32, 32)
    rdm[rdm < 0] = 0
    faded = add_event_fade([rdm], decay=0.9)
    print(rdm.shape)
    print(f"RDM min: {rdm.min()}, max: {rdm.max()}")
    print(faded[0].shape)
    print(f"Faded min: {faded[0].min()}, max: {faded[0].max()}")
