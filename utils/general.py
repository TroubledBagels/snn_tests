import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import tqdm
import snntorch as snn
import pandas as pd
import copy


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
    # test_acc_rec is a list of [top1, top3, top5, top10]
    top1_acc = [acc[0] for acc in test_acc_rec]
    top3_acc = [acc[1] for acc in test_acc_rec]
    top5_acc = [acc[2] for acc in test_acc_rec]
    top10_acc = [acc[3] for acc in test_acc_rec]
    test_acc_rec = top1_acc
    plt.scatter([i for i in range(len(acc_rec))], acc_rec, marker='x', s=1, color='c', label='Train Acc')
    plt.plot(np.convolve(acc_rec, np.ones(100)/100, mode='valid'), color='blue', label='Train Acc (smoothed)')
    plt.scatter([(i+1)*len(acc_rec)//len(test_acc_rec) for i in range(len(test_acc_rec))], [test_acc_rec[i] for i in range(len(test_acc_rec))], color='orange', label='Test Acc')
    points = [(i+1)*len(acc_rec)//len(test_acc_rec) for i in range(len(test_acc_rec))]
    a, b, c = np.polyfit(points, test_acc_rec, 2)
    plt.plot(points, [a*(x**2) + b*x + c for x in points], color='darkorange', label='Test Acc (trend)')
    plt.scatter(points, [top3_acc[i] for i in range(len(top3_acc))], color='lime', marker='x', label='Test Top-3 Acc', s=3)
    a, b, c = np.polyfit(points, top3_acc, 2)
    plt.plot(points, [a*(x**2) + b*x + c for x in points], color='green', label='Test Top-3 Acc (trend)')
    plt.scatter(points, [top5_acc[i] for i in range(len(top5_acc))], color='blueviolet', marker='x', label='Test Top-5 Acc', s=3)
    a, b, c = np.polyfit(points, top5_acc, 2)
    plt.plot(points, [a*(x**2) + b*x + c for x in points], color='purple', label='Test Top-5 Acc (trend)')
    plt.scatter(points, [top10_acc[i] for i in range(len(top10_acc))], color='palevioletred', marker='x', label='Test Top-10 Acc' ,s=3)
    a, b, c = np.polyfit(points, top10_acc, 2)
    plt.plot(points, [a*(x**2) + b*x + c for x in points], color='crimson', label='Test Top-10 Acc (trend)')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.show()
    plt.savefig(save_name)

def plot_f1(f1_rec: list[list], save_name="f1.png"):
    # List of type epochs x classes
    plt.figure(figsize=(8, 8))
    num_epochs = len(f1_rec)
    num_classes = len(f1_rec[0]) if num_epochs > 0 else 0
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # Scatter plot with different colour per class
    for c in range(num_classes):
        class_f1 = [f1_rec[e][c] for e in range(num_epochs)]
        plt.scatter([i for i in range(num_epochs)], class_f1, marker='x', s=5, label=f'Class {c}', color=colours[c % len(colours)])
        plt.plot(np.convolve(class_f1, np.ones(5)/5, mode='valid'), label=f'Class {c} (smoothed)', color=colours[c % len(colours)])
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.title("Class-wise F1 Score Over Epochs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.savefig(save_name, bbox_inches='tight')

def train(model, train_dl, test_dl, device, loss_fn=nn.CrossEntropyLoss(), lr=1e-4, epochs=10, weight_decay=0, save_name=None):
    print("Training model...")
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser)
    print(f"Using optimiser: {optimiser}")
    model.to(device)
    print(f"Model training on {device} for {epochs} epochs...")
    model.train()

    loss_rec = []
    test_loss_rec = []
    acc_rec = []
    test_acc_rec = []
    test_f1_rec = []
    lr_rec = []

    best_model = copy.deepcopy(model)
    current_best_acc = 0.0

    f1_df = pd.DataFrame(columns=['Epoch'] + [f'Class_{i}_F1' for i in range(75)], index=None)
    rec_df = pd.DataFrame(columns=['Epoch'] + [f'Class_{i}_Rec' for i in range(75)], index=None)
    prec_df = pd.DataFrame(columns=['Epoch'] + [f'Class_{i}_Prec' for i in range(75)], index=None)

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        print("----------------------------------------------------")
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
        test_acc, test_loss, test_f1, test_rec, test_prec = test(model, test_dl, device, loss_fn)
        test_acc_rec.append(test_acc)
        test_loss_rec.append(np.mean(test_loss))
        test_f1_rec.append(test_f1)
        lr_scheduler.step(np.mean(test_loss))

        if test_acc[0] > current_best_acc:
            current_best_acc = test_acc[0]
            best_model = copy.deepcopy(model)
            if save_name is not None:
                torch.save(best_model.state_dict(), save_name + "_best.pth")
                print(f"New best model saved with accuracy: {current_best_acc:.4f}")
                plot_loss(loss_rec, test_loss_rec, save_name=save_name + "_loss_best.png")
                plot_acc(acc_rec, test_acc_rec, save_name=save_name + "_acc_best.png")
                plot_f1(test_f1_rec, save_name=save_name + "_f1_best.png")

        # Save F1 scores to dataframe
        f1_row = {'Epoch': epoch + 1}
        rec_row = {'Epoch': epoch + 1}
        prec_row = {'Epoch': epoch + 1}
        for class_idx in range(len(test_f1)):
            f1_row[f'Class_{class_idx}_F1'] = test_f1[class_idx]
            rec_row[f'Class_{class_idx}_F1'] = test_rec[class_idx]
            prec_row[f'Class_{class_idx}_F1'] = test_prec[class_idx]
        f1_df = pd.concat([f1_df, pd.DataFrame([f1_row])], ignore_index=True)
        rec_df = pd.concat([rec_df, pd.DataFrame([rec_row])], ignore_index=True)
        prec_df = pd.concat([prec_df, pd.DataFrame([prec_row])], ignore_index=True)

        if save_name is not None:
            f1_df.to_csv(save_name + "_f1_scores.csv", index=False)
            rec_df.to_csv(save_name + "_recall_scores.csv", index=False)
            prec_df.to_csv(save_name + "_precision_scores.csv", index=False)

        lr_rec.append(lr_scheduler.get_lr())
    print("Training complete.")
    
    if save_name is None:
        plot_loss(loss_rec, test_loss_rec)
        plot_acc(acc_rec, test_acc_rec)
        plot_f1(test_f1_rec)
    else:
        plot_loss(loss_rec, test_loss_rec, save_name=save_name+"_loss_final.png")
        plot_acc(acc_rec, test_acc_rec, save_name=save_name+"_acc_final.png")
        plot_f1(test_f1_rec, save_name=save_name+"_f1_final.png")

    return best_model, f1_df

def test(model, test_dl, device, loss_fn):
    model.to(device)
    model.eval()

    pbar = tqdm.tqdm(test_dl)
    correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = 0
    test_loss = []
    class_f1 = {}
    class_precision = {}
    class_recall = {}
    class_tps = {}
    class_fps = {}
    class_fns = {}

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
            top3 = spikes.topk(3, dim=1).indices
            top5 = spikes.topk(5, dim=1).indices
            top10 = spikes.topk(10, dim=1).indices
            for j in range(labels.size(0)):
                if labels[j] in top10[j]:
                    top10_correct += 1
                if labels[j] in top5[j]:
                    top5_correct += 1
                if labels[j] in top3[j]:
                    top3_correct += 1
            for j in range(labels.min().item(), labels.max().item()+1):
                if j not in class_f1:
                    class_f1[j] = 0
                    class_tps[j] = 0
                    class_fps[j] = 0
                    class_fns[j] = 0
                class_tps[j] += ((preds == j) & (labels == j)).sum().item()
                class_fps[j] += ((preds == j) & (labels != j)).sum().item()
                class_fns[j] += ((preds != j) & (labels == j)).sum().item()
            pbar.set_description(f"Testing Batch {i+1}, Loss: {sum(test_loss)/len(test_loss):.4f}, Testing Acc: {correct/total:.4f}")
    pbar.close()
    accuracy = np.array([correct, top3_correct, top5_correct, top10_correct])
    accuracy = accuracy / total
    print(f"Test Accuracy: Top-1: {accuracy[0]:.4f}, Top-3: {accuracy[1]:.4f}, Top-5: {accuracy[2]:.4f}, Top-10: {accuracy[3]:.4f}")
    # Calculate F1 score for each class
    for j in class_f1.keys():
        precision = class_tps[j] / (class_tps[j] + class_fps[j] + 1e-8)
        recall = class_tps[j] / (class_tps[j] + class_fns[j] + 1e-8)
        class_precision[j] = precision
        class_recall[j] = recall
        class_f1[j] = 2 * (precision * recall) / (precision + recall + 1e-8)
    classes_in_order = sorted(class_f1.keys())
    classes_per_line = 15
    print("Class-wise F1 Scores:")
    for i in range(0, len(classes_in_order), classes_per_line):
        line_classes = classes_in_order[i:i+classes_per_line]
        line_f1s = [f"{class_f1[c]:.4f}" for c in line_classes]
        print("Classes:", " ".join(f"{c:>6}" for c in line_classes))
        print("F1 Scores:", " ".join(f"{f:>5}" for f in line_f1s))
        print("Precision:", " ".join(f"{class_precision[c]:>5.4f}" for c in line_classes))
        print("Recall:   ", " ".join(f"{class_recall[c]:>5.4f}" for c in line_classes))
    print(f"Mean F1 Score: {sum(class_f1.values())/len(class_f1):.4f}")
    class_f1_scores = [class_f1[c] for c in sorted(class_f1.keys())]
    return accuracy, test_loss, class_f1_scores

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
