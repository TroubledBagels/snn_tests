import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class BasicClassifier(nn.Module):
    def __init__(self, out_c):
        super(BasicClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 24, out_c)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan())
        self.ada_weight = 1.0

    def forward(self, x):
        B, C, T = x.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []

        for t in range(T):
            xt = x[:, :, t]
            xt = self.conv1(xt)
            xt, mem1 = self.lif1(xt, mem1)
            xt = xt.view(B, -1)
            xt = self.fc1(xt)
            xt, mem2 = self.lif2(xt, mem2)
            spk_rec.append(mem2)

        out = torch.stack(spk_rec).mean(dim=0)

        return out


class AdaEnsembleAudio(nn.Module):
    def __init__(self, out_c, max_classifiers):
        super(AdaEnsembleAudio, self).__init__()
        self.classifiers = nn.ModuleList()

    def forward(self, x):
        outputs = []
        for classifier in self.classifiers:
            out = classifier(x)
            outputs.append(out)

        out = self.interpret_results(outputs)

        return out

    def interpret_results(self, outs):
        new_outs = []
        for i in range(len(outs)):
            out = outs[i] * self.classifiers[i].ada_weight
            new_outs.append(out)

        return torch.stack(new_outs).mean(dim=0)

    def add_classifier(self, classifier):
        self.classifiers.append(classifier)


def train_new_classifier(c_out, train_loader, epochs=1, lr=1e-3):
    model = BasicClassifier(c_out)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(tqdm.tqdm(train_loader)):
            optimiser.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, targets.long())
            loss.backward()
            optimiser.step()
    return model

def test_classifier(model, test_loader, class_num):
    model.eval()
    correct = 0
    total = 0
    tps = {}
    fps = {}
    fns = {}
    for i in range(class_num):
        tps[i] = 0
        fps[i] = 0
        fns[i] = 0

    with torch.no_grad():
        for b_idx, (data, targets) in enumerate(tqdm.tqdm(test_loader)):
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            for i in range(len(targets)):
                if predicted[i] == targets[i]:
                    tps[targets[i].item()] += 1
                else:
                    fps[predicted[i].item()] += 1
                    fns[targets[i].item()] += 1
    accuracy = 100 * correct / total
    print()
    # Find most often misclassified classes
    misclassified = {}
    for i in range(class_num):
        misclassified[i] = (fps[i] + fns[i]) / (tps[i] + fps[i] + fns[i] + 1e-6)  # Avoid division by zero
    sorted_misclassified = sorted(misclassified.items(), key=lambda x: x[1], reverse=True)
    num_misclassified = np.array([count for cls, count in sorted_misclassified]).sum() * 0.5
    most_misclassified = []
    for i, (cls, count) in enumerate(sorted_misclassified):
        most_misclassified.append(cls)
        num_misclassified -= count
        if num_misclassified <= 0:
            break
    print("Most misclassified classes:", most_misclassified)
    print("Number focused on:", len(most_misclassified))
    return accuracy, most_misclassified

def get_ds_by_labels(dataset, mis_c_labels: list[int]):
    indices = [i for i, (data, label) in enumerate(dataset) if label.item() in mis_c_labels]
    subset = torch.utils.data.Subset(dataset, indices)
    # Calculate and print the distribution of labels in the subset
    # label_count = {}
    # for i in range(len(subset)):
    #     _, label = subset[i]
    #     label_item = label.item()
    #     if label_item in label_count:
    #         label_count[label_item] += 1
    #     else:
    #         label_count[label_item] = 1
    # print("Label distribution in the new training subset:")
    # for label, count in label_count.items():
    #     print(f"Label {label}: {count} samples")
    return subset

def train_ensemble(ensemble_model, c_out, train_ds, test_ds, num_classifiers, epochs_per_classifier=1):
    bs = 1
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_acc = []
    miscls = list(range(c_out))  # Start with all classes
    for i in range(num_classifiers):
        new_classifier = train_new_classifier(c_out, train_dl, epochs=epochs_per_classifier)
        new_test_ds = get_ds_by_labels(test_ds, miscls)
        new_test_dl = DataLoader(new_test_ds, batch_size=bs, shuffle=True)
        specific_acc, _ = test_classifier(new_classifier, new_test_dl, c_out)
        print(f"Trained classifier {i+1} with specific accuracy on focused classes: {specific_acc}%")
        new_classifier.ada_weight = specific_acc
        ensemble_model.add_classifier(new_classifier)

        c_acc, miscls = test_classifier(ensemble_model, DataLoader(test_ds, batch_size=bs), c_out)
        print(f"Trained new classifier with accuracy: {c_acc}%")
        new_train_ds = get_ds_by_labels(train_ds, miscls)
        train_dl = DataLoader(new_train_ds, batch_size=bs, shuffle=True)
        test_acc.append(c_acc)
        plt.plot([i+1 for i in range(i+1)], test_acc)
        plt.xlabel("Number of Classifiers in Ensemble")
        plt.ylabel("Test Accuracy (%)")
        plt.title("Ada Ensemble Audio Classifier Performance")
        plt.grid()
        plt.show()
        plt.savefig("ada_ensemble_audio.png")

    return ensemble_model, test_acc


if __name__ == "__main__":
    model = AdaEnsembleAudio(out_c=10, max_classifiers=5)
    classifier1 = BasicClassifier(out_c=10)
    classifier2 = BasicClassifier(out_c=10)
    model.add_classifier(classifier1)
    model.add_classifier(classifier2)

    dummy_input = torch.randn(1, 30, 196)
    output = model(dummy_input)
    print("Ensemble output:", output.shape)
