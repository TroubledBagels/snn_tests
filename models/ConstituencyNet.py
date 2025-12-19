import torch
import torch.nn as nn

import tqdm

class SmallConstituency(nn.Module):
    def __init__(self, class_list):
        super(SmallConstituency, self).__init__()
        self.class_list = class_list
        self.num_outputs = len(class_list)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.do = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(2)
        self.fc1 = nn.Linear(64 * 2 * 2, self.num_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.do(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.do(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.gap(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        return x

    def get_hidden_weights(self):
        weights = [self.conv2.weight.data.clone(), self.conv3.weight.data.clone(), self.conv4.weight.data.clone(), self.fc1.weight.data.clone()]
        biases = [self.conv2.bias.data.clone(), self.conv3.bias.data.clone(), self.conv4.bias.data.clone(), self.fc1.bias.data.clone()]
        return weights, biases

class ConstituencyNet(nn.Module):
    def __init__(self, constituency_structures, out_type='rp'):
        # out_type is 'rp' for ranked pairs, 'bin' for binary, 'ann' for ann, 'sum' for sum
        super(ConstituencyNet, self).__init__()
        self.num_constituencies = len(constituency_structures)
        self.rp = out_type == 'rp'
        self.bin = out_type == 'bin'
        self.ann = out_type == 'ann'
        self.sum = out_type == 'sum'
        for i in range(len(constituency_structures)):
            setattr(self, f"constituency_{i}", SmallConstituency(constituency_structures[i]))
        self.classifiers = [getattr(self, f"constituency_{i}") for i in range(len(constituency_structures))]

    def forward(self, x):
        if not x.is_cuda:
            out_list = []
            for classifier in self.classifiers:
                classifier.eval()
                out = classifier(x)
                out_list.append(nn.Softmax(dim=1)(out))
        else:
            stream_list = []
            for classifier in self.classifiers:
                classifier.eval()
                stream = torch.cuda.Stream()
                stream_list.append(stream)
            out_list = []
            for i, classifier in enumerate(self.classifiers):
                with torch.cuda.stream(stream_list[i]):
                    out = classifier(x)
                    out_list.append(nn.Softmax(dim=1)(out))
            torch.cuda.synchronize()

        final_out = torch.zeros_like(out_list[0])
        for out in out_list:
            final_out += out
        return final_out

    def train_classifiers(self, tr_ds, te_ds, epochs=3, lr=1e-3, device='cpu'):
        print("Training ConstituencyNet Classifiers")
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimisers = [torch.optim.Adam(classifier.parameters(), lr=lr) for classifier in self.classifiers]

        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=64, shuffle=True)
        te_dl = torch.utils.data.DataLoader(te_ds, batch_size=64, shuffle=False)

        acc_dict = {}
        test_loss_dict = {}
        for idx, classifier in enumerate(self.classifiers):
            print(f"Training Classifier {idx+1}/{self.num_constituencies} with {len(classifier.class_list)} classes")
            classifier.to(device)

            cur_best_acc = 0.0
            best_loss = float('inf')

            classifier.train()
            for epoch in range(epochs):
                pbar = tqdm.tqdm(tr_dl)
                mean_loss = 0.0
                for i, (data, target) in enumerate(pbar):
                    data = data.float()
                    data, target = data.to(device), target.to(device)

                    # Create 1-hot encoding for target if it is in classifer.class_list
                    # If not, set softmax target to all zeros
                    target_binary = torch.zeros(len(target), classifier.num_outputs, device=device)
                    for j, t in enumerate(target):
                        if t.item() in classifier.class_list:
                            class_idx = classifier.class_list.index(t.item())
                            target_binary[j][class_idx] = 1.0
                    target_binary = target_binary / target_binary.sum(dim=1, keepdim=True).clamp(min=1e-6)

                    optimisers[idx].zero_grad()
                    output = classifier(data)
                    log_output = torch.LogSoftmax(dim=1)(output)
                    loss = criterion(log_output, target_binary)
                    loss.backward()
                    mean_loss += loss.item()
                    optimisers[idx].step()
                    pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {mean_loss/(i+1):.4f}")


if __name__ == "__main__":
    model = ConstituencyNet([[0, 1, 2, 3, 4]])
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    print(output.shape)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")