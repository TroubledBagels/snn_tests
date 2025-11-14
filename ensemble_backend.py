import torch
import torch.nn as nn
import numpy as np
import models.EnsembleAudio as ea
import utils.general as g
from torch.utils.data import DataLoader
from utils.fft_dataset import MFCCDataset
import tqdm

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pretrained subnetworks
    num_models = 10
    subnetworks = []
    for i in range(num_models):
        model = ea.BaseModel(out_c=38).to(device)
        model.load_state_dict(torch.load(f"./ensemble/fft_ensemble_model_{i}_best.pth", map_location=device))
        subnetworks.append(model)

    ensemble_model = ea.EnsembleAudioModel(subnetworks).to(device)

    # Load dataset
    full_ds = MFCCDataset()
    train_ds, test_ds = torch.utils.data.random_split(full_ds, [int(0.8*len(full_ds)), len(full_ds)-int(0.8*len(full_ds))])
    batch_size = 1
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(ensemble_model.parameters(), lr=1e-4)
    ensemble_model.train()
    ensemble_model.lock_subnets()

    for epoch in range(epochs):
        pbar = tqdm.tqdm(total=len(train_ds))
        loss_total = 0
        for batch_idx, (data, targets) in enumerate(train_dl):
            data, targets = data.to(device), targets.to(device)
            optimiser.zero_grad()
            outputs, _ = ensemble_model(data)
            loss = loss_fn(outputs, targets.long())
            loss_total += loss.item()
            loss.backward()
            optimiser.step()
            pbar.set_postfix(loss=loss_total/(batch_idx+1))
            pbar.update(data.size(0))
        pbar.close()

        test_acc, _, _, _, _ = g.test(ensemble_model, test_dl, device, loss_fn)
        print(f"Test Accuracy after fine-tuning: Top-1: {test_acc[0]:.2f}%, Top-3: {test_acc[1]:.2f}%, Top-5: {test_acc[2]:.2f}%, Top-10: {test_acc[3]:.2f}%")
    torch.save(ensemble_model.state_dict(), "fft_ensemble_finetuned.pth")

    test_acc, _, _, _, _ = g.test(ensemble_model, test_dl, device, loss_fn)
    print(f"Test Accuracy after fine-tuning: Top-1: {test_acc[0]:.2f}%, Top-3: {test_acc[1]:.2f}%, Top-5: {test_acc[2]:.2f}%, Top-10: {test_acc[3]:.2f}%")
