import torch
import numpy as np
import tonic
from torch.utils.data import DataLoader, random_split
from models.EnsembleAudio import BaseModel, EnsembleAudioModel

import utils.general as g
import torch.nn as nn
import torchvision.transforms as transforms
import utils.fft_dataset as fft

import matplotlib.pyplot as plt

import sys

from random import shuffle

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    full_ds = fft.MFCCDataset()
    # Split into train and test
    torch.manual_seed(42)
    set_size = len(full_ds)
    train_size = int(0.8 * set_size)
    test_size = set_size - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])

    batch_size = 1
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

    num_models = 5
    epochs = 1

    part_size = len(full_ds) // num_models
    ds_parts = [train_ds[i*part_size:(i+1)*part_size] for i in range(num_models)]
    models = [BaseModel(out_c=full_ds.max_c).to(device) for _ in range(num_models)]
    trained_models = []
    loss_fn = nn.CrossEntropyLoss()
    for i, model in enumerate(models):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        temp_ds = []
        for j in range(num_models):
            if j != i:
                temp_ds += ds_parts[j]
        temp_dl = DataLoader(temp_ds, batch_size=batch_size, shuffle=True)
        print(f"Training model {i+1}/{num_models} on {len(temp_ds)} samples")
        model, _ = g.train(model, temp_dl, test_dl, device, loss_fn=loss_fn, lr=1e-3, epochs=epochs, save_name=f"./ensemble/fft_ensemble_model_{i}", weight_decay=1e-4, n_c=full_ds.max_c)
        trained_models.append(model)

    model = EnsembleAudioModel(trained_models).to(device)

    if len(sys.argv) > 1:
        model.load_state_dict(torch.load(sys.argv[1], map_location=device))
        print(f"Loaded model weights from {sys.argv[1]}")
    model.to(device)
    print(model)

    accuracy, test_loss, class_f1_scores, class_rec_scores, class_prec_scores = g.test_reg(model, test_dl, device, loss_fn=loss_fn)
    print(f"Ensemble Test Accuracy: {accuracy}, Test Loss: {test_loss}")
    # model, f1_df = g.train(model, train_dl, test_dl, device, loss_fn=loss_fn, lr=1e-3, epochs=epochs, save_name="fft_simpleaudio", weight_decay=1e-4, n_c=full_ds.max_c)
    # print(f1_df)