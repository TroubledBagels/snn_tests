import torch
import numpy as np
import tonic
from torch.utils.data import DataLoader, random_split
from models.SimpleAudio import RegMFCCModel as SimpleAudioModel

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
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

    model = SimpleAudioModel(out_c=full_ds.max_c).to(device)
    if len(sys.argv) > 1:
        model.load_state_dict(torch.load(sys.argv[1], map_location=device))
        print(f"Loaded model weights from {sys.argv[1]}")
    model.to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 30
    model, f1_df = g.train_reg(model, train_dl, test_dl, device, loss_fn=loss_fn, lr=1e-3, epochs=epochs, save_name="fft_simpleaudio", weight_decay=1e-4, n_c=full_ds.max_c)
    print(f1_df)