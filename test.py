import snntorch.functional
import snntorch as snn

import utils.load_dvs_lips as dvs
import torch
import numpy as np
import tonic
from torch.utils.data import DataLoader
from models.SimpleConv import SimpleConvModel
import tqdm
import utils.general as g
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from utils.load_dvs_lips import load_combined_ambiguous


def collate_fn(batch, train=True):
    events, labels = zip(*batch)
    downsample_factor = 1
    sensor_size = (int(128 * downsample_factor), int(128 * downsample_factor), 2)
    time_bins = 800
    if train:
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(spatial_factor=downsample_factor),
            tonic.transforms.RandomFlipLR(p=0.5, sensor_size=sensor_size),
            #tonic.transforms.RandomFlipUD(p=0.2, sensor_size=sensor_size),
            tonic.transforms.UniformNoise(sensor_size=sensor_size, n=150),
            tonic.transforms.EventDrop(sensor_size=sensor_size),
            # tonic.transforms.ToFrame(time_window=1000, sensor_size=sensor_size),
            tonic.transforms.ToFrame(n_time_bins=time_bins, sensor_size=sensor_size),
        ])
        torch_transforms = transforms.Compose([
            transforms.CenterCrop((96, 96)),
            transforms.RandomCrop((88, 88))
        ])
    else:
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(spatial_factor=downsample_factor),
            # tonic.transforms.ToFrame(time_window=1000, sensor_size=sensor_size),
            tonic.transforms.ToFrame(n_time_bins=time_bins, sensor_size=sensor_size)
        ])
        torch_transforms = transforms.Compose([
            transforms.CenterCrop((88, 88))
        ])
    frames = []
    for event in events:
        frame = torch.from_numpy(transform(event)).float()
        frame = torch_transforms(frame)
        frame_combined = frame[:, 0, :, :] - frame[:, 1, :, :]
        frame_combined = frame_combined.unsqueeze(1)
        #T, H, W = frame_combined.shape
        #frame_combined = frame_combined.view(T, 1, H, W)
        frames.append(frame_combined)
    frames = g.add_event_fade(frames, decay=0.9)

    # set frame num to max of batch

    frames = torch.stack(frames)


    labels = torch.tensor(labels)
    return frames, labels

def density_box_plot(ds):
    num_events = []
    for i in range(len(ds)):
        events, _ = ds[i]
        num_events.append(len(events) / (128 * 128 * 2 * 800))
        if i % 500 == 0:
            print(f"Processed {i}/{len(ds)} samples for density plot.")

    plt.boxplot(num_events, vert=False)
    plt.title("Event Count Distribution")
    plt.xlabel("Density")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda:0"
    print(f"Using device: {device}")

    bottom_label = 0
    top_label = 75

    train_ds = dvs.load_combined_ambiguous(train=True)
    density_box_plot(train_ds)
    # Print the highest and lowest labels in the training set
    labels = [label for _, label in train_ds]
    print(f"Training set labels: min {min(labels)}, max {max(labels)}")
    test_ds = dvs.load_combined_ambiguous(train=False)

    bs = 32

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=lambda x: collate_fn(x))
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=lambda x: collate_fn(x, train=False))

    sample, label = next(iter(train_dl))
    print(f"Sample shape: {sample.shape}")
    print(f"Sample min: {sample.min()}, max: {sample.max()}")
    # Count unique values in the sample
    unique, counts = np.unique(sample.numpy(), return_counts=True)
    print(f"Number of unique values: {len(dict(zip(unique, counts)))}")
    print(f"Number of 1s: {dict(zip(unique, counts))[1.0] if 1.0 in dict(zip(unique, counts)) else 0}")
    print(f"Number of -1s: {dict(zip(unique, counts))[-1.0] if -1.0 in dict(zip(unique, counts)) else 0}")
    # count nonzero
    print(f"Total Nonzero Entries: {(sample != 0).sum()}")
    print(f"Number of new entries after fade (ish): {(sample != 0).sum() - (sample == 1).sum() - (sample == -1).sum()}")

    model = SimpleConvModel(in_c=1, out_c=top_label-bottom_label)
    print(model)
    # loss_fn = snn.functional.ce_count_loss()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"Using loss function: {loss_fn}")

    model = g.train(model, train_dl, test_dl, device, loss_fn=loss_fn, lr=2e-3, epochs=50, save_name="w50", weight_decay=1e-4)

    # acc_hist = []
    # for i in range(5):
    #     acc_hist.append(g.test(model, test_dl, device=device, loss_fn=loss_fn)[0])

    # g.plot_acc([0.1, 0.2, 0.3, 0.4, 0.5], acc_hist)

    torch.save(model.state_dict(), "actually_windowed.pth")
