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

def collate_fn(batch, train=True):
    events, labels = zip(*batch)
    downsample_factor = 0.5
    sensor_size = (int(128 * downsample_factor), int(128 * downsample_factor), 2)
    if train:
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(spatial_factor=downsample_factor),
            tonic.transforms.RandomFlipLR(p=0.5, sensor_size=sensor_size),
            #tonic.transforms.RandomFlipUD(p=0.2, sensor_size=sensor_size),
            tonic.transforms.UniformNoise(sensor_size=sensor_size, n=5000),
            tonic.transforms.EventDrop(sensor_size=sensor_size),
            # tonic.transforms.ToFrame(time_window=1000, sensor_size=sensor_size),
            tonic.transforms.ToFrame(n_time_bins=200, sensor_size=sensor_size),
        ])
    else:
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(spatial_factor=downsample_factor),
            tonic.transforms.ToFrame(n_time_bins=200, sensor_size=sensor_size)
        ])
    frames = []
    for event in events:
        frame = torch.from_numpy(transform(event)).float()
        frame_combined = frame[:, 0, :, :] - frame[:, 1, :, :]
        frame_combined = frame_combined.unsqueeze(1)
        #T, H, W = frame_combined.shape
        #frame_combined = frame_combined.view(T, 1, H, W)
        frames.append(frame_combined)
    frames = g.add_event_fade(frames, decay=0.6)

    # set frame num to max of batch

    frames = torch.stack(frames)


    labels = torch.tensor(labels)
    return frames, labels

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda:0"
    print(f"Using device: {device}")

    bottom_label = 0
    top_label = 2

    train_ds = dvs.get_range(True, bottom_label, top_label)
    test_ds = dvs.get_range(False, bottom_label, top_label)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=lambda x: collate_fn(x))
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=lambda x: collate_fn(x, train=False))

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
    loss_fn = nn.CrossEntropyLoss()
    print(f"Using loss function: {loss_fn}")

    model = g.train(model, train_dl, test_dl, device, loss_fn=loss_fn, lr=1e-3, epochs=50, save_name="20")
    torch.save(model.state_dict(), "model_20.pth")
