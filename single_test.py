import torch
import torch.nn as nn
import utils.general as g
import numpy as np
import pandas as pd
import sys

from models import SimpleConv as SC
from torch.utils.data import DataLoader
import utils.load_dvs_lips as dvs
import torchvision.transforms as transforms
import tonic
import tonic.functional

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

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "./outputs 5/w_2fc_best.pth"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    net = SC.SimpleConvModel(1, 75)
    net.load_state_dict(torch.load(model_name, map_location=device))

    te_ds = dvs.load_combined_ambiguous(train=False)
    te_dl = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=lambda x: collate_fn(x, train=False))

    net.to(device)
    net.eval()
    test_acc, test_loss, test_f1, test_rec, test_prec = g.test(net, te_dl, device, nn.CrossEntropyLoss())
    print(f"Test Acc Top-1: {test_acc[0]:.4f}")
    print(f"Test Acc Top-3: {test_acc[1]:.4f}")
    print(f"Test Acc Top-5: {test_acc[2]:.4f}")
    print(f"Test Acc Top-10: {test_acc[3]:.4f}")
    print(f"Test Loss: {np.mean(test_loss):.4f}")
    f1_df = pd.DataFrame(np.array(test_f1).T, index=[1], columns=[f'Class_{i}_F1' for i in range(len(test_f1))])
    rec_df = pd.DataFrame(np.array(test_rec).T, index=[1], columns=[f'Class_{i}_Rec' for i in range(len(test_rec))])
    prec_df = pd.DataFrame(np.array(test_prec).T, index=[1], columns=[f'Class_{i}_Prec' for i in range(len(test_prec))])
    print("F1 Scores per class:")
    print(f1_df)
    print("Recall per class:")
    print(rec_df)
    print("Precision per class:")
    print(prec_df)

    f1_df.to_csv(f"single_f1.csv")
    rec_df.to_csv(f"single_recall.csv")
    prec_df.to_csv(f"single_precision.csv")


