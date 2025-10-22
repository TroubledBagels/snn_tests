import torch
import torch.nn
from torch.utils.data import DataLoader
import models.SimpleConv as SC
import numpy as np
import tqdm
import tonic
import tonic.functional
import torchvision.transforms as transforms
import utils.general as g
import utils.load_dvs_lips as dvs

def test_top_k(model, test_dl):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    top15_correct = 0
    top20_correct = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(test_dl, desc='Testing', unit='batch')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)

            top_20 = outputs.topk(20, dim=1).indices
            top_15 = outputs.topk(15, dim=1).indices
            top_10 = outputs.topk(10, dim=1).indices
            top_5 = outputs.topk(5, dim=1).indices
            top_3 = outputs.topk(3, dim=1).indices
            top_1 = outputs.topk(1, dim=1).indices


            for i in range(labels.size(0)):
                if labels[i] in top_20[i]:
                    top20_correct += 1
                if labels[i] in top_15[i]:
                    top15_correct += 1
                if labels[i] in top_10[i]:
                    top10_correct += 1
                if labels[i] in top_5[i]:
                    top5_correct += 1
                if labels[i] in top_3[i]:
                    top3_correct += 1
                if labels[i] in top_1[i]:
                    top1_correct += 1

        pbar.close()

    total_samples = len(test_dl.dataset)
    top1_acc = top1_correct / total_samples
    top3_acc = top3_correct / total_samples
    top5_acc = top5_correct / total_samples
    top10_acc = top10_correct / total_samples
    top15_acc = top15_correct / total_samples
    top20_acc = top20_correct / total_samples
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Top-10 Accuracy: {top10_acc:.4f}")
    print(f"Top-15 Accuracy: {top15_acc:.4f}")
    print(f"Top-20 Accuracy: {top20_acc:.4f}")

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

if __name__ == "__main__":
    model = SC.SimpleConvModel(in_c=1, out_c=75)
    # load model
    model.load_state_dict(torch.load('./outputs/windowed.pth', map_location=torch.device('cpu')))
    test_ds = dvs.load_combined_ambiguous(train=False)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=lambda x: collate_fn(x, train=False))
    test_top_k(model, test_dl)
