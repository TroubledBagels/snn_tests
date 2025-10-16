'''
Utility functions to load DVS LIPS dataset using tonic
Functions:
- get_raw(batch_size=64, shuffle=True, train=True)
- get_dataloader(batch_size=64, shuffle=True, train=True)
- get_dataloaders(batch_size=64) (N.B. returns train with shuffle, test without)
- visualize_sample(events, label)
'''

import tonic
from tonic import DiskCachedDataset

import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np

def get_dataset(train=True):
    """Load the DVS LIPS dataset."""
    ds = tonic.datasets.DVSLip(save_to="~/data", train=train)
    return ds

def get_raw(batch_size=64, shuffle=True, train=True):
    """Get raw event data loader."""
    ds = get_dataset(train)
    np_transforms = tonic.transforms.NumpyAsType(np.float32)
    cached_raw = DiskCachedDataset(ds, cache_path=f"~/cache/dvs_lips/raw_{'train' if train else 'test'}", transform=np_transforms)
    raw_loader = DataLoader(cached_raw, batch_size=batch_size, shuffle=shuffle, collate_fn=tonic.collation.PadTensors())
    return raw_loader

def get_dataloader(batch_size=64, shuffle=True, train=True):
    """Get processed event data loader."""
    ds = get_dataset(train)
    sensor_size = tonic.datasets.DVSLip.sensor_size
    transform = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    cached = DiskCachedDataset(ds, cache_path=f"~/cache/dvs_lips/{'train' if train else 'test'}", transform=transform)
    loader = DataLoader(cached, batch_size=batch_size, shuffle=shuffle, collate_fn=tonic.collation.PadTensors())
    return loader

def get_dataloaders(batch_size=64):
    """Get both train and test data loaders."""
    train_loader = get_dataloader(batch_size=batch_size, shuffle=True, train=True)
    test_loader = get_dataloader(batch_size=batch_size, shuffle=False, train=False)
    return train_loader, test_loader

def visualize_sample(events, label):  # events: (N, 4) tensor (x, y, p, t), label: int
    """Visualize a sample of events."""
    events = events.numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(events[:, 0], events[:, 1], c=events[:, 2], s=1, cmap='bwr', alpha=0.75)
    plt.colorbar(label='Polarity')
    plt.xlim(0, 128)
    plt.ylim(0, 128)
    plt.title(f'Event Frame - Label: {label}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.show()

def events_to_voxel(events, num_bins, width, height, device):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():

        events_torch = torch.from_numpy(events).float()
        events_torch = events_torch.to(device)

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)
        if events_torch.shape[0] == 0:
            return voxel_grid

        voxel_grid = voxel_grid.flatten()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = float(last_stamp - first_stamp)

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices]
                                    * width + tis_long[valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width
                                    + (tis_long[valid_indices] + 1) * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid

def visualise_voxel(voxel_grid):
    print(f"Voxel grid shape: {voxel_grid.shape}")
    print(f"Voxel grid min: {voxel_grid.min()}, max: {voxel_grid.max()}, mean: {voxel_grid.mean()}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.voxels = voxel_grid.numpy().transpose(2, 1, 0)  # Transpose to (X, Y, Time Bins)
    ax.set_xlim(0, voxel_grid.shape[2])
    ax.set_ylim(0, voxel_grid.shape[1])
    ax.set_zlim(0, voxel_grid.shape[0])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time Bins')
    ax.set_title('Voxel Grid Visualization')
    plt.show()

class DVS_LIPS(Dataset):
    def __init__(self, train=True):
        super(DVS_LIPS, self).__init__()
        base_ds = get_dataset(train)
        self.sensor_size = tonic.datasets.DVSLip.sensor_size
        self.high_transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToVoxelGrid(sensor_size=self.sensor_size, n_time_bins=210),
        ])
        self.low_transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToVoxelGrid(sensor_size=self.sensor_size, n_time_bins=30),
        ])

        self.high_ds = DiskCachedDataset(base_ds, cache_path=f"~/cache/dvs_lips/high_{'train' if train else 'test'}", transform=self.high_transform)
        self.low_ds = DiskCachedDataset(base_ds, cache_path=f"~/cache/dvs_lips/low_{'train' if train else 'test'}", transform=self.low_transform)
        assert len(self.high_ds) == len(self.low_ds)
        self.length = len(self.high_ds)
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        high_sample = self.high_ds[idx]
        low_sample = self.low_ds[idx]
        if self.train:
            high_events, low_events = CenterCrop(high_sample[0], low_sample[0], (96, 96))
            high_events, low_events = RandomCrop(high_events, low_events, (88, 88))
            high_events, low_events = HorizontalFlip(high_events, low_events)
        else:
            high_events, low_events = CenterCrop(high_sample[0], low_sample[0], (88, 88))
        high_events = torch.Tensor(high_events).float().permute(1, 0, 2, 3)
        low_events = torch.Tensor(low_events).float().permute(1, 0, 2, 3)
        label = high_sample[1]
        assert label == low_sample[1]
        return {'event_low': low_events, 'event_high': high_events, 'label': label}

class SplitPolarityAndFrame:
    def __init__(self, sensor_size, time_window=1000, filter_time=10000, num_bins=30):
        self.sensor_size = sensor_size
        self.time_window = time_window
        self.denoise = tonic.transforms.Denoise(filter_time=filter_time)
        self.framer = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=num_bins)

    def __call__(self, events):
        events = self.denoise(events)

        pos_events = events[events['p'] == 1]
        neg_events = events[events['p'] == 0]

        pos_frames = self.framer(pos_events)
        neg_frames = self.framer(neg_events)

        pos_tensor = torch.from_numpy(pos_frames).float().unsqueeze(1)  # Add channel dimension
        neg_tensor = torch.from_numpy(neg_frames).float().unsqueeze(1)
        print(pos_tensor.shape, neg_tensor.shape)
        combined = torch.cat([pos_tensor, neg_tensor], dim=0)  # Shape: (C=2, H, W)
        return combined.permute(1, 0, 2, 3, 4)

class POL_DVS_LIPS(Dataset):
    def __init__(self, train=True, num_bins=30):
        super(POL_DVS_LIPS, self).__init__()
        self.train = train
        base_ds = get_dataset(train)
        transform = SplitPolarityAndFrame(sensor_size=tonic.datasets.DVSLip.sensor_size, time_window=1000, filter_time=10000, num_bins=num_bins)

        self.ds = DiskCachedDataset(base_ds, cache_path=f"~/cache/dvs_lips/pol_{'train' if train else 'test'}", transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        new_sample = [sample[0], sample[1]]
        if self.train:
            new_sample[0], _ = CenterCrop(sample[0], sample[0], (96, 96))
            new_sample[0], _ = RandomCrop(new_sample[0], new_sample[0], (88, 88))
            new_sample[0], _ = HorizontalFlip(new_sample[0], new_sample[0])
        else:
            new_sample[0], _ = CenterCrop(sample[0], sample[0], (88, 88))
        new_sample[0] = new_sample[0].squeeze(0)
        out_sample = (new_sample[0], new_sample[1])
        return out_sample

def CenterCrop(e_low, e_high, size):
    w, h = e_low.shape[-1], e_low.shape[-2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    e_low = e_low[..., y1:y1+th, x1:x1+tw]
    e_high = e_high[..., y1:y1+th, x1:x1+tw]
    return e_low, e_high

def RandomCrop(e_low, e_high, size):
    w, h = e_low.shape[-1], e_low.shape[-2]
    th, tw = size
    if w == tw and h == th:
        return e_low, e_high
    x1 = np.random.randint(0, 8)
    y1 = np.random.randint(0, 8)
    e_low = e_low[..., y1:y1+th, x1:x1+tw]
    e_high = e_high[..., y1:y1+th, x1:x1+tw]
    return e_low, e_high

def HorizontalFlip(e_low, e_high):
    if type(e_low) == torch.Tensor:
        e_low = e_low.numpy()
    if type(e_high) == torch.Tensor:
        e_high = e_high.numpy()
    if np.random.random() < 0.5:
        e_low = np.ascontiguousarray(e_low[..., ::-1])
        e_high = np.ascontiguousarray(e_high[..., ::-1])
    return e_low, e_high

def get_differ(train=True, label1=25, label2=26):
    ds = get_dataset(train=train)
    ds_array = []
    print(f"Filtering dataset for labels {label1} and {label2} only...")
    for i in range(len(ds)):
        events, label = ds[i]
        if label == label1 or label == label2:
            if label == label1:
                label = 0
            else:
                label = 1
            ds_array.append((events, label))
    return ds_array

def get_range(train=True, label_min=0, label_max=50):
    ds = get_dataset(train=train)
    ds_array = []
    print(f"Filtering dataset for labels in range [{label_min}, {label_max})...")
    for i in range(len(ds)):
        events, label = ds[i]
        if label >= label_min and label < label_max:
            ds_array.append((events, label))
    return ds_array

def get_number(train=True, num_classes=5):
    ds = get_dataset(train=train)
    ds_array = []
    class_interval = 100 // num_classes
    req_classes = [i for i in range(0, 100, class_interval)]
    print(f"Getting {req_classes} classes from dataset...")
    labels = ['accused', 'action', 'allow', 'allowed', 'america', 'american', 'another', 'around', 'attacks', 'banks',
              'become', 'being', 'benefit', 'benefits', 'between', 'billion', 'called', 'capital', 'challenge',
              'change', 'chief', 'couple', 'court', 'death', 'described', 'difference', 'different', 'during',
              'economic', 'education', 'election', 'england', 'evening', 'everything', 'exactly', 'general', 'germany',
              'giving', 'ground', 'happen', 'happened', 'having', 'heavy', 'house', 'hundreds', 'immigration', 'judge',
              'labour', 'leaders', 'legal', 'little', 'london', 'majority', 'meeting', 'military', 'million', 'minutes',
              'missing', 'needs', 'number', 'numbers', 'paying', 'perhaps', 'point', 'potential', 'press', 'price',
              'question', 'really', 'right', 'russia', 'russian', 'saying', 'security', 'several', 'should',
              'significant', 'spend', 'spent', 'started', 'still', 'support', 'syria', 'syrian', 'taken', 'taking',
              'terms', 'these', 'thing', 'think', 'times', 'tomorrow', 'under', 'warning', 'water', 'welcome', 'words',
              'worst', 'years', 'young']
    print(f"These are:")
    for i in range(len(req_classes)):
        print(f"- {labels[req_classes[i]]} ({req_classes[i]})")
    for i in range(len(ds)):
        events, label = ds[i]
        if label in req_classes:
            new_label = req_classes.index(label)
            ds_array.append((events, new_label))
    return ds_array

def load_combined_ambiguous(train = True):
    training = "training" if train else "testing"
    print(f"Combining ambiguous classes in DVS LIPS dataset for {training}...")
    ds = get_dataset(train=train)
    ds_array = []
    labels = ['accused', 'action', 'allow', 'allowed', 'america', 'american', 'another', 'around', 'attacks', 'banks',
              'become', 'being', 'benefit', 'benefits', 'between', 'billion', 'called', 'capital', 'challenge',
              'change', 'chief', 'couple', 'court', 'death', 'described', 'difference', 'different', 'during',
              'economic', 'education', 'election', 'england', 'evening', 'everything', 'exactly', 'general', 'germany',
              'giving', 'ground', 'happen', 'happened', 'having', 'heavy', 'house', 'hundreds', 'immigration', 'judge',
              'labour', 'leaders', 'legal', 'little', 'london', 'majority', 'meeting', 'military', 'million', 'minutes',
              'missing', 'needs', 'number', 'numbers', 'paying', 'perhaps', 'point', 'potential', 'press', 'price',
              'question', 'really', 'right', 'russia', 'russian', 'saying', 'security', 'several', 'should',
              'significant', 'spend', 'spent', 'started', 'still', 'support', 'syria', 'syrian', 'taken', 'taking',
              'terms', 'these', 'thing', 'think', 'times', 'tomorrow', 'under', 'warning', 'water', 'welcome', 'words',
              'worst', 'years', 'young']
    ambigious_labels = ['action', 'allow', 'allowed', 'america', 'american', 'around', 'being', 'benefit', 'benefits',
                        'billion', 'called', 'challenge', 'change', 'court', 'difference', 'different', 'election',
                        'evening', 'giving', 'ground', 'happen', 'happened', 'having', 'heavy', 'legal', 'little',
                        'meeting', 'million', 'missing', 'needs', 'number', 'numbers', 'paying', 'press', 'price',
                        'russia', 'russian', 'spend', 'spent', 'syria', 'syrian', 'taken', 'taking', 'terms', 'these',
                        'thing', 'think', 'times', 'words', 'worst']
    word_pairs = {
        "action": "election", "election": "action",
        "allow": "allowed", "allowed": "allow",
        "america": "american", "american": "america",
        "around": "ground", "ground": "around",
        "being": "paying", "paying": "being",
        "benefit": "benefits", "benefits": "benefit",
        "billion": "million", "million": "billion",
        "called": "court", "court": "called",
        "challenge": "change", "change": "challenge",
        "difference": "different", "different": "difference",
        "evening": "giving", "giving": "evening",
        "happen": "happened", "happened": "happen",
        "having": "heavy", "heavy": "having",
        "legal": "little", "little": "legal",
        "meeting": "missing", "missing": "meeting",
        "number": "numbers", "numbers": "number",
        "press": "price", "price": "press",
        "russia": "russian", "russian": "russia",
        "spend": "spent", "spent": "spend",
        "syria": "syrian", "syrian": "syria",
        "taken": "taking", "taking": "taken",
        "terms": "times", "times": "terms",
        "thing": "think", "think": "thing",
        "words": "worst", "worst": "words",
        "these": "needs", "needs": "these"
    }

    new_labels = {
        'accused': 0,
        'action': 1,
        'allow': 2,
        'america': 3,
        'another': 4,
        'around': 5,
        'attacks': 6,
        'banks': 7,
        'become': 8,
        'being': 9,
        'benefit': 10,
        'between': 11,
        'billion': 12,
        'called': 13,
        'capital': 14,
        'challenge': 15,
        'chief': 16,
        'couple': 17,
        'death': 18,
        'described': 19,
        'difference': 20,
        'during': 21,
        'economic': 22,
        'education': 23,
        'england': 24,
        'evening': 25,
        'everything': 26,
        'exactly': 27,
        'general': 28,
        'germany': 29,
        'happen': 30,
        'having': 31,
        'house': 32,
        'hundreds': 33,
        'immigration': 34,
        'judge': 35,
        'labour': 36,
        'leaders': 37,
        'legal': 38,
        'london': 39,
        'majority': 40,
        'meeting': 41,
        'military': 42,
        'minutes': 43,
        'needs': 44,
        'number': 45,
        'perhaps': 46,
        'point': 47,
        'potential': 48,
        'press': 49,
        'question': 50,
        'really': 51,
        'right': 52,
        'russia': 53,
        'saying': 54,
        'security': 55,
        'several': 56,
        'should': 57,
        'significant': 58,
        'spend': 59,
        'started': 60,
        'still': 61,
        'support': 62,
        'syria': 63,
        'taken': 64,
        'terms': 65,
        'thing': 66,
        'tomorrow': 67,
        'under': 68,
        'warning': 69,
        'water': 70,
        'welcome': 71,
        'words': 72,
        'years': 73,
        'young': 74
    }

    for i in range(len(ds)):
        events, label = ds[i]
        label_word = labels[label]
        if label_word in ambigious_labels:
            pair_word = word_pairs[label_word]
            if pair_word < label_word:
                pair_label = new_labels[pair_word]
                ds_array.append((events, pair_label))  # Label word is second in pair -> class of earliest label in alphabet
            else:
                ds_array.append((events, new_labels[label_word]))  # Label word is first in pair -> class of itself
        else:
            ds_array.append((events, new_labels[label_word]))
    return ds_array

if __name__ == "__main__":
    ds = get_differ(train=True, label1=25, label2=26)
    print(f"Filtered dataset size: {len(ds)}")
    print(ds[0])