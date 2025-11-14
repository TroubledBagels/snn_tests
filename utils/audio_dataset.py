import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchaudio
from pathlib import Path

class Sent2PhoneDataset(Dataset):
    def __init__(self, source=f"{str(Path.home())}/data/sent2phone/"):
        self.source_path = source
        self.data = []
        self.labels = []
        self.str_labels = []

        data_files = sorted([f for f in os.listdir(self.source_path) if f.endswith(".wav")])
        label_files = sorted([f for f in os.listdir(self.source_path) if f.endswith(".txt")])

        for f in label_files:
            with open(self.source_path+f, 'r') as l:
                self.str_labels.append(l.readline().strip())

        self.sorted_unique = sorted(list(set(self.str_labels)))
        self.max_c = len(self.sorted_unique)

        for i, f in enumerate(data_files):
            waveform, sr = torchaudio.load(self.source_path+f)
            self.data.append(waveform)
            label_idx = self.sorted_unique.index(self.str_labels[i])
            self.labels.append(label_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]