import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt

def plot_spectrogram(spectrogram):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.log2()[0,:,:].numpy(), cmap='gray')
    plt.title('Spectrogram')
    plt.xlabel('Frame')
    plt.ylabel('Frequency Bin')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def plot_waveform(waveform, sr):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())
    plt.title('Waveform')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_fft(waveform):
    import matplotlib.pyplot as plt
    waveform_np = waveform.numpy().squeeze()
    fft_vals = np.fft.fft(waveform_np)
    fft_freq = np.fft.fftfreq(len(fft_vals), 1/16000)  # Assuming sample rate of 16kHz
    plt.figure(figsize=(10, 4))
    plt.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_vals)[:len(fft_vals)//2])
    plt.title('FFT of Waveform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

class MelSpecDataset(Dataset):
    def __init__(self, source=f"{str(Path.home())}/data/daps-phonemes/"):
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
            spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=512,
                hop_length=256,
                n_mels=32
            )(waveform)
            mel_db = torchaudio.transforms.AmplitudeToDB()(spec)
            if i == 0:
                plot_spectrogram(mel_db)
                plot_waveform(waveform, sr)
                plot_fft(waveform)
            data = mel_db.numpy()
            with open(self.source_path+label_files[i], 'r') as l:
                str_label = l.readline().strip()
                self.labels.append(torch.Tensor([self.sorted_unique.index(str_label)]))
            self.data.append(torch.from_numpy(data).float())


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].squeeze(0), self.labels[idx].squeeze()

def get_new_label(old_label: str, type: str) -> str:
    vowel_list = ['ih', 'eh', 'ae', 'ah', 'uh', 'aa', 'ao', 'iy', 'uw', 'er', 'ey', 'ay', 'ow', 'aw']
    cons_list = ['b', 'ch', 'd', 'dh', 'f', 'g', 'hh', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh']
    short_vowels = ['ih', 'eh', 'ae', 'ah', 'uh']
    long_vowels = ['iy', 'uw', 'aa', 'ao', 'er']
    diphthongs = ['ey', 'ay', 'ow', 'aw']
    stops = ['p', 'b', 't', 'd', 'k', 'g']
    fricatives = ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh']
    affricates = ['ch', 'jh']
    nasals = ['m', 'n', 'ng']
    liquids = ['l', 'r']
    glides = ['w', 'y']
    aspirate = ['hh']

    match type:
        case 'all':
            return old_label
        case 'v_or_c':
            if old_label in vowel_list:
                return 'vowel'
            else:
                return 'consonant'
        case 'mono_or_di':
            if old_label in diphthongs:
                return 'diphthong'
            elif old_label in short_vowels or old_label in long_vowels:
                return 'monophthong'
        case 'short_or_long':
            if old_label in short_vowels:
                return 'short'
            elif old_label in long_vowels:
                return 'long'
    return old_label

class MFCCDataset(Dataset):
    def __init__(self, source=f"{str(Path.home())}/data/daps-phonemes/", type='all'):
        self.type = type
        self.source_path = source
        self.data = []
        self.labels = []
        self.str_labels = []

        data_files = sorted([f for f in os.listdir(self.source_path) if f.endswith(".wav")])
        label_files = sorted([f for f in os.listdir(self.source_path) if f.endswith(".txt")])

        for f in label_files:
            with open(self.source_path+f, 'r') as l:
                temp = l.readline().strip().split('_')[0]
                new_label = get_new_label(temp, self.type)
                if new_label == temp and self.type != 'all':
                    continue
                self.str_labels.append(new_label)

        self.sorted_unique = sorted(list(set(self.str_labels)))
        self.max_c = len(self.sorted_unique)

        for i, f in enumerate(data_files):
            waveform, sr = torchaudio.load(self.source_path+f)
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=30,
                melkwargs={
                    'n_fft': 512,
                    'hop_length': 256,
                    'n_mels': 32
                }
            )(waveform)
            # if i == 0:
            #     plot_spectrogram(mfcc)
            #     plot_waveform(waveform, sr)
            #     plot_fft(waveform)
            data = mfcc.numpy()
            with open(self.source_path+label_files[i], 'r') as l:
                str_label = l.readline().strip().split('_')[0]
                new_label = get_new_label(str_label, self.type)
                if str_label == new_label and self.type != 'all':
                    continue
                self.labels.append(torch.Tensor([self.sorted_unique.index(new_label)]))
            self.data.append(torch.from_numpy(data).float())


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [(self.data[i].squeeze(0), self.labels[i].squeeze()) for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, list):
            return [(self.data[i].squeeze(0), self.labels[i].squeeze()) for i in idx]
        return self.data[idx].squeeze(0), self.labels[idx].squeeze()

    def get_by_label(self, label: list[int]):
        indices = [i for i, lbl in enumerate(self.labels) if lbl.item() in label]
        return self[indices]

    def get_dist(self):
        dist = {}
        for lbl in self.str_labels:
            lbl_item = lbl
            if lbl_item in dist:
                dist[lbl_item] += 1
            else:
                dist[lbl_item] = 1
        return sorted(dist.items(), key=lambda x: x[0])

def get_one_of_each():
    dataset = MFCCDataset()
    selected_indices = []
    seen_labels = set()
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_item = label.item()
        if label_item not in seen_labels:
            seen_labels.add(label_item)
            selected_indices.append(i)
        if len(seen_labels) == dataset.max_c:
            break
    return dataset[selected_indices]

def get_by_labels(dataset: MFCCDataset, target_labels: list[int], incl_else: bool = False):
    new_ds = []
    for i in range(len(dataset)):
        data, label = dataset[i]
        if label.item() == target_labels[0]:
            new_ds.append((data, 0))
        elif label.item() == target_labels[1]:
            new_ds.append((data, 1))
        elif incl_else:
            new_ds.append((data, 2))
    return new_ds

if __name__ == "__main__":
    dataset = MFCCDataset(type="all")
    dl = DataLoader(dataset, batch_size=1, shuffle=True)
    print(next(iter(dl))[0].shape)
    dist = dataset.get_dist()
    print(dist)
    print(dataset[0][0].shape)
    # Plot bar chart distribution of classes
    plt.figure(figsize=(12, 8))
    # make bar chart with class labels at 90 degree angle, dist is a list of tuples (label, count)
    plt.bar([x[0] for x in dist], [x[1] for x in dist])
    plt.xticks(rotation=90)
    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Phoneme Dataset')
    plt.show()