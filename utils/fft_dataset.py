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

    if type == 'all':
        return old_label
    elif type == 'v_or_c':
        if old_label in vowel_list:
            return 'vowel'
        else:
            return 'consonant'
    elif type == 'mono_or_di':
        if old_label in diphthongs:
            return 'diphthong'
        elif old_label in short_vowels or old_label in long_vowels:
            return 'monophthong'
    elif type == 'short_or_long':
        if old_label in short_vowels:
            return 'short'
        elif old_label in long_vowels:
            return 'long'
    return old_label

class MFCCDataset(Dataset):
    def __init__(self, source=f"/data/daps-phonemes/", type='all'):
        source = os.path.join(str(Path.home()), source.lstrip('/'))
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
            if i+1 % 1000 == 0:
                print(f"\rLoaded {i+1}/{len(data_files)} samples", end="")
        print("")


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

def get_by_labels(dataset: MFCCDataset, target_labels: list[int], incl_else: bool = False, with_balance=0):
    new_ds = []
    for i in range(len(dataset)):
        data, label = dataset[i]
        if label.item() == target_labels[0]:
            new_ds.append((data, 0))
        elif label.item() == target_labels[1]:
            new_ds.append((data, 1))
        elif incl_else:
            new_ds.append((data, 2))

    if with_balance:
        # 1 is oversampling, 2 is undersampling
        counts = [0, 0]
        for _, label in new_ds:
            if label in [0, 1]:
                counts[label] += 1
        if with_balance == 1:
            max_count = max(counts)
            balanced_ds = []
            for label_id in [0, 1]:
                items = [item for item in new_ds if item[1] == label_id]
                multiplier = max_count // counts[label_id]
                remainder = max_count % counts[label_id]
                balanced_ds.extend(items * multiplier)
                balanced_ds.extend(items[:remainder])
            new_ds = balanced_ds
        elif with_balance == 2:
            min_count = min(counts)
            balanced_ds = []
            for label_id in [0, 1]:
                items = [item for item in new_ds if item[1] == label_id]
                balanced_ds.extend(items[:min_count])
            new_ds = balanced_ds
    return new_ds

if __name__ == "__main__":
    dataset = MFCCDataset(type="all")
    print(dataset.sorted_unique)
    length_dict = {}
    for i in range(len(dataset)):
        data, label = dataset[i]
        if label.item() not in length_dict:
            length_dict[label.item()] = []
        length_dict[label.item()].append(data.shape[1])
    mean_lengths = np.array([np.mean(length_dict[key]) for key in sorted(length_dict, key=lambda x: x)])
    overall_mean = np.mean(mean_lengths)
    median_lengths = np.array([np.median(length_dict[key]) for key in sorted(length_dict, key=lambda x: x)])
    max_lengths = np.array([np.max(length_dict[key]) for key in sorted(length_dict, key=lambda x: x)])
    min_lengths = np.array([np.min(length_dict[key]) for key in sorted(length_dict, key=lambda x: x)])
    q1_lengths = np.array([np.percentile(length_dict[key], 25) for key in sorted(length_dict, key=lambda x: x)])
    q3_lengths = np.array([np.percentile(length_dict[key], 75) for key in sorted(length_dict, key=lambda x: x)])
    print(f"Overall Mean Length: {overall_mean:.2f}")
    print(f"Mean of Mean Lengths: {np.mean(mean_lengths):.2f}, Std: {np.std(mean_lengths):.2f}")
    # Create 1 big box plot with all these stats. Box plot per phoneme class stretching horizontally in a portrait image
    plt.figure(figsize=(8, 12))
    plt.boxplot([length_dict[key] for key in sorted(length_dict, key=lambda x: x)], vert=False, patch_artist=False, showfliers=False)
    plt.yticks(ticks=np.arange(1, len(dataset.sorted_unique)+1), labels=dataset.sorted_unique)
    plt.xlabel('Length (Number of Frames)')
    plt.title('Distribution of MFCC Frame Lengths per Phoneme Class')
    plt.show()