import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torchaudio
from pathlib import Path

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

class MFCCDataset(Dataset):
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
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=30,
                melkwargs={
                    'n_fft': 512,
                    'hop_length': 256,
                    'n_mels': 32
                }
            )(waveform)
            if i == 0:
                plot_spectrogram(mfcc)
                plot_waveform(waveform, sr)
                plot_fft(waveform)
            data = mfcc.numpy()
            with open(self.source_path+label_files[i], 'r') as l:
                str_label = l.readline().strip()
                self.labels.append(torch.Tensor([self.sorted_unique.index(str_label)]))
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

if __name__ == "__main__":
    dataset = MFCCDataset()
    dl = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dl:
        print(batch)
        print(batch[0].shape)
        print(batch[1].shape)
        break
