import scipy.io.wavfile as wav
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os
import numpy as np

SOURCE = "/Users/user/data/daps-phonemes/"
TARGET = "/Users/user/data/daps-fft/"

def process_sample(sample):
    fs, data = wav.read(sample)
    a = data.T
    if a.ndim > 1:
        a = a[0]
    b = [(ele/8**2)*2-1 for ele in a]
    try:
        c = fft(b)
    except ValueError:
        print(sample)
        exit()

    d = len(c)//2
    return abs(c[:d])

if __name__ == "__main__":
    file = f"{SOURCE}000000.wav"
    # List all wav files in SOURCE (mix of wav and txt)
    all_files = [f for f in os.listdir(SOURCE) if f.endswith(".wav")]
    all_files.sort()
    for i, f in enumerate(all_files):
        processed = process_sample(SOURCE+f)
        np.save(f"{TARGET}/{f.replace(".wav", ".npy")}", processed)
        if i % 100 == 0:
            print(f"\rProcessed {i+1}/{len(all_files)}", end="")
