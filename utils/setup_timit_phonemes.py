import json
import os
from pydub import AudioSegment

DATA_LOC = "/Users/user/data/TIMIT"
TARGET_DIR = "/Users/user/data/timit-phonemes"

class Sentence:
    def __init__(self, wav_file):
        self.phone_list = []
        self.wav_file = wav_file

    def add_word(self, word):
        self.phone_list.append(word)

    def __str__(self):
        return "|".join([str(p) for p in self.phone_list])

class Phoneme:
    def __init__(self, t_s, t_e, label):
        self.t_start = t_s
        self.t_end = t_e
        self.label = label

    def __str__(self):
        return self.label

def go_through_folder(dir_path):
    # Get all json files in the directory
    phn_files = [f for f in os.listdir(dir_path) if f.endswith('.phn')]
    wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]

    sent_list = []

    for phn_file in phn_files:
        # Read line by line
        phn_path = os.path.join(dir_path, phn_file)
        wav_name = phn_file.replace(".phn", ".wav")
        wav_path = os.path.join(dir_path, wav_name)
        sent = Sentence(wav_path)
        with open(phn_path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                parts = line.strip().split(" ")
                t_start = int(parts[0])
                t_end = int(parts[1])
                label = parts[2]
                if label == "h#":
                    continue
                phoneme = Phoneme(t_start, t_end, label)
                sent.add_word(phoneme)
        sent_list.append(sent)

    return sent_list

def process_sentence(sent: Sentence, counter, target_dir=TARGET_DIR):
    wav = AudioSegment.from_wav(sent.wav_file)
    for phoneme in sent.phone_list:
        t1 = phoneme.t_start
        t2 = phoneme.t_end
        print(t1, t2)
        new_wav = wav[t1:t2]
        new_wav.export(f"{target_dir}/{counter}.wav", format="wav")
        with open(f"{target_dir}/{counter}.txt", "w") as f:
            f.write(phoneme.label)
        counter = f"{int(counter)+1:07}"
    return counter

def process_folder(folder_list, counter, target_dir=TARGET_DIR):
    # folder_list is a list of sentence objects
    for sent in folder_list:
        counter = process_sentence(sent, counter, target_dir)
    return counter

def get_folders(train=True):
    # List all folders in data_loc
    app = "TRAIN" if train else "TEST"
    data_loc_app = os.path.join(DATA_LOC, app)
    folders = [f.path for f in os.scandir(data_loc_app) if f.is_dir()]
    nested_folders = [f.path for folder in folders for f in os.scandir(folder) if f.is_dir()]
    return nested_folders

if __name__ == "__main__":
    train = False
    if train: target_dir = os.path.join(TARGET_DIR, "train")
    else: target_dir = os.path.join(TARGET_DIR, "test")
    all_folders = get_folders(train)
    all_folders.sort()
    all_folders_processed = []
    for folder in all_folders:
        all_folders_processed.append(go_through_folder(folder))

    c = "0000000"

    for i, folder in enumerate(all_folders_processed):
        c = process_folder(folder, c, target_dir)
        print(f"Processed {i+1}/{len(all_folders_processed)}: at count {c}")
