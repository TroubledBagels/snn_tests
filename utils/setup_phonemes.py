import json
import os
from pydub import AudioSegment

DATA_LOC = "/Users/user/data/daps-segmented"
TARGET_DIR = "/Users/user/data/daps-phonemes"

class Sentence:
    def __init__(self, t_s, t_e, num_w, wav_file):
        self.t_s = t_s
        self.t_e = t_e
        self.num_w = num_w
        self.word_list = []
        self.wav_file = wav_file

    def add_word(self, word):
        self.word_list.append(word)

    def __str__(self):
        return "|".join([str(w) for w in self.word_list])

class Word:
    def __init__(self, t_s, t_e, num_p):
        self.t_start = t_s
        self.t_end = t_e
        self.num_phonemes = num_p
        self.phoneme_list = []

    def add_phoneme(self, phoneme):
        self.phoneme_list.append(phoneme)

    def __str__(self):
        return " ".join([str(p) for p in self.phoneme_list])

class Phoneme:
    def __init__(self, t_s, t_e, label):
        self.t_start = t_s
        self.t_end = t_e
        self.label = label

    def __str__(self):
        return self.label

def go_through_folder(dir_path):
    # Get all json files in the directory
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]

    sent_list = []

    for json_file in json_files:
        json_path = os.path.join(dir_path, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            t_s = data["context"]["t_begin"]
            t_e = data["context"]["t_end"]
            num_w = len(data['tokens'])
            wav_name = json_path.replace(".json", ".wav")
            wav_path = os.path.join(dir_path, wav_name)
            sentence = Sentence(t_s, t_e, num_w, wav_path)
            tokens = data['tokens']
            for token in tokens:
                if token["case"] == 'not-found':
                    print(f"Skipping {token} in {json_path}")
                    continue
                wt_s = token["t_begin"]
                wt_e = token["t_end"]
                phonemes = token["phonemes"]
                num_p = len(phonemes)
                word = Word(wt_s, wt_e, num_p)
                roll_t = wt_s
                for phoneme in phonemes:
                    t_s = roll_t
                    if t_s < 0:
                        print(f"Skipping {phoneme} in {json_path} as requires prior file")
                        continue
                    t_e = roll_t + phoneme['duration']
                    roll_t = t_e
                    p = Phoneme(t_s, t_e, phoneme['phone'])
                    word.add_phoneme(p)
                sentence.add_word(word)
            sent_list.append(sentence)
    print(f"Found {len(wav_files)} .wav files in {dir_path}")
    return sent_list

def process_sentence(sent: Sentence, counter):
    wav = AudioSegment.from_wav(sent.wav_file)
    for word in sent.word_list:
        for phoneme in word.phoneme_list:
            t1 = phoneme.t_start * 1000 // 1
            t2 = phoneme.t_end * 1000 // 1
            new_wav = wav[t1:t2]
            if counter == "010301":
                print(f"{t1}-{t2}")
                counter = f"{int(counter)+1:06}"
                continue
            new_wav.export(f"{TARGET_DIR}/{counter}.wav", format="wav")
            with open(f"{TARGET_DIR}/{counter}.txt", "w") as f:
                f.write(phoneme.label)
            counter = f"{int(counter)+1:06}"
    return counter

def process_folder(folder_list, counter):
    # folder_list is a list of sentence objects
    for sent in folder_list:
        counter = process_sentence(sent, counter)
    return counter

def get_folders():
    # List all folders in data_loc
    folders = [f.path for f in os.scandir(DATA_LOC) if f.is_dir()]
    return folders

if __name__ == "__main__":
    all_folders = get_folders()
    all_folders.sort()
    all_folders_processed = []
    for folder in all_folders:
        all_folders_processed.append(go_through_folder(folder))

    c = "000000"

    for i, folder in enumerate(all_folders_processed):
        c = process_folder(folder, c)
        print(f"Processed {i+1}/{len(all_folders_processed)}")
