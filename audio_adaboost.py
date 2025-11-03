import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils.fft_dataset import MFCCDataset
import torch

def extract_features(dataset):
    features = []
    labels = []
    for i in range(len(dataset)):
        mfcc, label = dataset[i]
        mfcc_mean = mfcc.mean(dim=-1).numpy()  # Average over time frames
        features.append(mfcc_mean)
        labels.append(label)
        if i % 500 == 0:
            print(f"Extracted features from {i}/{len(dataset)} samples.")
    return np.array(features), np.array(labels)

if __name__ == '__main__':
    dataset = MFCCDataset()
    torch.manual_seed(42)
    np.random.seed(42)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_features, train_labels = extract_features(train_ds)

    print("Training AdaBoost classifier...")
    adb_clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    adb_clf.fit(train_features, train_labels.ravel())

    print("Training complete.")
    # Perform test set classification
    test_features, test_labels = extract_features(test_ds)
    predicted_labels = adb_clf.predict(test_features)

    # Calculate accuracy
    accuracy = np.sum(test_labels.ravel() == predicted_labels) / len(test_labels)
    print(f"Accuracy of AdaBoost classifier on test set: {accuracy:.4f}")