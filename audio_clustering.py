import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils.fft_dataset import MFCCDataset
import torch

def extract_features(dataset):
    features = []
    for i in range(len(dataset)):
        mfcc, label = dataset[i]
        mfcc_mean = mfcc.mean(dim=-1).numpy()  # Average over time frames
        features.append(mfcc_mean)
        if i % 500 == 0:
            print(f"Extracted features from {i}/{len(dataset)} samples.")
    return np.array(features)

if __name__ == '__main__':
    dataset = MFCCDataset()
    train_ds, test_ds = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    features = extract_features(train_ds)

    print("Performing K-Means clustering...")
    num_clusters = 99
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

    print("Clustering complete. Cluster centers shape:", kmeans.cluster_centers_.shape)
    # Perform test set clustering
    test_features = extract_features(test_ds)
    test_labels = kmeans.predict(test_features)
    unique, counts = np.unique(test_labels, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Samples')
    plt.title('K-Means Clustering of MFCC Features on Test Set')
    plt.show()

    # Calculate agreement with true labels
    true_labels = [label for _, label in test_ds]
    agreement = np.sum(np.array(true_labels) == test_labels) / len(true_labels)
    print(f"Agreement between K-Means clusters and true labels: {agreement:.2f}")