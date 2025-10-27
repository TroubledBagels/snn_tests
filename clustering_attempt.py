import numpy as np
import matplotlib.pyplot as plt

import utils.load_dvs_lips as dvs

from torch.utils.data import DataLoader

from sklearn.cluster import KMeans

if __name__ == '__main__':
    tr_ds = dvs.load_combined_ambiguous(train=True)
    te_ds = dvs.load_combined_ambiguous(train=False)

    # Attempt k-means clustering on the training set
    num_samples = len(tr_ds)
    flattened_data = []
    for i in range(num_samples):
        events, _ = tr_ds[i]
        frame = np.zeros((128, 128))
        for event in events:
            x, y, p, t = event
            frame[y, x] += 1 if p == 1 else -1
        flattened_data.append(frame.flatten())
        if i % 500 == 0:
            print(f"Processed {i}/{num_samples} samples for clustering.")
    flattened_data = np.array(flattened_data)

    print("Performing K-Means clustering...")
    kmeans = KMeans(n_clusters=75, random_state=0).fit(flattened_data)

    print("Clustering complete. Cluster centers shape:", kmeans.cluster_centers_.shape)
    # Visualize some cluster centers
    num_centers_to_show = 10
    plt.figure(figsize=(15, 5))
    for i in range(num_centers_to_show):
        plt.subplot(2, 5, i + 1)
        center = kmeans.cluster_centers_[i].reshape(128, 128)
        plt.imshow(center, cmap='seismic', vmin=-np.max(np.abs(center)), vmax=np.max(np.abs(center)))
        plt.title(f'Cluster {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Assign test samples to clusters and visualize distribution
    test_flattened_data = []
    num_test_samples = len(te_ds)
    for i in range(num_test_samples):
        events, _ = te_ds[i]
        frame = np.zeros((128, 128))
        for event in events:
            x, y, p, t = event
            frame[y, x] += 1 if p == 1 else -1
        test_flattened_data.append(frame.flatten())
        if i % 500 == 0:
            print(f"Processed {i}/{num_test_samples} test samples for clustering.")
    test_flattened_data = np.array(test_flattened_data)
    test_labels = kmeans.predict(test_flattened_data)
    unique, counts = np.unique(test_labels, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Test Samples')
    plt.title('Test Sample Distribution Across Clusters')
    plt.show()
