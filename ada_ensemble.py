import models.AdaEnsembleAudio as ae
import numpy as np
import torch
import torch.nn as nn
import utils.fft_dataset as fd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_c = 99
    max_classifiers = 100
    ensemble = ae.AdaEnsembleAudio(out_c=n_c, max_classifiers=20)

    ds = fd.MFCCDataset()
    print(ds[0][0].shape)

    train_ds, test_ds = torch.utils.data.random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    trained_ensemble, test_acc = ae.train_ensemble(ensemble, n_c, train_ds, test_ds, num_classifiers=max_classifiers, epochs_per_classifier=10)
    torch.save(trained_ensemble.state_dict(), "ada_ensemble_audio.pth")

    plt.plot(range(1, max_classifiers+1), test_acc)
    plt.xlabel("Number of Classifiers in Ensemble")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Ada Ensemble Audio Classifier Performance")
    plt.grid()
    plt.savefig("ada_ensemble_performance_2.png")
    plt.show()