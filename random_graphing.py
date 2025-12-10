import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_known_bs_one = { # Example format: "Model Name": (accuracy, latency (ms))
    "ResNet18": (95, 52),
    "MobileNetV2": (94.73, 164),
    "MobileNetV3": (92.97, 197),
    "WideResNet 28-10": (96.11, 81),
    "SmallCNN Ensemble": (82.89, 653),
    "AlexNet": (90, 441),
    "LeNet": (80.86, 1520),
    "VGG19": (93.95, 261),
    "VGG11": (92.39, 371),
    "EfficientNetB0": (93.44, 222),
    "GoogLeNet": (93.57, 92)
}

current_known_bs_64 = {
    "MobileNetV2": (94.73, 21),
    "SmallCNN Ensemble": (82.89, 445),
    "WideResNet 28-10": (96.11, 21),
    "MobileNetV3": (92.97, 22.5),
    "ResNet18": (95, 19),
    "AlexNet": (90, 18.7),
    "LeNet": (80.86, 150),
    "VGG19": (93,95, 18.5),
    "VGG11": (92.39, 29),
    "EfficientNetB0": (93.44, 34),
}

# plot the above as a labelled scatter plot

plt.figure(figsize=(10, 6))
for model_name, (accuracy, latency) in current_known_bs_one.items():
    plt.scatter(latency, accuracy, label=model_name)
    plt.text(latency + 0.5, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Theoretical Latency it/s (Batch size 1)')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10')
plt.xlim(0, 1600)
plt.ylim(80, 100)
plt.grid(True)
plt.show()