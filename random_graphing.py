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
    "MobileNetV2": (94.73, 152),
    "SmallCNN Ensemble": (82.89, 228),
    "WideResNet 28-10": (96.11, 73),
    "MobileNetV3": (92.97, 148),
    "ResNet18": (95, 185),
    "AlexNet": (90, 220),
    "LeNet": (80.86, 248),
    "VGG19": (93.95, 193),
    "VGG11": (92.39, 205),
    "EfficientNetB0": (93.44, 152),
    "GoogLeNet": (93.57, 108)
}

current_known_bs_64_list_ds = {
    "MobileNetV2": (94.73, 325),
    "SmallCNN Ensemble": (82.89, 1170),
    "WideResNet 28-10": (96.11, 95),
    "MobileNetV3": (92.97, 306),
    "ResNet18": (95, 504),
    "AlexNet": (90, 900),
    "LeNet": (80.86, 1550),
    "VGG19": (93.95, 540),
    "VGG11": (92.39, 660),
    "EfficientNetB0": (93.44, 308),
    "GoogLeNet": (93.57, 174)
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

print(current_known_bs_64.items())

plt.figure(figsize=(10, 6))
for model_name, (accuracy, latency) in current_known_bs_64.items():
    plt.scatter(latency, accuracy, label=model_name)
    plt.text(latency + 0.5, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Theoretical Latency it/s (Batch size 64)')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10')
plt.xlim(0, 500)
plt.ylim(80, 100)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for model_name, (accuracy, latency) in current_known_bs_64_list_ds.items():
    plt.scatter(latency, accuracy, label=model_name)
    plt.text(latency + 0.5, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Theoretical Latency it/s (Batch size 64, ListDataset)')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10')
plt.ylim(80, 100)
plt.grid(True)
plt.show()