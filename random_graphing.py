import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_known = { # Example format: "Model Name": (accuracy, latency (ms))
    "ResNet18": (96.05, 52),
    "MobileNetV2": (94.73, 47),
    "MobileNetV3": (95.8, 45),
    "WideResNet 28-10": (99.0, 47),
    "SmallCNN Ensemble": (82.89, 2.2),
    "AlexNet": (90, 53.47),
    "LeNet":
}

# plot the above as a labelled scatter plot

plt.figure(figsize=(10, 6))
for model_name, (accuracy, latency) in current_known.items():
    plt.scatter(latency, accuracy, label=model_name)
    plt.text(latency + 0.5, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Theoretical Latency (ms)')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10')
plt.xlim(0, 60)
plt.ylim(80, 100)
plt.grid(True)
plt.show()