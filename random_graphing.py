import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_known_bs_one = { # Example format: "Model Name": (accuracy, latency (ms))
    "ResNet18": (95, 52),
    "MobileNetV2": (94.73, 164),
    "MobileNetV3": (92.97, 197),
    "WideResNet 28-10": (96.11, 81),
    "SmallCNN Ensemble": (83.93, 653),
    "AlexNet": (90, 441),
    "LeNet": (80.86, 1520),
    "VGG19": (93.95, 261),
    "VGG11": (92.39, 371),
    "EfficientNetB0": (93.44, 222),
    "GoogLeNet": (93.57, 92)
}

current_known_bs_64 = {
    "MobileNetV2": (94.73, 152),
    "SmallCNN Ensemble": (83.93, 228),
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
    "SmallCNN Ensemble": (83.93, 1170),
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

current_known_bs_1_list_ds = {
    "MobileNetV2": (94.73, 355),
    "SmallCNN Ensemble": (83.93, 1500),
    "WideResNet 28-10": (96.11, 504),
    "MobileNetV3": (92.97, 374),
    "ResNet18": (95, 665),
    "AlexNet": (90, 1520),
    "LeNet": (80.86, 2400),
    "VGG19": (93.95, 670),
    "VGG11": (92.39, 825),
    "EfficientNetB0": (93.44, 336),
    "GoogLeNet": (93.57, 197)
}

current_known_all = { # "Model Name": [acc, latency_bs1, latency_bs64, latency_bs1_list_ds, latency_bs64_list_ds]
    "MobileNetV2": [94.73, 164, 152, 355, 325],
    "SmallCNN Ensemble": [83.93, 653, 228, 1500, 1170],
    "WideResNet 28-10": [96.11, 81, 73, 504, 95],
    "MobileNetV3": [92.97, 197, 148, 374, 306],
    "ResNet18": [95, 52, 185, 665, 504],
    "AlexNet": [90, 441, 220, 1520, 900],
    "LeNet": [80.86, 1520, 248, 2400, 1550],
    "VGG19": [93.95, 261, 193, 670, 540],
    "VGG11": [92.39, 371, 205, 825, 660],
    "EfficientNetB0": [93.44, 222, 152, 336, 308],
    "GoogLeNet": [93.57, 92, 108, 197, 174]
}

acc_to_params = {
    "MobileNetV2": (94.73, 2236682),
    "MobileNetV3": (92.97, 261712),
    "WideResNet 28-10": (96.11, 36481402),
    "ResNet18": (95, 11173962),
    "AlexNet": (90, 28714826),
    "LeNet": (80.86, 657080),
    "VGG19": (93.95, 38958922),
    "VGG11": (92.39, 14728366),
    "EfficientNetB0": (93.44, 4277382),
    "GoogLeNet": (93.57, 2470842),
    "SmallCNN Ensemble": (83.93, 2990970)
}

acc_to_MACs = {
    "MobileNetV2": (94.73, 94557952),
    "MobileNetV3": (92.97, 56025720),
    "WideResNet 28-10": (96.11, 5253692288),
    "ResNet18": (95, 557889024),
    "AlexNet": (90, 205058048),
    "VGG11": (92.39, 314308608),
    "VGG19": (93.95, 418258944),
}

# plot the above as a labelled scatter plot

plt.figure(figsize=(10, 6))
for model_name, (accuracy, latency) in current_known_bs_one.items():
    plt.scatter(latency, accuracy, label=model_name)
    plt.text(latency + 0.5, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Iterations per Second')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10 BS: 1')
plt.xlim(0, 1600)
plt.ylim(80, 100)
# add line of best fit
x = np.array([latency for _, (_, latency) in current_known_bs_one.items()])
y = np.array([accuracy for _, (accuracy, _) in current_known_bs_one.items()])
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', label='Line of Best Fit')
plt.grid(True)
plt.show()

print(current_known_bs_64.items())

plt.figure(figsize=(10, 6))
for model_name, (accuracy, latency) in current_known_bs_64.items():
    plt.scatter(latency, accuracy, label=model_name)
    plt.text(latency + 0.5, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Iterations per Second')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10 BS: 64')
plt.xlim(0, 300)
plt.ylim(80, 100)
# add line of best fit
x = np.array([latency for _, (_, latency) in current_known_bs_64.items()])
y = np.array([accuracy for _, (accuracy, _) in current_known_bs_64.items()])
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', label='Line of Best Fit')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for model_name, (accuracy, latency) in current_known_bs_64_list_ds.items():
    plt.scatter(latency, accuracy, label=model_name)
    plt.text(latency + 0.5, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Iterations per Second')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10 ListDataset BS: 64')
plt.ylim(80, 100)
plt.xlim(0, 1600)
# add line of best fit
x = np.array([latency for _, (_, latency) in current_known_bs_64_list_ds.items()])
y = np.array([accuracy for _, (accuracy, _) in current_known_bs_64_list_ds.items()])
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', label='Line of Best Fit')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for model_name, (accuracy, latency) in current_known_bs_1_list_ds.items():
    plt.scatter(latency, accuracy, label=model_name)
    plt.text(latency + 0.5, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Iterations per Second')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10 ListDataset BS: 1')
plt.ylim(80, 100)
plt.xlim(0, 2400)
# add line of best fit
x = np.array([latency for _, (_, latency) in current_known_bs_1_list_ds.items()])
y = np.array([accuracy for _, (accuracy, _) in current_known_bs_1_list_ds.items()])
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', label='Line of Best Fit')
plt.grid(True)
plt.show()

# Plot all models with different shapes for each batchsize and dataset type, but same color for same model
plt.figure(figsize=(12, 8))

shapes = {
    'BS=1': 'o',
    'BS=64': 's',
    'BS=1 ListDS': '^',
    'BS=64 ListDS': 'D'
}

# stable per-model colors
model_names = list(current_known_all.keys())
cmap = plt.get_cmap('tab20')
model_color = {m: cmap(i % cmap.N) for i, m in enumerate(model_names)}

# plot points
for model_name, values in current_known_all.items():
    acc = values[0]
    lat_bs1, lat_bs64, lat_bs1_list, lat_bs64_list = values[1:]

    points = [
        ('BS=1', lat_bs1),
        ('BS=64', lat_bs64),
        ('BS=1 ListDS', lat_bs1_list),
        ('BS=64 ListDS', lat_bs64_list),
    ]

    for cond, lat in points:
        plt.scatter(lat, acc, marker=shapes[cond], color=model_color[model_name])
        # optional: shorter text so it doesn't explode
        # plt.text(lat + 5, acc - 0.15, f"{model_name}\n{cond}", fontsize=7)

plt.xlabel('Latency (ms)')  # (your data are ms, not iterations/sec)
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Latency on CIFAR-10 (Various Settings)')
plt.ylim(80, 100)
plt.xlim(0, 2500)
plt.grid(True)

# legend 1: models (colors)
model_handles = [
    plt.Line2D([], [], marker='o', linestyle='None',
               color=model_color[m], label=m, markersize=7)
    for m in model_names
]
leg1 = plt.legend(handles=model_handles, title="Model (color)",
                  bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().add_artist(leg1)

# legend 2: conditions (shapes)
shape_handles = [
    plt.Line2D([], [], marker=marker, linestyle='None',
               color='black', label=cond, markersize=7)
    for cond, marker in shapes.items()
]
plt.legend(handles=shape_handles, title="Setting (marker)",
           bbox_to_anchor=(1.05, 0.45), loc='upper left')

plt.tight_layout()
plt.show()

# Plot accuracy vs number of parameters
plt.figure(figsize=(10, 6))
for model_name, (accuracy, num_params) in acc_to_params.items():
    plt.scatter(num_params, accuracy, label=model_name)
    plt.text(num_params + 50000, accuracy - 0.2, model_name, fontsize=9)
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Number of Parameters on CIFAR-10')
plt.xlim(0, 40000000)
plt.ylim(80, 100)
# add line of best fit
x = np.array([num_params for _, (_, num_params) in acc_to_params.items()])
y = np.array([accuracy for _, (accuracy, _) in acc_to_params.items()])
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red', label='Line of Best Fit')
plt.grid(True)
plt.show()