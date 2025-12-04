import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_results():
    df = pd.read_csv('bclass_comp_res.csv')
    return df

if __name__ == '__main__':
    df = get_results()
    print(df)

    # Plot unique graph showing accuracy against model type for each class pair
    model_types = ["TinyCNN", "SmallCNN", "SeparableSmallCNN", "MediumCNN", "SeparableMediumCNN"]
    class_pairs = df["Class_Pair"].tolist()
    x = np.arange(len(model_types))
    width = 0.4

    for j in range(len(class_pairs)):
        fig, ax = plt.subplots(figsize=(12, 6))
        class_pair = class_pairs[j]
        accuracies = df[df["Class_Pair"] == class_pair][model_types].values.flatten()

        ax.bar(x, accuracies, width)

        ax.set_xlabel('Model Type')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Model Accuracy for {class_pair} and Model Type')
        ax.set_xticks(x)
        ax.set_xticklabels(model_types)
        # Draw dotted line at highest accuracy and lowest accuracy
        highest_acc = np.max(accuracies)
        lowest_acc = np.min(accuracies)
        mean = np.mean(accuracies)
        ax.set_ylim(lowest_acc - 0.05, 1.0)
        ax.axhline(y=highest_acc, color='r', linestyle='--', label='Highest Accuracy')
        ax.axhline(y=lowest_acc, color='g', linestyle='--', label='Lowest Accuracy')
        ax.axhline(y=mean, color=(1.0, 0.5, 0.0), linestyle='--', label='Mean Accuracy')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'comp_results/{class_pair}.png')
    plt.show()
