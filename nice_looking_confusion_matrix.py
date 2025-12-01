import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

class_names = sorted(
    ['ih', 'eh', 'ae', 'ah', 'uh', 'aa', 'ao', 'iy', 'uw', 'er', 'ey', 'ay', 'ow', 'aw', 'b', 'ch', 'd', 'dh', 'f', 'g',
     'hh', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z', 'zh'])


def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using matplotlib.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): List of class names.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # Make colour scale on logarithmic scale
    # cax = ax.matshow(cm, cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix', pad=20)
    # fig.colorbar(cax)
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, norm=matplotlib.colors.LogNorm(vmin=1, vmax=cm.max()))
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    cm = pd.read_csv('./confusion_matrices/binary_square_confusion_matrix_38_bal.csv', index_col=None, header=None)
    print(cm)

    cm = cm.to_numpy()
    class_names = sorted(['ih', 'eh', 'ae', 'ah', 'uh', 'aa', 'ao', 'iy', 'uw', 'er', 'ey', 'ay', 'ow', 'aw', 'b', 'ch',
                          'd', 'dh', 'f', 'g', 'hh', 'jh', 'k', 'l', 'm', 'n', 'ng', 'p', 'r', 's', 'sh', 't', 'th',
                          'v', 'w', 'y', 'z', 'zh'])
    plot_confusion_matrix(cm, class_names)