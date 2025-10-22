import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_f1_scores(f1_scores, class_names):
    # F1 scores: Epoch x Classes x 1 (score)
    plt.figure(figsize=(12, 8))
    plt.plot(f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class over Epochs')
    plt.legend(class_names, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.tight_layout()
    plt.show()

def get_ambiguous(f1_df):
    amb_class_list = [1, 2, 3, 5, 9, 10, 12, 13, 15, 20, 25, 30, 31, 38, 41, 44, 45,
                      49, 53, 59, 63, 64, 65, 66, 72]
    amb_df = pd.DataFrame()
    for num in amb_class_list:
        class_col = f"Class_{num}_F1"
        amb_df[class_col] = f1_df[class_col]

    return amb_df

def get_means(f1_df):
    mean_f1 = f1_df.filter(like='Class_').mean(axis=1)
    return mean_f1


if __name__ == "__main__":
    