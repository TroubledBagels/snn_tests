import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_word(idx):
    new_labels = {
        'accused': 0,
        'action': 1,
        'allow': 2,
        'america': 3,
        'another': 4,
        'around': 5,
        'attacks': 6,
        'banks': 7,
        'become': 8,
        'being': 9,
        'benefit': 10,
        'between': 11,
        'billion': 12,
        'called': 13,
        'capital': 14,
        'challenge': 15,
        'chief': 16,
        'couple': 17,
        'death': 18,
        'described': 19,
        'difference': 20,
        'during': 21,
        'economic': 22,
        'education': 23,
        'england': 24,
        'evening': 25,
        'everything': 26,
        'exactly': 27,
        'general': 28,
        'germany': 29,
        'happen': 30,
        'having': 31,
        'house': 32,
        'hundreds': 33,
        'immigration': 34,
        'judge': 35,
        'labour': 36,
        'leaders': 37,
        'legal': 38,
        'london': 39,
        'majority': 40,
        'meeting': 41,
        'military': 42,
        'minutes': 43,
        'needs': 44,
        'number': 45,
        'perhaps': 46,
        'point': 47,
        'potential': 48,
        'press': 49,
        'question': 50,
        'really': 51,
        'right': 52,
        'russia': 53,
        'saying': 54,
        'security': 55,
        'several': 56,
        'should': 57,
        'significant': 58,
        'spend': 59,
        'started': 60,
        'still': 61,
        'support': 62,
        'syria': 63,
        'taken': 64,
        'terms': 65,
        'thing': 66,
        'tomorrow': 67,
        'under': 68,
        'warning': 69,
        'water': 70,
        'welcome': 71,
        'words': 72,
        'years': 73,
        'young': 74
    }

    return list(new_labels.keys())[idx]

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

def create_f1_grid(f1_scores, grid_shape=(8, 10)):
    # Use CV2 to generate an image with colour mapping of F1 score magnitude
    # Each cell represents a class with the class number inside it
    num_classes = f1_scores.shape[1]
    cell_height = 100
    cell_width = 150
    grid_height = grid_shape[0] * cell_height
    grid_width = grid_shape[1] * cell_width
    f1_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        row = class_idx // grid_shape[1]
        col = class_idx % grid_shape[1]
        f1_score = f1_scores[-1, class_idx]  # Last epoch score
        color_intensity = int(f1_score * 255)
        color = (0, color_intensity, 255 - color_intensity)  # Green to Red gradient
        top_left = (col * cell_width, row * cell_height)
        bottom_right = ((col + 1) * cell_width, (row + 1) * cell_height)
        cv2.rectangle(f1_image, top_left, bottom_right, color, -1)
        cv2.putText(f1_image, str(get_word(class_idx)), (top_left[0] + 5, top_left[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(f1_image, f"{f1_score:.4f}", (top_left[0] + 5, top_left[1] + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return f1_image



if __name__ == "__main__":
    test_f1 = pd.read_csv("outputs 3/w_2fc_f1_scores.csv", index_col=0)
    print(test_f1)
    amb_f1 = get_ambiguous(test_f1)
    last_epoch = amb_f1.iloc[-1]
    img = create_f1_grid(last_epoch.values.reshape(1, -1))
    cv2.imwrite("./imgs/f1_grid.png", img)
    cv2.imshow("F1 Score Grid", img)
    cv2.waitKey(0)