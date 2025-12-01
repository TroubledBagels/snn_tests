import numpy as np
import matplotlib.pyplot as plt
import cv2

def apply_custom_colourmap(im_grey):
    im_grey = cv2.normalize(im_grey, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    # I want green to raise and red to lower proportionately so that 1 is bright green and 0 is bright red
    for i in range(256):
        lut[i, 0, 0] = 0        # Blue channel
        lut[i, 0, 1] = i        # Green channel
        lut[i, 0, 2] = 255 - i  # Red channel
    lut[0, 0, :] = [0, 0, 0]         # Black for 0

    im_color = cv2.LUT(cv2.cvtColor(im_grey, cv2.COLOR_GRAY2BGR), lut)
    return im_color


def generate_heatmap(data, num_to_str_labels, colourmap=cv2.COLORMAP_HOT, use_acc=True, tuple_based=False):
    # Generates heatmap for the triangle of binary classifiers
    # data is a dictionary of form ((class1, class2): accuracy)
    classes = set()
    for (c1, c2) in data.keys():
        classes.add(c1)
        classes.add(c2)
    classes = sorted(list(classes))
    n = len(classes)

    heatmap = np.zeros((n, n))
    for (c1, c2), acc in data.items():
        i = classes.index(c1)
        j = classes.index(c2)
        if use_acc:
            heatmap[i, j] = (acc - 0.5) * 2
            heatmap[j, i] = (acc - 0.5) * 2  # Symmetric
        elif tuple_based:
            heatmap[i, j] = acc[1]
            heatmap[j, i] = acc[0]
    pixel_size = 25
    # Normalize heatmap to 0-255
    heatmap = (heatmap * 255).astype(np.uint8)
    # Apply colour map
    # heatmap_img = cv2.applyColorMap(heatmap, colourmap)
    heatmap_img = apply_custom_colourmap(heatmap)
    # add border of 1 pixel in #C8C8C8 colour
    heatmap_img = cv2.copyMakeBorder(heatmap_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[200, 200, 200])
    # make heatmap_img have 10x10 pixels per cell
    heatmap_img = cv2.resize(heatmap_img, ((n+2) * pixel_size, (n+2) * pixel_size), interpolation=cv2.INTER_NEAREST)
    # add class labels
    for i, c in enumerate(classes):
        label = num_to_str_labels[c]
        cv2.putText(heatmap_img, label, ((i+1) * pixel_size + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(heatmap_img, label, (5, (i+1) * pixel_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return heatmap_img, classes

if __name__ == "__main__":
    # Example usage

    # Create dictionary from folling data:
    '''
     Classes 0 vs 1: 96.30%
     Classes 0 vs 2: 91.45%
     Classes 0 vs 3: 94.55%
     Classes 0 vs 4: 94.95%
     Classes 0 vs 5: 96.10%
     Classes 0 vs 6: 97.30%
     Classes 0 vs 7: 96.70%
     Classes 0 vs 8: 92.95%
     Classes 0 vs 9: 94.45%
     Classes 1 vs 2: 97.75%
     Classes 1 vs 3: 97.25%
     Classes 1 vs 4: 98.45%
     Classes 1 vs 5: 98.35%
     Classes 1 vs 6: 98.40%
     Classes 1 vs 7: 98.70%
     Classes 1 vs 8: 96.05%
     Classes 1 vs 9: 91.05%
     Classes 2 vs 3: 85.25%
     Classes 2 vs 4: 87.40%
     Classes 2 vs 5: 89.20%
     Classes 2 vs 6: 91.85%
     Classes 2 vs 7: 92.15%
     Classes 2 vs 8: 96.25%
     Classes 2 vs 9: 96.80%
     Classes 3 vs 4: 89.30%
     Classes 3 vs 5: 77.75%
     Classes 3 vs 6: 89.60%
     Classes 3 vs 7: 91.85%
     Classes 3 vs 8: 96.70%
     Classes 3 vs 9: 95.35%
     Classes 4 vs 5: 90.25%
     Classes 4 vs 6: 93.00%
     Classes 4 vs 7: 91.35%
     Classes 4 vs 8: 97.30%
     Classes 4 vs 9: 97.80%
     Classes 5 vs 6: 93.90%
     Classes 5 vs 7: 90.40%
     Classes 5 vs 8: 96.75%
     Classes 5 vs 9: 96.85%
     Classes 6 vs 7: 96.75%
     Classes 6 vs 8: 98.05%
     Classes 6 vs 9: 98.20%
     Classes 7 vs 8: 97.80%
     Classes 7 vs 9: 97.05%
     Classes 8 vs 9: 95.10%
    '''
    acc_dict = {
        (0, 1): 0.9630,
        (0, 2): 0.9145,
        (0, 3): 0.9455,
        (0, 4): 0.9495,
        (0, 5): 0.9610,
        (0, 6): 0.9730,
        (0, 7): 0.9670,
        (0, 8): 0.9295,
        (0, 9): 0.9445,
        (1, 2): 0.9775,
        (1, 3): 0.9725,
        (1, 4): 0.9845,
        (1, 5): 0.9835,
        (1, 6): 0.9840,
        (1, 7): 0.9870,
        (1, 8): 0.9605,
        (1, 9): 0.9105,
        (2, 3): 0.8525,
        (2, 4): 0.8740,
        (2, 5): 0.8920,
        (2, 6): 0.9185,
        (2, 7): 0.9215,
        (2, 8): 0.9625,
        (2, 9): 0.9680,
        (3, 4): 0.8930,
        (3, 5): 0.7775,
        (3, 6): 0.8960,
        (3, 7): 0.9185,
        (3, 8): 0.9670,
        (3, 9): 0.9535,
        (4, 5): 0.9025,
        (4, 6): 0.9300,
        (4, 7): 0.9135,
        (4, 8): 0.9730,
        (4, 9): 0.9780,
        (5, 6): 0.9390,
        (5, 7): 0.9040,
        (5, 8): 0.9675,
        (5, 9): 0.9685,
        (6, 7): 0.9675,
        (6, 8): 0.9805,
        (6, 9): 0.9820,
        (7, 8): 0.9780,
        (7, 9): 0.9705,
        (8, 9): 0.9510
    }
    num_to_str_labels = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9'
    }
    heatmap_img, classes = generate_heatmap(acc_dict, num_to_str_labels, use_acc=True)
    cv2.imshow("Heatmap", heatmap_img)
    cv2.imwrite("example_heatmap.png", heatmap_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()