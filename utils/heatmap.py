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


def generate_heatmap(data, num_to_str_labels, colourmap=cv2.COLORMAP_HOT, use_acc=True):
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
            heatmap[i, j] = acc
            heatmap[j, i] = acc  # Symmetric
        else:
            heatmap[i, j] = 1-acc
            heatmap[j, i] = acc
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
    acc_dict = {
        (0, 1): 0.9,
        (0, 2): 0.8,
        (0, 3): 0.6,
        (1, 2): 0.85321421,
        (1, 3):    0.7,
        (2, 3): 0.95
    }
    num_to_str_labels = {
        0: "a",
        1: "ay",
        2: "ag",
        3: "b"
    }
    heatmap_img, classes = generate_heatmap(acc_dict, num_to_str_labels)
    cv2.imwrite("example_heatmap.png", heatmap_img)