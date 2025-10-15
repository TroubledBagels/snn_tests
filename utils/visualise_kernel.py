import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.avg_pool2d(x, 2)
        return x

def generate_from_layer(conv_layer):
    C, D, H, W = conv_layer.weight.shape
    img_list = []
    for i in range(C):
        kernel = conv_layer.weight[i, 0, :, :].detach().cpu().numpy()
        kernel_img = generate_img(kernel)
        kernel_img = cv2.copyMakeBorder(kernel_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        kernel_img = cv2.resize(kernel_img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        img_list.append(kernel_img)

    grid_shape = get_grid_shape(C)
    grid_img = []
    for i in range(grid_shape[0]):
        row_imgs = img_list[i*grid_shape[1]:(i+1)*grid_shape[1]]
        row_img = cv2.hconcat(row_imgs)
        grid_img.append(row_img)
    grid_img = cv2.vconcat(grid_img)
    return grid_img

def generate_img(k):
    # Generate a heat map of the kernel, lowest value is 0, highest value is 255
    k = (k - k.min()) / (k.max() - k.min()) * 255
    k = k.astype('uint8')
    k = cv2.resize(k, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    k_img = cv2.applyColorMap(k, cv2.COLORMAP_HOT)
    return k_img

def get_grid_shape(num_items):
    best_shape = (num_items, 1)
    best_diff = num_items - 1
    for i in range(1, int(num_items**0.5) + 1):
        if num_items % i == 0:
            j = num_items // i
            diff = abs(i - j)
            if diff < best_diff:
                best_diff = diff
                best_shape = (i, j)
    return best_shape

if __name__ == "__main__":
    net = TestNet()

    grid_img = generate_from_layer(net.conv1)
    cv2.imwrite("../imgs/kernel.png", grid_img)
    cv2.imshow("Kernels", grid_img)
    cv2.waitKey(0)