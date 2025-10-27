import utils.visualise_kernel as vk
from models.SimpleConv import SimpleConvModel
import torch
import torch.nn as nn
import numpy as np
import cv2

if __name__ == "__main__":
    model_name = "./outputs 5/w_2fc_best.pth"
    net = SimpleConvModel(1, 75)
    net.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

    img_1 = vk.generate_from_layer(net.conv1)
    img_2 = vk.generate_from_layer(net.conv2)
    img_3 = vk.generate_from_layer(net.conv3)
    img_4 = vk.generate_from_layer(net.conv4)
    print(img_2.shape, img_3.shape, img_4.shape)
    border_colour = (0, 255, 0)
    b_thick = 3
    img_2 = cv2.copyMakeBorder(img_2, b_thick, b_thick, b_thick, b_thick, cv2.BORDER_CONSTANT, value=border_colour)
    img_3 = cv2.copyMakeBorder(img_3, b_thick, b_thick, b_thick, b_thick, cv2.BORDER_CONSTANT, value=border_colour)
    img_4 = cv2.copyMakeBorder(img_4, b_thick, b_thick, b_thick, b_thick, cv2.BORDER_CONSTANT, value=border_colour)


    final_img = np.vstack((img_2, img_3, img_4))
    cv2.imwrite("imgs/kernels.png", final_img)
    print("Kernels saved to kernels.png")
    cv2.imwrite("imgs/kernal_7x7.png", img_1)
    cv2.imshow("Kernels", final_img)
    cv2.waitKey(0)