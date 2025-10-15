import utils.visualise_kernel as vk
from models.SimpleConv import SimpleConvModel
import torch
import torch.nn as nn
import numpy as np
import cv2

if __name__ == "__main__":
    model_name = "model_20.pth"
    net = SimpleConvModel(1, 5)
    net.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

    img_1 = vk.generate_from_layer(net.conv1)
    img_2 = vk.generate_from_layer(net.conv2)

    final_img = np.vstack((img_1, img_2))
    cv2.imwrite("imgs/kernels.png", final_img)
    print("Kernels saved to kernels.png")
    cv2.imshow("Kernels", final_img)
    cv2.waitKey(0)