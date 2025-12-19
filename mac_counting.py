from models.AlexNet import AlexNet
from models.MobileNetV2 import MobileNetV2
from models.MobileNetV3 import MobileNetV3
from models.ResNet18 import (ResNet18)
from models.ConventionalBSquare import BSquareModel, SmallCNN
from models.VGG19 import VGG11, VGG19
from models.EfficientNetB0 import EfficientNetB0
from models.GoogLeNet import GoogLeNet
from models.LeNet import LeNet
from models.WideResNet import WideResNet

import pandas as pd

import torch
from thop import profile

models = [
    AlexNet(num_classes=10),
    MobileNetV2(num_classes=10),
    MobileNetV3(num_classes=10),
    ResNet18(num_classes=10),
    VGG11(num_classes=10),
    VGG19(num_classes=10),
    EfficientNetB0(num_classes=10),
    GoogLeNet(num_classes=10),
    LeNet(num_classes=10),
    WideResNet(num_classes=10),
    BSquareModel(
        num_classes=10,
        input_size=3,
        bclass=SmallCNN,
    )
]

stat_list = []

for model in models:
    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(input, ))
    print(f"{model.__class__.__name__}: MACs: {macs/1e6:.2f}M, Params: {params/1e6:.2f}M")
    stat_list.append((model.__class__.__name__, macs, params))

pd.DataFrame(stat_list, columns=["Model", "MACs", "Params"]).to_csv("model_mac_param_counts.csv", index=False)