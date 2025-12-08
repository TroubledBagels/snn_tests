import torchvision
import pathlib

home = pathlib.Path.home()
data_dir = home / 'data' / 'cifar10'

cf10_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)

print(cf10_ds[0][0])