import torch
import tonic
import pathlib

home = pathlib.Path.home()
data_path = home / "data" / "nmnist"

ds = tonic.datasets.NMNIST(train=True, save_to=data_path)
to_frame = tonic.transforms.ToFrame(
    n_time_bins=300,
    sensor_size=tonic.datasets.NMNIST.sensor_size
)
data, label = ds[0]
frame = to_frame(data)
print(f"Data shape after ToFrame transform: {frame.shape}")
