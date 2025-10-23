import torch
import utils.load_dvs_lips as dvs
from utils.gen_video_from_events import gen_video_from_events, play_in_window

ds = sorted(dvs.get_dataset(train=True), key=lambda x: x[1])
# get random sample
for i in range(len(ds)):
    if ds[i][1] == 87:
        sample = ds[i+31]
        break
gen_video_from_events(sample, "test_vid.mp4", target_time=2.0, fps=60, fade_time=0.2)
play_in_window("test_vid.mp4", scale=4.0)