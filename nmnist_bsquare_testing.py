import torch
import torch.nn as nn
import tonic
from torch.utils.data import DataLoader, random_split
from models.BSquare import BSquareModel, ListDataset, BSquareModelCombined
import utils.general as g
import pathlib
import tqdm
import random
import cv2
import utils.heatmap

def collate_fn(batch):
    pass

if __name__ == "__main__":
    home_dir = pathlib.Path.home()
    data_dir = home_dir / "data" / "nmnist"
    tr_ds = tonic.datasets.NMNIST(save_to=data_dir, train=True)
    te_ds = tonic.datasets.NMNIST(save_to=data_dir, train=False)
    print(f"Train size: {len(tr_ds)}, Test size: {len(te_ds)}")

    model = BSquareModel(num_classes=10, input_size=34*34*2, hidden_size=64, num_layers=2, binary_voting=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    # model.train_classifiers(tr_ds, te_ds, epochs=3, lr=1e-3)
    # torch.save(model.state_dict(), "./bsquares/nmnist_bsquare.pth")

    model.load_state_dict(torch.load("./bsquares/nmnist_bsquare.pth", map_location=device))

    comb_model = BSquareModelCombined(num_classes=10, input_size=34*34*2, hidden_size=64, num_layers=2, binary_voting=False, net_output=True)
    comb_model.load_ensemble(model)
    print(comb_model)
    comb_model.to(device)

    comb_model.train_ann_out(tr_ds, te_ds, epochs=5, lr=1e-3)

    to_frame = tonic.transforms.ToFrame(
        n_time_bins=300,
        sensor_size=tonic.datasets.NMNIST.sensor_size
    )

    comb_model.eval()
    test_ds = ListDataset(te_ds, transform=to_frame)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(test_dl)
        for data, target in pbar:
            data = data.float().to(device)
            target = target.long().to(device)
            votes = comb_model(data, 0)
            preds = votes.argmax(dim=1)
            correct += (preds == target.to(device)).sum().item()
            total += target.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
