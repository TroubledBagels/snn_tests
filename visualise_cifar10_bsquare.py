import torch
import torchvision
import pathlib
import random
import utils.graphing_bsquare
import models.ConventionalBSquare as CBS

if __name__ == "__main__":
    home = pathlib.Path.home()
    data_dir = home / 'data' / 'cifar10'
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor())

    torch.manual_seed(47)
    random.seed(47)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    sample, label = next(iter(dl))
    sample = sample.squeeze(0)

    print(f"Sample shape: {sample.shape}")

    model = CBS.BSquareModel(
        num_classes=10,
        input_size=3,
        hidden_size=64,
        num_layers=4,
        binary_voting=False,
        bclass=CBS.SmallCNN,
        net_out=False,
        threshold=0.25
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load("./bsquares/cifar10_bal_4conv_1fc_ac_full.pth", map_location=device))

    out, vote_dict = model(sample.unsqueeze(0).to(device))
    G = utils.graphing_bsquare.create_dependency_graph(vote_dict)
    utils.graphing_bsquare.visualise_graph(G, title="CIFAR-10 B-Square Dependency Graph")
    # utils.graphing_bsquare.save_graph(G, "./graphs/cifar10_bsquare_dependency_graph.png")
    print("Graph saved to ./graphs/cifar10_bsquare_dependency_graph.png")
    print(f"Prediction: {out.argmax(dim=1).item()}, Label: {label.item()}")
