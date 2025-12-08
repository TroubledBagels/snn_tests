import torch
import torchvision
import pathlib
import random
import utils.graphing_bsquare
import models.ConventionalBSquare as CBS

if __name__ == "__main__":
    home = pathlib.Path.home()
    data_dir = home / 'data' / 'cifar10'
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True).data

    idx = random.randint(0, len(ds) - 1)
    sample = torch.tensor(ds[idx]).permute(2, 0, 1).float() / 255.0

    print(f"Sample shape: {sample.shape}")

    model = CBS.BSquareModel(
        num_classes=10,
        input_size=3,
        hidden_size=64,
        num_layers=4,
        binary_voting=False,
        bclass=CBS.TinyCNN,
        net_out=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load("./bsquares/cifar10_bal.pth", map_location=device))

    _, vote_dict = model(sample.unsqueeze(0).to(device))
    G = utils.graphing_bsquare.create_dependency_graph(vote_dict)
    utils.graphing_bsquare.visualise_graph(G, title="CIFAR-10 B-Square Dependency Graph")
    # utils.graphing_bsquare.save_graph(G, "./graphs/cifar10_bsquare_dependency_graph.png")
    print("Graph saved to ./graphs/cifar10_bsquare_dependency_graph.png")
