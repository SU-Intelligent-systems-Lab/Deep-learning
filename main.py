import random
import ssl
import numpy as np
import torch

from parameters import get_params
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from train import run_training
from test  import run_test


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params):
    model_name = params["model"]
    dataset    = params["dataset"]
    nc         = params["num_classes"]

    if model_name == "mlp":
        return MLP(
            input_size   = params["input_size"],
            hidden_sizes = params["hidden_sizes"],
            num_classes  = nc,
            dropout      = params["dropout"],
        )

    if model_name == "cnn":
        # MNIST_CNN expects 1-channel 28×28; SimpleCNN expects 3-channel 32×32
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        else:
            return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10 with vgg.")
        return VGG(dept=params["vgg_depth"], num_class=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10 with resnet.")
        return ResNet(BasicBlock, params["resnet_layers"], num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")


def main():
    params = get_params()

    set_seed(params["seed"])
    print(f"Seed set to: {params['seed']}")
    print(f"Dataset: {params['dataset']}  |  Model: {params['model']}")

    device = torch.device(
        params["device"] if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    print(model)

    if params["mode"] in ("train", "both"):
        run_training(model, params, device)

    if params["mode"] in ("test", "both"):
        run_test(model, params, device)


if __name__ == "__main__":
    main()