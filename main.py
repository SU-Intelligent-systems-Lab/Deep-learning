import argparse
import random
import ssl
import numpy as np
import torch


from parameters import PARAMS
from models.MLP import MLP
from train import run_training
from test  import run_test


#Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params):
    return MLP(
        input_size   = params["input_size"],
        hidden_sizes = params["hidden_sizes"],
        num_classes  = params["num_classes"],
        dropout      = params["dropout"],
    )


def main():
    parser = argparse.ArgumentParser(description="MLP on MNIST")
    parser.add_argument("--mode",   choices=["train", "test", "both"],
                        default="both", help="Run mode")
    parser.add_argument("--epochs", type=int,   default=None)
    parser.add_argument("--lr",     type=float, default=None)
    parser.add_argument("--device", type=str,   default=None)
    args = parser.parse_args()

    # Allow CLI overrides
    if args.epochs: PARAMS["epochs"]        = args.epochs
    if args.lr:     PARAMS["learning_rate"] = args.lr
    if args.device: PARAMS["device"]        = args.device

    set_seed(PARAMS["seed"])
    print(f"Seed set to: {PARAMS['seed']}")

    device = torch.device(
        PARAMS["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(PARAMS).to(device)
    print(model)

    if args.mode in ("train", "both"):
        run_training(model, PARAMS, device)

    if args.mode in ("test", "both"):
        run_test(model, PARAMS, device)


if __name__ == "__main__":
    main()