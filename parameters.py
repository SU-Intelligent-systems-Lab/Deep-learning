import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",   choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model",     choices=["mlp", "cnn", "vgg", "resnet"], default="mlp")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    # VGG-specific
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    # ResNet-specific: map a simple int to a block config
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)")

    args = parser.parse_args()

    # Dataset-dependent settings
    if args.dataset == "mnist":
        input_size = 784          # 1 × 28 × 28
        mean, std  = (0.1307,), (0.3081,)
    else:                         # cifar10
        input_size = 3072         # 3 × 32 × 32
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)

    return {
        # Data
        "dataset":      args.dataset,
        "data_dir":     "./data",
        "num_workers":  2,
        "mean":         mean,
        "std":          std,

        # Model
        "model":        args.model,
        "input_size":   input_size,
        "hidden_sizes": [512, 256, 128],
        "num_classes":  10,
        "dropout":      0.3,
        "vgg_depth":    args.vgg_depth,
        "resnet_layers": args.resnet_layers,

        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  1e-4,

        # Misc
        "seed":         42,
        "device":       args.device,
        "save_path":    "best_model.pth",
        "log_interval": 100,

        # CLI
        "mode":         args.mode,
    }