PARAMS = {
    # Data
    "data_dir": "./data",
    "num_workers": 2,

    # Model
    "input_size": 784,       # 28x28
    "hidden_sizes": [512, 256, 128],
    "num_classes": 10,
    "dropout": 0.3,

    # Training
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,

    # Misc
    "seed": 42,
    "device": "cuda",        # "cuda" or "cpu"
    "save_path": "best_model.pth",
    "log_interval": 100,     # print every N batches
}