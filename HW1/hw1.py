#!/usr/bin/env python3
"""
HW1 experiment runner for MNIST MLP ablations.

Designed to live inside a folder named HW1 in the existing course repo and to
reuse the repo's general conventions while fixing experimental methodology:
- creates a true validation split from the training set
- runs baseline + ablations from a single entry point
- saves plots for baseline and variant comparisons
- saves best checkpoints and a run summary CSV

Example usage from repo root:
    python HW1/hw1_runner.py --run_all
    python HW1/hw1_runner.py --experiments baseline activation dropout
    python HW1/hw1_runner.py --final_on_test
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

# Allow running from inside the repo after placing this file in HW1/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class DataConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    train_size: int = 55_000
    val_size: int = 5_000
    mean: Tuple[float, ...] = (0.1307,)
    std: Tuple[float, ...] = (0.3081,)


@dataclass
class ModelConfig:
    input_dim: int = 784
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    num_classes: int = 10
    activation: str = "relu"  # relu | gelu
    dropout: float = 0.2
    use_batchnorm: bool = True


@dataclass
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    optimizer_name: str = "adam"  # adam | sgd
    momentum: float = 0.9
    weight_decay: float = 0.0  # L2
    l1_lambda: float = 0.0
    scheduler_name: str = "none"  # none | step | plateau
    step_size: int = 5
    gamma: float = 0.5
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    early_stopping: bool = True
    early_stopping_patience: int = 5
    min_delta: float = 0.0
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 100


@dataclass
class RunConfig:
    run_name: str
    experiment_group: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# -----------------------------
# Model
# -----------------------------
def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ConfigurableMLP(nn.Module):
    """MLP with configurable depth/width/activation/dropout/batchnorm.

    Block order follows the homework requirement discussion:
        Linear -> BatchNorm1d(optional) -> Activation -> Dropout(optional)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Flatten()]
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Data
# -----------------------------
def get_transforms(mean: Tuple[float, ...], std: Tuple[float, ...]) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_dataloaders(cfg: DataConfig, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = get_transforms(cfg.mean, cfg.std)
    full_train = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)

    if cfg.train_size + cfg.val_size != len(full_train):
        raise ValueError(
            f"train_size + val_size must equal {len(full_train)} for MNIST train split"
        )

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [cfg.train_size, cfg.val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


# -----------------------------
# Training helpers
# -----------------------------
def get_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float


@dataclass
class RunResult:
    run_name: str
    experiment_group: str
    hidden_dims: List[int]
    num_hidden_layers: int
    activation: str
    dropout: float
    batchnorm: bool
    optimizer: str
    learning_rate: float
    scheduler: str
    weight_decay: float
    l1_lambda: float
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    final_test_loss: Optional[float] = None
    final_test_acc: Optional[float] = None


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> Optimizer:
    if cfg.optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer_name}")


def build_scheduler(optimizer: Optimizer, cfg: TrainConfig):
    if cfg.scheduler_name == "none":
        return None
    if cfg.scheduler_name == "step":
        return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler_name}")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_l1_penalty(model: nn.Module) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        penalty = penalty + param.abs().sum()
    return penalty


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    l1_lambda: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if l1_lambda > 0:
            loss = loss + l1_lambda * compute_l1_penalty(model)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((10, 10), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    return total_loss / total_count, total_correct / total_count, confusion


# -----------------------------
# Filesystem helpers
# -----------------------------
def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    out = {
        "root": base_dir,
        "checkpoints": base_dir / "checkpoints",
        "logs": base_dir / "logs",
        "figures": base_dir / "figures",
        "reports": base_dir / "reports",
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def save_history_csv(history: List[EpochMetrics], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
        )
        writer.writeheader()
        for item in history:
            writer.writerow(asdict(item))


def save_json(data: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Plotting
# -----------------------------
def plot_single_history(history: List[EpochMetrics], title: str, out_path: Path) -> None:
    epochs = [x.epoch for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [x.train_loss for x in history], label="train_loss")
    plt.plot(epochs, [x.val_loss for x in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_loss.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [x.train_acc for x in history], label="train_acc")
    plt.plot(epochs, [x.val_acc for x in history], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_acc.png"))
    plt.close()


def plot_group_overlay(
    histories: Dict[str, List[EpochMetrics]],
    metric: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    for name, history in histories.items():
        epochs = [x.epoch for x in history]
        values = [getattr(x, metric) for x in history]
        plt.plot(epochs, values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(confusion: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(confusion, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    ticks = np.arange(10)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Core run logic
# -----------------------------
def execute_run(run_cfg: RunConfig, output_root: Path) -> Tuple[RunResult, List[EpochMetrics], Path]:
    set_seed(run_cfg.train.seed)
    dirs = ensure_dirs(output_root)
    device = get_device(run_cfg.train.device)

    train_loader, val_loader, test_loader = build_dataloaders(run_cfg.data, run_cfg.train.seed)

    model = ConfigurableMLP(
        input_dim=run_cfg.model.input_dim,
        hidden_dims=run_cfg.model.hidden_dims,
        num_classes=run_cfg.model.num_classes,
        activation=run_cfg.model.activation,
        dropout=run_cfg.model.dropout,
        use_batchnorm=run_cfg.model.use_batchnorm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, run_cfg.train)
    scheduler = build_scheduler(optimizer, run_cfg.train)

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: List[EpochMetrics] = []

    checkpoint_path = dirs["checkpoints"] / f"{run_cfg.run_name}.pt"
    config_path = dirs["logs"] / f"{run_cfg.run_name}_config.json"
    history_path = dirs["logs"] / f"{run_cfg.run_name}_history.csv"
    save_json(
        {
            "run_name": run_cfg.run_name,
            "experiment_group": run_cfg.experiment_group,
            "data": asdict(run_cfg.data),
            "model": asdict(run_cfg.model),
            "train": asdict(run_cfg.train),
        },
        config_path,
    )

    for epoch in range(1, run_cfg.train.epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, run_cfg.train.l1_lambda
        )
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr,
            )
        )

        improved = val_loss < (best_val_loss - run_cfg.train.min_delta)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(
                {
                    "model_state": best_state,
                    "run_config": {
                        "run_name": run_cfg.run_name,
                        "experiment_group": run_cfg.experiment_group,
                        "data": asdict(run_cfg.data),
                        "model": asdict(run_cfg.model),
                        "train": asdict(run_cfg.train),
                    },
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        print(
            f"[{run_cfg.run_name}] epoch {epoch:02d}/{run_cfg.train.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        if run_cfg.train.early_stopping and patience_counter >= run_cfg.train.early_stopping_patience:
            print(f"[{run_cfg.run_name}] Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    save_history_csv(history, history_path)
    plot_single_history(
        history,
        title=run_cfg.run_name,
        out_path=dirs["figures"] / f"{run_cfg.run_name}",
    )

    result = RunResult(
        run_name=run_cfg.run_name,
        experiment_group=run_cfg.experiment_group,
        hidden_dims=list(run_cfg.model.hidden_dims),
        num_hidden_layers=len(run_cfg.model.hidden_dims),
        activation=run_cfg.model.activation,
        dropout=run_cfg.model.dropout,
        batchnorm=run_cfg.model.use_batchnorm,
        optimizer=run_cfg.train.optimizer_name,
        learning_rate=run_cfg.train.learning_rate,
        scheduler=run_cfg.train.scheduler_name,
        weight_decay=run_cfg.train.weight_decay,
        l1_lambda=run_cfg.train.l1_lambda,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
    )
    return result, history, checkpoint_path


@torch.no_grad()
def evaluate_checkpoint_on_test(checkpoint_path: Path, output_root: Path) -> Tuple[float, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    data_cfg = DataConfig(**ckpt["run_config"]["data"])
    model_cfg = ModelConfig(**ckpt["run_config"]["model"])
    train_cfg = TrainConfig(**ckpt["run_config"]["train"])
    device = get_device(train_cfg.device)

    _, _, test_loader = build_dataloaders(data_cfg, train_cfg.seed)
    model = ConfigurableMLP(
        input_dim=model_cfg.input_dim,
        hidden_dims=model_cfg.hidden_dims,
        num_classes=model_cfg.num_classes,
        activation=model_cfg.activation,
        dropout=model_cfg.dropout,
        use_batchnorm=model_cfg.use_batchnorm,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, confusion = evaluate(model, test_loader, criterion, device)
    dirs = ensure_dirs(output_root)
    plot_confusion_matrix(
        confusion,
        title=f"Confusion Matrix - {checkpoint_path.stem}",
        out_path=dirs["figures"] / f"{checkpoint_path.stem}_confusion.png",
    )
    return test_loss, test_acc


def save_summary_csv(results: List[RunResult], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def overlay_group_figures(
    group_name: str,
    histories_by_run: Dict[str, List[EpochMetrics]],
    figures_dir: Path,
) -> None:
    plot_group_overlay(
        histories_by_run,
        metric="val_loss",
        title=f"{group_name} comparison - validation loss",
        out_path=figures_dir / f"{group_name}_val_loss_overlay.png",
    )
    plot_group_overlay(
        histories_by_run,
        metric="val_acc",
        title=f"{group_name} comparison - validation accuracy",
        out_path=figures_dir / f"{group_name}_val_acc_overlay.png",
    )


# -----------------------------
# Experiment definitions
# -----------------------------
def make_baseline() -> RunConfig:
    return RunConfig(run_name="baseline", experiment_group="baseline")


def clone_with(base: RunConfig, run_name: str, experiment_group: str, **updates) -> RunConfig:
    cfg = copy.deepcopy(base)
    cfg.run_name = run_name
    cfg.experiment_group = experiment_group
    for key, value in updates.items():
        if hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
        elif hasattr(cfg.train, key):
            setattr(cfg.train, key, value)
        elif hasattr(cfg.data, key):
            setattr(cfg.data, key, value)
        else:
            raise ValueError(f"Unknown config field: {key}")
    return cfg


def build_experiment_plan(selected: Sequence[str]) -> Dict[str, List[RunConfig]]:
    base = make_baseline()
    plan: Dict[str, List[RunConfig]] = {}

    if "baseline" in selected:
        plan["baseline"] = [base]

    if "architecture" in selected:
        plan["architecture"] = [
            clone_with(base, "arch_1x128", "architecture", hidden_dims=[128]),
            clone_with(base, "arch_2x256_128", "architecture", hidden_dims=[256, 128]),
            clone_with(base, "arch_3x512_256_128", "architecture", hidden_dims=[512, 256, 128]),
            clone_with(base, "arch_wide_512_256", "architecture", hidden_dims=[512, 256]),
        ]

    if "activation" in selected:
        plan["activation"] = [
            clone_with(base, "act_relu", "activation", activation="relu"),
            clone_with(base, "act_gelu", "activation", activation="gelu"),
        ]

    if "dropout" in selected:
        plan["dropout"] = [
            clone_with(base, "dropout_0p0", "dropout", dropout=0.0),
            clone_with(base, "dropout_0p2", "dropout", dropout=0.2),
            clone_with(base, "dropout_0p5", "dropout", dropout=0.5),
        ]

    if "batchnorm" in selected:
        plan["batchnorm"] = [
            clone_with(base, "bn_off", "batchnorm", use_batchnorm=False),
            clone_with(base, "bn_on", "batchnorm", use_batchnorm=True),
        ]

    if "regularization" in selected:
        plan["regularization"] = [
            clone_with(base, "l2_0", "regularization", weight_decay=0.0, l1_lambda=0.0),
            clone_with(base, "l2_1e5", "regularization", weight_decay=1e-5, l1_lambda=0.0),
            clone_with(base, "l2_1e4", "regularization", weight_decay=1e-4, l1_lambda=0.0),
            clone_with(base, "l2_1e3", "regularization", weight_decay=1e-3, l1_lambda=0.0),
            clone_with(base, "l1_1e6", "regularization", weight_decay=0.0, l1_lambda=1e-6),
            clone_with(base, "l1_1e5", "regularization", weight_decay=0.0, l1_lambda=1e-5),
        ]

    if "training" in selected:
        plan["training"] = [
            clone_with(base, "lr_1e2", "training", learning_rate=1e-2, scheduler_name="none"),
            clone_with(base, "lr_1e3", "training", learning_rate=1e-3, scheduler_name="none"),
            clone_with(base, "lr_3e4", "training", learning_rate=3e-4, scheduler_name="none"),
            clone_with(base, "sched_step", "training", scheduler_name="step"),
            clone_with(base, "sched_plateau", "training", scheduler_name="plateau"),
            clone_with(base, "no_early_stop", "training", early_stopping=False, epochs=20),
        ]

    return plan


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW1 MNIST MLP experiment runner")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "architecture", "activation", "dropout", "batchnorm", "regularization", "training"],
        help="Experiment groups to run",
    )
    parser.add_argument("--run_all", action="store_true", help="Run all experiment groups")
    parser.add_argument("--final_on_test", action="store_true", help="Evaluate best validation run on test set")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = (
        ["baseline", "architecture", "activation", "dropout", "batchnorm", "regularization", "training"]
        if args.run_all
        else args.experiments
    )

    output_root = Path(args.output_dir)
    dirs = ensure_dirs(output_root)
    plan = build_experiment_plan(selected)

    # propagate top-level CLI overrides
    for runs in plan.values():
        for run_cfg in runs:
            run_cfg.train.device = args.device
            run_cfg.data.batch_size = args.batch_size
            run_cfg.train.epochs = args.epochs
            run_cfg.train.seed = args.seed

    all_results: List[RunResult] = []
    all_histories: Dict[str, Dict[str, List[EpochMetrics]]] = {}
    best_result: Optional[RunResult] = None
    best_checkpoint: Optional[Path] = None

    for group_name, runs in plan.items():
        histories_by_run: Dict[str, List[EpochMetrics]] = {}
        for run_cfg in runs:
            result, history, checkpoint_path = execute_run(run_cfg, output_root)
            all_results.append(result)
            histories_by_run[run_cfg.run_name] = history

            if best_result is None or result.best_val_loss < best_result.best_val_loss:
                best_result = result
                best_checkpoint = checkpoint_path

        all_histories[group_name] = histories_by_run
        if len(histories_by_run) > 1:
            overlay_group_figures(group_name, histories_by_run, dirs["figures"])

    if all_results:
        save_summary_csv(all_results, dirs["reports"] / "experiment_summary.csv")
        print(f"Saved summary to {dirs['reports'] / 'experiment_summary.csv'}")

    if best_result is not None:
        print(
            "Best validation run: "
            f"{best_result.run_name} | val_loss={best_result.best_val_loss:.4f} | "
            f"val_acc={best_result.best_val_acc:.4f} | epoch={best_result.best_epoch}"
        )

    if args.final_on_test and best_checkpoint is not None and best_result is not None:
        test_loss, test_acc = evaluate_checkpoint_on_test(best_checkpoint, output_root)
        best_result.final_test_loss = test_loss
        best_result.final_test_acc = test_acc
        save_summary_csv(all_results, dirs["reports"] / "experiment_summary.csv")
        print(
            f"Final test evaluation for best validation run ({best_result.run_name}): "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
