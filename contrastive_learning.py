"""
Contrastive Learning (SimCLR) on CIFAR-10
------------------------------------------
Key ideas:
  - Self-supervised: no labels during representation learning
  - Two random augmentations of the same image = a positive pair
  - NT-Xent loss pulls positive pairs together, pushes all other pairs apart
  - After training, freeze the encoder and fit a linear classifier to check
    how useful the learned representations are (linear evaluation protocol)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset as hf_load_dataset



# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS_CONTRASTIVE = 10
EPOCHS_LINEAR      = 10
BATCH_SIZE         = 256
TEMPERATURE        = 0.5    # sharpens/softens the similarity distribution
PROJ_DIM           = 64     # projection head output dimension
LR                 = 3e-4
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ── Augmentation pipeline ─────────────────────────────────────────────────────
# This is the heart of contrastive learning: two independently augmented views
# of the same image must map to nearby points in embedding space.
augment = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# For linear evaluation we just normalize, no aggressive augmentation
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


# ── Dataset wrapper ───────────────────────────────────────────────────────────
class HFCIFARDataset(Dataset):
    """Wraps a HuggingFace CIFAR-10 split to behave like torchvision CIFAR10."""
    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img, label = item["img"], item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


class ContrastiveDataset(Dataset):
    """Returns two independent augmented views of each image."""
    def __init__(self, base_dataset, transform):
        self.ds        = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return self.transform(img), self.transform(img), label


_hf_cifar = hf_load_dataset("cifar10")
raw_train = HFCIFARDataset(_hf_cifar["train"])
raw_test  = HFCIFARDataset(_hf_cifar["test"])

contrastive_loader = DataLoader(
    ContrastiveDataset(raw_train, augment),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
)
linear_train_loader = DataLoader(
    HFCIFARDataset(_hf_cifar["train"], transform=eval_transform),
    batch_size=256, shuffle=True
)
linear_test_loader = DataLoader(
    HFCIFARDataset(_hf_cifar["test"], transform=eval_transform),
    batch_size=512, shuffle=False
)


# ── Encoder ───────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """Small CNN → 128-dim embedding (the part we keep after training)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 16x16
            # block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                         # 8x8
            # block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                 # 1x1
        )

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)   # (B, 128)


# ── Projection head ───────────────────────────────────────────────────────────
# Used only during contrastive training; discarded for downstream tasks.
# The extra non-linearity helps the encoder not overfit to the loss geometry.
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, PROJ_DIM)
        )

    def forward(self, h):
        return self.net(h)


# ── NT-Xent loss ──────────────────────────────────────────────────────────────
def nt_xent_loss(z1, z2, temperature):
    """
    Normalized Temperature-scaled Cross Entropy loss.

    For a batch of N pairs:
      - 2N embeddings total (z1 and z2 concatenated)
      - Each z_i has exactly one positive: its augmentation partner
      - All other 2N-2 embeddings are negatives
      - We minimize -log( exp(sim(zi,zj)/T) / sum_negatives )
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)               # (2N, proj_dim)
    z = F.normalize(z, dim=1)                    # cosine similarity via dot product

    # Similarity matrix (2N x 2N)
    sim = torch.mm(z, z.T) / temperature

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive indices: (i, i+N) and (i+N, i)
    pos_indices = torch.cat([
        torch.arange(N, 2 * N),   # for z1_i, positive is z2_i (at i+N)
        torch.arange(0, N)        # for z2_i, positive is z1_i (at i)
    ]).to(z.device)

    loss = F.cross_entropy(sim, pos_indices)
    return loss


# ── Contrastive training ──────────────────────────────────────────────────────
encoder    = Encoder().to(DEVICE)
proj_head  = ProjectionHead().to(DEVICE)
optimizer  = torch.optim.Adam(
    list(encoder.parameters()) + list(proj_head.parameters()), lr=LR
)

print("=== Contrastive pre-training ===")
for epoch in range(1, EPOCHS_CONTRASTIVE + 1):
    encoder.train(); proj_head.train()
    total_loss = 0
    for view1, view2, _ in contrastive_loader:   # labels unused during pretraining
        view1, view2 = view1.to(DEVICE), view2.to(DEVICE)

        z1 = proj_head(encoder(view1))
        z2 = proj_head(encoder(view2))
        loss = nt_xent_loss(z1, z2, TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d}/{EPOCHS_CONTRASTIVE}  loss: {total_loss / len(contrastive_loader):.4f}")


# ── Linear evaluation ─────────────────────────────────────────────────────────
# Freeze the encoder and train only a linear layer.
# Accuracy here tells us how much semantic structure the encoder captured.
print("\n=== Linear evaluation ===")
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

classifier = nn.Linear(128, 10).to(DEVICE)
lin_opt    = torch.optim.Adam(classifier.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS_LINEAR + 1):
    classifier.train()
    for x, y in linear_train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            h = encoder(x)
        loss = F.cross_entropy(classifier(h), y)
        lin_opt.zero_grad()
        loss.backward()
        lin_opt.step()

    # Evaluate
    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in linear_test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = classifier(encoder(x)).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    print(f"Epoch {epoch:02d}/{EPOCHS_LINEAR}  test acc: {correct / total:.3f}")
