"""
Semantic Segmentation Tutorial — Pascal VOC 2012 with U-Net
============================================================
Task   : assign a class label to every pixel in an image (21 VOC classes)
Model  : U-Net trained from scratch
Metric : mean Intersection-over-Union (mIoU)

Run:
    python segmentation.py

The best checkpoint is saved to best_unet.pth and predictions are written to
segmentation_results.png after training.
"""

import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

from models.unet import UNet

ssl._create_default_https_context = ssl._create_unverified_context

# ---- Device ---------------------------------------------------------------
device = torch.device("cuda"  if torch.cuda.is_available()
                  else "mps"   if torch.backends.mps.is_available()
                  else "cpu")

# ---- VOC class metadata ---------------------------------------------------
VOC_CLASSES = [
    "background", "aeroplane", "bicycle",    "bird",     "boat",
    "bottle",     "bus",       "car",         "cat",      "chair",
    "cow",        "diningtable","dog",        "horse",    "motorbike",
    "person",     "pottedplant","sheep",      "sofa",     "train",
    "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)  # 21 (index 0 = background)
VOID_IDX    = 255               # VOC marks boundary / ignore pixels with 255

# Official VOC colour palette: class index → (R, G, B)
VOC_COLORMAP = [
    (  0,   0,   0), (128,   0,   0), (  0, 128,   0), (128, 128,   0),
    (  0,   0, 128), (128,   0, 128), (  0, 128, 128), (128, 128, 128),
    ( 64,   0,   0), (192,   0,   0), ( 64, 128,   0), (192, 128,   0),
    ( 64,   0, 128), (192,   0, 128), ( 64, 128, 128), (192, 128, 128),
    (  0,  64,   0), (128,  64,   0), (  0, 192,   0), (128, 192,   0),
    (  0,  64, 128),
]


# ---- Dataset --------------------------------------------------------------
class VOCSegDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around torchvision.VOCSegmentation that applies joint
    image + mask transforms (both must be resized the same way).

    Returns:
        image : (3, H, W) float tensor, ImageNet-normalised
        mask  : (H, W) long tensor — class indices; VOID_IDX = ignore
    """
    def __init__(self, root, image_set="train", size=256, augment=False):
        self.base = VOCSegmentation(root, year="2012", image_set=image_set,
                                    download=True)
        self.size = size
        self.augment = augment
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        self.img_tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        # Resize — use NEAREST for mask so class indices are never interpolated
        img  = TF.resize(img,  [self.size, self.size])
        mask = TF.resize(mask, [self.size, self.size],
                         interpolation=T.InterpolationMode.NEAREST)
        if self.augment:
            if torch.rand(1) > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            img = self.color_jitter(img)
        img  = self.img_tf(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask


# ---- Loss -----------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) for semantic segmentation.

    Suppresses the gradient from easy, correctly-classified pixels by (1-p)^gamma,
    so misclassified foreground pixels dominate training instead of the abundant
    background class.  gamma=2 is the standard default.
    """
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets,
                                ignore_index=self.ignore_index, reduction='none')
        pt    = torch.exp(-ce)                      # probability of correct class
        loss  = (1 - pt) ** self.gamma * ce
        mask  = targets != self.ignore_index
        return loss[mask].mean()


# ---- Metric ---------------------------------------------------------------
def compute_miou(pred_mask, true_mask, num_classes, void_idx=255):
    """
    Mean IoU over all classes that appear in true_mask (skips void pixels).

    IoU_c = |pred==c & true==c| / |pred==c | true==c|
    """
    valid = true_mask != void_idx
    pred  = pred_mask[valid]
    true  = true_mask[valid]
    ious  = []
    for c in range(num_classes):
        inter = ((pred == c) & (true == c)).sum().float()
        union = ((pred == c) | (true == c)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return float(np.mean(ious)) if ious else 0.0


# ---- Train / Val loops ----------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)                        # (B, C, H, W)
        loss   = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        preds = logits.argmax(dim=1)
        for p, m in zip(preds.cpu(), masks.cpu()):
            total_iou += compute_miou(p, m, NUM_CLASSES)
            n += 1
    return total_loss / len(loader), total_iou / max(n, 1)


# ---- Visualisation --------------------------------------------------------
def mask_to_rgb(mask_np):
    """Convert (H, W) class-index array to (H, W, 3) colour image."""
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cls, colour in enumerate(VOC_COLORMAP):
        rgb[mask_np == cls] = colour
    return rgb


@torch.no_grad()
def visualise(model, dataset, n=4):
    model.eval()
    indices  = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).squeeze().cpu()

        img_show = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(img_show);              axes[row, 0].set_title("Image")
        axes[row, 1].imshow(mask_to_rgb(mask.numpy())); axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(mask_to_rgb(pred.numpy())); axes[row, 2].set_title("Prediction")
        for ax in axes[row]:
            ax.axis("off")

    patches = [mpatches.Patch(color=[c/255 for c in VOC_COLORMAP[i]], label=VOC_CLASSES[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig("segmentation_results.png", bbox_inches="tight", dpi=150)
    plt.show()
    print("Saved → segmentation_results.png")


# ---- Main -----------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyper-parameters
    DATA_ROOT    = "./data"
    IMAGE_SIZE   = 256
    BATCH_SIZE   = 8
    EPOCHS       = 20
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    BASE_FILTERS = 32   # 32 trains faster on CPU/MPS; set to 64 for full U-Net

    # Data
    train_ds = VOCSegDataset(DATA_ROOT, image_set="train", size=IMAGE_SIZE, augment=True)
    val_ds   = VOCSegDataset(DATA_ROOT, image_set="val",   size=IMAGE_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {device}")

    # Model — CrossEntropyLoss ignores void (255) pixels automatically
    model     = UNet(in_channels=3, num_classes=NUM_CLASSES,
                     base_filters=BASE_FILTERS).to(device)
    criterion = FocalLoss(gamma=2, ignore_index=VOID_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_iou  = 0.0
    best_path = "best_unet.pth"
    for epoch in range(1, EPOCHS + 1):
        train_loss          = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_miou  = validate(model, val_loader, criterion)
        scheduler.step()

        flag = ""
        if val_miou > best_iou:
            best_iou = val_miou
            torch.save(model.state_dict(), best_path)
            flag = " ← best"
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"mIoU={val_miou:.4f}{flag}")

    # Final visualisation with the best checkpoint
    print(f"\nLoading best model (mIoU={best_iou:.4f}) …")
    model.load_state_dict(torch.load(best_path, map_location=device))
    visualise(model, val_ds, n=4)
