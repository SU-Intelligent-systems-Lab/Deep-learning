import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation

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
    def __init__(self, root, image_set="train", size=384, augment=False):
        self.base = VOCSegmentation(root, year="2012", image_set=image_set,
                                    download=True)
        self.size    = size
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


    Numerical example:
        Confident prediction (p=0.9):
            ce  = -log(0.9) = 0.105
            pt  = exp(-0.105) = 0.9
            weight = (1 - 0.9)^2 = 0.01  → loss nearly zeroed out

        Wrong prediction (p=0.2):
            ce  = -log(0.2) = 1.609
            pt  = exp(-1.609) = 0.2
            weight = (1 - 0.2)^2 = 0.64  → loss stays strong


    """
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets,
                               ignore_index=self.ignore_index,
                               reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        mask = targets != self.ignore_index
        return loss[mask].mean()


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice Loss.

    For every class c:
        Dice_c = 1 - (2·|pred_c ∩ gt_c| + smooth) / (|pred_c| + |gt_c| + smooth)

    Mean is taken over classes that appear in the batch (union > 0).
    """
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits, targets):
        valid  = (targets != self.ignore_index).float()
        probs  = torch.softmax(logits, dim=1)

        dice_sum = 0.0
        count    = 0
        for c in range(self.num_classes):
            gt   = (targets == c).float() * valid
            pred = probs[:, c] * valid

            inter = (pred * gt).sum()
            union = pred.sum() + gt.sum()

            if union > 0:
                dice_sum += 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
                count    += 1

        return dice_sum / max(count, 1)


class CombinedLoss(nn.Module):
    """
    FocalLoss + DiceLoss weighted sum.

        Loss = (1 - dice_weight)·Focal  +  dice_weight·Dice

    """
    def __init__(self, num_classes, gamma=2, ignore_index=255, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        self.dice  = DiceLoss(num_classes, ignore_index=ignore_index)
        self.w     = dice_weight

    def forward(self, logits, targets):
        return (1.0 - self.w) * self.focal(logits, targets) \
             +        self.w  * self.dice(logits, targets)


# ---- LR schedule ----------------------------------------------------------

def poly_lr(optimizer, epoch, max_epochs, base_lrs, power=0.9):
    """
    Polynomial LR decay — the standard schedule for segmentation.

    Keeps LR high for most of training and decays smoothly toward the end,
    preventing the premature LR collapse that CosineAnnealingLR caused around
    epoch 10 in the original code.

    base_lrs is a list aligned with optimizer.param_groups so each group
    (encoder / decoder) decays from its own starting LR.
    """
    factor = (1 - epoch / max_epochs) ** power
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


# ---- Metric ---------------------------------------------------------------
def compute_miou(pred_mask, true_mask, num_classes, void_idx=255):
    """
    Mean IoU over all classes that appear in true_mask (skips void pixels).
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
        logits = model(imgs)            # (B, C, H, W) — same API as custom UNet
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
    indices   = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).squeeze().cpu()

        img_show = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(img_show);                    axes[row, 0].set_title("Image")
        axes[row, 1].imshow(mask_to_rgb(mask.numpy()));   axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(mask_to_rgb(pred.numpy()));   axes[row, 2].set_title("Prediction")
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
    IMAGE_SIZE   = 384          # raised from 256; better spatial detail for small objects
    BATCH_SIZE   = 8
    EPOCHS       = 50
    LR_ENCODER   = 1e-5         # tiny: preserve the ImageNet features
    LR_DECODER   = 1e-3         # full speed: learn to decode segmentation masks
    WEIGHT_DECAY = 1e-4

    # ── Model ─────────────────────────────────────────────────────────────────
    # smp.Unet with a pretrained ResNet-34 encoder.
    #
    # encoder_name    : any torchvision / timm backbone string
    # encoder_weights : "imagenet" downloads and caches pretrained weights
    # classes         : number of output channels (21 for VOC)
    # activation      : None → raw logits, which our loss functions expect
    #
    # The forward pass returns a (B, 21, H, W) tensor — identical shape to the
    # custom UNet — so train_one_epoch / validate need zero changes.
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,
    ).to(device)

    print(f"Model  : smp.Unet  |  Encoder: ResNet-34 (ImageNet pretrained)")
    print(f"Device : {device}")

    # ── Two-group optimiser ───────────────────────────────────────────────────
    # The encoder already knows how to extract rich features; it only needs a
    # tiny nudge.  The decoder and segmentation head start from random weights
    # and need a full learning rate.
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": LR_ENCODER},
            {"params": list(model.decoder.parameters()) +
                       list(model.segmentation_head.parameters()),
             "lr": LR_DECODER},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    base_lrs = [LR_ENCODER, LR_DECODER]    # used by poly_lr

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = VOCSegDataset(DATA_ROOT, image_set="train", size=IMAGE_SIZE, augment=True)
    val_ds   = VOCSegDataset(DATA_ROOT, image_set="val",   size=IMAGE_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f"Train  : {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # dice_weight raised to 0.5: with a strong pretrained encoder, FocalLoss no
    # longer needs to dominate early — equal Dice weight from epoch 1 prevents
    # background collapse more reliably.
    criterion = CombinedLoss(
        num_classes  = NUM_CLASSES,
        gamma        = 2,
        ignore_index = VOID_IDX,
        dice_weight  = 0.5,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_iou  = 0.0
    best_path = "best_unet.pth"

    for epoch in range(1, EPOCHS + 1):
        # PolyLR: decay both param groups from their own base LR
        poly_lr(optimizer, epoch - 1, EPOCHS, base_lrs, power=0.9)
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]

        train_loss         = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_miou = validate(model, val_loader, criterion)

        flag = ""
        if val_miou > best_iou:
            best_iou = val_miou
            torch.save(model.state_dict(), best_path)
            flag = " ← best"

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"lr_enc={current_lrs[0]:.2e}  lr_dec={current_lrs[1]:.2e}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"mIoU={val_miou:.4f}{flag}")

    # ── Final visualisation ───────────────────────────────────────────────────
    print(f"\nLoading best model (mIoU={best_iou:.4f}) …")
    model.load_state_dict(torch.load(best_path, map_location=device))
    visualise(model, val_ds, n=4)