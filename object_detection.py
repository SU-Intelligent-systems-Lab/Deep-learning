"""
Object Detection Tutorial — Pascal VOC 2012 with Faster R-CNN
=============================================================
Task   : predict bounding boxes + class labels for objects in an image
Model  : Faster R-CNN (ResNet-50 + FPN backbone) pre-trained on COCO,
         then fine-tuned on VOC's 20 object classes
Metric : precision @ IoU = 0.5

The key insight for fine-tuning: only the detection head (roi_heads.box_predictor)
is replaced.  The backbone stays pretrained on ImageNet/COCO — we only update it
with a much smaller learning rate to avoid destroying useful features.

Run:
    python object_detection.py

Best checkpoint → best_detector.pth
Prediction visualisation → detection_results.png
"""

import os
import ssl
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

# ---- Device ---------------------------------------------------------------
device = torch.device("cuda"  if torch.cuda.is_available()
                  else "mps"   if torch.backends.mps.is_available()
                  else "cpu")

# ---- VOC class list (index 0 = background, required by Faster R-CNN) -----
VOC_CLASSES = [
    "__background__", "aeroplane", "bicycle",  "bird",      "boat",
    "bottle",         "bus",       "car",       "cat",       "chair",
    "cow",            "diningtable","dog",      "horse",     "motorbike",
    "person",         "pottedplant","sheep",    "sofa",      "train",
    "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)  # 21


# ---- Dataset --------------------------------------------------------------
class VOCDetDataset(torch.utils.data.Dataset):
    """
    Wraps torchvision.VOCDetection and converts the XML annotation dict
    into the tensor format Faster R-CNN expects.

    Each item returns:
        image  : (3, H, W) float tensor in [0, 1]
        target : dict with
                   "boxes"    — (N, 4) float32 [x1, y1, x2, y2]
                   "labels"   — (N,)   int64   class indices (≥1; 0 = background)
                   "image_id" — (1,)   int64
    """
    def __init__(self, root, image_set="train"):
        self.base       = VOCDetection(root, year="2012", image_set=image_set,
                                       download=True)
        self.to_tensor  = T.ToTensor()
        self.cls_to_idx = {name: i for i, name in enumerate(VOC_CLASSES)}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, anno = self.base[idx]
        img = self.to_tensor(img)

        boxes, labels = [], []
        for obj in anno["annotation"]["object"]:
            cls = obj["name"]
            if cls not in self.cls_to_idx:
                continue
            bb = obj["bndbox"]
            boxes.append([float(bb["xmin"]), float(bb["ymin"]),
                           float(bb["xmax"]), float(bb["ymax"])])
            labels.append(self.cls_to_idx[cls])

        target = {
            "boxes":    torch.tensor(boxes,  dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx],  dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    """Faster R-CNN expects a list of (image, target) — not a stacked tensor."""
    return tuple(zip(*batch))


# ---- Model ----------------------------------------------------------------
def build_model(num_classes, backbone="resnet50"):
    """
    Backbones:
      "resnet50"   — Faster R-CNN ResNet-50+FPN (accurate, slower on MPS/CPU)
      "mobilenet"  — Faster R-CNN MobileNet-v3  (faster, ~4x less compute)

    The backbone is fully frozen (no gradients) so only the detection head
    trains. This is the biggest single speed-up on MPS/CPU.

    Image scale is reduced from the default 800/1333 → 600/1000 to shrink
    feature maps and speed up RPN + RoI pooling.
    """
    kwargs = dict(weights="DEFAULT", min_size=600, max_size=1000)
    if backbone == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_fpn(**kwargs)
    else:
        model = fasterrcnn_resnet50_fpn(**kwargs)

    for p in model.backbone.parameters():
        p.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(model)
    return model


# ---- Metric ---------------------------------------------------------------
def _box_iou(a, b):
    """IoU between two [x1, y1, x2, y2] tensors."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area  = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (area + 1e-6)


@torch.no_grad()
def evaluate(model, loader, iou_thresh=0.5, score_thresh=0.5):
    """
    Micro-averaged precision @ IoU = iou_thresh.

    A predicted box is a true positive when:
      - its confidence ≥ score_thresh
      - it overlaps a ground-truth box of the same class with IoU ≥ iou_thresh
    """
    model.eval()
    tp, total = 0, 0
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for imgs, targets in pbar:
        imgs  = [img.to(device) for img in imgs]
        preds = model(imgs)
        for pred, target in zip(preds, targets):
            gt_boxes  = target["boxes"]
            gt_labels = target["labels"]
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if score < score_thresh:
                    continue
                total += 1
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    if label == gt_label and _box_iou(box.cpu().tolist(),
                                                       gt_box.tolist()) >= iou_thresh:
                        tp += 1
                        break
        pbar.set_postfix({"precision": f"{tp / max(total, 1):.3f}"})
    return tp / max(total, 1)


# ---- Train loop -----------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, epoch, total_epochs):
    """
    Faster R-CNN returns a loss dict directly when called with targets,
    so we just sum the components rather than writing a manual loss formula.

    Loss components:
      loss_classifier   — RoI class cross-entropy
      loss_box_reg      — RoI box regression (smooth-L1)
      loss_objectness   — RPN foreground/background
      loss_rpn_box_reg  — RPN box regression
    """
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs} [train]", leave=True)
    for imgs, targets in pbar:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss_dict = model(imgs, targets)
            loss      = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()

        pbar.set_postfix({
            "loss":    f"{loss.item():.3f}",
            "cls":     f"{loss_dict['loss_classifier'].item():.3f}",
            "box":     f"{loss_dict['loss_box_reg'].item():.3f}",
            "rpn_cls": f"{loss_dict['loss_objectness'].item():.3f}",
            "rpn_box": f"{loss_dict['loss_rpn_box_reg'].item():.3f}",
        })

    return running_loss / len(loader)


# ---- Visualisation --------------------------------------------------------
COLORS = plt.colormaps["tab20"].resampled(NUM_CLASSES).colors


@torch.no_grad()
def visualise(model, dataset, n=4, score_thresh=0.5):
    """
    Side-by-side comparison for n random validation images.
    Ground-truth boxes: green dashed.  Predicted boxes: solid, coloured by class.
    """
    model.eval()
    indices   = np.random.choice(len(dataset), n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img, target = dataset[idx]
        pred        = model([img.to(device)])[0]
        img_np      = img.permute(1, 2, 0).numpy()
        ax.imshow(img_np)

        # Ground-truth boxes (dashed green)
        for box, lbl in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--"))
            ax.text(x1, y1 - 2, VOC_CLASSES[lbl.item()],
                    color="lime", fontsize=6, va="bottom")

        # Predicted boxes (solid, class colour)
        for box, lbl, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.cpu().tolist()
            color = COLORS[lbl.item() % NUM_CLASSES]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"))
            ax.text(x1, y2 + 2, f"{VOC_CLASSES[lbl.item()]}  {score:.2f}",
                    color=color, fontsize=6, va="top")

        ax.axis("off")
        ax.set_title(f"sample {idx}")

    plt.suptitle("Green dashed = ground truth  |  Solid = predictions", fontsize=10)
    plt.tight_layout()
    plt.savefig("detection_results.png", dpi=150)
    plt.show()
    print("Saved → detection_results.png")


# ---- Main -----------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Hyper-parameters
    DATA_ROOT    = "./data"
    BATCH_SIZE   = 32
    EPOCHS       = 20
    LR           = 5e-4
    WEIGHT_DECAY = 5e-4

    # Data
    train_ds = VOCDetDataset(DATA_ROOT, image_set="train")
    val_ds   = VOCDetDataset(DATA_ROOT, image_set="val")
    num_workers = min(os.cpu_count(), 4)
    pin_memory  = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin_memory, persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin_memory, persistent_workers=num_workers > 0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {device} | Workers: {num_workers}")

    # Backbone is frozen; only the detection head params need an optimizer.
    # Switch BACKBONE = "mobilenet" for a ~4x faster alternative.
    BACKBONE = "resnet50"
    model  = build_model(NUM_CLASSES, backbone=BACKBONE).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Training loop
    best_prec = 0.0
    best_path = "best_detector.pth"
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch, EPOCHS)
        scheduler.step()

        # Detection evaluation is slow — run every 2 epochs
        if epoch % 2 == 0 or epoch == EPOCHS:
            prec = evaluate(model, val_loader)
            flag = ""
            if prec > best_prec:
                best_prec = prec
                torch.save(model.state_dict(), best_path)
                flag = " ← best"
            print(f"Epoch {epoch:02d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  "
                  f"precision@0.5={prec:.4f}{flag}")
        else:
            print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.4f}")

    # Final visualisation
    print(f"\nLoading best model (precision@0.5={best_prec:.4f}) …")
    model.load_state_dict(torch.load(best_path, map_location=device))
    visualise(model, val_ds, n=4)