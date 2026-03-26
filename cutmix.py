"""
CutMix Augmentation
===================
CutMix mixes two training samples by cutting a rectangular patch from image B
and pasting it onto image A, then mixing the labels proportionally to the area:

    ỹ = λ · y_A  +  (1 - λ) · y_B
    L = λ · CE(f(x̃), y_A)  +  (1-λ) · CE(f(x̃), y_B)

Why it helps
------------
• Unlike Cutout (which zero-fills), every pixel still carries real signal.
• Strong regularizer — forces the model to use the whole image, not just the
  most discriminative patch.
• Improves calibration and often outperforms Mixup on natural images.

Reference: Yun et al., "CutMix: Training Strategy that Makes Use of Sample
Pairings," ICCV 2019.  https://arxiv.org/abs/1905.04899
"""

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import load_dataset


# ── Core ──────────────────────────────────────────────────────────────────────

def rand_bbox(H, W, lam):
    """
    Sample a random box whose area ≈ (1-lam)·H·W.

    Box size derivation
    -------------------
    cut_ratio = sqrt(1 - lam)  ensures  area(box)/area(image) ≈ 1 - lam.
    Center is sampled uniformly; edges are clipped to image boundaries.
    lam is recomputed from the actual clipped area so the label mix is exact.
    """
    r = np.sqrt(1.0 - lam)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - int(W*r)//2, 0, W), np.clip(cx + int(W*r)//2, 0, W)
    y1, y2 = np.clip(cy - int(H*r)//2, 0, H), np.clip(cy + int(H*r)//2, 0, H)
    return x1, y1, x2, y2, 1.0 - (x2-x1)*(y2-y1)/(H*W)


def cutmix_criterion(criterion, outputs, y_a, y_b, lam):
    """
    CutMix-aware loss: weighted sum of two cross-entropy terms.
    lam is the fraction of image A retained, so it weights CE toward y_a.
    """
    lam = lam.to(outputs.device)
    return (lam * criterion(outputs, y_a.to(outputs.device)) +
            (1 - lam) * criterion(outputs, y_b.to(outputs.device))).mean()

# ── Data ──────────────────────────────────────────────────────────────────────

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

tf = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def denorm(t):
    t = t.clone() * torch.tensor(STD).view(3,1,1) + torch.tensor(MEAN).view(3,1,1)
    return np.clip(t.permute(1,2,0).numpy(), 0, 1)

def get_diverse_samples(n):
    ds = load_dataset("zh-plus/tiny-imagenet", split="valid", streaming=True)
    seen, images, labels = set(), [], []
    for s in ds:
        idx = s["label"]
        if idx in seen: continue
        seen.add(idx)
        images.append(tf(s["image"].convert("RGB")))
        labels.append(ds.features["label"].int2str(idx))
        if len(images) == n: break
    return torch.stack(images), labels


# ── Demos ─────────────────────────────────────────────────────────────────────

def visualise_cutmix(n_examples=5, alpha=1.0):
    images, label_names = get_diverse_samples(n_examples * 2)
    imgs_A, lbls_A = images[:n_examples],  label_names[:n_examples]
    imgs_B, lbls_B = images[n_examples:],  label_names[n_examples:]

    _, _, H, W = imgs_A.shape
    lam_vals = np.random.beta(alpha, alpha, size=n_examples)
    mixed, boxes, lam_out = imgs_A.clone(), [], []
    for i in range(n_examples):
        x1, y1, x2, y2, la = rand_bbox(H, W, lam_vals[i])
        mixed[i, :, y1:y2, x1:x2] = imgs_B[i, :, y1:y2, x1:x2]
        boxes.append((x1, y1, x2, y2)); lam_out.append(la)

    fig, axes = plt.subplots(n_examples, 4, figsize=(14, 3.2 * n_examples))
    fig.suptitle(f"CutMix  (α={alpha})\nImage A  |  Image B (donor)  |  Mixed  |  Soft label",
                 fontsize=13, fontweight="bold")
    for j, t in enumerate(["Image A", "Image B (donor)", "Mixed", "Soft label"]):
        axes[0, j].set_title(t, fontsize=10, fontweight="bold")

    for i, (lam, (x1,y1,x2,y2)) in enumerate(zip(lam_out, boxes)):
        ca, cb = lbls_A[i], lbls_B[i]
        axes[i,0].imshow(denorm(imgs_A[i])); axes[i,0].set_xlabel(ca, fontsize=8, color="navy")
        axes[i,1].imshow(denorm(imgs_B[i])); axes[i,1].set_xlabel(cb, fontsize=8, color="darkred")
        axes[i,1].add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                            lw=2, edgecolor="red", facecolor="none", linestyle="--"))
        axes[i,2].imshow(denorm(mixed[i]))
        axes[i,2].add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                            lw=2, edgecolor="red", facecolor="none"))
        axes[i,2].set_xlabel(f"{lam:.2f}·{ca} + {1-lam:.2f}·{cb}", fontsize=7)
        ax = axes[i,3]
        ax.barh([ca, cb], [lam, 1-lam], color=["steelblue","tomato"])
        ax.set_xlim(0,1); ax.axvline(0.5, color="grey", ls=":", lw=1)
        ax.set_xlabel("label weight", fontsize=8)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for ax_ in axes[i,:3]: ax_.axis("off")

    plt.tight_layout()
    plt.savefig("cutmix_examples.png", dpi=150, bbox_inches="tight")

def visualise_alpha_effect(alpha_values=(0.2, 0.5, 1.0, 2.0), n_pairs=3):
    images, label_names = get_diverse_samples(n_pairs * 2)
    img_A, lbl_A = images[:n_pairs],  label_names[:n_pairs]
    img_B, lbl_B = images[n_pairs:],  label_names[n_pairs:]
    _, _, H, W = img_A.shape

    fig, axes = plt.subplots(n_pairs, len(alpha_values),
                             figsize=(4*len(alpha_values), 3.5*n_pairs))
    fig.suptitle("CutMix — Effect of α\nLow α → extreme patches; High α → medium patches",
                 fontsize=12, fontweight="bold")
    for col, alpha in enumerate(alpha_values):
        axes[0, col].set_title(f"α = {alpha}", fontsize=11, fontweight="bold")
        for row in range(n_pairs):
            lam = np.random.beta(alpha, alpha)
            x1, y1, x2, y2, la = rand_bbox(H, W, lam)
            m = img_A[row].clone()
            m[:, y1:y2, x1:x2] = img_B[row, :, y1:y2, x1:x2]
            axes[row, col].imshow(denorm(m))
            axes[row, col].set_xlabel(f"λ={la:.2f}  {lbl_A[row]}+{lbl_B[row]}", fontsize=7)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("cutmix_alpha_effect.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    visualise_cutmix(n_examples=5, alpha=1.0)
    visualise_alpha_effect(alpha_values=(0.2, 0.5, 1.0, 2.0))