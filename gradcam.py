"""
GradCAM — Gradient-weighted Class Activation Mapping
======================================================
GradCAM answers the question: "Which parts of the image made the model
predict class C?"

Key idea
--------
After a forward pass, we:
  1. Pick a convolutional layer (usually the last one — it has the richest
     spatial features while still being high-resolution enough to be useful).
  2. Compute the gradient of the class score  y^c  w.r.t. every activation
     map  A^k  in that layer.
  3. Average those gradients spatially to get importance weights  α_k^c.
  4. Take a weighted sum of the activation maps, then ReLU it.

     GradCAM(x) = ReLU( Σ_k  α_k^c · A^k )

     The ReLU keeps only the activations that *increase* the class score
     (negatively-contributing regions are ignored — they'd correspond to a
     different class).

  5. Resize the heatmap back to input resolution and overlay it.

References
----------
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization," ICCV 2017.
https://arxiv.org/abs/1610.02391
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datasets import load_dataset
import ssl
from huggingface_hub import login
import os


# Fix for macOS SSL certificate verification error when downloading
ssl._create_default_https_context = ssl._create_unverified_context


# ── GradCAM core ──────────────────────────────────────────────────────────────

class GradCAM:
    """
    Computes GradCAM heatmaps for a given model and target layer.

    How the hooks work
    ------------------
    PyTorch's hook system lets us intercept:
      • forward_hook       — captures activations A^k after the forward pass.
      • full_backward_hook — captures gradients dY/dA^k after the backward pass.

    We register both on the target conv layer, then read them off after calling
    loss.backward() to compute the weighted heatmap.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._fwd_hook   = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        self._bwd_hook   = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(x)                                   # (1, num_classes)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[0, class_idx].backward()

        # alpha_k^c = global-average-pool of gradients -> weighted sum -> ReLU
        weights  = self.gradients.mean(dim=(2, 3), keepdim=True) # (1, K, 1, 1) ex. for resnet18 layer4[-1] K=512 so weights shape is (1, 512, 1, 1)
        cam      = torch.relu((weights * self.activations).sum(dim=1, keepdim=True)) # (1, 1, H', W') ex. for resnet18 layer4[-1] H'=W'=7 so cam shape is (1, 1, 7, 7)

        # Normalise and upsample to input resolution
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        h, w    = x.shape[2], x.shape[3]
        heatmap = np.array(Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)) / 255.0 # (H, W) ex. for resnet18 layer4[-1] (224, 224)
        return heatmap, class_idx

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Helpers ───────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def denormalize(t):
    t = t.clone().squeeze(0) * torch.tensor(IMAGENET_STD).view(3,1,1) \
        + torch.tensor(IMAGENET_MEAN).view(3,1,1)
    return np.clip(t.permute(1,2,0).numpy(), 0, 1)

def overlay(img, heatmap, alpha=0.5):
    rgb = cm.get_cmap("jet")(heatmap)[:, :, :3]
    return (np.clip((1 - alpha) * img + alpha * rgb, 0, 1) * 255).astype(np.uint8)

def get_diverse_samples(n: int = 8):
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
    # ds = load_dataset("zh-plus/tiny-imagenet", split="valid", streaming=True)
    seen, images, labels = set(), [], []
    for sample in ds:
        label_idx  = sample["label"]
        label_name = ds.features["label"].int2str(label_idx)
        if label_idx in seen:
            continue
        seen.add(label_idx)
        images.append(transform(sample["image"].convert("RGB")))
        labels.append(label_name)
        if len(images) == n:
            break
    return torch.stack(images), labels


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_multiclass(n_images: int = 8):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    images, label_names = get_diverse_samples(n_images)

    fig, axes = plt.subplots(n_images, 3, figsize=(11, 3.5 * n_images))
    for col, title in enumerate(["Original", "Heatmap (layer4)", "Overlay"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(n_images):
        x = images[i].unsqueeze(0)

        cam = GradCAM(model, target_layer=model.layer4[-1])
        heatmap, pred_class = cam(x)
        cam.remove_hooks()

        img_np = denormalize(x)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(f"True: {label_names[i]}\nPred idx: {pred_class}",
                               fontsize=8, rotation=0, labelpad=90, va="center")
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 2].imshow(overlay(img_np, heatmap))
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle("GradCAM — ResNet-18 on imagenet (one image per class)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("gradcam_multiclass.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    login()
    demo_multiclass(n_images=10)
    os._exit(0)