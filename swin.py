"""Swin-T for CIFAR-10 (32×32). Liu et al., ICCV 2021. https://arxiv.org/abs/2103.14030"""

import argparse, copy, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")


# ── Window helpers ─────────────────────────────────────────────────────────────

def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)

def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws ** 2))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── Window Attention ───────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """W-MSA with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        flat = torch.flatten(coords, 1)
        rel  = flat[:, :, None] - flat[:, None, :]
        rel  = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B_, N, C)))


# ── Swin Block ─────────────────────────────────────────────────────────────────

class SwinBlock(nn.Module):
    """Even blocks: W-MSA (shift=0). Odd blocks: SW-MSA (shift=ws//2)."""

    def __init__(self, dim, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.shift_size, self.window_size = shift_size, window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads,
                                     qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def _make_attn_mask(self, H, W, dev):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=dev)
        cnt = 0
        for hs in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
            for ws in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)):
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mw   = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.).masked_fill(mask == 0, 0.)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        mask     = self._make_attn_mask(H, W, x.device)
        x_win    = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        attn_win = self.attn(x_win, mask=mask).view(-1, self.window_size, self.window_size, C)
        x        = window_reverse(attn_win, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), (1, 2))
        x = shortcut + x.view(B, L, C)
        return x + self.mlp(self.norm2(x))


# ── Patch Merging ──────────────────────────────────────────────────────────────

class PatchMerging(nn.Module):
    """2× spatial downsampling: concatenate 2×2 neighbours, project 4C → 2C."""

    def __init__(self, dim):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


# ── Swin Stage ─────────────────────────────────────────────────────────────────

class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 drop=0., attn_drop=0., downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                      shift_size=0 if i % 2 == 0 else window_size // 2,
                      mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W), H // 2, W // 2
        return x, H, W


# ── Swin Transformer ───────────────────────────────────────────────────────────

class SwinTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_channels=3, num_classes=10,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=4, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm0    = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)

        # Only downsample when next grid size >= window_size; dims only double after a merge.
        grid = img_size // patch_size
        will_downsample = [(i < len(depths) - 1) and (grid // (2 ** (i + 1)) >= window_size)
                           for i in range(len(depths))]
        dims = [embed_dim]
        for ds in will_downsample[:-1]:
            dims.append(dims[-1] * 2 if ds else dims[-1])

        self.stages = nn.ModuleList([
            SwinStage(dim=dims[i], depth=depths[i], num_heads=num_heads[i],
                      window_size=window_size, mlp_ratio=mlp_ratio,
                      drop=drop_rate, attn_drop=attn_drop_rate,
                      downsample=will_downsample[i])
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = self.pos_drop(self.norm0(x.flatten(2).transpose(1, 2)))
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        return self.head(self.norm(x).mean(dim=1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Data ───────────────────────────────────────────────────────────────────────

def get_loaders(batch_size=128):
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    kw = dict(num_workers=2, pin_memory=True)
    train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=val_tf)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw))


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type in ("cuda", "cpu"))):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        n       += imgs.size(0)
    return correct / n


# ── Main ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup",       type=int,   default=10)
    p.add_argument("--embed_dim",    type=int,   default=96)
    p.add_argument("--drop_rate",    type=float, default=0.1)
    p.add_argument("--save_path",    type=str,   default="best_swin.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    model = SwinTransformer(
        img_size=32, patch_size=2, in_channels=3, num_classes=10,
        embed_dim=args.embed_dim, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        window_size=4, mlp_ratio=4., drop_rate=args.drop_rate,
    ).to(device)
    print(model)
    print(f"\nTrainable parameters: {model.count_params():,}\n")

    train_loader, val_loader = get_loaders(args.batch_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.999))

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_acc, best_weights = 0.0, None
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Acc':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_acc = evaluate(model, val_loader)
        scheduler.step()

        flag = ""
        if val_acc > best_acc:
            best_acc, best_weights = val_acc, copy.deepcopy(model.state_dict())
            torch.save(best_weights, args.save_path)
            flag = " ✓"

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
              f"{val_acc:>8.4f}  {optimizer.param_groups[0]['lr']:>8.2e}  "
              f"{time.time()-t0:>5.1f}s{flag}")

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print(f"Weights saved to: {args.save_path}")