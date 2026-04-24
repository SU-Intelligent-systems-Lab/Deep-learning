import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks (used at every U-Net level)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool 2× then DoubleConv — one encoder step."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear 2× upsample, concat skip connection, then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad x to match skip's spatial size (handles odd input dimensions)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    U-Net for semantic segmentation (Ronneberger et al., 2015).

    Architecture: 4 encoder levels → bottleneck → 4 decoder levels.
    Skip connections concatenate encoder feature maps into the decoder.

    Args:
        in_channels  : input image channels (3 for RGB)
        num_classes  : number of segmentation classes (output channels)
        base_filters : feature channels at level 1; doubles each level (default 64)

    Input  : (B, in_channels, H, W)
    Output : (B, num_classes, H, W)  raw logits — apply argmax for prediction
    """
    def __init__(self, in_channels=3, num_classes=21, base_filters=64):
        super().__init__()
        f = base_filters
        # Encoder
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,     f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)   # bottleneck
        # Decoder (in_ch = upsampled + skip concatenation)
        self.up1   = Up(f * 16 + f * 8, f * 8)
        self.up2   = Up(f * 8  + f * 4, f * 4)
        self.up3   = Up(f * 4  + f * 2, f * 2)
        self.up4   = Up(f * 2  + f,     f)
        self.out   = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.out(x)
