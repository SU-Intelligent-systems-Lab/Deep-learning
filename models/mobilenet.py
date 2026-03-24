'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
### Why Is MobileNetV2 So Efficient?

For a 3×3 conv, 32 input channels, 64 output channels, on a 16×16 feature map:

**Standard conv cost:**
3 × 3 × 32 × 64 × 16 × 16 = 9,437,184 multiplications


**Depthwise + Pointwise cost:**
depthwise:  3 × 3 × 32 × 16 × 16        =   73,728
pointwise:  1 × 1 × 32 × 64 × 16 × 16   = 524,288
total:                                   = 598,016


ResNet-18:   ~556M
MobileNetV2:  ~91M 
─────────────────────────
Ratio:          6.1×  lighter

"""

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False) #The `groups=planes` argument is what makes it depthwise, each channel gets its own filter.
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()



"""
Input:  (64, 3, 32, 32)

Conv2d(3→32, k=3, s=1, pad=1) + BN + ReLU → (64, 32, 32, 32)
                                              ↑ stride=1 for CIFAR

── 7 stages of Inverted Residual Blocks ──

Stage 1: t=1, 1 block,  32→16,  s=1  → (64, 16,  32, 32)
Stage 2: t=6, 2 blocks, 16→24,  s=1  → (64, 24,  32, 32)  ← s=1 for CIFAR
Stage 3: t=6, 3 blocks, 24→32,  s=2  → (64, 32,  16, 16)
Stage 4: t=6, 4 blocks, 32→64,  s=2  → (64, 64,  8,  8)
Stage 5: t=6, 3 blocks, 64→96,  s=1  → (64, 96,  8,  8)
Stage 6: t=6, 3 blocks, 96→160, s=2  → (64, 160, 4,  4)
Stage 7: t=6, 1 block,  160→320,s=1  → (64, 320, 4,  4)

Each block internally:
  (64, 32, H, W)
  → expand:     Conv1×1(32→192) + BN + ReLU    (t=6, so 32×6=192)
  → depthwise:  Conv3×3(192, groups=192) + BN + ReLU
  → project:    Conv1×1(192→32) + BN            ← NO ReLU (linear bottleneck)
  → + shortcut if stride==1

Conv2d(320→1280, k=1) + BN + ReLU    → (64, 1280, 4, 4)
AvgPool2d(4)                          → (64, 1280, 1, 1)  ← kernel=4 for CIFAR
x.view(B, -1)                         → (64, 1280)
Linear(1280→10)                        → (64, 10)

Output: (64, 10) logits

"""
