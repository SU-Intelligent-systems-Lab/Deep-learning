import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module): #Residual block 
    """
    Basic residual block used in shallow ResNet architectures (e.g., ResNet-18, ResNet-34).

    This block consists of two 3×3 convolutional layers with Batch Normalization
    and ReLU activation. A residual (skip) connection adds the input tensor to
    the output of the stacked convolutions.

    If the spatial resolution or number of channels changes (due to stride > 1
    or channel mismatch), a projection shortcut (typically a 1×1 convolution)
    is applied to the input to match dimensions before addition.

    Structure:
        Conv3x3(in_channels → channels, stride)
        BatchNorm
        ReLU
        Conv3x3(channels → channels, stride=1)
        BatchNorm
        Add shortcut
        ReLU

    Args:
        in_channels (int):
            Number of input channels.
        channels (int):
            Number of output channels produced by the block.
        stride (int, optional):
            Stride for the first convolution layer. Default is 1.
        downsample (nn.Module, optional):
            Optional module to match spatial or channel dimensions
            for the residual connection.

    Attributes:
        expansion (int):
            Expansion factor for output channels. For BasicBlock, expansion = 1.

    Shape:
        Input:
            (N, in_channels, H, W)
        Output:
            (N, channels * expansion, H_out, W_out)

        where:
            H_out = H / stride
            W_out = W / stride
    """
  
    expansion = 1 # For BasicBlock, output channels = channels * expansion = channels
    def __init__(self, in_channels, channels, stride=1,norm=nn.BatchNorm2d, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm(channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                Information gets lost 
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, channels//4, channels//4), "constant", 0))
            elif option == 'B':
                #More efficient 
                #Learnable parameter but more computationally expensive    1X1 convultion --> reducing ot increasing dimesion 
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                     norm(self.expansion * channels)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) #Residual Part 
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet (Residual Network) implementation for image classification.

    This class builds a ResNet architecture using either BasicBlock or Bottleneck
    residual blocks. The network consists of an initial convolution and
    normalization, followed by four residual layers, global average pooling,
    and a final fully connected layer for classification.

    Args:
    -----
        block (nn.Module):
            Residual block class to use (BasicBlock or Bottleneck).
        num_blocks (list of int):
            Number of blocks in each of the four layers.
            Example: [2, 2, 2, 2] for ResNet-18.
        norm (nn.Module, optional):
            Normalization layer to use after convolutions. Default: nn.BatchNorm2d.
        num_classes (int, optional):
            Number of output classes for classification. Default: 10.

    Attributes:
    -----------
        in_channels (int):
            Number of input channels for the next block; updated after each layer.
        conv1 (nn.Conv2d):
            Initial convolution layer (3×3 kernel).
        bn1 (nn.Module):
            Normalization after conv1.
        layer1, layer2, layer3, layer4 (nn.Sequential):
            Residual layers composed of the specified block type.
        avgpool (nn.AdaptiveAvgPool2d):
            Global average pooling layer reducing spatial dimensions to 1×1.
        linear (nn.Linear):
            Fully connected layer mapping features to `num_classes`.

    Shape:
        Input:
            (N, 3, H, W) where H and W are image height and width.
        Output:
            (N, num_classes) — class logits for each input sample.

    Example:
        >>> model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        >>> x = torch.randn(8, 3, 32, 32)
        >>> logits = model(x)

    References:
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    [2] https://github.com/KaimingHe/deep-residual-networks
    
    """
    def __init__(self, block, num_blocks, norm=nn.BatchNorm2d, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], norm=norm,stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm=norm,stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm=norm,stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm=norm,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  #just need to specify the final output 
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, norm, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride,norm))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



