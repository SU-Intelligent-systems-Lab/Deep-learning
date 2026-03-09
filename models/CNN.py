import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):

    def __init__(self,norm,num_classes=10):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1) #format: (in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500) # assuming input images are 28x28, after two conv+pool layers we get 4x4 feature maps
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        # formula for output size: (W - F + 2P) / S + 1, where W=input size, F=filter size, P=padding, S=stride
        x = F.relu(self.conv1(x)) # (28 - 5 + 2*0) / 1 + 1 = 24
        x = F.max_pool2d(x, 2, 2) # (24 - 2) / 2 + 1 = 12
        x = F.relu(self.conv2(x)) # (12 - 5 + 2*0) / 1 + 1 = 8
        x = F.max_pool2d(x, 2, 2) # (8 - 2) / 2 + 1 = 4, so we have 50 feature maps of size 4x4
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """
    Example of CNN architecture with Kaiming (He) initialization applied.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # assuming input images 32x32
        self.fc2 = nn.Linear(128, num_classes)

        # Apply Kaiming initialization
        self._initialize_weights()

        """
        Kaiming initialization is typically applied to convolutional and linear layers when using ReLU activations.
        It helps maintain the variance of activations through the layers, which can lead to better convergence during training.
        For convolutional layers, we use 'fan_in' mode, which considers the number of input units to the layer.
        'fan_out' mode can also be used if you want to consider the number of output units, but 'fan_in' is more common for ReLU.
        In practice, 'fan_in' is often preferred for ReLU activations because it helps prevent the variance from exploding as we go deeper into the network.

        Formula for Kaiming initialization (for ReLU):
        weight ~ N(0, sqrt(2 / fan_in))

        2.0 is the ReLU correction factor. ReLU zeros out ~half of all activations
        (anything negative), which halves the signal variance at each layer.
        Doubling the initial variance compensates for this, keeping the signal stable.

        """

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Conv layers + ReLU + MaxPool
        x = F.relu(self.conv1(x)) # 32 - 3 + 2*1 = 32 ((padding=1 keeps size))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16

        x = F.relu(self.conv2(x)) # 16 - 3 + 2*1 = 16 (padding=1 keeps size)
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # output logits

        return x
