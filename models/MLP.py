import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten (B, 1, 28, 28) → (B, 784)
        return self.net(x)