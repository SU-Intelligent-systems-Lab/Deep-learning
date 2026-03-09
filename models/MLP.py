import torch.nn as nn
import torch.nn.functional as F  

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

class MLP2(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], num_classes=10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        x = self.output_layer(x)  # logits
        return x
