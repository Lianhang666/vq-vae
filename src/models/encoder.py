# encoder.py
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=256, type='simple'):
        super().__init__()
        if type == 'simple':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(128, hidden_dims, 3, 1, 1)
            )

        elif type == 'h8xw8':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, hidden_dims, 3, 1, 1),
                nn.BatchNorm2d(hidden_dims),
                nn.SiLU(),
            )

        elif type == 'h32xw32':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, hidden_dims, 3, 1, 1),
                nn.BatchNorm2d(hidden_dims),
                nn.SiLU(),
            )

        elif type == 'maxpool':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, 1, 1),
                nn.MaxPool2d(2, 2),
                nn.GELU(),
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, hidden_dims, 1)
            )

        else:
            raise ValueError('Invalid encoder type')
    
    def forward(self, x):
        return self.encoder(x)
