# decoder.py
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dims=256, type='simple'):
        super().__init__()
        if type == 'simple':
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden_dims, 128, 3, 1, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
                nn.Tanh()
            )

        elif type == 'h8xw8':
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
                nn.Tanh()
            )

        elif type == 'h32xw32':
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.ConvTranspose2d(32, out_channels, 3, 1, 1),
                nn.Tanh()
            )
        
        else:
            raise ValueError('Invalid decoder type')
    
    def forward(self, x):
        return self.decoder(x)
    