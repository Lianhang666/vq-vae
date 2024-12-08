import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
import torch.optim as optim
from src.models.vqvae import VQVAE

import numpy as np


class vae_classifier(nn.Module):
    def __init__(self, vae: VQVAE, n_classes = 10):
        super(vae_classifier, self).__init__()
        self.vae = vae
        self.encoder = vae.encoder
        self.quantizer = vae.quantizer
        self.quantized_type = vae.quantized_type
        # self.head = nn.Linear(vae.hidden_dims, n_classes)
        self.head = nn.Sequential(
            nn.Conv2d(vae.hidden_dims, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        if self.quantized_type=='vq':
            z = self.encoder(x)
            z = z.permute(0, 2, 3, 1)
            shape = z.shape
            z = z.reshape(shape[0], shape[1] * shape[2], shape[3])
            quantized_z, _, _ = self.quantizer(z)
            quantized_z = quantized_z.reshape(shape)
            quantized_z = quantized_z.permute(0, 3, 1, 2)
            # quantized_z = quantized_z.mean(dim=[2, 3])
            logits = self.head(quantized_z)
        elif self.quantized_type == 'fsq':
            z = self.encoder(x)
            z = z.permute(0, 2, 3, 1)
            shape = z.shape
            z = z.reshape(shape[0], shape[1] * shape[2], shape[3])
            quantized_z, _ = self.quantizer(z)
            quantized_z = quantized_z.reshape(shape)
            quantized_z = quantized_z.permute(0, 3, 1, 2)
            logits = self.head(quantized_z)
        return logits
    
class WarmUpCosine(LRScheduler):
    def __init__(self, optimizer: optim,
                 total_steps: int,
                 warmup_steps: int,
                 learning_rate_base: float,
                 warmup_learning_rate: float,
                 last_epoch: int = -1):
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = np.pi
        super(WarmUpCosine, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step > self.total_steps:
            return [0.0 for _ in self.base_lrs]

        cos_annealed_lr = 0.5 * self.learning_rate_base * (1 + np.cos(self.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))

        if step < self.warmup_steps:
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * step + self.warmup_learning_rate
            return [warmup_rate for _ in self.base_lrs]
        else:
            return [cos_annealed_lr for _ in self.base_lrs]