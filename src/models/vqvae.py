import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, FSQ
from .encoder import Encoder
from .decoder import Decoder

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_dims=256,
        codebook_size=512,
        decay=0.8,
        commitment_weight=1.0,
        quantized_type='fsq'
    ):
        super().__init__()
        self.quantized_type=quantized_type
        self.encoder = Encoder(in_channels, hidden_dims)
        self.decoder = Decoder(in_channels, hidden_dims)
        if self.quantized_type=='vq':     
            self.quantizer = VectorQuantize(
                dim=hidden_dims,
                codebook_size=codebook_size,
                decay=decay,
                commitment_weight=commitment_weight
            )
        elif self.quantized_type=='fsq': 
            self.fsq_levels=[8, 5, 5, 5]    
            self.quantizer = FSQ(
                self.fsq_levels
            )
            
        self.hidden_dims = hidden_dims
        
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        if self.quantized_type=='vq':    
            # Reshape for VQ
            z = z.permute(0, 2, 3, 1)  # [B, H, W, C]
            shape = z.shape
            z = z.reshape(-1, shape[-1])  # [B*H*W, C]
            
 
            quantized, indices, commit_loss = self.quantizer(z)
            
            # Reshape back
            quantized = quantized.reshape(shape)  # [B, H, W, C]
            quantized = quantized.permute(0, 3, 1, 2)  # [B, C, H, W]
            indices=indices.reshape(shape[0],shape[1],shape[2])
        elif self.quantized_type == 'fsq':
            # Reshape for FSQ
            z = z.permute(0, 2, 3, 1)  # [B, H, W, C]
            shape = z.shape          
            z = z.reshape(shape[0] * shape[1] * shape[2], -1, len(self.fsq_levels))
            
            # FSQ Quantization
            quantized, indices = self.quantizer(z)
            commit_loss = torch.tensor(0.0, device=z.device)
            
            # Reshape back
            quantized = quantized.reshape(shape)
            quantized = quantized.permute(0, 3, 1, 2)           
            indices = indices[:, 0].reshape(shape[0], shape[1], shape[2])
 
        # Decode
        x_recon = self.decoder(quantized)
        
        return x_recon, commit_loss, indices

    def encode(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1)
        shape = z.shape
        z = z.reshape(-1, shape[-1])
        _, indices, _ = self.vq(z)
        return indices.reshape(shape[0], shape[1], shape[2])

    def decode(self, indices):
        shape = indices.shape
        indices = indices.reshape(-1)
        quantized = self.vq.codebook[indices]
        quantized = quantized.reshape(shape[0], shape[1], shape[2], -1)
        quantized = quantized.permute(0, 3, 1, 2)
        return self.decoder(quantized)