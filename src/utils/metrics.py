import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from scipy import linalg
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.output_dim = 2048

    def forward(self, x):
        x = self.blocks(x)
        return x

def calculate_activation_statistics(images, model, batch_size=128, dims=2048, device='cuda'):
    model.eval()
    act = np.empty((len(images), dims))
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i + batch_size]).to(device)
            batch = batch.type(torch.FloatTensor).to(device)
            
            # 调整输入大小为inception的要求
            if batch.size(-1) != 299:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # 获取特征
            pred = model(batch)
            
            # 如果输出是元组，取第一个元素
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # 平均池化到特征向量
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            act[i:i + batch.shape[0]] = pred.squeeze().cpu().numpy()
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(model, dataloader, device, batch_size=50):
    inception = InceptionV3().to(device)
    real_images = []
    generated_images = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            recon_images, _, _ = model(images)
            
            real_images.extend(images.cpu())
            generated_images.extend(recon_images.cpu())
            
            if len(real_images) >= batch_size:
                break
    
    mu_real, sigma_real = calculate_activation_statistics(real_images, inception, device=device, dims=192)
    mu_fake, sigma_fake = calculate_activation_statistics(generated_images, inception, device=device, dims=192)
    
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_value