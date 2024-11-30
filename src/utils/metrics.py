import torch
import torch.nn as nn
import torchvision.models as models
from scipy import linalg
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F

def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

#construct the feature extracter
class Inception(nn.Module):
    def __init__(self,device='cuda'):
        super().__init__()
        self.device = device
        #define the inception
        weights=models.Inception_V3_Weights.DEFAULT
        self.inception=models.inception_v3(weights=weights)
        #delete the layers in the inception we do not need
        self.inception.fc=nn.Identity()
        self.inception.aux_logits=False
        #freeze the gradient
        for param in self.inception.parameters():
            param.requires_grad=False
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # 确保在GPU上计算
        return self.inception(x.to(self.device))

#calculate the FID
class FIDcalculator():
    def __init__(self,device='cuda'):
        self.inception = Inception(device=device).to(device)
        self.device=device
    
    def calculate_activation_statistics(self, images, batch_size=128):
        self.inception.eval()
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                # 图片已经在GPU上，直接stack
                batch = torch.stack(images[i:i + batch_size])
                batch_features = self.inception(batch)
                features_list.append(batch_features)
                
            # 在GPU上合并和计算统计量
            features = torch.cat(features_list, dim=0)
            mu = features.mean(0)
            sigma = torch.cov(features.T)
            
            return mu, sigma  # 直接返回tensor而不是numpy数组
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        # 已经是tensor，不需要从numpy转换
        diff = mu1 - mu2
        diff_squared = torch.dot(diff, diff)
        
        # 使用svd计算矩阵平方根
        try:
            # sigma1和sigma2应该是对称矩阵
            prod = torch.mm(sigma1, sigma2)
            u, s, vh = torch.linalg.svd(prod)
            covmean = torch.mm(torch.mm(u, torch.diag(torch.sqrt(s))), vh)
        except:
            # 如果SVD失败，添加一个小的对角矩阵
            print(f"SVD失败，添加 {eps} 到对角线")
            offset = torch.eye(sigma1.shape[0], device=self.device) * eps
            prod = torch.mm(sigma1 + offset, sigma2 + offset)
            u, s, vh = torch.linalg.svd(prod)
            covmean = torch.mm(torch.mm(u, torch.diag(torch.sqrt(s))), vh)
        
        tr_covmean = torch.trace(covmean)
        
        return (diff_squared + torch.trace(sigma1) + 
                torch.trace(sigma2) - 2 * tr_covmean).item()
            
    def calculate_fid(self, real_images, fake_images, n_samples=5):
        """
        计算随机n张图片的FID分数
        
        Args:
            real_images: 真实图片列表
            fake_images: 生成图片列表
            n_samples: 要计算的图片数量（默认5张）
        """
        # 确保有足够的图片
        assert len(real_images) >= n_samples, f"真实图片数量不足{n_samples}张"
        assert len(fake_images) >= n_samples, f"生成图片数量不足{n_samples}张"
        
        # 随机选择n张图片
        indices = torch.randperm(len(real_images))[:n_samples]
        real_samples = [real_images[i] for i in indices]
        fake_samples = [fake_images[i] for i in indices]
        
        # 计算统计量
        mu_real, sigma_real = self.calculate_activation_statistics(real_samples)
        mu_fake, sigma_fake = self.calculate_activation_statistics(fake_samples)
        
        # 计算FID距离
        fid_value = self.calculate_frechet_distance(
            mu_real, sigma_real,
            mu_fake, sigma_fake
        )
        
        return fid_value
        