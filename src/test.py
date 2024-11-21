import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from tqdm import tqdm
from .utils.metrics import FIDcalculator

def test_model(model, test_loader, device,args):
    """Evaluate the VQ-VAE model and calculate FID score."""
    model.eval()
    test_loss = 0
    test_n_samples = 0
    
    # 准备FID计算
    real_images = []
    recon_images = []
    
    # 创建结果目录
    os.makedirs('test_results', exist_ok=True)
    
    # 使用tqdm显示进度
    with torch.no_grad():
        with tqdm(test_loader, desc='Testing') as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(device)
                
                # 前向传播
                recon_batch, commit_loss, _ = model(data)
                
                # 计算重建损失
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                test_loss += recon_loss.item()
                test_n_samples += data.size(0)
                
                # 收集图片用于FID计算
                real_images.extend(data.cpu())
                recon_images.extend(recon_batch.cpu())
                
                # 更新进度条
                pbar.set_postfix({
                    'test_loss': test_loss / test_n_samples
                })
                
                # 保存第一个批次的重建结果
                if batch_idx == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch[:n]])
                    save_image(
                        comparison.cpu(),
                        'test_results/reconstruction.png',
                        nrow=n
                    )
    
    # 计算平均损失
    avg_test_loss = test_loss / test_n_samples
    
    # 计算FID分数
    fid_calculator = FIDcalculator(device)
    fid_score = fid_calculator.calculate_fid(real_images, recon_images, -1)
    
    # 打印结果
    print(f'====> Test set loss: {avg_test_loss:.4f}')
    print(f'====> Test set FID score: {fid_score:.2f}')
    
    return {
        'test_loss': avg_test_loss,
        'fid_score': fid_score
    }