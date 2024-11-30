import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from tqdm import tqdm
from .utils.metrics import FIDcalculator, multiplyList
from .models.vqvae import fsq_levels_lookup


def test_model(model, test_loader, device, codebook_size, model_type, args):
    """Evaluate the VQ-VAE model and calculate FID score."""
    model.eval()
    test_loss = 0
    test_n_samples = 0
    
    # Prepare for FID calculation
    real_images = []
    recon_images = []
    
    # Create results directory
    os.makedirs(f'model_{codebook_size}/{model_type}_{codebook_size}', exist_ok=True)
    
    # Initialize a set to store all unique indices used
    total_indices = set()
    
    # Use tqdm to display progress
    with torch.no_grad():
        with tqdm(test_loader, desc='Testing') as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(device)
                
                # Forward pass
                recon_batch, commit_loss, indices = model(data)
                
                # Update total_indices with unique indices from the current batch
                # index shape: [B, H * W]
                for idx in indices.cpu().numpy().flatten().tolist():
                    total_indices.add(idx)


                # if model_type == 'fsq':
                #     total_indices.update(indices.cpu().numpy().flatten().tolist())
                #     # for idx in indices.cpu().numpy().flatten().tolist():
                #     #     total_indices.add(idx.item())
                # else:
                #     total_indices.update(indices.cpu().numpy().flatten().tolist())
                
                # Compute reconstruction loss
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
                test_loss += recon_loss.item()
                test_n_samples += data.size(0)
                
                # Collect images for FID calculation
                real_images.extend(data.cpu())
                recon_images.extend(recon_batch.cpu())
                
                # Update progress bar
                pbar.set_postfix({
                    'test_loss': test_loss / test_n_samples,
                    'active %': indices.unique().numel() / model.codebook_size * 100
                })
                
                # Save reconstruction results of the first batch
                if batch_idx == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch[:n]])
                    save_image(
                        comparison.cpu(),
                        f'model_{codebook_size}/{model_type}_{codebook_size}/reconstruction.png',
                        nrow=n
                    )
    
    # Compute average loss
    avg_test_loss = test_loss / test_n_samples
    
    # Compute total codebook usage percentage
    # print(f'Total indices: {len(total_indices)}')
    if model_type == 'fsq':
        codebook_size = multiplyList(fsq_levels_lookup[codebook_size])
    total_active_percentage = len(set(total_indices)) / codebook_size * 100
    
    # Compute FID score
    fid_calculator = FIDcalculator(device)
    #skip fid for now
    fid_score = fid_calculator.calculate_fid(real_images, recon_images, -1)
    # fid_score = 0
    
    # Print results
    print(f'====> Test set loss: {avg_test_loss:.4f}')
    print(f'====> Test set FID score: {fid_score:.2f}')
    print(f'Total codebook usage on test dataset: {total_active_percentage:.2f}%')
    
    return {
        'mse_loss': avg_test_loss,
        'fid_score': fid_score,
        'codebook_usage': total_active_percentage
    }
