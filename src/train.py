import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from datetime import datetime
from .utils.metrics import FIDcalculator
import wandb

######################train######################################
def train_one_epoch(model,train_loader,optimizer,device,epoch,codebook_size, args):
    model.train()
    total_recon_loss=0
    total_commit_loss=0
    total_loss = 0
    with tqdm(train_loader,desc=f'Train Epoch{epoch+1}/{args.epochs}') as pbar:
        for batch_idx,(data,_) in enumerate(pbar):
            data=data.to(device)
            optimizer.zero_grad()
            recon_batch,commit_loss,indices=model(data)
            recon_loss=F.mse_loss(recon_batch,data)
            loss=recon_loss+commit_loss
            loss.backward()
            optimizer.step()
            # Update metrics
            total_recon_loss += recon_loss.item()
            total_commit_loss += commit_loss.item()
            total_loss += loss.item()
            # Update progress bar
            pbar.set_postfix({
                'recon_loss': total_recon_loss / (batch_idx + 1),
                'commit_loss': total_commit_loss / (batch_idx + 1),
                'total_loss': total_loss / (batch_idx + 1),
                'active %': indices.unique().numel() / model.codebook_size * 100
            })
    #calculate the average loss and return 
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_commit_loss = total_commit_loss / len(train_loader)
    avg_total_loss = total_loss / len(train_loader)
    avg_active = indices.unique().numel() / model.codebook_size * 100
    #plot the reults in wandb and need to distinguish between train and validation the log name must contiaon train or validation
    #wandb.log({ 'train_recon_loss': avg_recon_loss, 'train_commit_loss': avg_commit_loss, 'train_total_loss': avg_total_loss, 'active ': avg_active})
    
    
    return {
        'recon_loss': avg_recon_loss,
        'commit_loss': avg_commit_loss,
        'total_loss': avg_total_loss,
        'active': avg_active
    }
    
#################validation#################################
def validate_one_epoch(model,val_loader,device,epoch,codebook_size, args):
    model.eval()
    total_recon_loss=0
    total_commit_loss=0
    total_loss = 0
    fid_score=0
    recon_images=[]
    real_images=[]
    with torch.no_grad():
        with tqdm(val_loader,desc=f'Validation Epoch{epoch+1}/{args.epochs}') as pbar:
            for batch_idx, (data,_) in enumerate(pbar):
                data=data.to(device)
                recon_batch,commit_loss,indices=model(data)
                recon_loss=F.mse_loss(recon_batch,data)
                # loss=recon_loss+commit_loss
                loss=recon_loss
                # Update metrics
                total_recon_loss += recon_loss.item()
                total_commit_loss += commit_loss.item()
                total_loss += loss.item()
                # Update progress bar
                pbar.set_postfix({
                    'recon_loss': total_recon_loss / (batch_idx + 1),
                    'commit_loss': total_commit_loss / (batch_idx + 1),
                    'total_loss': total_loss / (batch_idx + 1),
                    'active %': indices.unique().numel() / model.codebook_size * 100
                })
                recon_images.extend(recon_batch)
                real_images.extend(data)
    # fid_calculator = FIDcalculator(device=device)
    # fid_score = fid_calculator.calculate_fid(real_images, recon_images,args.num_sample)
    # print(f'FID Score: {fid_score:.2f}')
    val_avg_recon_loss = total_recon_loss / len(val_loader)
    val_avg_commit_loss = total_commit_loss / len(val_loader)
    val_avg_total_loss = total_loss / len(val_loader)
    val_avg_active = indices.unique().numel() / model.codebook_size
    #plot the reults in wandb and need to distinguish between train and validation
    #wandb.log({ 'val_recon_loss': val_avg_recon_loss, 'val_commit_loss': val_avg_commit_loss, 'val_total_loss': val_avg_total_loss, 'val_active ':val_avg_active})
    return {
        'recon_loss': avg_recon_loss,
        'commit_loss': avg_commit_loss,
        'total_loss': avg_total_loss,
        'active': avg_active
    }
   
def train_model(model, train_loader, val_loader, optimizer, device, codebook_size, args):
    checkpoint_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    num_bad_epochs = 0  # Number of epochs since last improvement

    # Ensure 'patience' is set in args; default to 5 if not provided
    if not hasattr(args, 'patience'):
        args.patience = 10

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, codebook_size, args)
        # wandb.log(train_metrics)
        if model.quantized_type == 'vq':
            wandb.log({f'vq_vae_{codebook_size}_train_recon_loss': train_metrics['recon_loss'], 
                          f'vq_vae_{codebook_size}_train_commit_loss': train_metrics['commit_loss'], 
                          f'vq_vae_{codebook_size}_train_total_loss': train_metrics['total_loss'], 
                          f'vq_vae_{codebook_size}_train_active': train_metrics['active']})
        elif model.quantized_type == 'fsq':
            wandb.log({f'fsq_vae_{codebook_size}_train_recon_loss': train_metrics['recon_loss'], 
                          f'fsq_vae_{codebook_size}_train_total_loss': train_metrics['total_loss'], 
                          f'fsq_vae_{codebook_size}_train_active': train_metrics['active']})
        elif model.quantized_type == 'vq_rotation':
            wandb.log({f'vq_rotation_vae_{codebook_size}_train_recon_loss': train_metrics['recon_loss'], 
                          f'vq_rotation_vae_{codebook_size}_train_commit_loss': train_metrics['commit_loss'], 
                          f'vq_rotation_vae_{codebook_size}_train_total_loss': train_metrics['total_loss'], 
                          f'vq_rotation_vae_{codebook_size}_train_active': train_metrics['active']})
        # Validate    
        val_metrics = validate_one_epoch(model, val_loader, device, epoch, codebook_size, args)
        if model.quantized_type == 'vq':
            wandb.log({f'vq_vae_{codebook_size}_val_recon_loss': val_metrics['recon_loss'], 
                          f'vq_vae_{codebook_size}_val_commit_loss': val_metrics['commit_loss'], 
                          f'vq_vae_{codebook_size}_val_total_loss': val_metrics['total_loss'], 
                          f'vq_vae_{codebook_size}_val_active': val_metrics['active']})
        elif model.quantized_type == 'fsq':
            wandb.log({f'fsq_vae_{codebook_size}_val_recon_loss': val_metrics['recon_loss'], 
                          f'fsq_vae_{codebook_size}_val_total_loss': val_metrics['total_loss'], 
                          f'fsq_vae_{codebook_size}_val_active': val_metrics['active']})
        elif model.quantized_type == 'vq_rotation':
            wandb.log({f'vq_rotation_vae_{codebook_size}_val_recon_loss': val_metrics['recon_loss'], 
                          f'vq_rotation_vae_{codebook_size}_val_commit_loss': val_metrics['commit_loss'], 
                          f'vq_rotation_vae_{codebook_size}_val_total_loss': val_metrics['total_loss'], 
                          f'vq_rotation_vae_{codebook_size}_val_active': val_metrics['active']})
        current_val_loss = val_metrics['total_loss']

        # Check for improvement
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            num_bad_epochs = 0
            # Save the best model
            # torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            # print(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.6f}. Model saved.")
        else:
            num_bad_epochs += 1
            # print(f"Epoch {epoch+1}: No improvement in validation loss for {num_bad_epochs} epoch(s).")

        # Early stopping
        if num_bad_epochs >= args.patience:
            print(f"Early stopping triggered after {num_bad_epochs} epochs with no improvement.")
            break
        