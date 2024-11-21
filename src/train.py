import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from datetime import datetime
from .utils.metrics import FIDcalculator

######################train######################################
def train_one_epoch(model,train_loader,optimizer,device,epoch,args):
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
                'active %': indices.unique().numel() / args.codebook_size * 100
            })
    #calculate the average loss and return 
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_commit_loss = total_commit_loss / len(train_loader)
    avg_total_loss = total_loss / len(train_loader)
    return {
        'recon_loss': avg_recon_loss,
        'commit_loss': avg_commit_loss,
        'total_loss': avg_total_loss
    }
    
#################validation#################################
def validate_one_epoch(model,val_loader,device,epoch,args):
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
                loss=recon_loss+commit_loss
                # Update metrics
                total_recon_loss += recon_loss.item()
                total_commit_loss += commit_loss.item()
                total_loss += loss.item()
                # Update progress bar
                pbar.set_postfix({
                    'recon_loss': total_recon_loss / (batch_idx + 1),
                    'commit_loss': total_commit_loss / (batch_idx + 1),
                    'total_loss': total_loss / (batch_idx + 1),
                    'active %': indices.unique().numel() / args.codebook_size * 100
                })
                recon_images.extend(recon_batch)
                real_images.extend(data)
    fid_calculator = FIDcalculator(device=device)
    fid_score = fid_calculator.calculate_fid(real_images, recon_images,args.num_sample)
    print(f'FID Score: {fid_score:.2f}')
    avg_recon_loss = total_recon_loss / len(val_loader)
    avg_commit_loss = total_commit_loss / len(val_loader)
    avg_total_loss = total_loss / len(val_loader)
    
    return {
        'recon_loss': avg_recon_loss,
        'commit_loss': avg_commit_loss,
        'total_loss': avg_total_loss,
        'fid_score': fid_score
    }
   
def train_model(model,train_loader,val_loader,optimizer,device,args):
    checkpoint_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_fid = float('inf')
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        val_metrics = validate_one_epoch(model, val_loader, device, epoch, args)
        


          
# def train_model(model, train_loader,val_loader,optimizer, device, args):
#     """Train the VQ-VAE model."""
#     model.train()
#     # Create checkpoint directory
#     checkpoint_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
#     os.makedirs(checkpoint_dir, exist_ok=True)
    
#     best_fid = float('inf')
    
#     for epoch in range(args.epochs):
#         total_recon_loss = 0
#         total_commit_loss = 0
        
#         with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}') as pbar:
#             for batch_idx, (data, _) in enumerate(pbar):
#                 data = data.to(device)
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 recon_batch, commit_loss, _ = model(data)
                
#                 # Calculate losses
#                 recon_loss = F.mse_loss(recon_batch, data)
#                 loss = recon_loss + commit_loss
                
#                 # Backward pass
#                 loss.backward()
#                 optimizer.step()
                
#                 # Update metrics
#                 total_recon_loss += recon_loss.item()
#                 total_commit_loss += commit_loss.item()
                
#                 # Update progress bar
#                 pbar.set_postfix({
#                     'recon_loss': total_recon_loss / len(train_loader),
#                     'commit_loss': total_commit_loss / (batch_idx + 1)
#                 })
        
#         # Calculate FID score and save checkpoint every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             fid_score = calculate_fid(model, train_loader, device)
#             print(f'FID Score: {fid_score:.2f}')
            
#             # Save checkpoint if best FID
#             if fid_score < best_fid:
#                 best_fid = fid_score
#                 checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pt')
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'fid_score': fid_score,
#                 }, checkpoint_path)