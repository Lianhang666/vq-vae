import os
import torch
import argparse
import wandb
from tqdm import tqdm

from src.models.vqvae import VQVAE
from src.models.classifier import vae_classifier, WarmUpCosine
from src.data.dataset import get_cifar10_dataloaders

BATCH_SIZE = 128
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.05

# Give a large max epoch number to rely on early stopping.
MAX_EPOCHS = 500
PATIENCE = 200     # Early stopping patience

###################################
# 修改开始：引入autocast和GradScaler #
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
###################################

def main():
    experiment_names = ['ft']
    codebook_sizes = [16384]
    # codebook_sizes = [16, 64, 256, 1024, 4096, 16384, 65536]
    model_types = ['vqvae', 'fsqvae']

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(BATCH_SIZE, 4)
    train_set = train_loader.dataset

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    for experiment_name in experiment_names:
        for codebook_size in codebook_sizes:
            for model_type in model_types:
                run = wandb.init(project="Classification_FT_16384", 
                                 name=f'{model_type}_{codebook_size}_{experiment_name}',
                                 reinit=True)
                if model_type == 'vqvae':
                    model_struct = VQVAE(
                        in_channels=3,
                        hidden_dims=64,
                        codebook_size=codebook_size,
                        decay=0.8,
                        commitment_weight=1.0,
                        quantized_type='vq'
                    )
                elif model_type == 'fsqvae':
                    model_struct = VQVAE(
                        in_channels=3,
                        hidden_dims=64,
                        codebook_size=codebook_size,
                        decay=0.8,
                        commitment_weight=1.0,
                        quantized_type='fsq'
                    )
                elif model_type == 'vqvae_rotation':
                    model_struct = VQVAE(
                        in_channels=3,
                        hidden_dims=64,
                        codebook_size=codebook_size,
                        decay=0.8,
                        commitment_weight=1.0,
                        quantized_type='vq',
                        rotation=True
                    )

                model_path_pre = f'./model_{codebook_size}'
                model_name = f'{model_type}_{codebook_size}.pt'
                model_path = os.path.join(model_path_pre, model_name)
                params = torch.load(model_path)
                model_struct.load_state_dict(params)
                model = vae_classifier(model_struct, n_classes=NUM_CLASSES)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                if torch.cuda.device_count() > 1:
                    print(f"Using {torch.cuda.device_count()} GPUs!")
                    model = torch.nn.DataParallel(model)

                optim = torch.optim.AdamW(model.parameters(),
                                          lr=LEARNING_RATE,
                                          betas=(0.9, 0.999),
                                          weight_decay=WEIGHT_DECAY)

                total_steps = int((len(train_set) / BATCH_SIZE) * MAX_EPOCHS)
                warmup_epoch_percentage = 0.1
                warmup_steps = int(total_steps * warmup_epoch_percentage)

                scheduler = WarmUpCosine(optim, total_steps=total_steps, warmup_steps=warmup_steps, learning_rate_base=LEARNING_RATE, warmup_learning_rate=0.0)

                step_count = 0
                optim.zero_grad()

                ###################################
                # 修改开始：初始化GradScaler实例     #
                scaler = GradScaler()
                ###################################

                best_val_acc = 0.0
                best_val_loss = float('inf')
                epochs_no_improve = 0
                best_model_state = None

                for e in range(MAX_EPOCHS):
                    if experiment_name == 'lp':
                        (model.module if hasattr(model, 'module') else model).encoder.eval()
                        (model.module if hasattr(model, 'module') else model).quantizer.eval()
                        (model.module if hasattr(model, 'module') else model).head.train()
                    else:
                        model.train()

                    losses = []
                    acces = []
                    for img, label in tqdm(iter(train_loader), desc=f"Training Epoch {e}/{MAX_EPOCHS}"):
                        step_count += 1
                        img = img.to(device)
                        label = label.to(device)
                        
                        ###################################
                        # 修改开始：使用autocast进行前向计算 #
                        with autocast(device_type="cuda"):
                            logits = model(img)
                            loss = loss_fn(logits, label)
                        ###################################

                        acc = acc_fn(logits, label)

                        ###################################
                        # 修改开始：使用scaler进行梯度缩放和更新 #
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()
                        ###################################

                        optim.zero_grad()
                        losses.append(loss.item())
                        acces.append(acc.item())
                    scheduler.step()
                    avg_train_loss = sum(losses) / len(losses)
                    avg_train_acc = sum(acces) / len(acces)

                    model.eval()
                    with torch.no_grad():
                        losses = []
                        acces = []
                        for img, label in tqdm(iter(val_loader), desc=f"Validation Epoch {e}/{MAX_EPOCHS}"):
                            img = img.to(device)
                            label = label.to(device)
                            ###################################
                            # 修改开始：验证过程也可使用autocast   #
                            with autocast(device_type="cuda"):
                                logits = model(img)
                                loss = loss_fn(logits, label)
                            ###################################
                            acc = acc_fn(logits, label)
                            losses.append(loss.item())
                            acces.append(acc.item())
                        avg_val_loss = sum(losses) / len(losses)
                        avg_val_acc = sum(acces) / len(acces)

                    wandb.log({
                        "train_loss": avg_train_loss,
                        "train_acc": avg_train_acc,
                        "val_loss": avg_val_loss,
                        "val_acc": avg_val_acc,
                        "epoch": e
                    })

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_val_acc = avg_val_acc
                        epochs_no_improve = 0
                        best_model_state = (model.module if hasattr(model, 'module') else model).state_dict()
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= PATIENCE:
                        print(f"Early stopping triggered at epoch {e}. Best val_acc: {best_val_acc:.4f}")
                        break

                if best_model_state is not None:
                    (model.module if hasattr(model, 'module') else model).load_state_dict(best_model_state)

                model.eval()
                with torch.no_grad():
                    losses = []
                    acces = []
                    for img, label in tqdm(iter(test_loader), desc="Testing"):
                        img = img.to(device)
                        label = label.to(device)
                        ###################################
                        # 修改开始：测试过程也可使用autocast   #
                        with autocast(device_type="cuda"):
                            logits = model(img)
                            loss = loss_fn(logits, label)
                        ###################################
                        acc = acc_fn(logits, label)
                        losses.append(loss.item())
                        acces.append(acc.item())
                    avg_test_loss = sum(losses) / len(losses)
                    avg_test_acc = sum(acces) / len(acces)
                    wandb.log({
                        "test_loss": avg_test_loss,
                        "test_acc": avg_test_acc
                    })


if __name__ == '__main__':
    main()
