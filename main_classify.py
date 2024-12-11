import os
import torch
import argparse
import wandb
from tqdm import tqdm

from src.models.vqvae import VQVAE
from src.models.classifier import vae_classifier, WarmUpCosine
from src.data.dataset import get_cifar10_dataloaders
from torch.profiler import profile, record_function, ProfilerActivity

BATCH_SIZE = 64
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05

# Give a large max epoch number to rely on early stopping.
MAX_EPOCHS = 500
PATIENCE = 10      # Early stopping patience
ENABLE_AMP = True  # Automatic Mixed Precision

def main():
    experiment_names = ['lp', 'ft']
    # task: lph from 65536 to 16
    codebook_sizes = [65536, 16384, 4096, 1024, 256, 64, 16]
    # task: llh from 16 to 65536
    codebook_sizes = [4096]
    model_types = ['vqvae_rotation', 'vqvae', 'fsqvae']

    # wandb.init(project="Classification_Experiment")

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(BATCH_SIZE, 8)
    train_set = train_loader.dataset

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    for experiment_name in experiment_names:
        for codebook_size in codebook_sizes:
            for model_type in model_types:
                run = wandb.init(project="Classification_Experiment_test", 
                                 name=f'{model_type}_{codebook_size}_{experiment_name}',
                                 reinit=True)
                # reconstruct model
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

                # load the parameters
                model_path_pre = f'./model_{codebook_size}'
                model_name = f'{model_type}_{codebook_size}.pt'
                model_path = os.path.join(model_path_pre, model_name)
                # params = torch.load(model_path, map_location=torch.device('cpu'))
                params = torch.load(model_path)
                model_struct.load_state_dict(params)
                model = vae_classifier(model_struct, n_classes=NUM_CLASSES)

                # Optimizer
                optim = torch.optim.AdamW(model.parameters(),
                                          lr=LEARNING_RATE,
                                          betas=(0.9, 0.999),
                                          weight_decay=WEIGHT_DECAY)

                total_steps = int((len(train_set) / BATCH_SIZE) * MAX_EPOCHS)
                warmup_epoch_percentage = 0.15
                warmup_steps = int(total_steps * warmup_epoch_percentage)

                scheduler = WarmUpCosine(optim, total_steps=total_steps, warmup_steps=warmup_steps, learning_rate_base=LEARNING_RATE, warmup_learning_rate=0.0)

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)

                step_count = 0
                optim.zero_grad()

                best_val_acc = 0.0
                best_val_loss = float('inf')
                epochs_no_improve = 0
                best_model_state = None

                for e in range(MAX_EPOCHS):
                    if experiment_name == 'lp':
                        model.encoder.eval()
                        model.quantizer.eval()
                        model.head.train()
                    else:
                        model.train()

                    # Training loop
                    losses = []
                    acces = []
                    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    for img, label in tqdm(iter(train_loader), desc=f"Training Epoch {e}/{MAX_EPOCHS}"):
                        with torch.autocast(device_type="cuda", enabled=ENABLE_AMP):
                            step_count += 1
                            img = img.to(device)
                            label = label.to(device)
                            with profile(activities=activities, record_shapes=True) as prof:
                                logits = model(img)
                                loss = loss_fn(logits, label)
                                acc = acc_fn(logits, label)
                                loss.backward()
                                optim.step()
                                optim.zero_grad()
                            losses.append(loss.item())
                            acces.append(acc.item())
                    scheduler.step()
                    sort_by_keyword = device + "_time_total"
                    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

                    avg_train_loss = sum(losses) / len(losses)
                    avg_train_acc = sum(acces) / len(acces)

                    # Validation loop
                    model.eval()
                    with torch.no_grad():
                        losses = []
                        acces = []
                        for img, label in tqdm(iter(val_loader), desc=f"Validation Epoch {e}/{MAX_EPOCHS}"):
                            img = img.to(device)
                            label = label.to(device)
                            logits = model(img)
                            loss = loss_fn(logits, label)
                            acc = acc_fn(logits, label)
                            losses.append(loss.item())
                            acces.append(acc.item())
                        avg_val_loss = sum(losses) / len(losses)
                        avg_val_acc = sum(acces) / len(acces)

                    # Log metrics
                    wandb.log({
                        "train_loss": avg_train_loss,
                        "train_acc": avg_train_acc,
                        "val_loss": avg_val_loss,
                        "val_acc": avg_val_acc,
                        "epoch": e
                    })

                    exit()

                    # Early Stopping Check
                    # if avg_val_acc > best_val_acc:
                    #     best_val_acc = avg_val_acc
                    #     epochs_no_improve = 0
                    #     best_model_state = model.state_dict()
                    # else:
                    #     epochs_no_improve += 1

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_val_acc = avg_val_acc
                        epochs_no_improve = 0
                        best_model_state = model.state_dict()
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= PATIENCE:
                        print(f"Early stopping triggered at epoch {e}. Best val_acc: {best_val_acc:.4f}")
                        break

                # After training/early stopping, load the best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)

                # Final test evaluation
                model.eval()
                with torch.no_grad():
                    losses = []
                    acces = []
                    for img, label in tqdm(iter(test_loader), desc="Testing"):
                        img = img.to(device)
                        label = label.to(device)
                        logits = model(img)
                        loss = loss_fn(logits, label)
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
