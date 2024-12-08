import os
import torch
import argparse
import wandb
from tqdm import tqdm

from src.models.vqvae import VQVAE
from src.models.classifier import vae_classifier, WarmUpCosine
from src.data.dataset import get_cifar10_dataloaders

BATCH_SIZE = 512
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05

EPOCHS = 100

def main():
    experiment_name = 'fine-tune'
    # codebook_sizes = [65536, 16384, 4096, 1024, 256, 64, 16]
    codebook_sizes = [65536]
    # model_types = ['vqvae', 'fsqvae', 'vqvae_rotation']
    model_types = ['vqvae']

    wandb.init(project="Classification_Experiment")

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(BATCH_SIZE, 4)
    train_set = train_loader.dataset

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    for codebook_size in codebook_sizes:
        for model_type in model_types:
            #set current wandb run name
            wandb.run.name = f'{model_type}_{codebook_size}_{experiment_name}'
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
            params = torch.load(model_path, map_location=torch.device('cpu'))
            model_struct.load_state_dict(params)
            model = vae_classifier(model_struct, n_classes=NUM_CLASSES)

            # Optimizer
            optim = torch.optim.AdamW(model.parameters(),
                                    lr=LEARNING_RATE * BATCH_SIZE / 256,
                                    betas=(0.9, 0.999),
                                    weight_decay=WEIGHT_DECAY)

            total_steps = int((len(train_set) / BATCH_SIZE) * EPOCHS)
            warmup_epoch_percentage = 0.15
            warmup_steps = int(total_steps * warmup_epoch_percentage)

            scheduler = WarmUpCosine(optim, total_steps=total_steps, warmup_steps=warmup_steps, learning_rate_base=LEARNING_RATE, warmup_learning_rate=0.0)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # if torch.cuda.device_count() > 1:
            #     print(f"Use {torch.cuda.device_count()} GPUs.")
            #     model = nn.DataParallel(model)
            model = model.to(device)


            step_count = 0
            optim.zero_grad()

            for e in range(EPOCHS):
                if experiment_name == 'linear_probe':
                    model.encoder.eval()
                    model.quantizer.eval()
                    model.head.train()
                else:
                    model.train()

                losses = []
                acces = []
                for img, label in tqdm(iter(train_loader)):
                    step_count += 1
                    img = img.to(device)
                    label = label.to(device)
                    logits = model(img)
                    loss = loss_fn(logits, label)
                    acc = acc_fn(logits, label)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    losses.append(loss.item())
                    acces.append(acc.item())
                scheduler.step()
                avg_train_loss = sum(losses) / len(losses)
                avg_train_acc = sum(acces) / len(acces)
                # print(f'Epoch {e} - avg_train_loss: {avg_train_loss}, avg_train_acc: {avg_train_acc}')
                wandb.log({
                    "train_loss": avg_train_loss,
                    "train_acc": avg_train_acc,
                    # "epoch": e
                })
                

            model.eval()
            with torch.no_grad():
                losses = []
                acces = []
                for img, label in tqdm(iter(test_loader)):
                    img = img.to(device)
                    label = label.to(device)
                    logits = model(img)
                    loss = loss_fn(logits, label)
                    acc = acc_fn(logits, label)
                    losses.append(loss.item())
                    acces.append(acc.item())
                avg_val_loss = sum(losses) / len(losses)
                avg_val_acc = sum(acces) / len(acces)
                wandb.log({
                    "val_loss": avg_val_loss,
                    "val_acc": avg_val_acc
                })


if __name__ == '__main__':
    main()
