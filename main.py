import os
import torch
import argparse
import wandb
from src.train import train_model
from src.test import test_model
from src.models.vqvae import VQVAE
from src.data.dataset import get_cifar10_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_sample', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dims', type=int, default=64)
    # parser.add_argument('--codebook_size', type=int, default=[16, 64, 256, 1024, 4096, 16384, 65536], nargs='+')
    parser.add_argument('--codebook_size', type=int, default=[65536, 16384, 4096, 1024, 256, 64, 16], nargs='+')
    parser.add_argument('--decay', type=float, default=0.8)
    parser.add_argument('--commitment_weight', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--experiment', type=str, default="VQ")

    print('Arguments:', parser.parse_args())

    return parser.parse_args()

def main():
    args = parse_args()
    # wandb.login(key='b969446317599f594b3ad992680f7e8db1a4bfb8')
    wandb.init(project='Reconstruction_Experiment', name='Reconstruction_Experiment_0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(args.batch_size, args.workers)

    # Initialize test results dictionary
    test_results = {
        'vqvae': {'codebook_size': [], 'mse_loss': [], 'fid_score': [], 'codebook_usage': []},
        'fsqvae': {'codebook_size': [], 'mse_loss': [], 'fid_score': [], 'codebook_usage': []},
        'vqvae_rotation': {'codebook_size': [], 'mse_loss': [], 'fid_score': [], 'codebook_usage': []}
    }

    # For each of the experiments, iterate through all codebook sizes
    for codebook_size in args.codebook_size:
        os.makedirs(f'model_{codebook_size}', exist_ok=True)

        # Training VQ-VAE
        model_type = 'vqvae'
        print("Training VQ-VAE")
        model = VQVAE(
            in_channels=3,
            hidden_dims=args.hidden_dims,
            codebook_size=codebook_size,
            decay=args.decay,
            commitment_weight=args.commitment_weight,
            quantized_type='vq'
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_model(model, train_loader, val_loader, optimizer, device, codebook_size, args)
        test_result = test_model(model, test_loader, device, codebook_size, model_type, args)
        # wandb log the modeltype + codebooksize and results.
        wandb.log({f'{model_type}_{codebook_size}': test_result})

        torch.save(model.state_dict(), f'model_{codebook_size}/{model_type}_{codebook_size}.pt')
        test_results[model_type]['codebook_size'].append(codebook_size)
        test_results[model_type]['mse_loss'].append(test_result['mse_loss'])
        test_results[model_type]['fid_score'].append(test_result['fid_score'])
        test_results[model_type]['codebook_usage'].append(test_result['codebook_usage'])

        # Training FSQ-VAE
        model_type = 'fsqvae'
        print("Training FSQ-VAE")
        model = VQVAE(
            in_channels=3,
            hidden_dims=args.hidden_dims,
            codebook_size=codebook_size,
            decay=args.decay,
            commitment_weight=args.commitment_weight,
            quantized_type='fsq'
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_model(model, train_loader, val_loader, optimizer, device, codebook_size, args)
        test_result = test_model(model, test_loader, device, codebook_size, model_type, args)
        wandb.log({f'{model_type}_{codebook_size}': test_result})
        torch.save(model.state_dict(), f'model_{codebook_size}/{model_type}_{codebook_size}.pt')
        test_results[model_type]['codebook_size'].append(codebook_size)
        test_results[model_type]['mse_loss'].append(test_result['mse_loss'])
        test_results[model_type]['fid_score'].append(test_result['fid_score'])
        test_results[model_type]['codebook_usage'].append(test_result['codebook_usage'])

        # Training VQ-VAE with Rotation
        model_type = 'vqvae_rotation'
        print("Training VQ-VAE with Rotation")
        model = VQVAE(
            in_channels=3,
            hidden_dims=args.hidden_dims,
            codebook_size=codebook_size,
            decay=args.decay,
            commitment_weight=args.commitment_weight,
            quantized_type='vq',
            rotation=True
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_model(model, train_loader, val_loader, optimizer, device, codebook_size, args)
        test_result = test_model(model, test_loader, device, codebook_size, model_type, args)
        wandb.log({f'{model_type}_{codebook_size}': test_result})
        torch.save(model.state_dict(), f'model_{codebook_size}/{model_type}_{codebook_size}.pt')
        test_results[model_type]['codebook_size'].append(codebook_size)
        test_results[model_type]['mse_loss'].append(test_result['mse_loss'])
        test_results[model_type]['fid_score'].append(test_result['fid_score'])
        test_results[model_type]['codebook_usage'].append(test_result['codebook_usage'])

    # Create a wandb Table
    table = wandb.Table(columns=["codebook_size", "model_type", "mse_loss", "fid_score", "codebook_usage"])

    # Populate the table with your test results
    for model_type in test_results:
        for i in range(len(test_results[model_type]['codebook_size'])):
            codebook_size = test_results[model_type]['codebook_size'][i]
            mse_loss = test_results[model_type]['mse_loss'][i]
            fid_score = test_results[model_type]['fid_score'][i]
            codebook_usage = test_results[model_type]['codebook_usage'][i]
            table.add_data(codebook_size, model_type, mse_loss, fid_score, codebook_usage)

    # Log the table and create plots
    wandb.log({'test_results_table': table})
    wandb.log({
        'mse_loss_plot': wandb.plot_table(
            'wandb/line',
            table,
            {'x': 'codebook_size', 'y': 'mse_loss', 'groupKey': 'model_type'},
            {'title': 'MSE Loss vs Codebook Size'}
        ),
        'fid_score_plot': wandb.plot_table(
            'wandb/line',
            table,
            {'x': 'codebook_size', 'y': 'fid_score', 'groupKey': 'model_type'},
            {'title': 'FID Score vs Codebook Size'}
        ),
        'codebook_usage_plot': wandb.plot_table(
            'wandb/line',
            table,
            {'x': 'codebook_size', 'y': 'codebook_usage', 'groupKey': 'model_type'},
            {'title': 'Codebook Usage vs Codebook Size'}
        )
    })

if __name__ == '__main__':
    main()
