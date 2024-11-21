import torch
import argparse
from src.train import train_model
from src.test import test_model
from src.models.vqvae import VQVAE
from src.data.dataset import get_cifar10_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_sample', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dims', type=int, default=256)
    parser.add_argument('--codebook_size', type=int, default=512)
    parser.add_argument('--decay', type=float, default=0.8)
    parser.add_argument('--commitment_weight', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)

    print('Arguments:', parser.parse_args())

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(args.batch_size, args.workers)
    
    # Model
    model = VQVAE(
        in_channels=3,
        hidden_dims=args.hidden_dims,
        codebook_size=args.codebook_size,
        decay=args.decay,
        commitment_weight=args.commitment_weight,
        quantized_type='fsq'
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    train_model(model, train_loader, val_loader, optimizer, device, args)
    
    # Testing
    test_model(model, test_loader, device,args)

if __name__ == '__main__':
    main()