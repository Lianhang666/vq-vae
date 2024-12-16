import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_full = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )
    total_size=len(train_full)
    train_size=int(0.8*total_size)
    val_size=total_size-train_size
    
    train_dataset,val_dataset=random_split(train_full,[train_size,val_size])
    
    test_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader