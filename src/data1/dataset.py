from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(batch_size=64, root='./data/raw'):
    train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader