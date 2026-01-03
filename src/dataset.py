import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import load_config

BATCH_SIZE = 64
NUM_WORKERS = 2

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    ),
])

def get_dataloaders():
    cfg = load_config()
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    root = cfg["data"]["root"]

    pin = torch.cuda.is_available()
    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )

    return train_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    train_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)
    print("Train batches:", len(train_loader))
