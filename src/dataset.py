from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir="./data",
    batch_size=128,
    train_subset=5000,
    val_subset=1000,
    num_workers=2,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = Subset(
        train_dataset,
        range(min(train_subset, len(train_dataset))),
    )

    val_dataset = Subset(
        val_dataset,
        range(min(val_subset, len(val_dataset))),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
