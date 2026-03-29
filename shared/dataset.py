import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


BATCH_SIZE = 64


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "sampleNetwork")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_dataloaders(data_root=None, batch_size=BATCH_SIZE):
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    os.makedirs(data_root, exist_ok=True)

    transform_conf = get_transforms()

    train_dataset = datasets.FashionMNIST(
        data_root,
        train=True,
        download=True,
        transform=transform_conf,
    )

    test_dataset = datasets.FashionMNIST(
        data_root,
        train=False,
        download=True,
        transform=transform_conf,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_dataset.classes

    return train_dataset, test_dataset, train_loader, test_loader, class_names