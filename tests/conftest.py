import pytest
import torch
from torchvision.datasets import CIFAR10

from train import (
    get_datasets,
    get_dataloaders,
    get_model,
    train_one_batch,
    evaluate,
    train,
)


@pytest.fixture
def train_dataset() -> CIFAR10:
    """Fixture to load the CIFAR10 train dataset."""
    train_dataset, _ = get_datasets()
    return train_dataset


@pytest.fixture
def test_dataset() -> CIFAR10:
    """Fixture to load the CIFAR10 test dataset."""
    _, test_dataset = get_datasets()
    return test_dataset


@pytest.fixture
def train_loader(train_dataset: CIFAR10) -> torch.utils.data.DataLoader:
    """Fixture to create a train dataloader."""
    train_loader, _ = get_dataloaders(train_dataset, None, batch_size=32)
    return train_loader


@pytest.fixture
def test_loader(test_dataset: CIFAR10) -> torch.utils.data.DataLoader:
    """Fixture to create a test dataloader."""
    _, test_loader = get_dataloaders(None, test_dataset, batch_size=32)
    return test_loader


@pytest.fixture
def model() -> torch.nn.Module:
    """Fixture to create and return the model."""
    device = torch.device("cpu")
    return get_model(device)
