import os
import pytest
import torch
from unittest.mock import patch
from torchvision.datasets import CIFAR10

from train import evaluate, train, train_one_batch


def test_train_dataset(train_dataset: CIFAR10):
    """Test that the train dataset is loaded correctly."""
    assert len(train_dataset) == 50000  # CIFAR10 train set has 50,000 train images (10000 test images)
    assert isinstance(train_dataset[0][0], torch.Tensor)  # Check that images are tensors
    assert isinstance(train_dataset[0][1], int)  # Check that labels are integers


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_train_on_one_batch(
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
):
    """Test one training step on a single batch."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU not available")

    # Move model to the specified device
    model = model.to(device)

    # Get one batch of data
    images, labels = next(iter(train_loader))

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train on one batch
    loss = train_one_batch(model, images, labels, criterion, optimizer, device)

    # Assertions
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0  # Loss should be positive


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training(
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
):
    """Test the full training loop, including main(), with Wandb mocked."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU not available")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Train for one epoch
    for images, labels in train_loader:
        loss = train_one_batch(model, images, labels, criterion, optimizer, device)

    # Evaluate the model
    accuracy = evaluate(model, test_loader, device)

    # Assertions
    assert isinstance(accuracy, torch.Tensor)
    assert 0 <= accuracy <= 1  # Accuracy should be between 0 and 1

    # Mock Wandb to avoid logging to the cloud
    with patch("train.wandb.log") as mock_wandb_log:
        train()
        assert mock_wandb_log.call_count > 0  # Ensure metrics were logged

    # Verify that the model file was saved
    assert os.path.exists("model.pt")

    # Verify that run_id.txt was written
    with open("run_id.txt", "r") as f:
        run_id = f.read().strip()
        assert len(run_id) > 0  # Check that the run ID is not empty
