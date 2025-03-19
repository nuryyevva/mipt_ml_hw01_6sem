import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm, trange

from hparams import config


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute accuracy given predictions and targets.

    :param preds: A tensor containing the predicted class labels (indices of the maximum logits).
    :param targets: A tensor containing the ground truth class labels.

    :return: A scalar tensor containing the accuracy (ratio of correct predictions to total predictions).
    """
    result = (targets == preds).float().mean()
    return result


def get_datasets() -> tuple[CIFAR10, CIFAR10]:
    """
    Load and return CIFAR10 train and test datasets.

    :return: A tuple containing the CIFAR10 train dataset and test dataset.
    """
    # Define a transformation pipeline for the dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),  # Normalize with CIFAR-10 mean and std
        ]
    )

    train_dataset = CIFAR10(
        root="CIFAR10/train",  # Directory where the training data is stored
        train=True,  # Load the training split
        transform=transform,  # Apply the defined transformations
        download=True,  # Download the dataset if it doesn't exist
    )

    test_dataset = CIFAR10(
        root="CIFAR10/test",  # Directory where the test data is stored
        train=False,  # Load the training split
        transform=transform,  # Apply the defined transformations
        download=True,  # Download the dataset if it doesn't exist
    )

    return train_dataset, test_dataset


def get_dataloaders(
    train_dataset: CIFAR10 | None, test_dataset: CIFAR10 | None, batch_size: int
) -> tuple[torch.utils.data.DataLoader | None, torch.utils.data.DataLoader | None]:
    """
    Create and return train and test dataloaders.

    :param train_dataset: The CIFAR10 training dataset (can be None if not needed).
    :param test_dataset: The CIFAR10 test dataset (can be None if not needed).
    :param batch_size: The batch size for the DataLoader.

    :return: A tuple containing the train DataLoader and test DataLoader (either can be None if the corresponding
             dataset is None).
    """
    train_loader = None
    test_loader = None

    if train_dataset is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size
        )

    return train_loader, test_loader


def get_model(device: torch.device) -> nn.Module:
    """
    Create and return the model.

    :param device: The device (e.g., CPU or GPU) to which the model should be moved.

    :return: The ResNet-18 model configured for CIFAR-10 classification.
    """
    model = resnet18(
        pretrained=False,  # Do not use pre-trained weights
        num_classes=10,  # Number of output classes (CIFAR-10 has 10 classes)
        zero_init_residual=config[
            "zero_init_residual"
        ],  # Use zero initialization for residual blocks if specified
    )
    model.to(device)
    wandb.watch(model)
    return model


def train_one_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Train the model on one batch of data.

    :param model: The neural network model to be trained.
    :param images: A batch of input images.
    :param labels: A batch of ground truth labels corresponding to the images.
    :param criterion: The loss function used to compute the loss.
    :param optimizer: The optimizer used to update the model's parameters.
    :param device: The device (e.g., CPU or GPU) on which the computation is performed.

    :return: The computed loss for the batch.
    """
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)  # Forward pass: compute predictions
    loss = criterion(outputs, labels)  # Compute the loss

    loss.backward()  # Backward pass: compute gradients
    optimizer.step()  # Update model parameters
    optimizer.zero_grad()  # Clear gradients for the next iteration

    return loss


def evaluate(
    model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> torch.Tensor:
    """
    Evaluate the model on the test dataset.

    :param model: The neural network model to be evaluated.
    :param test_loader: The DataLoader for the test dataset.
    :param device: The device (e.g., CPU or GPU) on which the computation is performed.

    :return: A scalar tensor containing the accuracy of the model on the test dataset.
    """
    all_preds = []  # Store all predictions
    all_labels = []  # Store all ground truth labels

    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        with torch.inference_mode():  # Disable gradient computation for evaluation
            outputs = model(test_images)  # Forward pass: compute predictions
            preds = torch.argmax(
                outputs, 1
            )  # Get the predicted class labels (indices of the maximum logits)

            all_preds.append(preds)
            all_labels.append(test_labels)

    # Concatenate all predictions and labels, then compute accuracy
    accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))
    return accuracy


def train() -> None:
    """
    Main function to run the training pipeline.

    Initializes Wandb, loads datasets, trains the model, evaluates it periodically,
    and saves the trained model and Wandb run ID.
    """
    wandb.init(
        config=config, project="mipt_ml_hw01_6sem_project", name="baseline"
    )  # Initialize Wandb for tracking
    train_dataset, test_dataset = get_datasets()
    train_loader, test_loader = get_dataloaders(
        train_dataset, test_dataset, config["batch_size"]
    )

    device = torch.device("cpu")
    model = get_model(device)  # Initialize the model

    criterion = (
        nn.CrossEntropyLoss()
    )  # Define the loss function (CrossEntropyLoss for classification)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )  # Define the optimizer (AdamW with learning rate and weight decay from config)

    for epoch in trange(config["epochs"]):  # Loop over the number of epochs
        for i, (images, labels) in enumerate(
            tqdm(train_loader)
        ):  # Loop over batches in the training DataLoader
            loss = train_one_batch(model, images, labels, criterion, optimizer, device)

            # Evaluate the model every 100 batches
            if i % 100 == 0:
                accuracy = evaluate(
                    model, test_loader, device
                )  # Compute accuracy on the test set
                metrics = {"test_acc": accuracy, "train_loss": loss}  # Log metrics
                wandb.log(
                    metrics,
                    step=epoch * len(train_dataset)
                    + (i + 1) * config["batch_size"],  # Log step in Wandb
                )

    torch.save(model.state_dict(), "model.pt")  # Save the trained model

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)  # Save the Wandb run ID to a file


if __name__ == "__main__":
    train()
