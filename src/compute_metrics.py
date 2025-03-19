import json

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from hparams import config


def compute_metrics() -> None:
    """
    Compute the accuracy of the trained model on the CIFAR-10 test dataset.
    The model is loaded from 'model.pt', and the accuracy is saved to 'final_metrics.json'.
    """
    # Define the transformation to be applied to the test data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),  # Normalize with CIFAR-10 mean and std
        ]
    )

    # Load the CIFAR-10 test dataset
    test_dataset = CIFAR10(
        root="CIFAR10/test",
        train=False,
        transform=transform,
        download=False,
    )

    # Create a DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"]
    )

    device = torch.device("cpu")

    # Load the pre-trained ResNet-18 model with 10 output classes (for CIFAR-10)
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(
        torch.load("model.pt")
    )  # Load the model weights from the saved file 'model.pt'
    model.to(device)  # Move the model to the specified device

    correct = (
        0.0  # Initialize a variable to keep track of the number of correct predictions
    )

    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        # Perform inference (no gradient computation)
        with torch.inference_mode():
            outputs = model(
                test_images
            )  # Forward pass: compute the model's predictions
            preds = torch.argmax(
                outputs, 1
            )  # Get the predicted class labels (indices of the maximum logits)
            correct += (
                preds == test_labels
            ).sum()  # Count the number of correct predictions

    accuracy = correct / len(test_dataset)

    # Save the accuracy to a JSON file
    with open("final_metrics.json", "w+") as f:
        json.dump({"accuracy": accuracy.item()}, f)  # Save accuracy as a JSON object
        print("\n", file=f)  # Add a newline for better readability in the file


if __name__ == "__main__":
    compute_metrics()
