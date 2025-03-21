import torch
import pytest

from train import compute_accuracy


def test_arange_elems():
    """Test that torch.arange generates a tensor with the correct elements."""
    arr = torch.arange(0, 10, dtype=torch.float)
    assert torch.allclose(arr[-1], torch.tensor([9.0]))


def test_div_zero():
    """Test division by zero in PyTorch, which should result in infinity."""
    a = torch.zeros(1, dtype=torch.long)
    b = torch.ones(1, dtype=torch.long)

    assert torch.isinf(b / a)


def test_div_zero_python():
    """Test division by zero in Python, which should raise a ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        1 / 0


def test_accuracy():
    """Test the compute_accuracy function with simple cases."""
    preds = torch.randint(0, 2, size=(100,))
    targets = preds.clone()

    assert compute_accuracy(preds, targets) == 1.0

    preds = torch.tensor([1, 2, 3, 0, 0, 0])
    targets = torch.tensor([1, 2, 3, 4, 5, 6])

    assert compute_accuracy(preds, targets) == 0.5


@pytest.mark.parametrize(
    "preds,targets,result",
    [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), 1.0),
        (torch.tensor([1, 2, 3]), torch.tensor([0, 0, 0]), 0.0),
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 0]), 2 / 3),
    ],
)
def test_accuracy_parametrized(preds: torch.Tensor, targets: torch.Tensor, result: float):
    """
    Test the compute_accuracy function with multiple input cases using pytest parametrization.

    :param preds: Tensor of predictions.
    :param targets: Tensor of ground truth labels.
    :param result: Expected accuracy value.
    """
    assert torch.allclose(compute_accuracy(preds, targets), torch.tensor([result]), rtol=0, atol=1e-5)
