import os
import json

from compute_metrics import compute_metrics


def test_compute_metrics():
    """Test compute_metrics."""
    # Ensure the model exists
    assert os.path.exists("model.pt"), "model.pt not found. Run test_training first."

    compute_metrics()

    # Verify that final_metrics.json was created
    assert os.path.exists("final_metrics.json")

    # Verify the contents of final_metrics.json
    with open("final_metrics.json", "r") as f:
        metrics = json.load(f)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1  # Accuracy should be between 0 and 1
