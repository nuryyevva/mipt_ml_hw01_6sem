# MIPT ML Homework 01 (6th Semester)

This project is a machine learning pipeline for training and evaluating a ResNet-18 model on the CIFAR-10 dataset.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [Tests](#tests)
4. [Reproducibility with DVC](#reproducibility-with-dvc)

---

## Project Overview

This project implements a machine learning pipeline for the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. The pipeline includes:

- **Data Loading**: Loading and preprocessing the CIFAR-10 dataset.
- **Model Training**: Training a ResNet-18 model using PyTorch.
- **Evaluation**: Computing accuracy on the test dataset.
- **Logging**: Logging training metrics (e.g., loss, accuracy) using Weights & Biases (Wandb).
- **Reproducibility**: Using DVC (Data Version Control) to manage data and model versions.

---

## Setup

### Prerequisites
- Python 3.10 or higher
- `pip` for installing dependencies
- `make` for running commands

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nuryyevva/mipt_ml_hw01_6sem.git
   cd mipt_ml_hw01_6sem
   ```

2. Install the required dependencies:
   ```bash
   make install
   ```

   This command installs all the dependencies listed in `requirements.txt`.

3. Set up Weights & Biases (Wandb):
   - Create an account at [wandb.ai](https://wandb.ai/).
   - Log in to Wandb from the command line:
     ```bash
     wandb login
     ```

---

## Tests

Run the tests:
```bash
make test
```

This command runs all the tests in the `tests/` directory using `pytest` and coverage flags. The tests cover:
- Data loading and preprocessing.
- Model training and evaluation.
- Accuracy computation.

---

## Reproducibility with DVC

- Initialize DVC:
    ```bash
    dvc init
    ```
- Pull data and models:
  ```bash
  dvc pull
  ```
- Reproduce the pipeline:
  ```bash
  dvc repro
  ```

For more information, refer to the [DVC documentation](https://dvc.org/doc).

---

## Acknowledgments

- The CIFAR-10 dataset is provided by the Canadian Institute for Advanced Research (CIFAR).
- This project uses PyTorch and Weights & Biases for model training and logging.
