import os
import pickle
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
EPOCHS = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()
HARD_IMBALANCE_DIR = 'hardnessImbalance/'
CONFIDENCES_SAVE_DIR = "confidences/"
DATA_SAVE_DIR = "data/"
MODEL_SAVE_DIR = "models/"
ACCURACIES_SAVE_DIR = "accuracies/"
DIVISIONS_SAVE_DIR = 'dataset_divisions/'
CORRELATIONS_SAVE_DIR = 'correlations/'
for directory in [HARD_IMBALANCE_DIR, CONFIDENCES_SAVE_DIR, DATA_SAVE_DIR, MODEL_SAVE_DIR, ACCURACIES_SAVE_DIR,
                  DIVISIONS_SAVE_DIR, CORRELATIONS_SAVE_DIR]:
    os.makedirs(directory, exist_ok=True)


def save_data(data, file_name: str):
    def move_to_cpu(data):
        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {key: move_to_cpu(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [move_to_cpu(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(move_to_cpu(item) for item in data)
        else:
            return data

    # Move all tensors in the data to CPU before saving
    data = move_to_cpu(data)
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def calculate_mean_std(accuracies: List[float]) -> Tuple[float, float]:
    return np.mean(accuracies), np.std(accuracies)


def load_full_data_and_normalize(dataset_name: str) -> TensorDataset:
    """ Used to load the data from common datasets available in torchvision, and normalize them. The normalization
    is based on the mean and std of a random subset of the dataset of the size subset_size.

    :param dataset_name: name of the dataset to load. It has to be available in `torchvision.datasets`
    :return: random, normalized subset of dataset_name of size subset_size with (noise_rate*subset_size) labels changed
    to introduce label noise
    """
    # Load the train and test datasets based on the 'dataset_name' parameter
    train_dataset = getattr(datasets, dataset_name)(root="./data", train=True, download=True,
                                                    transform=transforms.ToTensor())
    test_dataset = getattr(datasets, dataset_name)(root="./data", train=False, download=True,
                                                   transform=transforms.ToTensor())
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2).float()  # (N, H, W, C) -> (N, C, H, W)
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2).float()
    else:
        train_data = train_dataset.data.unsqueeze(1).float()  # Add channel dimension for MNIST/other datasets
        test_data = test_dataset.data.unsqueeze(1).float()
    # Concatenate train and test datasets
    full_data = torch.cat([train_data, test_data])
    full_targets = torch.cat([torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)])
    # Normalize the data
    data_means = torch.mean(full_data, dim=(0, 2, 3)) / 255.0
    data_vars = torch.sqrt(torch.var(full_data, dim=(0, 2, 3)) / 255.0 ** 2 + EPSILON)
    # Apply the calculated normalization to the subset
    normalize_transform = transforms.Normalize(mean=data_means, std=data_vars)
    # TODO: maybe weird results come from applying normalization to data when computing metrics (not for training)...
    normalized_subset_data = normalize_transform(full_data / 255.0)  # TODO: see the impact of removing normalization
    return TensorDataset(normalized_subset_data, full_targets)


def load_data_and_normalize(dataset_name: str) -> Tuple[TensorDataset, TensorDataset]:
    """
    Load and normalize the dataset. Optionally, modify the dataset to be long-tailed for the training set.

    :param dataset_name: name of the dataset to load. It has to be available in `torchvision.datasets`.
    :return: Two TensorDatasets - one for training data and another for test data.
    """
    train_dataset = getattr(datasets, dataset_name)(root="./data", train=True, download=True,
                                                    transform=transforms.ToTensor())
    test_dataset = getattr(datasets, dataset_name)(root="./data", train=False, download=True,
                                                   transform=transforms.ToTensor())
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2).float()
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2).float()
    else:
        train_data = train_dataset.data.unsqueeze(1).float()
        test_data = test_dataset.data.unsqueeze(1).float()

    train_targets = torch.tensor(train_dataset.targets)
    test_targets = torch.tensor(test_dataset.targets)

    data_means = torch.mean(train_data, dim=(0, 2, 3)) / 255.0
    data_vars = torch.sqrt(torch.var(train_data, dim=(0, 2, 3)) / 255.0 ** 2 + EPSILON)

    normalize_transform = transforms.Normalize(mean=data_means, std=data_vars)
    normalized_train_data = normalize_transform(train_data / 255.0)
    normalized_test_data = normalize_transform(test_data / 255.0)

    return TensorDataset(normalized_train_data, train_targets), TensorDataset(normalized_test_data, test_targets)


def train(dataset_name: str, model: torch.nn.Module, loader: DataLoader, optimizer: Union[Adam, SGD],
          epochs: int = EPOCHS):
    # TODO: modify to save learning-based hardness metrics like forgetting or memorization
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for data, target in loader:
            inputs, labels = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()
        if dataset_name == 'CIFAR10':
            scheduler.step()


def class_level_test(model: torch.nn.Module, loader: DataLoader, num_classes: int) -> List[float]:
    """Compute accuracy per class."""
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            for i in range(num_classes):
                correct_per_class[i] += (predictions[target == i] == i).sum().item()
                total_per_class[i] += (target == i).sum().item()
    accuracies = (correct_per_class.float() / total_per_class.float()).tolist()
    return accuracies


def test(model: torch.nn.Module, loader: DataLoader) -> float:
    """Measures the accuracy of the 'model' on the test set.

    :param model: The model to evaluate.
    :param loader: DataLoader containing test data.
    :return: Dictionary with accuracy on the test set rounded to 2 decimal places.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            total += target.size(0)  # Increment the total count
            correct += (predicted == target).sum().item()  # Increment the correct count
    accuracy = 100 * correct / total
    return round(accuracy, 2)


def dataset_to_tensors(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a PyTorch Dataset (or Subset) into two Tensors: one for the data and one for the targets.

    :param dataset: The dataset to convert, can be a Dataset or Subset object.
    :return: A tuple containing two tensors (data_tensor, target_tensor).
    """
    data_list, target_list = [], []
    for data, target in dataset:
        data_list.append(data.unsqueeze(0))  # Add an extra dimension to stack properly later
        target_list.append(torch.tensor(target))
    return torch.cat(data_list, dim=0), torch.tensor(target_list)


def combine_and_split_data(hard_dataset: Subset, easy_dataset: Subset,
                           dataset_name: str) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Combines easy and hard data samples into a single dataset, then splits it into train and test sets while
    maintaining the same easy:hard ratio in both splits and keeping the overall train:test ratio as defined.

    :param hard_dataset: identified hard samples (Subset)
    :param easy_dataset: identified easy samples (Subset)
    :param dataset_name: name of the used dataset
    :return: A list containing 2 training DataLoaders (for hard, easy), and 2 test loaders (for hard, easy, all data).
    """
    train_test_ratio = 6 / 7 if dataset_name == 'MNIST' else 5 / 6 if dataset_name == 'CIFAR10' else 8 / 10

    # Convert Subsets into Tensors
    hard_data, hard_target = dataset_to_tensors(hard_dataset)
    easy_data, easy_target = dataset_to_tensors(easy_dataset)

    # Randomly shuffle hard and easy samples
    hard_perm, easy_perm = torch.randperm(hard_data.size(0)), torch.randperm(easy_data.size(0))
    hard_data, hard_target = hard_data[hard_perm], hard_target[hard_perm]
    easy_data, easy_target = easy_data[easy_perm], easy_target[easy_perm]

    # Calculate the number of training and test samples
    train_size_hard = int(len(hard_data) * train_test_ratio)
    train_size_easy = int(len(easy_data) * train_test_ratio)

    # Split hard and easy samples into training and test sets
    hard_train_data, hard_test_data = hard_data[:train_size_hard], hard_data[train_size_hard:]
    hard_train_target, hard_test_target = hard_target[:train_size_hard], hard_target[train_size_hard:]
    easy_train_data, easy_test_data = easy_data[:train_size_easy], easy_data[train_size_easy:]
    easy_train_target, easy_test_target = easy_target[:train_size_easy], easy_target[train_size_easy:]

    # Combine easy and hard samples into full training and test data
    train_data = torch.cat((hard_train_data, easy_train_data), dim=0)
    train_targets = torch.cat((hard_train_target, easy_train_target), dim=0)
    test_data = torch.cat((hard_test_data, easy_test_data), dim=0)
    test_targets = torch.cat((hard_test_target, easy_test_target), dim=0)

    # Shuffle the full training split
    train_permutation = torch.randperm(train_data.size(0))
    train_data, train_targets = train_data[train_permutation], train_targets[train_permutation]

    # Create DataLoaders for full, hard-only, full easy, and easy subset training sets
    train_loaders = [
        DataLoader(TensorDataset(hard_train_data, hard_train_target), batch_size=BATCH_SIZE, shuffle=True),  # Hard-only
        DataLoader(TensorDataset(easy_train_data, easy_train_target), batch_size=BATCH_SIZE, shuffle=True),  # Easy-only
    ]

    # Create DataLoaders for the test sets (hard, easy, all)
    test_loaders = [
        DataLoader(TensorDataset(hard_test_data, hard_test_target), batch_size=len(hard_test_data)),  # Hard test set
        DataLoader(TensorDataset(easy_test_data, easy_test_target), batch_size=len(easy_test_data)),  # Easy test set
        DataLoader(TensorDataset(test_data, test_targets), batch_size=len(test_data))  # Full test set
    ]

    return train_loaders, test_loaders
