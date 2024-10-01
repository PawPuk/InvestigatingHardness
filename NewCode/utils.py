import os
import pickle
from typing import List, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from neural_networks import LeNet, SimpleMLP


EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
EPOCHS = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()


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


def initialize_models(dataset_name: str, model_type: str):
    if dataset_name == 'CIFAR10':
        # Create three instances of each model type with fresh initializations
        if model_type == 'simple':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False).to(DEVICE)
        else:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=False).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-4)
    else:
        if model_type == 'simple':
            model = SimpleMLP().to(DEVICE)
        else:
            model = LeNet().to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.001)
    return model, optimizer


def calculate_mean_std(accuracies: List[float]) -> Tuple[float, float]:
    return np.mean(accuracies), np.std(accuracies)


def reduce_dimensionality(dataset_name: str, data: torch.Tensor, apply_pca: bool = False) -> torch.Tensor:
    """
    Optionally reduces the dimensionality of the data to match the data/dimensionality ratio of MNIST if the dataset is
    CIFAR10 or CIFAR100. For other datasets like MNIST, it simply returns the data as is.

    :param dataset_name: Name of the dataset.
    :param data: Data tensor of shape (N, C, H, W).
    :param apply_pca: Whether to apply PCA for dimensionality reduction.
    :return: Data tensor with reduced dimensionality or original data.
    """
    if apply_pca and dataset_name in ['CIFAR10', 'CIFAR100']:
        N = data.shape[0]
        # Flatten the data: (N, C, H, W) -> (N, C*H*W)
        data_flat = data.view(N, -1)
        # Perform PCA to reduce dimensions to 672
        pca = PCA(n_components=672)
        data_reduced = pca.fit_transform(data_flat.numpy())
        # Convert back to tensor
        data_reduced = torch.from_numpy(data_reduced).float()
        return data_reduced  # Shape: (N, 672)
    else:
        return data  # Return data as is


def load_full_data_and_normalize(dataset_name: str, to_grayscale: bool = False,
                                 apply_pca: bool = False) -> TensorDataset:
    """Loads and normalizes the full dataset (train + test). Optionally reduces dimensionality.

    :param dataset_name: Name of the dataset to load.
    :param to_grayscale: If True and dataset_name is 'CIFAR10', transforms images to grayscale.
    :param apply_pca: Whether to apply PCA for dimensionality reduction.
    :return: A TensorDataset containing the normalized data and targets.
    """
    # Load the train and test datasets
    train_dataset = getattr(datasets, dataset_name)(root="./data", train=True, download=True)
    test_dataset = getattr(datasets, dataset_name)(root="./data", train=False, download=True)

    # Convert datasets into tensors
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2).float()  # (N, H, W, C) -> (N, C, H, W)
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2).float()
        if to_grayscale and dataset_name == 'CIFAR10':
            train_data = train_data.mean(dim=1, keepdim=True)  # Convert to single channel
            test_data = test_data.mean(dim=1, keepdim=True)    # Convert to single channel
    else:
        train_data = torch.tensor(train_dataset.data).unsqueeze(1).float()
        test_data = torch.tensor(test_dataset.data).unsqueeze(1).float()

    # Concatenate train and test data
    full_data = torch.cat([train_data, test_data], dim=0)
    full_targets = torch.cat([torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)], dim=0)

    # Reduce dimensionality if necessary
    full_data = reduce_dimensionality(dataset_name, full_data, apply_pca=apply_pca)

    # Normalize the data
    if apply_pca and dataset_name in ['CIFAR10', 'CIFAR100']:
        # Data is flattened to (N, 784) after PCA
        mean = torch.mean(full_data, dim=0)
        std = torch.std(full_data, dim=0) + EPSILON
        normalized_full_data = (full_data - mean) / std
    else:
        # Data is in (N, C, H, W)
        data_means = torch.mean(full_data, dim=(0, 2, 3)) / 255.0
        data_stds = torch.sqrt(torch.var(full_data, dim=(0, 2, 3)) / 255.0 ** 2 + EPSILON)
        # Apply normalization
        normalize_transform = transforms.Normalize(mean=data_means, std=data_stds)
        normalized_full_data = normalize_transform(full_data / 255.0)

    return TensorDataset(normalized_full_data, full_targets)


def load_data_and_normalize(dataset_name: str, to_grayscale: bool = False,
                            apply_pca: bool = False) -> Tuple[TensorDataset, TensorDataset]:
    """
    Loads and normalizes the dataset. Optionally reduces dimensionality.

    :param dataset_name: Name of the dataset to load.
    :param to_grayscale: If True and dataset_name is 'CIFAR10', it will transform images to grayscale.
    :param apply_pca: Whether to apply PCA for dimensionality reduction.
    :return: Two TensorDatasets - one for training data and another for test data.
    """
    # Load the train and test datasets
    train_dataset = getattr(datasets, dataset_name)(root="./data", train=True, download=True)
    test_dataset = getattr(datasets, dataset_name)(root="./data", train=False, download=True)

    # Convert datasets into tensors
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        train_data = torch.as_tensor(train_dataset.data).permute(0, 3, 1, 2).float()
        test_data = torch.as_tensor(test_dataset.data).permute(0, 3, 1, 2).float()
        if to_grayscale:
            train_data = train_data.mean(dim=1, keepdim=True)
            test_data = test_data.mean(dim=1, keepdim=True)
    else:
        # Assuming dataset like MNIST where images are single channel (grayscale)
        train_data = torch.as_tensor(train_dataset.data).unsqueeze(1).float()
        test_data = torch.as_tensor(test_dataset.data).unsqueeze(1).float()

    train_targets = torch.as_tensor(train_dataset.targets)
    test_targets = torch.as_tensor(test_dataset.targets)

    # Reduce dimensionality if necessary
    train_data = reduce_dimensionality(dataset_name, train_data, apply_pca=apply_pca)
    test_data = reduce_dimensionality(dataset_name, test_data, apply_pca=apply_pca)

    # Normalize the data
    if apply_pca and dataset_name in ['CIFAR10', 'CIFAR100']:
        # Data is flattened to (N, 784) after PCA
        mean = torch.mean(train_data, dim=0)
        std = torch.std(train_data, dim=0) + EPSILON
        normalized_train_data = (train_data - mean) / std
        normalized_test_data = (test_data - mean) / std  # Use the same mean and std as training data
    else:
        # Data is in (N, C, H, W)
        data_means = torch.mean(train_data, dim=(0, 2, 3)) / 255.0
        data_stds = torch.sqrt(torch.var(train_data, dim=(0, 2, 3)) / 255.0 ** 2 + EPSILON)
        # Apply normalization
        normalize_transform = transforms.Normalize(mean=data_means, std=data_stds)
        normalized_train_data = normalize_transform(train_data / 255.0)
        normalized_test_data = normalize_transform(test_data / 255.0)

    return TensorDataset(normalized_train_data, train_targets), TensorDataset(normalized_test_data, test_targets)


def train(dataset_name: str, model: torch.nn.Module, loader, optimizer: Union[Adam, SGD],
          epochs: int = EPOCHS):
    # TODO: modify to save learning-based hardness metrics like forgetting or memorization
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses, accuracies = [], []
    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for data, target in loader:
            inputs, labels = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        if dataset_name == 'CIFAR10':
            scheduler.step()
        train_losses.append(train_loss / len(loader.dataset))
        accuracies.append(100. * correct / len(loader.dataset))
    return train_losses, accuracies


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


def test(model: torch.nn.Module, loader) -> Tuple[float, float]:
    """Measures the accuracy of the 'model' on the test set.

    :param model: The model to evaluate.
    :param loader: DataLoader containing test data.
    :return: Dictionary with accuracy on the test set rounded to 2 decimal places.
    """
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            test_loss += F.cross_entropy(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            correct += predicted.eq(target.view_as(predicted)).sum().item()
    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return test_loss, accuracy


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
