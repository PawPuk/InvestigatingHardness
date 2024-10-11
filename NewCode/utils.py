import pickle
import random
from typing import Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset
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


def initialize_models(dataset_name: str, model_type: str, width: int):
    if dataset_name == 'CIFAR10':
        # Create three instances of each model type with fresh initializations
        if model_type == 'simple':
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False).to(DEVICE)
        else:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=False).to(DEVICE)
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        if model_type == 'simple':
            model = SimpleMLP(width).to(DEVICE)
        else:
            model = LeNet().to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.001)
    return model, optimizer


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


def load_data_and_normalize(dataset_name: str, to_grayscale: bool = False,
                            apply_pca: bool = False, label_noise: float = 0.0) -> Tuple[TensorDataset, TensorDataset]:
    """
    Loads and normalizes the dataset. Optionally reduces dimensionality and adds label noise.

    :param dataset_name: Name of the dataset to load.
    :param to_grayscale: If True and dataset_name is 'CIFAR10', it will transform images to grayscale.
    :param apply_pca: Whether to apply PCA for dimensionality reduction.
    :param label_noise: Fraction of training labels to be randomly changed (0 to 1).
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

    # Apply label noise to the training set
    if label_noise > 0.0:
        train_targets = add_label_noise(train_targets, noise_fraction=label_noise, num_classes=len(torch.unique(train_targets)))

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


def add_label_noise(targets: torch.Tensor, noise_fraction: float, num_classes: int) -> torch.Tensor:
    """
    Introduce label noise by changing a fraction of the labels to random incorrect labels.

    :param targets: Original labels as a tensor.
    :param noise_fraction: Fraction of labels to be randomly changed.
    :param num_classes: Number of classes in the dataset.
    :return: Tensor with noisy labels.
    """
    num_samples = len(targets)
    num_noisy_samples = int(noise_fraction * num_samples)

    # Randomly select a subset of indices to corrupt labels
    noisy_indices = random.sample(range(num_samples), num_noisy_samples)

    # Corrupt labels by randomly changing them to a different class
    for idx in noisy_indices:
        original_label = targets[idx].item()
        new_label = original_label
        while new_label == original_label:
            new_label = random.randint(0, num_classes - 1)  # Random class different from the original
        targets[idx] = new_label

    return targets


def train(dataset_name: str, model: torch.nn.Module, training_loader, test_loader, optimizer: Union[Adam, SGD],
          epochs: int = 10):
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0

        for data, target in training_loader:
            inputs, labels = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss, test_accuracy, _, _ = test(model, test_loader)

        if dataset_name == 'CIFAR10':
            scheduler.step()

        train_losses.append(train_loss / len(training_loader.dataset))
        train_accuracies.append(100. * correct / len(training_loader.dataset))
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    return train_losses, train_accuracies, test_losses, test_accuracies


def test(model: torch.nn.Module, loader) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Measures the accuracy and loss per class on the test set.

    :param model: The model to evaluate.
    :param loader: DataLoader containing test data.
    :return: Tuple containing overall test loss, test accuracy, class-level losses, and accuracies.
    """
    model.eval()
    test_loss, correct = 0, 0
    class_correct = np.zeros(10)  # Assuming 10 classes
    class_total = np.zeros(10)
    class_losses = np.zeros(10)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            batch_loss = CRITERION(outputs, target).item()

            test_loss += batch_loss * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            # Gather class-wise statistics
            for i in range(len(target)):
                label = target[i].item()
                pred_label = predicted[i].item()
                class_correct[label] += (pred_label == label)
                class_total[label] += 1
                class_losses[label] += batch_loss

    # Calculate overall test loss and accuracy
    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)

    # Compute class-level accuracies and losses
    class_accuracies = 100. * class_correct / class_total
    class_losses /= class_total

    return test_loss, accuracy, class_losses, class_accuracies

