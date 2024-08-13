from typing import Union
import pickle

import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()


def save_data(data, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_data_and_normalize(dataset_name: str) -> TensorDataset:
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
    if dataset_name == 'CIFAR10':
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2).float()
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2).float()
    else:
        train_data = train_dataset.data.unsqueeze(1).float()
        test_data = test_dataset.data.unsqueeze(1).float()
    # Concatenate train and test datasets
    full_data = torch.cat([train_data, test_data])
    full_targets = torch.cat([torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)])
    # Shuffle the combined dataset
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(full_data))
    full_data, full_targets = full_data[torch.tensor(shuffled_indices)], full_targets[torch.tensor(shuffled_indices)]
    # Normalize the data
    data_means = torch.mean(full_data, dim=(0, 2, 3)) / 255.0
    data_vars = torch.sqrt(torch.var(full_data, dim=(0, 2, 3)) / 255.0 ** 2 + EPSILON)
    # Apply the calculated normalization to the subset
    normalize_transform = transforms.Normalize(mean=data_means, std=data_vars)
    normalized_subset_data = normalize_transform(full_data / 255.0)
    return TensorDataset(normalized_subset_data, full_targets)


def train(dataset_name: str, model: torch.nn.Module, loader: DataLoader, optimizer: Union[Adam, SGD], epochs=EPOCHS):
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


def test(model: torch.nn.Module, loader: DataLoader) -> float:
    """Measures the accuracy of the 'model' on the test set.

    :param model: The model to evaluate.
    :param loader: DataLoader containing test data.
    :return: Dictionary with accuracy on the test set rounded to 2 decimal places.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            total += target.size(0)  # Increment the total count
            correct += (predicted == target).sum().item()  # Increment the correct count
    accuracy = 100 * correct / total
    return round(accuracy, 2)
