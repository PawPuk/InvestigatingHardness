from typing import List, Tuple, Union
import pickle

import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()


def save_data(data, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


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


def train(dataset_name: str, model: torch.nn.Module, loader: DataLoader, optimizer: Union[Adam, SGD],
          epochs: int = EPOCHS):
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
                           remove_hard: bool, sample_removal_rate: float) -> Tuple[DataLoader, List[DataLoader]]:
    """
    Combines easy and hard data samples into a single dataset, then splits it into train and test sets.

    :param hard_dataset: identified hard samples (Subset)
    :param easy_dataset: identified easy samples (Subset)
    :param remove_hard: flag indicating whether to reduce the number of hard (True) or easy (False) samples
    :param sample_removal_rate: ratio of samples to be removed from the train set
    :return: train loader and 3 test loaders - 1) with all data samples; 2) with only hard samples; and 3) with only
    easy samples.
    """
    # Convert Subsets into Tensors
    hard_data, hard_target = dataset_to_tensors(hard_dataset)
    easy_data, easy_target = dataset_to_tensors(easy_dataset)
    # Randomly shuffle hard and easy samples
    hard_perm, easy_perm = torch.randperm(hard_data.size(0)), torch.randperm(easy_data.size(0))
    hard_data, hard_target = hard_data[hard_perm], hard_target[hard_perm]
    easy_data, easy_target = easy_data[easy_perm], easy_target[easy_perm]
    # Split data into initial train/test sets
    train_size_hard = int(len(hard_data) * (1 - (10000 / (len(hard_data) + len(easy_data)))))
    train_size_easy = int(len(easy_data) * (1 - (10000 / (len(hard_data) + len(easy_data)))))
    hard_train_data, hard_test_data = hard_data[:train_size_hard], hard_data[train_size_hard:]
    hard_train_target, hard_test_target = hard_target[:train_size_hard], hard_target[train_size_hard:]
    easy_train_data, easy_test_data = easy_data[:train_size_easy], easy_data[train_size_easy:]
    easy_train_target, easy_test_target = easy_target[:train_size_easy], easy_target[train_size_easy:]
    # Validate sample_removal_rate
    if not 0 <= sample_removal_rate <= 1:
        raise ValueError(f'The parameter sample_removal_rate must be in [0, 1]; {sample_removal_rate} not allowed.')
    # Adjust the number of samples in the training set based on sample_removal_rate
    if remove_hard:
        reduced_hard_train_size = int(train_size_hard * (1 - sample_removal_rate))
        reduced_easy_train_size = train_size_easy
    else:
        reduced_hard_train_size = train_size_hard
        reduced_easy_train_size = int(train_size_easy * (1 - sample_removal_rate))
    print(f'Proceeding with {reduced_hard_train_size} hard samples, and {reduced_easy_train_size} easy samples.')
    # Combine easy and hard samples into train data
    train_data = torch.cat((hard_train_data[:reduced_hard_train_size],
                            easy_train_data[:reduced_easy_train_size]), dim=0)
    train_targets = torch.cat((hard_train_target[:reduced_hard_train_size],
                               easy_train_target[:reduced_easy_train_size]), dim=0)
    # Shuffle the final train dataset
    train_permutation = torch.randperm(train_data.size(0))
    train_data, train_targets = train_data[train_permutation], train_targets[train_permutation]
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=128, shuffle=True)
    # Create two test sets - one containing only hard samples, and the other only easy samples
    hard_and_easy_test_sets = [(hard_test_data, hard_test_target), (easy_test_data, easy_test_target)]
    full_test_data = torch.cat((hard_and_easy_test_sets[0][0], hard_and_easy_test_sets[1][0]), dim=0)
    full_test_targets = torch.cat((hard_and_easy_test_sets[0][1], hard_and_easy_test_sets[1][1]), dim=0)
    # Create 3 test loaders: 1) with all data samples; 2) with only hard data samples; 3) with only easy data samples
    test_loaders = []
    for data, target in [(full_test_data, full_test_targets)] + hard_and_easy_test_sets:
        test_loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
        test_loaders.append(test_loader)
    return train_loader, test_loaders
