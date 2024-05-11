import argparse
import pickle
from typing import List, Set, Tuple

import numpy as np
from numpy import array
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

from neural_network import BasicBlock, ResNet, OldSimpleNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()
EPOCHS = 500
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def identify_hard_samples_based_on_average_confidence(confidences: List[array],
                                                      threshold: float) -> Tuple[Set[int], Set[int]]:
    """Divide the CIFAR10 dataset into the easy and hard ('threshold' percent of the dataset) samples."""
    all_confidences = np.stack(confidences)
    average_confidences = np.mean(all_confidences, axis=0)
    sorted_indices = np.argsort(average_confidences)  # This sorts from lowest to highest
    hard_sample_indices = set(sorted_indices[:int(threshold*len(confidences[0]))])
    easy_sample_indices = set([x for x in range(len(confidences[0]))]).difference(hard_sample_indices)
    return easy_sample_indices, hard_sample_indices


def identify_hard_samples_based_on_overlap(confidences: List[List[array]],
                                           threshold: float) -> Tuple[Set[int], Set[int]]:
    """Divide the CIFAR10 dataset into the easy and hard ('threshold' percent of the dataset) samples."""
    low_conf_sets = []
    # Collect lowest confidence indices from selected models
    for model_index in [0, 1, 2, 3]:
        indices = torch.topk(torch.tensor(confidences[model_index]), int(len(confidences[0]) * threshold)).indices
        low_conf_sets.append(set(indices.numpy()))
    # Calculate the intersection of all low confidence sets
    hard_sample_indices = set.intersection(*low_conf_sets)
    easy_sample_indices = set([x for x in range(len(confidences[0]))]).difference(hard_sample_indices)
    return easy_sample_indices, hard_sample_indices


def identify_hard_samples(confidences: List[Tuple[int, float]], threshold: float) -> Tuple[Set[int], Set[int]]:
    """Divide the CIFAR10 dataset into the easy and hard ('threshold' percent of the dataset) samples."""
    confidences = confidences[0]
    # Corrected list comprehension to directly unpack index and confidence
    confidence_indices = [(confidence, index) for index, confidence in confidences]

    # Sort the list by confidence in ascending order to find the most uncertain samples
    sorted_confidence_indices = sorted(confidence_indices, key=lambda x: x[0], reverse=False)

    # Select the indices of the hard and easy samples based on the threshold
    num_hard_samples = int(threshold * len(confidences))
    hard_sample_indices = {index for _, index in sorted_confidence_indices[:num_hard_samples]}
    easy_sample_indices = {index for _, index in sorted_confidence_indices[num_hard_samples:]}

    # Calculate average, min, and max confidences for hard and easy samples
    hard_confidences = [confidence for confidence, index in sorted_confidence_indices[:num_hard_samples]]
    easy_confidences = [confidence for confidence, index in sorted_confidence_indices[num_hard_samples:]]

    print("Hard Samples:")
    print("Average Confidence:", np.mean(hard_confidences))
    print("Minimum Confidence:", np.min(hard_confidences))
    print("Maximum Confidence:", np.max(hard_confidences))

    print("Easy Samples:")
    print("Average Confidence:", np.mean(easy_confidences))
    print("Minimum Confidence:", np.min(easy_confidences))
    print("Maximum Confidence:", np.max(easy_confidences))

    return hard_sample_indices, easy_sample_indices


def load_dataset(flag: str) -> TensorDataset:
    """Load either CIFAR10 or MNIST dataset based on the 'flag'."""
    if flag == 'CIFAR':
        dataset_class = datasets.CIFAR10
        channel = 3  # CIFAR-10 images are RGB
    elif flag == 'MNIST':
        dataset_class = datasets.MNIST
        channel = 1  # MNIST images are grayscale
    else:
        raise ValueError("Unsupported dataset flag!")

    # Load the datasets
    train_dataset = dataset_class(root='./data', train=True, download=True,
                                  transform=transforms.ToTensor())
    test_dataset = dataset_class(root='./data', train=False, download=True,
                                 transform=transforms.ToTensor())

    # Concatenate train and test datasets
    full_data = torch.cat([train_dataset.data.unsqueeze(1).float(), test_dataset.data.unsqueeze(1).float()])
    full_targets = torch.cat([torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)])

    """# Shuffle the combined dataset
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(full_data))
    full_data, full_targets = full_data[torch.tensor(shuffled_indices)], full_targets[torch.tensor(shuffled_indices)]"""

    # Calculate mean and variance for normalization
    data_mean = full_data.mean(dim=(0, 2, 3)) / 255.0
    # data_std = full_data.std(dim=(0, 2, 3)) / 255.0
    data_std = torch.sqrt(torch.var(full_data, dim=(0, 2, 3)) / 255.0 ** 2 + 0.0000001)

    # Apply the normalization transform
    normalize_transform = transforms.Normalize(mean=data_mean.tolist(), std=data_std.tolist())
    full_data = normalize_transform(full_data / 255.0)  # Ensure scaling to [0, 1] before normalization

    return TensorDataset(full_data, full_targets)


def split_data(dataset_name: str, dataset: Dataset, hard_samples: Set[int], easy_samples: Set[int],
               sample_removal_rate: float, remove_hard: bool) -> Tuple[DataLoader, List[DataLoader]]:
    """Use the indices from 'easy_samples' and 'hard_samples' to divide the dataset into training and test set. Before
    creating a DataLoader remove sample_removal_rate hard/easy (depending on the 'remove_hard' flag) samples from the
    training set. Note that for test set we give 3 DataLoaders - one for easy samples, one for hard, and one for all."""
    # Randomly shuffle hard and easy samples
    hard_perm, easy_perm = torch.randperm(torch.tensor(list(hard_samples)).size(0)), \
        torch.randperm(torch.tensor(list(easy_samples)).size(0))
    if dataset_name == 'CIFAR':
        hard_data, hard_target = dataset[:][0][hard_perm].squeeze(1), dataset[:][1][hard_perm]
        easy_data, easy_target = dataset[:][0][easy_perm].squeeze(1), dataset[:][1][easy_perm]
    else:
        hard_data, hard_target = dataset[:][0][hard_perm], dataset[:][1][hard_perm]
        easy_data, easy_target = dataset[:][0][easy_perm], dataset[:][1][easy_perm]
    # Split hard and easy samples into train and test sets (use 80:20 training:test ratio)
    train_size_hard, train_size_easy = int(len(hard_data) * 0.8), int(len(easy_data) * 0.8)
    hard_train_data, hard_test_data = hard_data[:train_size_hard], hard_data[train_size_hard:]
    hard_train_target, hard_test_target = hard_target[:train_size_hard], hard_target[train_size_hard:]
    easy_train_data, easy_test_data = easy_data[:train_size_easy], easy_data[train_size_easy:]
    easy_train_target, easy_test_target = easy_target[:train_size_easy], easy_target[train_size_easy:]
    # Combine easy and hard samples into train and test data
    train_data = torch.cat((hard_train_data, easy_train_data), dim=0)
    train_targets = torch.cat((hard_train_target, easy_train_target), dim=0)
    # Shuffle the final train dataset
    train_permutation = torch.randperm(train_data.size(0))
    train_data, train_targets = train_data[train_permutation], train_targets[train_permutation]
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=len(train_data))
    # Create two test sets - one containing only hard samples, and the other only easy samples
    hard_and_easy_test_sets = [(hard_test_data, hard_test_target), (easy_test_data, easy_test_target)]
    full_test_data = torch.cat((hard_and_easy_test_sets[0][0], hard_and_easy_test_sets[1][0]), dim=0)
    full_test_targets = torch.cat((hard_and_easy_test_sets[0][1], hard_and_easy_test_sets[1][1]), dim=0)
    # Create 3 test loaders: 1) with all data samples; 2) with only hard data samples; 3) with only easy data samples
    test_loaders = []
    for data, target in [(full_test_data, full_test_targets)] + hard_and_easy_test_sets:
        test_loader = DataLoader(TensorDataset(data, target), batch_size=1000, shuffle=False)
        test_loaders.append(test_loader)
    return train_loader, test_loaders


def initialize_model(dataset_name: str) -> Tuple[torch.nn.Module, Adam]:
    if dataset_name == 'CIFAR':
        model = ResNet(BasicBlock, [3, 3, 3, 3]).to(DEVICE)
    else:
        model =OldSimpleNN(20, 2).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    return model, optimizer


def train(dataset_name: str, model: torch.nn.Module, loader: DataLoader, optimizer: Adam):
    for epoch in tqdm(range(EPOCHS)):
        if dataset_name == 'CIFAR':
            # Adjust the learning rate
            lr = 0.001 * (0.1 ** (epoch // 30))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # Proceed with training
        model.train()
        running_loss = 0.0
        for i, data in enumerate(loader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


def test(model: torch.nn.Module, loader: DataLoader) -> dict[str, float]:
    """ Measures the accuracy of the 'model' on the test set.

    :param model: model, which performance we want to evaluate
    :param loader: DataLoader containing test data
    :return: accuracy on the test set rounded to 2 decimal places
    """
    model.eval()
    num_classes = 10
    # Initialize metrics
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(DEVICE)
    precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(DEVICE)
    recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(DEVICE)
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(DEVICE)

    confidences = []  # List to store all max confidences per batch

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_confidences = probabilities.max(dim=1)[0]  # Get max confidence for each item in the batch
            confidences.extend(max_confidences.tolist())  # Store the confidences

            # Update metrics
            accuracy.update(outputs, target)
            precision.update(outputs, target)
            recall.update(outputs, target)
            f1_score.update(outputs, target)

    # Compute confidence statistics
    average_confidence = round(sum(confidences) / len(confidences), 4)
    min_confidence = round(min(confidences), 4)
    max_confidence = round(max(confidences), 4)

    # Print confidence statistics
    print(f"Average Confidence: {average_confidence}")
    print(f"Minimum Confidence: {min_confidence}")
    print(f"Maximum Confidence: {max_confidence}")

    # Compute final results
    accuracy_result = round(accuracy.compute().item() * 100, 2)
    precision_result = round(precision.compute().item() * 100, 2)
    recall_result = round(recall.compute().item() * 100, 2)
    f1_result = round(f1_score.compute().item() * 100, 2)
    return {'accuracy': accuracy_result, 'precision': precision_result, 'recall': recall_result, 'f1': f1_result}


def save_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def main(threshold: float, dataset_name: str, sample_removal_rates: List[float], remove_hard: bool, runs: int):
    results = load_results(f'../Results/{dataset_name}_1_metrics.pkl')
    hard_samples, easy_samples = identify_hard_samples(results, threshold)
    dataset = load_dataset(dataset_name)
    metrics = {setting: {sample_removal_rate: {metric: []
                                               for metric in ['accuracy', 'precision', 'recall', 'f1']}
                         for sample_removal_rate in sample_removal_rates}
               for setting in ['full', 'hard', 'easy']}
    for sample_removal_rate in sample_removal_rates:
        train_loader, test_loaders = split_data(dataset_name, dataset, hard_samples, easy_samples, sample_removal_rate,
                                                remove_hard)
        for _ in range(runs):
            model, optimizer = initialize_model(dataset_name)
            train(dataset_name, model, train_loader, optimizer)
            # Evaluate the model on test set
            for i, setting in enumerate(['full', 'hard', 'easy']):
                current_metrics = test(model, test_loaders[i])
                # Save the obtained metrics
                for metric_name, metric_values in current_metrics.items():
                    metrics[setting][sample_removal_rate][metric_name].append(metric_values)
                print(current_metrics['accuracy'])
    print(metrics)
    save_data(metrics, f"../Results/{remove_hard}_{threshold}_metrics.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='')
    parser.add_argument('--dataset_name', type=str, default='MNIST', choices=['MNIST', 'CIFAR'])
    parser.add_argument('--sample_removal_rates', nargs='+', type=float,
                        default=[0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0],
                        help='')
    parser.add_argument('--remove_hard', action='store_true', default=False,
                        help='Flag indicating whether we want to see the effect of changing the number of easy (False) '
                             'or hard (True) samples in the training set on the generalization.')
    parser.add_argument('--runs', type=int, default=3,
                        help='')
    args = parser.parse_args()
    main(**vars(args))
