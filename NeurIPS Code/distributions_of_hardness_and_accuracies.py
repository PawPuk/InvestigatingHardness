import argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from tqdm import tqdm

from utils import find_stragglers, identify_hard_samples_with_confidences_or_energies, initialize_models,\
    load_data_and_normalize, load_results, save_data, train


def compute_hardness_distribution(hard_target):
    """
    Computes the distribution of hard samples across different classes.

    Parameters:
    - hard_target (torch.Tensor): The tensor containing class labels of the hard samples.

    Returns:
    - distribution (dict): A dictionary with class labels as keys and the count of hard samples in each class as values.
    """
    unique_classes = torch.unique(hard_target)
    distribution = {}
    for cls in unique_classes:
        class_mask = (hard_target == cls)
        distribution[cls.item()] = class_mask.sum().item()
    return distribution


def load_data(dataset_name: str):
    """
    Load and normalize datasets based on the dataset name.
    """
    if dataset_name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError("Unsupported dataset")

    train_set = datasets.__dict__[dataset_name](root='./data', train=True, download=True, transform=transform)
    test_set = datasets.__dict__[dataset_name](root='./data', train=False, download=True, transform=transform)
    return train_set, test_set


def compute_class_accuracy(model, loader):
    """
    Compute per-class accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    accuracies = {i: class_correct[i] / class_total[i] for i in range(10)}
    return accuracies


def find_universal_stragglers(dataset: TensorDataset, filename: str,
                              threshold: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Load the results which contain lists of tensors with straggler flags
    hard_samples_indices = load_results(filename)

    # Check the device of the first tensor in the first list (assuming all tensors are on the same device)
    if len(hard_samples_indices) > 0 and len(hard_samples_indices[0]) > 0:
        device = hard_samples_indices[0][0].device
    else:
        raise ValueError("No straggler data found in the results.")

    num_samples = dataset.tensors[0].size(0)
    straggler_counts = torch.zeros(num_samples, dtype=torch.int32, device=device)

    # Aggregate the counts of straggler flags across all runs
    for run_list in hard_samples_indices:
        for tensor in run_list:
            straggler_counts += tensor.to(device).int()  # Ensure tensor is on the correct device

    # Identify indices that meet the threshold
    hard_indices = torch.where(straggler_counts >= threshold)[0].cpu()  # Move indices to CPU
    easy_indices = torch.where(straggler_counts < threshold)[0].cpu()  # Move indices to CPU

    # Extract the hard and easy data and targets from the TensorDataset
    hard_data = dataset.tensors[0][hard_indices]  # Indexing on CPU
    hard_target = dataset.tensors[1][hard_indices]  # Indexing on CPU
    easy_data = dataset.tensors[0][easy_indices]  # Indexing on CPU
    easy_target = dataset.tensors[1][easy_indices]  # Indexing on CPU

    return hard_data, hard_target, easy_data, easy_target


def main(dataset_name: str, thresholds: List[float], strategy: str, runs: int, depends_on_stragglers: bool):
    confidences_and_energies = load_results(f'Results/{dataset_name}_20_metrics.pkl')
    dataset = load_data_and_normalize(dataset_name, 70000)
    if dataset_name != 'CIFAR10':
        filename = f'Results/straggler_indices_{dataset_name}_20.pkl'
        _, hard_target, _, _ = find_universal_stragglers(dataset, filename)
    full_dataset = load_data_and_normalize(dataset_name, 70000)
    train_set, test_set = load_data(dataset_name)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    results = {'class_level_accuracies': [], 'hardness_distribution': []}
    if strategy == 'stragglers':
        results['hardness_distribution'].append(compute_hardness_distribution(hard_target))
    else:
        if depends_on_stragglers:
            _, hard_target, _, _ = identify_hard_samples_with_confidences_or_energies(confidences_and_energies,
                                                                                      full_dataset, strategy,
                                                                                      len(hard_target))
            results['hardness_distribution'].append(compute_hardness_distribution(hard_target))
        else:
            for threshold in thresholds:
                _, hard_target, _, _ = identify_hard_samples_with_confidences_or_energies(confidences_and_energies,
                                                                                          full_dataset, strategy,
                                                                                          int(70000*threshold))
                results['hardness_distribution'].append(compute_hardness_distribution(hard_target))
    print(f'Distribution of hardness among classes - {results["hardness_distribution"]}')
    for _ in tqdm(range(runs)):
        models, optimizers = initialize_models(dataset_name)
        model, optimizer = models[0], optimizers[0]
        if dataset_name == 'MNIST':
            epochs = 15
        elif dataset_name == 'KMNIST':
            epochs = 25
        elif dataset_name == 'FashionMNIST':
            epochs = 35
        else:
            epochs = 100
        train(dataset_name, model, train_loader, optimizer, epochs=epochs)
        accuracies = compute_class_accuracy(model, test_loader)
        print(f'Class-level accuracies - {accuracies}')
        results['class_level_accuracies'].append(accuracies)
    save_data(results, f'Results/distributions_{strategy}_{dataset_name}_{depends_on_stragglers}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST, KMNIST, and CIFAR10.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.05, 0.1, 0.2],
                        help='')
    parser.add_argument('--strategy', type=str, choices=['stragglers', 'confidence', 'energy'],
                        default='confidence', help='Strategy (method) to use for identifying hard samples.')
    parser.add_argument('--runs', type=int, default=20,
                        help='Specifies how many straggler sets will be computed for the experiment, and how many '
                             'networks will be trained per a straggler set (for every ratio in remaining_train_ratios. '
                             'The larger this value the higher the complexity and the statistical significance.')
    parser.add_argument('--depends_on_stragglers', action='store_true', default=False,
                        help='flag indicating whether we want to see the effect of changing the number of easy (False) '
                             'or hard (True) samples.')
    args = parser.parse_args()
    main(**vars(args))
