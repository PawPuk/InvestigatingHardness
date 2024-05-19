import argparse
from typing import List

import torch
from torch.utils.data import DataLoader
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


def main(dataset_name: str, thresholds: List[float], strategy: str, runs: int):
    confidences_and_energies = load_results(f'Results/{dataset_name}_20_metrics.pkl')
    full_dataset = load_data_and_normalize(dataset_name, 70000)
    train_set, test_set = load_data(dataset_name)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    results = []
    for _ in tqdm(range(runs)):
        if dataset_name != 'CIFAR10':
            _, hard_target, _, _, stragglers = find_stragglers(full_dataset)
            # Compute the class-level number of stragglers and convert it to threshold
            aggregated_stragglers = [int(tensor.sum().item()) for tensor in stragglers]
            thresholds = [sum(aggregated_stragglers)]
        hardness_distributions = []
        for threshold in thresholds:
            print(threshold)
            if isinstance(threshold, float):
                # Threshold is based on the training set, the size of which is 10k in all used datasets
                threshold = int(10000 * threshold)
            print(threshold)
            if strategy != 'stragglers':
                _, hard_target, _, _ = identify_hard_samples_with_confidences_or_energies(confidences_and_energies,
                                                                                          full_dataset, strategy,
                                                                                          threshold)
            hardness_distribution = compute_hardness_distribution(hard_target)
            hardness_distributions.append(hardness_distribution)
            print(f'Distribution of hardness among classes - {hardness_distribution}')
        models, optimizers = initialize_models(dataset_name)
        model, optimizer = models[0], optimizers[0]
        train(dataset_name, model, train_loader, optimizer)
        accuracies = compute_class_accuracy(model, test_loader)
        print(f'Class-level accuracies - {accuracies}')
        results.append({
            'hardness_distribution': hardness_distributions,
            'class_level_accuracies': accuracies
        })
    save_data(results, f'distributions_{strategy}_{dataset_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.05, 0.1, 0.2],
                        help='')
    parser.add_argument('--strategy', type=str, choices=['stragglers', 'confidence', 'energy'],
                        default='stragglers', help='Strategy (method) to use for identifying hard samples.')
    parser.add_argument('--runs', type=int, default=10,
                        help='Specifies how many straggler sets will be computed for the experiment, and how many '
                             'networks will be trained per a straggler set (for every ratio in remaining_train_ratios. '
                             'The larger this value the higher the complexity and the statistical significance.')
    args = parser.parse_args()
    main(**vars(args))
