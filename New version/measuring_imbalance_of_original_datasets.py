import argparse
from collections import defaultdict
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader

from compute_confidences import compute_curvatures, compute_disjuncts, compute_proximity_metrics
from train_ensembles import EnsembleTrainer
import utils as u

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def extract_hard_samples(metrics: List[List[float]], labels: List[int], invert: List[bool], dataset_name: str,
                         threshold: float = 0.05) -> Dict:
    """
    Extract the 'threshold' percentage of the hardest samples for each metric and compute the average of each metric
    for every class.

    :param metrics: List of lists containing metrics (proximity and curvature) for all samples in the dataset.
    :param labels: List of class labels corresponding to the samples.
    :param invert: List of booleans indicating whether to invert the selection (i.e., select highest instead of lowest)
                   for each metric.
    :param dataset_name: Name of the used dataset.
    :param threshold: The percentage of samples to consider (default: 0.05 for the hardest samples).
    :return: A dictionary containing:
             - 'class_distributions': A list of dictionaries, each containing class labels as keys and the count of
               the hardest samples as values for each metric.
             - 'class_averages': A list of dictionaries, each containing class labels as keys and the average metric
               value as values for each metric.
    """
    num_metrics = len(metrics)
    class_distributions = []
    class_averages = []

    for metric_idx in range(num_metrics):
        selected_metric = metrics[metric_idx]
        num_samples = len(selected_metric)
        num_hard_samples = int(threshold * num_samples)

        # Sort the samples by the selected metric
        if invert[metric_idx]:
            sorted_indices = np.argsort(selected_metric)[-num_hard_samples:]  # Select top samples for harder metrics
        else:
            sorted_indices = np.argsort(selected_metric)[:num_hard_samples]  # Select bottom samples for easier metrics
        hard_samples_indices = sorted_indices
        # Compute the distribution of these hard samples across different classes
        class_distribution = defaultdict(int)
        for idx in hard_samples_indices:
            class_label = labels[idx]
            class_distribution[class_label] += 1
        class_distributions.append(class_distribution)

        # Compute the average of each metric for every class
        class_average = defaultdict(list)
        for metric_value, label in zip(selected_metric, labels):
            class_average[label].append(metric_value)
        # Compute the mean of each class
        for class_label, values in class_average.items():
            class_average[class_label] = list(np.mean(values))
        class_averages.append(class_average)

        # Plot and save sorted metric values
        plt.figure(figsize=(10, 6))
        sorted_metric = sorted(selected_metric)
        plt.plot(sorted_metric, marker='o', linestyle='-')
        plt.title(f'Metric {metric_idx + 1} Sorted Values')
        plt.xlabel('Sample Index (Sorted)')
        plt.ylabel('Metric Value')
        plt.grid(True)

        # Ensure the directory exists
        output_dir = 'metric_plots'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{dataset_name}metric_{metric_idx + 1}_distribution.pdf'))
        plt.close()

    return {
        'class_distributions': class_distributions,
        'class_averages': class_averages
    }


def compare_metrics_to_class_accuracies(class_distributions: List[Dict[int, int]],
                                        avg_class_accuracies: np.ndarray,
                                        num_classes: int,
                                        output_filename: str):
    """
    Compare the class-level distribution of hard samples to the class-level accuracies
    for each metric by computing Pearson Correlation Coefficient (PCC) and plotting the results.

    :param class_distributions: List of dictionaries, each containing class labels as keys and the count of hardest
    samples as values for each metric.
    :param avg_class_accuracies: The average accuracies for each class.
    :param num_classes: The number of classes in the dataset.
    :param output_filename: The filename for saving the PCC bar plot.
    :return: A list of PCCs for each metric.
    """
    correlations = []
    metric_abbreviations = [
        'SCD', 'OCD', 'CR', 'CSC', 'COC', 'CDR',
        'ASD', 'AOD', 'AAD', 'ADR', 'PSK',
        'POK', 'ASC', 'AOC', 'AAC', 'GC',
        'MC', 'CCS', 'GCS', 'DCS'
    ]  # Abbreviations for each metric to keep plot readable.

    # Compute PCC for each metric
    for i, class_distribution in enumerate(class_distributions):
        class_level_distribution = [class_distribution.get(cls, 0) for cls in range(num_classes)]

        # Normalize the distribution to [0, 1]
        normalized_distribution = (class_level_distribution - np.min(class_level_distribution)) / \
                                  (np.max(class_level_distribution) - np.min(class_level_distribution))

        # Compute Pearson correlation coefficient
        correlation, _ = pearsonr(avg_class_accuracies, normalized_distribution)
        correlations.append(correlation)

    # Plot PCCs in a bar chart
    plt.figure(figsize=(14, 8))
    plt.bar(metric_abbreviations, correlations, color='skyblue')
    plt.ylabel('Pearson Correlation Coefficient (PCC)')
    plt.title('PCC Between Metrics and Class-Level Accuracies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = 'pcc_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close()

    return correlations


def main(dataset_name: str, models_count: int, threshold: float):
    # Define file paths for saving and loading cached results
    accuracies_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_avg_class_accuracies.npy"
    proximity_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_proximity_indicators.pkl"
    curvatures_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_curvature_indicators.pkl"
    disjuncts_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_disjuncts_indicators.pkl"

    # Load the dataset (full for proximity_indicators, and official training+test splits for ratio)
    dataset = u.load_full_data_and_normalize(dataset_name)
    labels = dataset.tensors[1].numpy()
    num_classes = len(np.unique(labels))

    if os.path.exists(accuracies_file):
        print('Loading accuracies.')
        avg_class_accuracies = np.load(accuracies_file)
    else:
        print('Computing accuracies.')
        trainer = EnsembleTrainer(dataset_name, models_count)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        trainer.train_ensemble(loader)

        # Compute average class-level accuracies
        class_accuracies = np.zeros((models_count, num_classes))
        for model_idx, model in enumerate(trainer.get_trained_models()):
            class_accuracies[model_idx] = u.class_level_test(model, loader, num_classes)
        avg_class_accuracies = class_accuracies.mean(axis=0)
        np.save(accuracies_file, avg_class_accuracies)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    if os.path.exists(curvatures_file):
        print('Loading curvatures.')
        gaussian_curvatures, mean_curvatures = u.load_data(curvatures_file)
    else:
        print('Calculating curvatures.')
        gaussian_curvatures, mean_curvatures = compute_curvatures(loader)
        u.save_data((gaussian_curvatures, mean_curvatures), curvatures_file)

    if os.path.exists(proximity_file):
        print('Loading proximities.')
        proximity_metrics = u.load_data(proximity_file)
    else:
        print('Calculating proximities.')
        proximity_metrics = compute_proximity_metrics(loader, gaussian_curvatures)
        u.save_data(proximity_metrics, proximity_file)

    if os.path.exists(disjuncts_file):
        print('Loading disjuncts.')
        disjunct_metrics = u.load_data(disjuncts_file)
    else:
        print('Calculating disjuncts.')
        disjunct_metrics = compute_disjuncts(loader)
        u.save_data(disjunct_metrics, disjuncts_file)

    gaussian_curvatures = [abs(gc) for gc in gaussian_curvatures]
    # Combine proximity metrics, curvature metrics, and disjunct metrics
    all_metrics = proximity_metrics + (gaussian_curvatures, mean_curvatures) + disjunct_metrics

    # Extract the hardest samples for each metric and compute their class distributions
    distributions_top, class_averages = extract_hard_samples(all_metrics, labels, [True] * 11, dataset_name, threshold)
    distributions_bottom, _ = extract_hard_samples(all_metrics, labels, [False] * 11, dataset_name, threshold)

    # Find the hardest and easiest classes, analyze hard sample distribution and visualize results
    hardest_class = np.argmin(avg_class_accuracies)
    easiest_class = np.argmax(avg_class_accuracies)
    print(f"\nHardest class accuracy (class {hardest_class}): {avg_class_accuracies[hardest_class]:.5f}%")
    print(f"Easiest class accuracy (class {easiest_class}): {avg_class_accuracies[easiest_class]:.5f}%")

    # Compare and plot all metrics against class-level accuracies
    compare_metrics_to_class_accuracies(distributions_top, avg_class_accuracies, num_classes)
    compare_metrics_to_class_accuracies(distributions_bottom, avg_class_accuracies, num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze hard samples in the official training and test splits using precomputed hardness '
                    'indicators.'
    )
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset (MNIST, CIFAR10, CIFAR100).')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='The percentage of the most extreme (hardest) samples that will be considered as hard.')
    args = parser.parse_args()

    main(**vars(args))
