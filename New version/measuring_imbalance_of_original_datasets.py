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


def extract_hard_samples(metrics: List[List[float]], labels: List[int], threshold: float = 0.05,
                         invert: List[bool] = None):
    """
    Extract the 'threshold' percentage of the hardest samples for each metric.

    :param metrics: List of lists containing metrics (proximity and curvature) for all samples in the dataset.
    :param labels: List of class labels corresponding to the samples.
    :param threshold: The percentage of samples to consider (default: 0.05 for the hardest samples).
    :param invert: List of booleans indicating whether to invert the selection (i.e., select highest instead of lowest)
                   for each metric.
    :return: A list of dictionaries, each containing class labels as keys and the count of hardest samples as values
    for each metric.
    """
    num_metrics = len(metrics)
    class_distributions = []

    if invert is None:
        invert = [False] * num_metrics

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

    return class_distributions


def compare_metrics_to_class_accuracies(class_distributions: List[Dict[int, int]],
                                        avg_class_accuracies: np.ndarray,
                                        num_classes: int):
    """
    Compare the class-level distribution of the hardest samples to the class-level accuracies
    for each metric (both proximity and curvature).

    :param class_distributions: List of dictionaries, each containing class labels as keys and the count of hardest
    samples as values for each metric.
    :param avg_class_accuracies: The average accuracies for each class.
    :param num_classes: The number of classes in the dataset.
    """
    correlations = []

    fig, ax1 = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(num_classes)

    # Plot class error rates
    ax1.bar(index, 1 - avg_class_accuracies, bar_width, label='Class Error Rate', color='lightblue')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('Class-Level Error Rates and Hard Sample Distribution')

    # Colors and labels for the different metrics
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'blue', 'cyan', 'magenta']
    metric_labels = [
        'Proximity Ratio (Centroids)',
        'Distance to Closest Other Class Centroid',
        'Distance to Same Class Centroid',
        'Sample-to-Sample Distance Ratio',
        'KNN Ratio',
        'Gaussian Curvature',
        'Mean Curvature',
        'Average KNN Curvature'
    ]

    ax2 = ax1.twinx()

    # Normalize and plot each metric
    for i, class_distribution in enumerate(class_distributions):
        bottom_samples_distribution = [class_distribution.get(cls, 0) for cls in range(num_classes)]

        # Normalize the distribution to [0, 1]
        normalized_distribution = (bottom_samples_distribution - np.min(bottom_samples_distribution)) / \
                                  (np.max(bottom_samples_distribution) - np.min(bottom_samples_distribution))

        # Compute and display Pearson correlation coefficient
        correlation, _ = pearsonr(1 - avg_class_accuracies, normalized_distribution)
        correlations.append(correlation)
        print(f"Pearson Correlation Coefficient for {metric_labels[i]}: {correlation:.4f}")

        # Plot the normalized distribution
        ax2.plot(index, normalized_distribution, label=f'{metric_labels[i]}', color=colors[i], marker='o')

    ax2.set_ylabel('Normalized Number of Hard Samples')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    plt.show()

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
    class_sample_counts = np.bincount(labels)
    max_class_samples = np.max(class_sample_counts)

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
        proximity_metrics = compute_proximity_metrics(loader, max_class_samples, gaussian_curvatures)
        u.save_data(proximity_metrics, proximity_file)

    if os.path.exists(disjuncts_file):
        print('Loading disjuncts.')
        disjunct_metrics = u.load_data(disjuncts_file)
    else:
        print('Calculating disjuncts.')
        disjunct_metrics = compute_disjuncts(loader, max_class_samples)
        u.save_data(disjunct_metrics, disjuncts_file)

    gaussian_curvatures = [abs(gc) for gc in gaussian_curvatures]
    # Combine proximity metrics, curvature metrics, and disjunct metrics
    all_metrics = proximity_metrics + (gaussian_curvatures, mean_curvatures) + disjunct_metrics

    # Define which metrics should be inverted (set True if higher values mean harder samples)
    invert = [
        False,  # Proximity Ratio (Centroids)
        False,  # Distance to the Closest Other Class Centroid
        True,   # Distance to the Same Class Centroid
        False,  # Sample-to-Sample Distance Ratio
        True,   # KNN Ratio
        True,   # Gaussian Curvature
        True,   # Mean Curvature
        True,   # Average KNN Curvature
        False,  # Disjunct Size Based on Custom Algorithm
        False,  # Disjunct Size Based on GMM
        False   # Disjunct Size Based on DBSCAN
    ]

    # Extract the hardest samples for each metric and compute their class distributions
    class_distributions = extract_hard_samples(all_metrics, labels, threshold=threshold, invert=invert)

    # Find the hardest and easiest classes, analyze hard sample distribution and visualize results
    hardest_class = np.argmin(avg_class_accuracies)
    easiest_class = np.argmax(avg_class_accuracies)
    print(f"\nHardest class accuracy (class {hardest_class}): {avg_class_accuracies[hardest_class]:.5f}%")
    print(f"Easiest class accuracy (class {easiest_class}): {avg_class_accuracies[easiest_class]:.5f}%")

    # Compare and plot all metrics against class-level accuracies
    compare_metrics_to_class_accuracies(class_distributions, avg_class_accuracies, num_classes)


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
