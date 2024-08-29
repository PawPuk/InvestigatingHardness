import argparse
from collections import defaultdict
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader

from compute_confidences import compute_hardness_indicators
from train_ensembles import EnsembleTrainer
import utils as u

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def extract_bottom_proximity_samples(proximity_values: List[float], labels: List[int], threshold: float = 0.05):
    """
    Extract the bottom 'threshold' percentage of samples with the smallest proximity values from the entire dataset.

    :param proximity_values: List of proximity values for all samples in the dataset.
    :param labels: List of class labels corresponding to the samples.
    :param threshold: The percentage of samples to consider (default: 0.05 for the bottom 5%).
    :return: A dictionary with class labels as keys and the count of bottom proximity samples as values.
    """
    num_samples = len(proximity_values)
    num_bottom_samples = int(threshold * num_samples)

    # Sort the samples by proximity in ascending order and get the indices of the bottom 'threshold' percentage
    sorted_indices = np.argsort(proximity_values)[:num_bottom_samples]
    bottom_proximity_samples_indices = sorted_indices

    # Compute the distribution of these bottom samples across different classes
    class_distribution = defaultdict(int)
    for idx in bottom_proximity_samples_indices:
        class_label = labels[idx]
        class_distribution[class_label] += 1

    return class_distribution


def compare_top_proximity_samples_to_class_accuracies(class_distribution: Dict[int, int],
                                                      avg_class_accuracies: np.ndarray,
                                                      num_classes: int):
    """
    Compare the class-level distribution of the bottom proximity samples to the class-level accuracies.

    :param class_distribution: Dictionary with class labels as keys and the count of bottom proximity samples as values.
    :param avg_class_accuracies: The average accuracies for each class.
    :param num_classes: The number of classes in the dataset.
    """
    # Convert class distribution to a list where index corresponds to class label
    bottom_proximity_samples_distribution = [class_distribution.get(i, 0) for i in range(num_classes)]

    # Compute and display Pearson correlation coefficient
    correlation, _ = pearsonr(1 - avg_class_accuracies, bottom_proximity_samples_distribution)
    print(f"Pearson Correlation Coefficient between class errors and bottom proximity sample distribution: "
          f"{correlation:.4f}")

    # Visualize the class-level distribution of bottom proximity samples and compare it to class-level accuracies
    fig, ax1 = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(num_classes)

    # Plot class error rates
    ax1.bar(index, 1 - avg_class_accuracies, bar_width, label='Class Error Rate', color='lightblue')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('Class-Level Error Rates and Bottom Proximity Sample Distribution')

    # Plot bottom proximity sample distribution
    ax2 = ax1.twinx()
    ax2.plot(index, bottom_proximity_samples_distribution, label='Bottom Proximity Sample Distribution', color='orange',
             marker='o')
    ax2.set_ylabel('Number of Bottom Proximity Samples')

    fig.tight_layout()
    plt.show()


def main(dataset_name: str, models_count: int, threshold: float):
    # Define file paths for saving and loading cached results
    accuracies_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_avg_class_accuracies.npy"
    proximity_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_proximity_indicators.pkl"

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

    if os.path.exists(proximity_file):
        print('Loading proximities.')
        proximity_values = u.load_data(proximity_file)
    else:
        print('Calculating proximities.')
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        proximity_values = compute_hardness_indicators(loader)
        u.save_data(proximity_values, proximity_file)

    # Extract bottom 5% proximity samples and compute their class distribution
    class_distribution = extract_bottom_proximity_samples(proximity_values, labels, threshold=threshold)

    # Display class distribution for bottom proximity samples
    print("\nClass distribution of bottom 5% proximity samples:")
    for cls, count in class_distribution.items():
        print(f"Class {cls}: {count} samples")

    # Find the hardest and easiest classes, analyze hard sample distribution and visualize results
    hardest_class = np.argmin(avg_class_accuracies)
    easiest_class = np.argmax(avg_class_accuracies)
    print(f"\nHardest class accuracy (class {hardest_class}): {avg_class_accuracies[hardest_class]:.5f}%")
    print(f"Easiest class accuracy (class {easiest_class}): {avg_class_accuracies[easiest_class]:.5f}%")
    compare_top_proximity_samples_to_class_accuracies(class_distribution, avg_class_accuracies, num_classes)


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
