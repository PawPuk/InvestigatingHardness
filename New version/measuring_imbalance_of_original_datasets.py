import argparse
from collections import defaultdict
import os
from typing import List, Tuple

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


def extract_curvy_samples(curvature_values: List[float], threshold: float = 0.05):
    """
    Extract the top 'threshold' percentage of samples with the highest curvature values.

    :param curvature_values: List of curvature values for each sample.
    :param threshold: The percentage of samples to consider (default: 0.05 for the top 5%).
    :return: List of indices corresponding to the curviest samples.
    """
    num_samples = len(curvature_values)
    num_curvy_samples = int(threshold * num_samples)

    # Sort the samples by curvature in descending order and get the indices of the top 'threshold' percentage
    sorted_indices = np.argsort(curvature_values)[::-1][:num_curvy_samples]

    return sorted_indices


def compare_curvy_samples_to_class_accuracies(curvy_sample_indices,
                                              labels: List[int],
                                              avg_class_accuracies: np.ndarray,
                                              num_classes: int):
    """
    Compare the class-level distribution of the curvy samples to the class-level accuracies.

    :param curvy_sample_indices: List of indices corresponding to the curviest samples.
    :param labels: List of class labels corresponding to the samples.
    :param avg_class_accuracies: The average accuracies for each class.
    :param num_classes: The number of classes in the dataset.
    """
    # Compute the class-level distribution of curvy samples
    curvy_samples_distribution = defaultdict(int)
    for idx in curvy_sample_indices:
        class_label = labels[idx]
        curvy_samples_distribution[class_label] += 1

    # Convert the distribution to a list for easier comparison
    curvy_samples_distribution_list = [curvy_samples_distribution[i] for i in range(num_classes)]

    # Visualize the class-level distribution of curvy samples and compare it to class-level accuracies
    fig, ax1 = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(num_classes)
    ax1.bar(index, avg_class_accuracies, bar_width, label='Class Accuracy', color='lightblue')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Class-Level Accuracies and Curvy Sample Distribution')

    ax2 = ax1.twinx()
    ax2.plot(index, curvy_samples_distribution_list, label='Curvy Sample Distribution', color='orange', marker='o')
    ax2.set_ylabel('Number of Curvy Samples')

    fig.tight_layout()
    plt.show()

    # Compute and display Pearson correlation coefficient
    correlation, _ = pearsonr(1 - avg_class_accuracies, curvy_samples_distribution_list)
    print(f"Pearson Correlation Coefficient between class errors and curvy sample distribution: {correlation:.4f}")


def main(dataset_name: str, models_count: int, threshold: float):
    # Define file paths for saving and loading cached results
    accuracies_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_avg_class_accuracies.npy"
    hardness_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_hardness_indicators.pkl"

    # Load the dataset (full for hardness_indicators, and official training+test splits for ratio)
    dataset = u.load_full_data_and_normalize(dataset_name)
    train_dataset, test_dataset = u.load_data_and_normalize(dataset_name)
    num_classes = len(np.unique(dataset.tensors[1].numpy()))
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    labels = dataset.tensors[1].numpy()

    # Check if the results are already cached
    if os.path.exists(accuracies_file) and os.path.exists(hardness_file):
        print("Loading cached results...")
        avg_class_accuracies = np.load(accuracies_file)
        hardness_indicators = u.load_data(hardness_file)
    else:
        curvature_values = compute_hardness_indicators(loader)
        curvy_sample_indices = extract_curvy_samples(curvature_values, threshold=0.05)
        loader.shuffle = True
        print("Running expensive operations (training and computing indicators)...")
        trainer = EnsembleTrainer(dataset_name, models_count)
        trainer.train_ensemble(loader)

        # Compute average class-level accuracies
        class_accuracies = np.zeros((models_count, num_classes))
        for model_idx, model in enumerate(trainer.get_trained_models()):
            class_accuracies[model_idx] = u.class_level_test(model, loader, num_classes)
        avg_class_accuracies = class_accuracies.mean(axis=0)
        compare_curvy_samples_to_class_accuracies(curvy_sample_indices, labels, avg_class_accuracies, num_classes)
    # Find the hardest and easiest classes, analyze hard sample distribution and visualize results
    hardest_class = np.argmin(avg_class_accuracies)
    easiest_class = np.argmax(avg_class_accuracies)
    print(f"Hardest class accuracy (class {hardest_class}): {avg_class_accuracies[hardest_class]:.2f}%")
    print(f"Easiest class accuracy (class {easiest_class}): {avg_class_accuracies[easiest_class]:.2f}%")


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
