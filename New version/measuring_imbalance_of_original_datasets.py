import argparse
from collections import defaultdict
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
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
    
    
def identify_hard_samples(hardness_indicators: List[Tuple[float, float, int, float, float, float]],
                          threshold: float) -> Dict[str, List[int]]:
    """
    Identify the hardest samples in the dataset based on different metrics.

    :param hardness_indicators: List of tuples containing various hardness indicators (confidence, margin,
                                misclassification, loss, gradient, entropy) for each sample.
    :param threshold: Proportion of the hardest samples to consider (e.g., 0.05 for the top 5% of the hardest samples).
    :return: A dictionary mapping each metric to a list of indices corresponding to the hardest samples.
    """
    metrics = ['Confidence', 'Margin', 'Misclassification', 'Loss', 'Gradient', 'Entropy']
    hard_sample_indices_for_different_metrics = {metric: [] for metric in metrics}
    num_samples = len(hardness_indicators)
    num_hard_samples = int(threshold * num_samples)

    confidence_sorted = sorted(enumerate(hardness_indicators), key=lambda x: x[1][0])[:num_hard_samples]
    margin_sorted = sorted(enumerate(hardness_indicators), key=lambda x: x[1][1])[:num_hard_samples]
    misclassified_sorted = sorted(enumerate(hardness_indicators), key=lambda x: x[1][2], reverse=True)[:num_hard_samples]
    loss_sorted = sorted(enumerate(hardness_indicators), key=lambda x: x[1][3], reverse=True)[:num_hard_samples]
    gradient_sorted = sorted(enumerate(hardness_indicators), key=lambda x: x[1][4], reverse=True)[:num_hard_samples]
    entropy_sorted = sorted(enumerate(hardness_indicators), key=lambda x: x[1][5], reverse=True)[:num_hard_samples]

    hard_sample_indices_for_different_metrics['Confidence'] = [idx for idx, _ in confidence_sorted]
    hard_sample_indices_for_different_metrics['Margin'] = [idx for idx, _ in margin_sorted]
    hard_sample_indices_for_different_metrics['Misclassification'] = [idx for idx, _ in misclassified_sorted]
    hard_sample_indices_for_different_metrics['Loss'] = [idx for idx, _ in loss_sorted]
    hard_sample_indices_for_different_metrics['Gradient'] = [idx for idx, _ in gradient_sorted]
    hard_sample_indices_for_different_metrics['Entropy'] = [idx for idx, _ in entropy_sorted]

    return hard_sample_indices_for_different_metrics


def compute_hardness_distributions(hard_samples: Dict[str, List[int]],
                                   labels: List[int]) -> Dict[str, defaultdict]:
    """
    Compute the class-level distribution of the hardest samples based on different hardness metrics.

    :param hard_samples: Dictionary containing the hardest samples for each metric.
    :param labels: List of class labels corresponding to the samples.
    :return: A dictionary mapping each metric to a class-level distribution of hard samples.
    """
    metrics = ['Confidence', 'Margin', 'Misclassification', 'Loss', 'Gradient', 'Entropy']
    class_distributions = {metric: defaultdict(int) for metric in metrics}

    for metric in metrics:
        for idx in hard_samples[metric]:
            class_label = labels[idx]
            class_distributions[metric][class_label] += 1

    return class_distributions


def compute_hard_easy_ratios(hard_samples: Dict[str, List[int]], train_indices: List[int], test_indices: List[int]):
    """
    Compute and print the ratio of hard to easy samples in the training and test datasets.

    :param hard_samples: Dictionary containing the hardest samples for each metric.
    :param train_indices: List of indices corresponding to the training dataset.
    :param test_indices: List of indices corresponding to the test dataset.
    """
    metrics = hard_samples.keys()
    for metric in metrics:
        hard_train = len([idx for idx in hard_samples[metric] if idx in train_indices])
        hard_test = len([idx for idx in hard_samples[metric] if idx in test_indices])

        easy_train = len(train_indices) - hard_train
        easy_test = len(test_indices) - hard_test

        print(f"Metric: {metric}")
        print(f"Training dataset - Hard:Easy Ratio = {hard_train}:{easy_train}")
        print(f"Test dataset - Hard:Easy Ratio = {hard_test}:{easy_test}")
        print("-" * 40)



def visualize_class_accuracies_and_hardness(avg_class_accuracies: np.ndarray, hardness_distributions: dict,
                                            num_classes: int):
    """
    Visualize the class-level accuracies as a bar chart and overlay the hardness indicators as line plots.
    Also, compute and display the Pearson correlation between class accuracies and hardness indicators.

    :param avg_class_accuracies: The average accuracies for each class.
    :param hardness_distributions: A dictionary containing class-level hardness counts for each indicator.
    :param num_classes: The number of classes in the dataset.
    """
    metrics = list(hardness_distributions.keys())
    avg_class_errors = 1 - avg_class_accuracies

    # Plot class-level errors as a bar chart
    fig, ax1 = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    index = np.arange(num_classes)
    ax1.bar(index, avg_class_errors, bar_width, label='Class Accuracy', color='lightcoral')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('Class-Level Errors and Hardness Indicators')

    # Create a second y-axis and plot the hardness indicators
    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Hard Samples')
    for metric in metrics:
        ax2.plot(index, hardness_distributions[metric], label=metric, marker='o', linestyle='--')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

    # Compute and display Pearson correlation coefficients
    print("\nPearson Correlation Coefficients:")
    correlation_table = PrettyTable()
    correlation_table.field_names = ["Metric", "Pearson Correlation Coefficient"]
    for metric in metrics:
        correlation, _ = pearsonr(avg_class_errors, hardness_distributions[metric])
        correlation_table.add_row([metric, f"{correlation:.4f}"])
    print(correlation_table)


def main(dataset_name: str, models_count: int, threshold: float):
    # Define file paths for saving and loading cached results
    accuracies_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_avg_class_accuracies.npy"
    hardness_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_hardness_indicators.pkl"

    # Load the dataset (full for hardness_indicators, and official training+test splits for ratio)
    dataset = u.load_full_data_and_normalize(dataset_name)
    train_dataset, test_dataset = u.load_data_and_normalize(dataset_name)
    num_classes = len(np.unique(dataset.tensors[1].numpy()))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    labels = dataset.tensors[1].numpy()

    # Check if the results are already cached
    if os.path.exists(accuracies_file) and os.path.exists(hardness_file):
        print("Loading cached results...")
        avg_class_accuracies = np.load(accuracies_file)
        hardness_indicators = u.load_data(hardness_file)
    else:
        print("Running expensive operations (training and computing indicators)...")
        trainer = EnsembleTrainer(dataset_name, models_count)
        trainer.train_ensemble(loader)

        # Compute average class-level accuracies
        class_accuracies = np.zeros((models_count, num_classes))
        for model_idx, model in enumerate(trainer.get_trained_models()):
            class_accuracies[model_idx] = u.class_level_test(model, loader, num_classes)
        avg_class_accuracies = class_accuracies.mean(axis=0)

        # Identify hard samples and measure their distribution among classes
        loader.shuffle = False
        hardness_indicators = compute_hardness_indicators(trainer.get_trained_models(), loader, models_count)

        np.save(accuracies_file, avg_class_accuracies)
        u.save_data(hardness_indicators, hardness_file)
    # Find the hardest and easiest classes, analyze hard sample distribution and visualize results
    hardest_class = np.argmin(avg_class_accuracies)
    easiest_class = np.argmax(avg_class_accuracies)
    print(f"Hardest class accuracy (class {hardest_class}): {avg_class_accuracies[hardest_class]:.2f}%")
    print(f"Easiest class accuracy (class {easiest_class}): {avg_class_accuracies[easiest_class]:.2f}%")
    hard_samples = identify_hard_samples(hardness_indicators, threshold)
    hardness_distributions = compute_hardness_distributions(hard_samples, labels)
    visualize_class_accuracies_and_hardness(avg_class_accuracies, hardness_distributions, num_classes)


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
