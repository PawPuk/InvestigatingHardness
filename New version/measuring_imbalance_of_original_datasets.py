import argparse
from collections import defaultdict
import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
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


def detect_family(normalized_metric: np.ndarray, avg_gradients: List[float], second_derivatives: List[float]):
    if is_first_family_metric(avg_gradients):
        return 1
    elif is_second_family_metric(normalized_metric, avg_gradients):
        return 2
    return 3


def is_first_family_metric(avg_gradients: List[float]) -> bool:
    """Check if the metric belongs to the first family."""
    # Check if the rightmost points have the highest value
    right_most = np.mean(avg_gradients[-1000:])
    # Check if the leftmost points are higher than the mean of the middle samples
    left_most = np.mean(avg_gradients[:1000])
    middle_mean = np.mean(avg_gradients[len(avg_gradients)//2 - 10000: len(avg_gradients)//2 + 10000])

    # Conditions: rightmost should be highest, leftmost should be higher than the middle mean
    return right_most > 3 * middle_mean and left_most > 3 * middle_mean


def is_second_family_metric(normalized_metric: np.ndarray, avg_gradients: List[float]) -> int:
    """Check if the metric belongs to the new family based on distribution and gradient."""
    # Check the normalized distribution (left side low, right side high)
    left_side_distribution = np.mean(normalized_metric[:500])
    right_side_distribution = np.mean(normalized_metric[-20000:])

    # Check the first derivative (left side high, right side low)
    left_side_gradient = np.mean(avg_gradients[:500])
    right_side_gradient = np.mean(avg_gradients[-20000:])
    # Check the condition: distribution (left low, right high) and first derivative (left high, right low)
    if left_side_distribution < right_side_distribution and left_side_gradient > right_side_gradient:
        return True


def find_division_points_for_first_family(second_derivatives: np.ndarray, epsilon_fraction: float = 0.01,
                                          window_size_1: int = 10000) -> Tuple[int, int]:
    """Find the first and second division points based on the second derivative analysis."""
    max_second_derivative = np.max(second_derivatives)
    epsilon = epsilon_fraction * max_second_derivative

    # Initialize division points
    first_division_point = None
    second_division_point = None

    # Find the second division point (from left to right)
    for i in range(window_size_1, len(second_derivatives) - window_size_1):
        if np.all(np.abs(second_derivatives[i:i + window_size_1]) < epsilon):
            first_division_point = i
            break

    # Find the first division point (from right to left)
    for i in range(len(second_derivatives) - 1, window_size_1 - 1, -1):
        if np.all(np.abs(second_derivatives[i - window_size_1:i]) < epsilon):
            second_division_point = i
            break

    return first_division_point, second_division_point


def find_division_points_for_second_family(first_derivatives: np.ndarray, window_size: int = 500,
                                           epsilon_fraction: float = 0.01) -> Tuple[int, int]:
    right_most_value = np.mean(first_derivatives[-window_size:])
    epsilon = epsilon_fraction * right_most_value

    # Start from the rightmost point and move left
    for i in range(len(first_derivatives) - window_size, 0, -1):
        window_mean = np.mean(first_derivatives[i:i + window_size])
        if abs(window_mean - right_most_value) > epsilon:
            return i + 500, i + 500  # Move the point slightly to the right
    raise Exception


def find_division_points_for_third_family(second_derivatives: List[float], window_size1: int = 20000,
                                          window_size2: int = 100, epsilon_factor: float = 100) -> Tuple[int, int]:
    left_most_value = np.mean(second_derivatives[:window_size1])
    epsilon = epsilon_factor * left_most_value
    print(left_most_value)
    # Start from the rightmost point and move left
    for i in range(len(second_derivatives) - window_size2):
        window_mean = np.mean(second_derivatives[i:i + window_size2])
        if abs(window_mean - left_most_value) > epsilon:
            return i + 500, i + 500  # Move the point slightly to the right
    raise Exception


def plot_metric_results(metric_idx: int, sorted_normalized_metric: np.ndarray, avg_gradients: List[float],
                        avg_second_gradients: List[float], first_division_point: int, second_division_point: int,
                        dataset_name: str, invert: bool):
    """Plot the results with division points marked and areas colored as easy, medium, and hard."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot sorted normalized metric
    axes[0].plot(sorted_normalized_metric, marker='o', linestyle='-')

    # Define the regions (easy, medium, hard)
    if first_division_point is not None and second_division_point is not None:
        if first_division_point != second_division_point:
            pass
            # Color the medium region (between first and second division points) blue
            axes[0].axvspan(first_division_point, second_division_point, facecolor='blue', alpha=0.3, label='Medium')

        # Color the easy and hard regions based on the invert flag
        if invert:
            # If invert is True, left is hard (red), right is easy (green)
            axes[0].axvspan(0, first_division_point, facecolor='red', alpha=0.3, label='Hard')
            axes[0].axvspan(second_division_point, len(sorted_normalized_metric), facecolor='green', alpha=0.3, label='Easy')
        else:
            # If invert is False, left is easy (green), right is hard (red)
            axes[0].axvspan(0, first_division_point, facecolor='green', alpha=0.3, label='Easy')
            axes[0].axvspan(second_division_point, len(sorted_normalized_metric), facecolor='red', alpha=0.3, label='Hard')

    # Add division lines
    if first_division_point is not None:
        axes[0].axvline(x=first_division_point, color='blue', linestyle='--', label='First Division')
    if second_division_point is not None:
        axes[0].axvline(x=second_division_point, color='blue', linestyle='--', label='Second Division')

    axes[0].set_title(f'Metric {metric_idx + 1} Normalized Distribution')
    axes[0].set_xlabel('Sample Index (Sorted)')
    axes[0].set_ylabel('Normalized Metric Value')
    axes[0].grid(True)

    # Plot average gradient
    axes[1].plot(avg_gradients, marker='x', linestyle='-', color='r')
    if first_division_point is not None:
        axes[1].axvline(x=first_division_point, color='blue', linestyle='--', label='First Division')
    if second_division_point is not None:
        axes[1].axvline(x=second_division_point, color='blue', linestyle='--', label='Second Division')
    axes[1].set_title(f'Metric {metric_idx + 1} Average Gradients (First Derivative)')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Average Gradient')
    axes[1].grid(True)

    # Plot second derivative
    axes[2].plot(avg_second_gradients, marker='x', linestyle='-', color='g')
    if first_division_point is not None:
        axes[2].axvline(x=first_division_point, color='blue', linestyle='--', label='First Division')
    if second_division_point is not None:
        axes[2].axvline(x=second_division_point, color='blue', linestyle='--', label='Second Division')
    axes[2].set_title(f'Metric {metric_idx + 1} Second Derivatives')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Second Derivative')
    axes[2].grid(True)

    # Save plot
    output_dir = 'metric_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_metric_{metric_idx + 1}_distribution_gradients_second_derivatives.pdf'))
    plt.savefig(os.path.join(output_dir,
                             f'{dataset_name}_metric_{metric_idx + 1}_distribution_gradients_second_derivatives.png'))

    plt.close()



def extract_extreme_samples_via_hard_threshold(metrics: List[List[float]], labels: List[int], invert: List[bool],
                                               dataset_name: str, threshold: float = 0.05):
    num_metrics = len(metrics)
    class_distributions = []
    extreme_indices = []

    for metric_idx in range(num_metrics):
        selected_metric = np.array(metrics[metric_idx])
        num_samples = len(selected_metric)
        num_extreme_samples = int(threshold * num_samples)

        # Replace inf values with the maximum finite value
        max_finite_value = np.max(selected_metric[np.isfinite(selected_metric)])
        selected_metric[np.isinf(selected_metric)] = max_finite_value

        # Normalize the selected metric (min-max normalization)
        min_val = np.min(selected_metric)
        max_val = np.max(selected_metric)
        normalized_metric = (selected_metric - min_val) / (max_val - min_val)

        # Sort the samples by the normalized metric
        if invert[metric_idx]:
            sorted_indices = np.argsort(normalized_metric)[-num_extreme_samples:]
        else:
            sorted_indices = np.argsort(normalized_metric)[:num_extreme_samples]

        # Store the indices of the extreme samples
        extreme_indices.append(sorted_indices.tolist())

        # Compute the distribution of these samples across different classes
        class_distribution = defaultdict(int)
        for idx in sorted_indices:
            class_label = labels[idx]
            class_distribution[class_label] += 1
        class_distributions.append(class_distribution)

    return class_distributions, extreme_indices

def extract_extreme_samples_via_soft_threshold(metrics: List[List[float]], labels: List[int], dataset_name: str,
                                               invert: List[bool]) -> Tuple[List[int], List[int], List[Dict[int, int]], List[Dict[int, int]]]:
    """Extract easy and hard samples based on the division points and invert logic, returning their indices and distributions."""
    num_metrics = len(metrics)
    easy_samples = []
    hard_samples = []
    easy_distributions = []
    hard_distributions = []

    for metric_idx in range(num_metrics):
        selected_metric = np.array(metrics[metric_idx])
        num_samples = len(selected_metric)

        # Replace inf values with the maximum finite value
        max_finite_value = np.max(selected_metric[np.isfinite(selected_metric)])
        selected_metric[np.isinf(selected_metric)] = max_finite_value

        # Normalize the selected metric (min-max normalization)
        min_val = np.min(selected_metric)
        max_val = np.max(selected_metric)
        normalized_metric = (selected_metric - min_val) / (max_val - min_val)

        # Sort normalized metric for gradient computation
        sorted_indices = np.argsort(normalized_metric)  # Get sorted indices to map back
        sorted_normalized_metric = normalized_metric[sorted_indices]

        # Compute the first derivative (gradient) for the sorted normalized metric
        gradients = np.gradient(sorted_normalized_metric)

        # Compute average gradient using 500 points to the left and 500 to the right
        avg_gradients = []
        window_size = 1000
        for i in range(window_size, num_samples - window_size):
            start = max(0, i - window_size)
            end = min(num_samples, i + window_size + 1)
            avg_gradients.append(np.mean(gradients[start:end]))

        # Smooth the first derivative using Savitzky-Golay filter
        smoothed_avg_gradients = savgol_filter(avg_gradients, window_length=1000, polyorder=2)

        # Compute the second derivative (gradient of the smoothed first derivative)
        second_derivatives = np.gradient(smoothed_avg_gradients)

        # Detect family and find division points
        first_division_point, second_division_point = None, None
        family = detect_family(sorted_normalized_metric, avg_gradients, second_derivatives)
        print(f'Metric {metric_idx + 1} is of family {family}.')
        if family == 1:
            first_division_point, second_division_point = find_division_points_for_first_family(second_derivatives)
        elif family == 2:
            first_division_point, second_division_point = find_division_points_for_second_family(smoothed_avg_gradients)
        elif family == 3:
            first_division_point, second_division_point = find_division_points_for_third_family(second_derivatives)

        # Dictionaries to hold the distribution of easy and hard samples per class
        easy_dist = defaultdict(int)
        hard_dist = defaultdict(int)

        # Extract easy and hard samples based on division points and invert logic
        if first_division_point is not None and second_division_point is not None:
            if invert[metric_idx]:
                # If invert is True: Left side (before first division point) is hard, right side (after second) is easy
                hard_indices = sorted_indices[:first_division_point]
                easy_indices = sorted_indices[second_division_point:]
            else:
                # If invert is False: Left side (before first division point) is easy, right side (after second) is hard
                easy_indices = sorted_indices[:first_division_point]
                hard_indices = sorted_indices[second_division_point:]

            # Add indices to the global easy and hard sample lists
            easy_samples.append(easy_indices.tolist())
            hard_samples.append(hard_indices.tolist())

            # Compute class distributions for easy and hard samples
            for idx in easy_indices:
                class_label = labels[idx]
                easy_dist[class_label] += 1

            for idx in hard_indices:
                class_label = labels[idx]
                hard_dist[class_label] += 1

        # Append the distributions for this metric
        easy_distributions.append(easy_dist)
        hard_distributions.append(hard_dist)

        # Plot results
        plot_metric_results(metric_idx, sorted_normalized_metric, avg_gradients, second_derivatives,
                            first_division_point, second_division_point, dataset_name, invert[metric_idx])

    return easy_samples, hard_samples, easy_distributions, hard_distributions




def compare_metrics_to_class_accuracies(class_distributions, avg_class_accuracies, num_classes, output_filename):
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
        'MC'
    ]  # Abbreviations for each metric to keep plot readable.

    # Compute PCC for each metric
    for class_distribution in class_distributions:
        class_level_distribution = [class_distribution.get(cls, 0) for cls in range(num_classes)]

        # Compute Pearson correlation coefficient
        correlation, _ = pearsonr(avg_class_accuracies, class_level_distribution)
        correlations.append(correlation)

    # Define colors based on correlation strength
    def get_color(pcc):
        abs_pcc = abs(pcc)
        if abs_pcc < 0.2:
            return 'lightgray'  # very weak
        elif abs_pcc < 0.4:
            return 'lightblue'  # weak
        elif abs_pcc < 0.6:
            return 'skyblue'  # moderate
        elif abs_pcc < 0.8:
            return 'dodgerblue'  # strong
        else:
            return 'blue'  # very strong

    colors = [get_color(corr) for corr in correlations]

    # Plot PCCs in a bar chart with horizontal lines
    plt.figure(figsize=(14, 8))
    plt.bar(metric_abbreviations, correlations, color=colors)
    plt.title('PCC Between Metrics and Class-Level Accuracies')
    plt.ylabel('Pearson Correlation Coefficient (PCC)')
    plt.xticks(rotation=45, ha='right')

    # Add horizontal lines for better readability
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axhline(0.2, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(-0.2, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(0.4, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(-0.4, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(0.6, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(-0.6, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(0.8, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(-0.8, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = 'pcc_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close()


def compute_class_averages_of_metrics(metrics, labels):
    class_averages = []

    for metric in metrics:
        # Handle inf values: replace them with the now-largest values (max * 2)
        metric = np.array(metric)
        max_finite_value = np.max(metric[np.isfinite(metric)])
        metric[np.isinf(metric)] = max_finite_value * 2

        # Normalize by dividing by the maximum value in the metric
        max_val = np.max(metric)
        if max_val != 0:
            metric = metric / max_val
        else:
            raise Exception('This should not happen.')

        # Compute the average of each normalized metric for every class
        class_average = defaultdict(list)
        for metric_value, label in zip(metric, labels):
            class_average[label].append(metric_value)

        # Compute the mean of each class
        for class_label, values in class_average.items():
            class_average[class_label] = np.mean(values)

        class_averages.append(class_average)

    return class_averages


def compute_correlation_heatmaps(easy_distribution, hard_distribution, output_dir="heatmaps"):
    num_metrics = len(easy_distribution)
    print(num_metrics)

    easy_overlap = np.zeros((num_metrics, num_metrics))
    hard_overlap = np.zeros((num_metrics, num_metrics))

    for i in range(num_metrics):
        for j in range(num_metrics):
            easy_overlap[i, j] = len(set(easy_distribution[i]) & set(easy_distribution[j])) / len(
                    easy_distribution[j])
            hard_overlap[i, j] = len(set(hard_distribution[i]) & set(hard_distribution[j])) / len(
                    hard_distribution[j])

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(easy_overlap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Easy Samples Overlap Heatmap")
    plt.savefig(os.path.join(output_dir, "easy_overlap_heatmap.pdf"))
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(hard_overlap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Hard Samples Overlap Heatmap")
    plt.savefig(os.path.join(output_dir, "hard_overlap_heatmap.pdf"))
    plt.show()
    plt.close()


def compute_easy_hard_ratios(dataset_name: str, easy_indices: List[np.ndarray], hard_indices: List[np.ndarray]) -> None:
    """Compute the ratio of easy to hard samples in both training and test splits for each metric and display as a table."""

    # Load the training and test datasets using the provided function
    train_dataset, test_dataset = u.load_data_and_normalize(dataset_name)

    # Extract the targets (labels) from the datasets
    train_targets = train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset.tensors[1]
    test_targets = test_dataset.targets if hasattr(test_dataset, 'targets') else test_dataset.tensors[1]

    # Extract indices of the training and test sets
    train_indices = list(range(len(train_targets)))
    test_indices = list(range(len(test_targets)))

    # Initialize lists to store the data for the table
    metrics_data = []

    # Loop over each metric's array of easy and hard indices
    for metric_idx in range(len(easy_indices)):
        easy_idx = easy_indices[metric_idx]
        hard_idx = hard_indices[metric_idx]

        # Compute counts of easy and hard samples in the training set for this metric
        easy_train_count = sum([1 for idx in easy_idx if idx in train_indices])
        hard_train_count = sum([1 for idx in hard_idx if idx in train_indices])

        # Compute counts of easy and hard samples in the test set for this metric
        easy_test_count = sum([1 for idx in easy_idx if idx in test_indices])
        hard_test_count = sum([1 for idx in hard_idx if idx in test_indices])

        # Compute the ratios for training and test sets for this metric
        train_ratio = easy_train_count / hard_train_count if hard_train_count > 0 else float('inf')
        test_ratio = easy_test_count / hard_test_count if hard_test_count > 0 else float('inf')

        # Append the results to the metrics_data list
        metrics_data.append({
            'Metric': metric_idx + 1,
            'Easy Train': easy_train_count,
            'Hard Train': hard_train_count,
            'Train Ratio (Easy/Hard)': train_ratio,
            'Easy Test': easy_test_count,
            'Hard Test': hard_test_count,
            'Test Ratio (Easy/Hard)': test_ratio
        })

    # Create a DataFrame to display the results
    df = pd.DataFrame(metrics_data)

    # Print the table
    print("\nEasy-to-Hard Sample Ratios Across Metrics:\n")
    print(df.to_string(index=False))


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
    all_metrics = proximity_metrics + (gaussian_curvatures, mean_curvatures)

    class_averages = compute_class_averages_of_metrics(all_metrics, labels)

    invert_metrics = [False, True, False, False, True, False, False, True, False, False, True, False, False, False,
                      False, False, False]

    # Extract the hardest samples for each metric and compute their class distributions
    easy_indices, hard_indices, easy_distribution, hard_distribution = (
        extract_extreme_samples_via_soft_threshold(all_metrics, labels, dataset_name, invert_metrics))

    print(hard_distribution)

    print()
    print('-'*20)
    print()

    print(easy_distribution)

    # Compute and visualize the correlation heatmaps
    # compute_correlation_heatmaps(easy_indices, hard_indices)

    # Find the hardest and easiest classes, analyze hard sample distribution and visualize results
    hardest_class = np.argmin(avg_class_accuracies)
    easiest_class = np.argmax(avg_class_accuracies)
    print(f"\nHardest class accuracy (class {hardest_class}): {avg_class_accuracies[hardest_class]:.5f}%")
    print(f"Easiest class accuracy (class {easiest_class}): {avg_class_accuracies[easiest_class]:.5f}%")

    # Compare and plot all metrics against class-level accuracies
    compare_metrics_to_class_accuracies(easy_distribution, avg_class_accuracies, num_classes,
                                        f'{dataset_name}_easyPCC.pdf')
    compare_metrics_to_class_accuracies(hard_distribution, avg_class_accuracies, num_classes,
                                        f'{dataset_name}_hardPCC.pdf')
    compare_metrics_to_class_accuracies(class_averages, avg_class_accuracies, num_classes,
                                        f'{dataset_name}_avgPCC.pdf')

    compute_easy_hard_ratios(dataset_name, easy_indices, hard_indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze hard samples in the official training and test splits using precomputed hardness '
                    'indicators.'
    )
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset (MNIST, CIFAR10, CIFAR100).')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='The percentage of the most extreme (hardest) samples that will be considered as hard.')
    args = parser.parse_args()

    main(**vars(args))
