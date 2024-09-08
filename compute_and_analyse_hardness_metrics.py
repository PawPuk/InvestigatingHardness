import argparse
from collections import defaultdict
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kendalltau, pearsonr, spearmanr
import torch
from torch.utils.data import DataLoader

from compute_confidences import compute_curvatures, compute_proximity_metrics
import utils as u

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def detect_family(normalized_metric: np.ndarray, avg_gradients: List[np.ndarray]):
    if is_first_family_metric(avg_gradients):
        return 1
    elif is_second_family_metric(normalized_metric, avg_gradients):
        return 2
    return 3


def is_first_family_metric(avg_gradients: List[np.ndarray]) -> bool:
    """Check if the metric belongs to the first family."""
    # Check if the rightmost points have the highest value
    right_most = np.mean(avg_gradients[-1000:])
    # Check if the leftmost points are higher than the mean of the middle samples
    left_most = np.mean(avg_gradients[:1000])
    middle_mean = np.mean(avg_gradients[len(avg_gradients) // 2 - 10000: len(avg_gradients) // 2 + 10000])

    # Conditions: rightmost should be highest, leftmost should be higher than the middle mean
    return right_most > 3 * middle_mean and left_most > 3 * middle_mean


def is_second_family_metric(normalized_metric: np.ndarray, avg_gradients: List[np.ndarray]) -> int:
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


def find_division_points_for_third_family(second_derivatives: np.ndarray, window_size1: int = 20000,
                                          window_size2: int = 500, epsilon_factor: float = 250) -> Tuple[int, int]:
    left_most_value = np.mean(second_derivatives[:window_size1])
    epsilon = epsilon_factor * left_most_value
    print(left_most_value)
    # Start from the rightmost point and move left
    for i in range(len(second_derivatives) - window_size2):
        window_mean = np.mean(second_derivatives[i:i + window_size2])
        if abs(window_mean - left_most_value) > epsilon:
            return i + 500, i + 500  # Move the point slightly to the right
    raise Exception


def plot_metric_results(metric_idx: int, sorted_normalized_metric: np.ndarray, avg_gradients: List[np.ndarray],
                        avg_second_gradients: np.ndarray, first_division_point: int, second_division_point: int,
                        dataset_name: str, invert: bool, hard_threshold_percent: float, training: str):
    """Plot the results with division points marked and areas colored as easy, medium, and hard, along with hard
    thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    num_samples = len(sorted_normalized_metric)
    hard_threshold_count = int(hard_threshold_percent * num_samples)

    # Compute the locations of the hard threshold lines
    hard_easy_index = hard_threshold_count  # This is the last index for the easy threshold
    hard_hard_index = num_samples - hard_threshold_count  # This is the first index for the hard threshold

    # Plot sorted normalized metric
    axes[0].plot(sorted_normalized_metric, marker='o', linestyle='-')

    # Define the regions (easy, medium, hard)
    if first_division_point is not None and second_division_point is not None:
        if first_division_point != second_division_point:
            # Color the medium region (between first and second division points) blue
            axes[0].axvspan(first_division_point, second_division_point, facecolor='blue', alpha=0.3, label='Medium')

        # Color the easy and hard regions based on the invert flag
        if invert:
            # If invert is True, left is hard (red), right is easy (green)
            axes[0].axvspan(0, first_division_point, facecolor='red', alpha=0.3, label='Hard')
            axes[0].axvspan(second_division_point, len(sorted_normalized_metric), facecolor='green', alpha=0.3,
                            label='Easy')
        else:
            # If invert is False, left is easy (green), right is hard (red)
            axes[0].axvspan(0, first_division_point, facecolor='green', alpha=0.3, label='Easy')
            axes[0].axvspan(second_division_point, len(sorted_normalized_metric), facecolor='red', alpha=0.3,
                            label='Hard')

    # Add division lines for adaptive (soft) thresholds
    if first_division_point is not None:
        axes[0].axvline(x=first_division_point, color='blue', linestyle='--', label='First Division')
    if second_division_point is not None:
        axes[0].axvline(x=second_division_point, color='blue', linestyle='--', label='Second Division')

    # Add hard threshold lines (black vertical lines)
    axes[0].axvline(x=hard_easy_index, color='black', linestyle='-', label='Hard Easy Threshold')
    axes[0].axvline(x=hard_hard_index, color='black', linestyle='-', label='Hard Hard Threshold')

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
    axes[1].axvline(x=hard_easy_index, color='black', linestyle='-', label='Hard Easy Threshold')
    axes[1].axvline(x=hard_hard_index, color='black', linestyle='-', label='Hard Hard Threshold')
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
    axes[2].axvline(x=hard_easy_index, color='black', linestyle='-', label='Hard Easy Threshold')
    axes[2].axvline(x=hard_hard_index, color='black', linestyle='-', label='Hard Hard Threshold')
    axes[2].set_title(f'Metric {metric_idx + 1} Second Derivatives')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Second Derivative')
    axes[2].grid(True)

    # Save plot
    output_dir = 'metric_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{training}{dataset_name}_metric_{metric_idx + 1}_distribution_gradients_'
                                         f'second_derivatives.pdf'))
    plt.savefig(os.path.join(output_dir, f'{training}{dataset_name}_metric_{metric_idx + 1}_distribution_gradients_'
                                         f'second_derivatives.png'))

    plt.close()


def extract_extreme_samples_threshold(metrics: List[List[float]], labels: List[int], dataset_name: str,
                                      invert: List[bool], training: str, hard_threshold_percent: float = 0.05):
    """Extract easy and hard samples using both adaptive and hard thresholds, returning their indices and
    distributions."""
    num_metrics = len(metrics)

    # Initialize lists for soft (adaptive) and hard threshold indices and distributions
    adaptive_easy_samples = []
    adaptive_hard_samples = []
    adaptive_easy_distributions = []
    adaptive_hard_distributions = []

    hard_easy_samples = []
    hard_hard_samples = []
    hard_easy_distributions = []
    hard_hard_distributions = []

    for metric_idx in range(num_metrics):
        selected_metric = np.array(metrics[metric_idx])
        num_samples = len(selected_metric)
        hard_threshold_count = int(hard_threshold_percent * num_samples)

        # Replace inf values with the maximum finite value
        max_finite_value = np.max(selected_metric[np.isfinite(selected_metric)])
        selected_metric[np.isinf(selected_metric)] = max_finite_value * 2

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

        # Detect family and find division points for adaptive (soft) thresholds
        first_division_point, second_division_point = None, None
        family = detect_family(sorted_normalized_metric, avg_gradients)
        print(f'Metric {metric_idx + 1} is of family {family}.')
        if family == 1:
            first_division_point, second_division_point = find_division_points_for_first_family(second_derivatives)
        elif family == 2:
            first_division_point, second_division_point = find_division_points_for_second_family(smoothed_avg_gradients)
        elif family == 3:
            first_division_point, second_division_point = find_division_points_for_third_family(second_derivatives)

        # Dictionaries to hold the distribution of easy and hard samples per class
        adaptive_easy_dist = defaultdict(int)
        adaptive_hard_dist = defaultdict(int)

        hard_easy_dist = defaultdict(int)
        hard_hard_dist = defaultdict(int)

        # Adaptive (soft) threshold logic based on division points and invert flag
        if first_division_point is not None and second_division_point is not None:
            if invert[metric_idx]:
                # If invert is True: Left side is hard, right side is easy
                adaptive_hard_indices = sorted_indices[:first_division_point]
                adaptive_easy_indices = sorted_indices[second_division_point:]
                hard_hard_indices = sorted_indices[:hard_threshold_count]
                hard_easy_indices = sorted_indices[-hard_threshold_count:]
            else:
                # If invert is False: Left side is easy, right side is hard
                adaptive_easy_indices = sorted_indices[:first_division_point]
                adaptive_hard_indices = sorted_indices[second_division_point:]
                hard_easy_indices = sorted_indices[:hard_threshold_count]
                hard_hard_indices = sorted_indices[-hard_threshold_count:]

            # Store those samples
            adaptive_easy_samples.append(adaptive_easy_indices.tolist())
            adaptive_hard_samples.append(adaptive_hard_indices.tolist())
            hard_easy_samples.append(hard_easy_indices.tolist())
            hard_hard_samples.append(hard_hard_indices.tolist())

            # Compute class distributions for those samples
            for idx in adaptive_easy_indices:
                class_label = labels[idx]
                adaptive_easy_dist[class_label] += 1

            for idx in adaptive_hard_indices:
                class_label = labels[idx]
                adaptive_hard_dist[class_label] += 1

            for idx in hard_easy_indices:
                class_label = labels[idx]
                hard_easy_dist[class_label] += 1

            for idx in hard_hard_indices:
                class_label = labels[idx]
                hard_hard_dist[class_label] += 1
        else:
            raise Exception

        # Append the distributions
        adaptive_easy_distributions.append(adaptive_easy_dist)
        adaptive_hard_distributions.append(adaptive_hard_dist)
        hard_easy_distributions.append(hard_easy_dist)
        hard_hard_distributions.append(hard_hard_dist)

        # Plot results and pass hard thresholds to the plotting function
        plot_metric_results(metric_idx, sorted_normalized_metric, avg_gradients, second_derivatives,
                            first_division_point, second_division_point, dataset_name, invert[metric_idx],
                            hard_threshold_percent, training)

    return (adaptive_easy_samples, adaptive_hard_samples, adaptive_easy_distributions, adaptive_hard_distributions,
            hard_easy_samples, hard_hard_samples, hard_easy_distributions, hard_hard_distributions)


def compare_metrics_to_class_accuracies(class_distributions, avg_class_accuracies, num_classes, pcc_output_filename,
                                        spearman_output_filename):
    """
    Compare the class-level distribution of hard samples to the class-level accuracies
    for each metric by computing Pearson Correlation Coefficient (PCC) and Spearman's Rank Correlation,
    and plotting the results for both correlations.

    :param class_distributions: List of dictionaries, each containing class labels as keys and the count of hardest
    samples as values for each metric.
    :param avg_class_accuracies: The average accuracies for each class.
    :param num_classes: The number of classes in the dataset.
    :param pcc_output_filename: The filename for saving the PCC bar plot.
    :param spearman_output_filename: The filename for saving the Spearman bar plot.
    """
    correlations_pcc = []
    p_values_pcc = []
    correlations_spearman = []
    p_values_spearman = []

    metric_abbreviations = [
        'SameCentroidDist', 'OtherCentroidDist', 'CentroidDistRatio', 'Same1NN', 'Other1NN', '1NNRatio',
        'AvgSame40NN', 'AvgOther40NN', 'AvgAll40NN', 'Avg40NNRatio', '40NNPercSame', '40NNPercOther',
        'AvgSame40NNCurv', 'AvgOther40NNCurv', 'AvgAll40NNCurv', 'GaussCurv', 'MeanCurv'
    ]  # Abbreviations for each metric to keep plot readable.

    # Compute both PCC and Spearman for each metric
    for class_distribution in class_distributions:
        class_level_distribution = [class_distribution.get(cls, 0) for cls in range(num_classes)]

        # Compute Pearson correlation coefficient
        correlation_pcc, p_value_pcc = pearsonr(avg_class_accuracies, class_level_distribution)
        correlations_pcc.append(correlation_pcc)
        p_values_pcc.append(p_value_pcc)

        # Compute Spearman's rank correlation coefficient
        correlation_spearman, p_value_spearman = spearmanr(avg_class_accuracies, class_level_distribution)
        correlations_spearman.append(correlation_spearman)
        p_values_spearman.append(p_value_spearman)

    print()
    print('-' * 20)
    print("PCC p-values:", p_values_pcc)
    print("Spearman p-values:", p_values_spearman)
    print('-' * 20)
    print()

    # Define colors based on p-value significance for PCC
    def get_color_pcc(p_value):
        if p_value < 0.005:
            return 'blue'  # Highly significant
        elif p_value < 0.01:
            return 'dodgerblue'  # Significant
        elif p_value < 0.05:
            return 'lightblue'  # Moderately significant
        else:
            return 'lightgray'  # Not significant

    colors_pcc = [get_color_pcc(p_val) for p_val in p_values_pcc]

    # Define colors based on p-value significance for Spearman
    def get_color_spearman(p_value):
        if p_value < 0.001:
            return 'darkgreen'  # Highly significant
        elif p_value < 0.01:
            return 'green'  # Significant
        elif p_value < 0.05:
            return 'lightgreen'  # Moderately significant
        elif p_value < 0.1:
            return 'yellowgreen'  # Marginally significant
        else:
            return 'lightgray'  # Not significant

    colors_spearman = [get_color_spearman(p_val) for p_val in p_values_spearman]

    # Plot PCCs in a bar chart with horizontal lines
    plt.figure(figsize=(14, 8))
    plt.bar(metric_abbreviations, correlations_pcc, color=colors_pcc)
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

    plt.savefig(os.path.join(u.CORRELATIONS_SAVE_DIR, pcc_output_filename))
    plt.close()

    # Plot Spearman's correlations in a separate bar chart
    plt.figure(figsize=(14, 8))
    plt.bar(metric_abbreviations, correlations_spearman, color=colors_spearman)
    plt.title('Spearman Rank Correlation Between Metrics and Class-Level Accuracies')
    plt.ylabel('Spearman Rank Correlation Coefficient')
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

    plt.savefig(os.path.join(u.CORRELATIONS_SAVE_DIR, spearman_output_filename))
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
            raise Exception('This should not happen.')  # Sanity check

        # Compute the average of each normalized metric for every class
        class_average = defaultdict(list)
        for metric_value, label in zip(metric, labels):
            class_average[label].append(metric_value)

        # Compute the mean of each class
        for class_label, values in class_average.items():
            class_average[class_label] = np.mean(values)

        class_averages.append(class_average)

    print('\nThe computed class averages are as follows:')
    for metric_idx in range(len(metrics)):
        print(f'\t{class_averages[metric_idx]}')
    print('-'*20, '\n')
    return class_averages


def compute_correlation_heatmaps(easy_distribution, hard_distribution, threshold, output_dir="heatmaps"):
    num_metrics = len(easy_distribution)

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
    plt.savefig(os.path.join(output_dir, f"{threshold}_easy_overlap_heatmap.pdf"))
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(hard_overlap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Hard Samples Overlap Heatmap")
    plt.savefig(os.path.join(output_dir, f"{threshold}_hard_overlap_heatmap.pdf"))
    plt.close()


def compute_easy_hard_ratios(dataset_name: str, easy_indices: List[np.ndarray], hard_indices: List[np.ndarray]) -> None:
    """Compute the ratio of easy to hard samples in both training and test splits for each metric and display as a
    table."""

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


def plot_correlation_metrics(metric_abbreviations, iou_values, spearman_values, kendall_values, output_filename):
    """
    Plot IoU, Spearman's, and Kendall's Tau for each metric.

    :param metric_abbreviations: List of abbreviations for each metric.
    :param iou_values: List of IoU values for each metric.
    :param spearman_values: List of Spearman's correlations for each metric.
    :param kendall_values: List of Kendall's Tau correlations for each metric.
    :param output_filename: The filename for saving the plot.
    """
    num_metrics = len(metric_abbreviations)
    x = range(num_metrics)

    plt.figure(figsize=(14, 8))

    # Plot IoU
    plt.bar(x, iou_values, width=0.2, label='IoU', color='lightblue', align='center')

    # Plot Spearman's
    plt.bar([i + 0.2 for i in x], spearman_values, width=0.2, label="Spearman's", color='lightgreen', align='center')

    # Plot Kendall's Tau
    plt.bar([i + 0.4 for i in x], kendall_values, width=0.2, label="Kendall's Tau", color='lightcoral', align='center')

    plt.xticks(x, metric_abbreviations, rotation=45, ha='right')
    plt.title('Comparison of IoU, Spearman’s, and Kendall’s Tau Across Metrics')
    plt.ylabel('Correlation / Overlap Value')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_filename)
    plt.close()


def compute_iou(adaptive_indices, full_indices):
    """
    Compute the IoU (Intersection Over Union) of samples from adaptive_indices and full_indices.

    :param adaptive_indices: List of indices for the adaptive set.
    :param full_indices: List of indices for the full set.
    :return: IoU value as a percentage.
    """
    adaptive_set = set(adaptive_indices)
    full_set = set(full_indices)
    overlap_count = len(adaptive_set.intersection(full_set))
    iou = (overlap_count / len(adaptive_set)) * 100
    return iou


def main(dataset_name: str, models_count: int, training: str, threshold: float, k2: int):
    # Define file paths for saving and loading cached results
    accuracies_file = f"{u.HARD_IMBALANCE_DIR}{dataset_name}_avg_class_accuracies.pkl"
    proximity_file = f"{u.HARD_IMBALANCE_DIR}{training}{dataset_name}_proximity_indicators.pkl"

    # Load the dataset
    if training == 'full':
        training_dataset = u.load_full_data_and_normalize(dataset_name)
    else:
        training_dataset, _ = u.load_data_and_normalize(dataset_name)
    training_labels = training_dataset.tensors[1].numpy()
    num_classes = len(np.unique(training_labels))
    metric_abbreviations = [
        'SameCentroidDist', 'OtherCentroidDist', 'CentroidDistRatio', 'Same1NNDist', 'Other1NNDist', '1NNRatioDist',
        'AvgSame40NNDist', 'AvgOther40NNDist', 'AvgAll40NNDist', 'Avg40NNDistRatio', '40NNPercSame', '40NNPercOther'
    ]

    if os.path.exists(accuracies_file):
        print('Loading accuracies.')
        avg_class_accuracies = u.load_data(accuracies_file)
    else:
        raise Exception('Train an ensemble via `train_ensembles.py --training full` before running this program.')
    loader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=False)

    if os.path.exists(proximity_file):
        print('Loading proximities.')
        proximity_metrics = u.load_data(proximity_file)
    else:
        print('Calculating proximities.')
        proximity_metrics = compute_proximity_metrics(loader, k2)
        u.save_data(proximity_metrics, proximity_file)

    all_metrics = proximity_metrics
    class_averages = compute_class_averages_of_metrics(all_metrics, training_labels)
    invert_metrics = [False, True, False, False, True, False, False, True, False, False, True, False]

    # Extract the hardest samples for each metric and compute their class distributions
    adaptive_easy_indices, adaptive_hard_indices, adaptive_easy_distributions, adaptive_hard_distributions, \
        fixed_easy_indices, fixed_hard_indices, fixed_easy_distributions, fixed_hard_distributions = \
        extract_extreme_samples_threshold(all_metrics, training_labels, dataset_name, invert_metrics, training)

    # Compute and visualize the correlation heatmaps
    # compute_correlation_heatmaps(adaptive_easy_indices, adaptive_hard_indices, 'adaptive')
    # compute_correlation_heatmaps(fixed_easy_indices, fixed_hard_indices, 'fixed')

    if training == 'full':
        # Compare and plot all metrics against class-level accuracies
        compare_metrics_to_class_accuracies(adaptive_easy_distributions, avg_class_accuracies, num_classes,
                                            f'{training}{dataset_name}_adaptive_easyPCC.pdf',
                                            f'{training}{dataset_name}_adaptive_easySRC.pdf')
        compare_metrics_to_class_accuracies(adaptive_hard_distributions, avg_class_accuracies, num_classes,
                                            f'{training}{dataset_name}_adaptive_hardPCC.pdf',
                                            f'{training}{dataset_name}_adaptive_hardSRC.pdf')
        compare_metrics_to_class_accuracies(fixed_easy_distributions, avg_class_accuracies, num_classes,
                                            f'{training}{dataset_name}_fixed_easyPCC.pdf',
                                            f'{training}{dataset_name}_fixed_easySRC.pdf')
        compare_metrics_to_class_accuracies(fixed_hard_distributions, avg_class_accuracies, num_classes,
                                            f'{training}{dataset_name}_fixed_hardPCC.pdf',
                                            f'{training}{dataset_name}_fixed_hardSRC.pdf')
        compare_metrics_to_class_accuracies(class_averages, avg_class_accuracies, num_classes,
                                            f'{training}{dataset_name}_avgPCC.pdf',
                                            f'{training}{dataset_name}_avgSRC.pdf')
        # compute_easy_hard_ratios(dataset_name, adaptive_easy_indices, adaptive_hard_indices)
    else:
        full_easy_indices = u.load_data(f'{u.DIVISIONS_SAVE_DIR}/full{dataset_name}_adaptive_easy_indices.pkl')
        full_hard_indices = u.load_data(f'{u.DIVISIONS_SAVE_DIR}/full{dataset_name}_adaptive_hard_indices.pkl')

        iou_values_easy = []
        iou_values_hard = []
        spearman_values_easy = []
        spearman_values_hard = []
        kendall_values_easy = []
        kendall_values_hard = []

        for metric_idx, (adaptive_easy, adaptive_hard) in enumerate(zip(adaptive_easy_indices, adaptive_hard_indices)):
            # Compute IoU
            easy_iou = compute_iou(adaptive_easy, full_easy_indices[metric_idx])
            hard_iou = compute_iou(adaptive_hard, full_hard_indices[metric_idx])

            iou_values_easy.append(easy_iou)
            iou_values_hard.append(hard_iou)

            # Compute Spearman's
            easy_spearman, _ = spearmanr(adaptive_easy, full_easy_indices[metric_idx])
            hard_spearman, _ = spearmanr(adaptive_hard, full_hard_indices[metric_idx])

            spearman_values_easy.append(easy_spearman)
            spearman_values_hard.append(hard_spearman)

            # Compute Kendall's Tau
            easy_kendall, _ = kendalltau(adaptive_easy, full_easy_indices[metric_idx])
            hard_kendall, _ = kendalltau(adaptive_hard, full_hard_indices[metric_idx])

            kendall_values_easy.append(easy_kendall)
            kendall_values_hard.append(hard_kendall)

        # Call the plotting function
        plot_correlation_metrics(metric_abbreviations, iou_values_easy, spearman_values_easy, kendall_values_easy,
                                 f'{training}{dataset_name}_easy_correlations.pdf')
        plot_correlation_metrics(metric_abbreviations, iou_values_hard, spearman_values_hard, kendall_values_hard,
                                 f'{training}{dataset_name}_hard_correlations.pdf')

    u.save_data(adaptive_easy_indices, f'{u.DIVISIONS_SAVE_DIR}/{training}{dataset_name}_adaptive_easy_indices.pkl')
    u.save_data(adaptive_hard_indices, f'{u.DIVISIONS_SAVE_DIR}/{training}{dataset_name}_adaptive_hard_indices.pkl')
    u.save_data(fixed_easy_indices, f'{u.DIVISIONS_SAVE_DIR}/{training}{dataset_name}_fixed_easy_indices.pkl')
    u.save_data(fixed_hard_indices, f'{u.DIVISIONS_SAVE_DIR}/{training}{dataset_name}_fixed_hard_indices.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze hard samples in the official training and test splits using precomputed hardness '
                    'indicators.'
    )
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset (MNIST, CIFAR10, CIFAR100).')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--training', type=str, choices=['full', 'part'], default='full',
                        help='Indicates which models to choose for evaluations - the ones trained on the entire dataset'
                             ' (full), or the ones trained only on the training set (part).')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='The percentage of the most extreme (hardest) samples that will be considered as hard.')
    parser.add_argument('--k2', type=int, default=10, help='k parameter for the kNN in proximity computations.')
    args = parser.parse_args()

    main(**vars(args))
