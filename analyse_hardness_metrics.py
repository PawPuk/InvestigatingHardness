import argparse
from collections import defaultdict
from glob import glob
import os
from typing import List, Tuple

from cleanlab.rank import get_label_quality_scores
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kendalltau, pearsonr, spearmanr
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import torch
from tqdm import tqdm

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
    # Check if the rightmost gradients are larger than the leftmost ones
    condition1 = True if np.mean(avg_gradients[-100:]) > np.mean(avg_gradients[:100]) else False
    # Check if the leftmost gradients are higher than average gradient
    condition2 = True if np.mean(avg_gradients[:100]) > np.mean(avg_gradients) else False
    # Check if the rightmost gradients are higher than average gradient
    condition3 = True if np.mean(avg_gradients[-100:]) > np.mean(avg_gradients) else False
    # Check if the leftmost gradients are higher than the threshold
    threshold = 0.025 * np.max(avg_gradients)
    condition4 = True if threshold < np.mean(avg_gradients[:100]) else False
    print(condition1, condition2, condition3, condition4)
    return condition1 and condition4


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


def find_division_points_for_first_family(first_derivatives: np.ndarray, percentage: float = 2.5,
                                          window_size: int = 10000) -> Tuple[int, int]:
    """Find the first and second division points based on the second derivative analysis. U-shaped first derivative"""
    max_value = np.max(first_derivatives)
    min_value = 0 if np.min(first_derivatives) < max_value / 100 else np.min(first_derivatives)
    values_range = max_value - min_value
    left_threshold = min_value + (percentage / 100) * values_range
    right_threshold = min_value + (percentage / 100) * values_range
    first_division_point, second_division_point = None, None

    # Find the first division point (from left to right)
    for i in range(0, len(first_derivatives) - window_size):
        window = first_derivatives[i:i + window_size]
        if np.all(window < left_threshold):
            first_division_point = i
            break

    # Find the second division point (from right to left)
    for i in range(len(first_derivatives) - 1, window_size, -1):
        window = first_derivatives[i - window_size: i]
        if np.all(window < right_threshold):
            second_division_point = i
            break
    if first_division_point is None or second_division_point is None:
        print()
        print(first_division_point, second_division_point)
        print(left_threshold, right_threshold)
        print(max_value)
        raise Exception
    return max(first_division_point, 1000), min(second_division_point, len(first_derivatives) - 1000)


def find_division_points_for_second_family(first_derivatives: np.ndarray, window_size: int = 100,
                                           percentage: float = 2.5) -> Tuple[int, int]:
    max_value = np.max(first_derivatives)
    min_value = 0 if np.min(first_derivatives) < max_value / 100 else np.min(first_derivatives)
    values_range = max_value - min_value
    threshold_value = min_value + (percentage / 100) * values_range
    for i in range(len(first_derivatives) - 1, window_size, -1):
        window = first_derivatives[i - window_size: i]
        if np.all(window > threshold_value):
            return i, i
    print('b' * 25)
    return 1000, 1000


def find_division_points_for_third_family(first_derivatives: np.ndarray, window_size: int = 100,
                                          percentage: float = 2.5) -> Tuple[int, int]:

    if np.mean(first_derivatives[:100]) == 0:
        start_index = 0  # Start from zero if the first 500 derivatives are zero
    else:
        start_index = 20000  # Otherwise, start from 20000

    max_value = np.max(first_derivatives[-window_size:])
    min_value = 0 if np.min(first_derivatives) < max_value / 100 else np.min(first_derivatives)
    values_range = max_value - min_value
    threshold_value = min_value + (percentage / 100) * values_range

    for i in range(start_index, len(first_derivatives) - window_size + 1):
        window = first_derivatives[i:i + window_size]
        if np.all(window > threshold_value):
            return i, i
    print(start_index, 'c' * 25)
    return len(first_derivatives) - 1000, len(first_derivatives) - 1000


def plot_metric_results(metric_idx: int, sorted_normalized_metric: np.ndarray, avg_gradients: List[np.ndarray],
                        first_division_point: int, second_division_point: int, dataset_name: str, invert: bool,
                        hard_threshold_percent: float, training: str, model_type: str, abbreviations: List[str]):
    """Plot the results with division points marked and areas colored as easy, medium, and hard, along with hard
    thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

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

    axes[0].set_title(f'Distribution of Values of {abbreviations[metric_idx]}')
    axes[0].set_xlabel('Sample Index (Sorted)')
    axes[0].set_ylabel('Normalized Value')
    axes[0].grid(True)
    # Plot average gradient
    axes[1].plot(avg_gradients, marker='x', linestyle='-', color='r')
    if first_division_point is not None:
        axes[1].axvline(x=first_division_point, color='blue', linestyle='--', label='First Division')
    if second_division_point is not None:
        axes[1].axvline(x=second_division_point, color='blue', linestyle='--', label='Second Division')
    axes[1].axvline(x=hard_easy_index, color='black', linestyle='-', label='Hard Easy Threshold')
    axes[1].axvline(x=hard_hard_index, color='black', linestyle='-', label='Hard Hard Threshold')
    axes[1].set_title(f'Gradients of Normalized Values')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Average Gradient')
    axes[1].grid(True)

    # Save plot
    output_dir = 'metric_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{training}{model_type}{dataset_name}_metric_{metric_idx + 1}_'
                                         f'distribution_gradients_.pdf'))
    plt.savefig(os.path.join(output_dir, f'{training}{model_type}{dataset_name}_metric_{metric_idx + 1}_'
                                         f'distribution_gradients_.png'))

    plt.close()


def extract_extreme_samples_threshold(metrics: List[List[float]], labels: List[int], dataset_name: str, model_type: str,
                                      invert: List[bool], training: str, metric_abbreviations: List[str],
                                      fixed_threshold_percent: float = 0.15):
    """Extract easy and hard samples using both adaptive and hard thresholds, returning their indices and
    distributions."""
    num_metrics = len(metrics)

    # Initialize lists for soft (adaptive) and hard threshold indices and distributions
    adaptive_easy_samples = []
    adaptive_hard_samples = []
    adaptive_easy_distributions = []
    adaptive_hard_distributions = []

    fixed_easy_samples = []
    fixed_hard_samples = []
    fixed_easy_distributions = []
    fixed_hard_distributions = []

    for metric_idx in range(num_metrics):
        selected_metric = np.array(metrics[metric_idx])
        num_samples = len(selected_metric)
        fixed_threshold = int(fixed_threshold_percent * num_samples)

        # Replace inf values with twice the maximum finite value
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

        # Compute average gradient using 1000 points to the left and 1000 to the right
        avg_gradients = []
        window_size = 1000
        for i in range(window_size, num_samples - window_size):
            start = max(0, i - window_size)
            end = min(num_samples, i + window_size + 1)
            avg_gradients.append(np.mean(gradients[start:end]))
        # Smooth the first derivative using Savitzky-Golay filter
        smoothed_avg_gradients = savgol_filter(avg_gradients, window_length=1001, polyorder=2)

        # Detect family and find division points for adaptive (soft) thresholds
        first_division_point, second_division_point = None, None
        family = detect_family(sorted_normalized_metric, avg_gradients)
        print(f'Metric {metric_idx + 1} is of family {family}.')
        if family == 1:
            first_division_point, second_division_point = find_division_points_for_first_family(np.array(avg_gradients))
        elif family == 2:
            first_division_point, second_division_point = find_division_points_for_second_family(smoothed_avg_gradients)
        elif family == 3:
            first_division_point, second_division_point = find_division_points_for_third_family(np.array(avg_gradients))

        # Dictionaries to hold the distribution of easy and hard samples per class
        adaptive_easy_dist = defaultdict(int)
        adaptive_hard_dist = defaultdict(int)

        fixed_easy_dist = defaultdict(int)
        fixed_hard_dist = defaultdict(int)

        # Adaptive (soft) threshold logic based on division points and invert flag
        if first_division_point is not None and second_division_point is not None:
            if invert[metric_idx]:
                # If invert is True: Left side is hard, right side is easy
                adaptive_hard_indices = sorted_indices[:first_division_point]
                adaptive_easy_indices = sorted_indices[second_division_point:]
                hard_hard_indices = sorted_indices[:fixed_threshold]
                hard_easy_indices = sorted_indices[-fixed_threshold:]
            else:
                # If invert is False: Left side is easy, right side is hard
                adaptive_easy_indices = sorted_indices[:first_division_point]
                adaptive_hard_indices = sorted_indices[second_division_point:]
                hard_easy_indices = sorted_indices[:fixed_threshold]
                hard_hard_indices = sorted_indices[-fixed_threshold:]

            # Store those samples
            adaptive_easy_samples.append(adaptive_easy_indices.tolist())
            adaptive_hard_samples.append(adaptive_hard_indices.tolist())
            fixed_easy_samples.append(hard_easy_indices.tolist())
            fixed_hard_samples.append(hard_hard_indices.tolist())

            # Compute class distributions for those samples
            for idx in adaptive_easy_indices:
                class_label = labels[idx]
                adaptive_easy_dist[class_label] += 1

            for idx in adaptive_hard_indices:
                class_label = labels[idx]
                adaptive_hard_dist[class_label] += 1

            for idx in hard_easy_indices:
                class_label = labels[idx]
                fixed_easy_dist[class_label] += 1

            for idx in hard_hard_indices:
                class_label = labels[idx]
                fixed_hard_dist[class_label] += 1
        else:
            raise Exception

        # Append the distributions
        adaptive_easy_distributions.append(adaptive_easy_dist)
        adaptive_hard_distributions.append(adaptive_hard_dist)
        fixed_easy_distributions.append(fixed_easy_dist)
        fixed_hard_distributions.append(fixed_hard_dist)

        # Plot results and pass hard thresholds to the plotting function
        plot_metric_results(metric_idx, sorted_normalized_metric, avg_gradients, first_division_point,
                            second_division_point, dataset_name, invert[metric_idx], fixed_threshold_percent, training,
                            model_type, metric_abbreviations)

    return (adaptive_easy_samples, adaptive_hard_samples, fixed_easy_samples, fixed_hard_samples), \
        (adaptive_easy_distributions, adaptive_hard_distributions, fixed_easy_distributions, fixed_hard_distributions)


def compare_metrics_to_class_accuracies(class_distributions, avg_class_accuracies, num_classes,  metric_abbreviations,
                                        dataset_name, grayscale, pca, pcc_output_filename, spearman_output_filename):
    """
    Compare the class-level distribution of hard samples to the class-level accuracies
    for each metric by computing Pearson Correlation Coefficient (PCC) and Spearman's Rank Correlation,
    and plotting the results for both correlations.

    :param class_distributions: List of dictionaries, each containing class labels as keys and the count of the hardest
    samples as values for each metric.
    :param avg_class_accuracies: The average accuracies for each class.
    :param num_classes: The number of classes in the dataset.
    :param metric_abbreviations: Abbreviations of the metric used in the Figure
    :param pcc_output_filename: The filename for saving the PCC bar plot.
    :param spearman_output_filename: The filename for saving the Spearman bar plot.
    """
    if dataset_name == 'CIFAR10':
        if grayscale:
            pcc_output_filename += 'gray'
            spearman_output_filename += 'gray'
        if pca:
            pcc_output_filename += 'pca'
            spearman_output_filename += 'pca'
    pcc_output_filename += '.pdf'
    spearman_output_filename += '.pdf'
    correlations_pcc, p_values_pcc, correlations_spearman, p_values_spearman = [], [], [], []

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
    print("PCC correlations:", correlations_pcc)
    print("PCC p-values:", p_values_pcc)
    print("Spearman correlations:", correlations_spearman)
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


def visualize_evaluation_results(hard_indices, abbreviations, dataset_name, grayscale, pca, save_path,
                                 num_benchmark_methods=3):
    """
    Visualize the evaluation of normal methods against benchmark methods using heatmaps and tables.

    :param hard_indices: List of lists, where the last `num_benchmark_methods` are benchmark methods,
                         and the others are normal methods.
    :param abbreviations: Abbreviations for each of the metrics used.
    :param dataset_name: name of the dataset
    :param grayscale: Bool flag.
    :param pca: Bool flag.
    :param save_path: Path used to save the heatmaps.
    :param num_benchmark_methods: Number of benchmark methods at the end of the list.
    """
    results = compute_correlation_heatmaps(hard_indices, num_benchmark_methods)

    # Reshape the recall, and AUC results into matrices for heatmap plotting
    num_normal_methods = len(hard_indices) - num_benchmark_methods
    recall_matrix = np.array(results['recall']).reshape(num_normal_methods, num_benchmark_methods)
    auc_matrix = np.array(results['auc']).reshape(num_normal_methods, num_benchmark_methods)

    # Create a dataframe for each metric for the table view
    df_recall = pd.DataFrame(recall_matrix, index=[f'{abbreviations[i]}' for i in range(num_normal_methods)],
                             columns=[f'{abbreviations[num_normal_methods + i]}' for i in range(num_benchmark_methods)])
    df_auc = pd.DataFrame(auc_matrix, index=[f'{abbreviations[i]}' for i in range(num_normal_methods)],
                          columns=[f'{abbreviations[num_normal_methods + i]}' for i in range(num_benchmark_methods)])

    # Print the tables
    print("\nRecall Scores:\n", df_recall)
    print("\nAUC Scores:\n", df_auc)

    # Heatmap for Recall (range 0 to 1)
    plt.figure(figsize=(7, 6))
    sns.heatmap(recall_matrix, annot=True, cmap='Greens', cbar=True, vmin=0, vmax=1,  # Fix the color range
                xticklabels=abbreviations[num_normal_methods:], yticklabels=abbreviations[:num_normal_methods])
    plt.yticks(rotation=0)
    plt.title('Recall Heatmap')
    plt.tight_layout()
    plt.savefig(save_path + 'recall_heatmap.pdf')  # Save recall heatmap
    plt.close()  # Close the plot to avoid overwriting

    # Heatmap for AUC (range 0 to 1)
    plt.figure(figsize=(7, 6))
    sns.heatmap(auc_matrix, annot=True, cmap='Purples', cbar=True, vmin=0, vmax=1,  # Fix the color range
                xticklabels=abbreviations[num_normal_methods:], yticklabels=abbreviations[:num_normal_methods])
    plt.yticks(rotation=0)
    plt.title('AUC Heatmap')
    plt.tight_layout()
    if dataset_name == 'CIFAR10':
        if grayscale:
            save_path += 'gray'
        if pca:
            save_path += 'pca'
    plt.savefig(save_path + 'auc_heatmap.pdf')  # Save AUC heatmap
    plt.close()  # Close the plot to avoid overwriting


def compute_correlation_heatmaps(hard_indices, num_benchmark_methods=3):
    benchmark_methods = hard_indices[-num_benchmark_methods:]  # Ground truth (benchmark)
    evaluated_methods = hard_indices[:-num_benchmark_methods]  # Methods to evaluate

    # Initialize lists to store results
    recall_values, auc_scores = [], []

    for metric_idx, metric_hard_indices in enumerate(evaluated_methods):
        metric_hard_set = set(metric_hard_indices)
        for benchmark_idx, benchmark_hard_indices in enumerate(benchmark_methods):
            benchmark_hard_set = set(benchmark_hard_indices)

            # Recall
            tp = len(metric_hard_set & benchmark_hard_set)
            fn = len(benchmark_hard_set - metric_hard_set)
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            recall_values.append(recall)

            # ROC Curve and AUC
            all_samples = set(range(max(max(metric_hard_indices), max(benchmark_hard_indices)) + 1))
            method_binary = np.array([1 if i in metric_hard_set else 0 for i in all_samples])
            benchmark_binary = np.array([1 if i in benchmark_hard_set else 0 for i in all_samples])

            # ROC curve and AUC
            fpr, tpr, _ = roc_curve(benchmark_binary, method_binary)
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)

    # Return collected precision, recall, F1, and AUC scores (could also return or save these)
    return {
        "recall": recall_values,
        "auc": auc_scores
    }


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
    test_indices = list(range(len(train_targets), len(train_targets) + len(test_targets)))

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


def plot_consistency_metrics(metric_abbreviations, iou_values, spearman_values, kendall_values, output_filename):
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
    iou = (overlap_count / len(adaptive_set))
    return iou


def main(dataset_name: str, model_type: str, ensemble_size: str, grayscale: bool, pca: bool):
    # Define file paths for saving and loading cached results
    full_accuracies_file = f"{u.METRICS_SAVE_DIR}{ensemble_size}_full{dataset_name}_avg_class_accuracies_on_" \
                           f"{model_type}ensemble.pkl"
    part_accuracies_file = f"{u.METRICS_SAVE_DIR}{ensemble_size}_part{dataset_name}_avg_class_accuracies_on_" \
                           f"{model_type}ensemble.pkl"
    full_proximity_file = f"{u.METRICS_SAVE_DIR}full{dataset_name}"
    part_proximity_file = f"{u.METRICS_SAVE_DIR}part{dataset_name}"
    full_curvature_file = f"{u.METRICS_SAVE_DIR}full{dataset_name}"
    part_curvature_file = f"{u.METRICS_SAVE_DIR}part{dataset_name}"
    full_save_dir = f"{u.DIVISIONS_SAVE_DIR}/full{model_type}{dataset_name}_indices"
    part_save_dir = f"{u.DIVISIONS_SAVE_DIR}/part{model_type}{dataset_name}_indices"
    if dataset_name == 'CIFAR10':
        if grayscale:
            full_proximity_file += 'gray'
            part_proximity_file += 'gray'
            full_curvature_file += 'gray'
            part_curvature_file += 'gray'
            full_save_dir += 'gray'
            part_save_dir += 'gray'
        if pca:
            full_proximity_file += 'pca'
            part_proximity_file += 'pca'
            full_curvature_file += 'pca'
            part_curvature_file += 'pca'
            full_save_dir += 'pca'
            part_save_dir += 'pca'
    full_proximity_file += "_proximity_indicators.pkl"
    part_proximity_file += "_proximity_indicators.pkl"
    full_curvature_file += "_curvature_indicators.pkl"
    part_curvature_file += "_curvature_indicators.pkl"
    full_save_dir += '.pkl'
    part_save_dir += '.pkl'
    full_model_file = f"{u.METRICS_SAVE_DIR}{ensemble_size}_full{dataset_name}_{model_type}_model_based_indicators.pkl"
    part_model_file = f"{u.METRICS_SAVE_DIR}{ensemble_size}_part{dataset_name}_{model_type}_model_based_indicators.pkl"

    metric_abbreviations = [
        'SameCentroidDist', 'Same1NNDist', 'AvgSame40NNDist',
        'OtherCentroidDist', 'Other1NNDist', 'AvgOther40NNDist', '40NNPercOther', 'N3',
        'CentroidDistRatio', '1NNRatioDist', 'Avg40NNDistRatio',
        'AvgAll40NNDist',
        'GaussCurv', 'MeanCurv',
        'Cleanlab', 'EL2N', 'VoG', 'Margin'
    ]

    # Load the dataset and metrics
    full_training_dataset = u.load_full_data_and_normalize(dataset_name)
    part_training_dataset, _ = u.load_data_and_normalize(dataset_name)
    full_avg_class_accuracies = u.load_data(full_accuracies_file)
    full_avg_class_accuracies = np.sum(full_avg_class_accuracies, axis=0) / len(full_avg_class_accuracies)
    part_avg_class_accuracies = u.load_data(part_accuracies_file)
    part_avg_class_accuracies = np.sum(part_avg_class_accuracies, axis=0) / len(part_avg_class_accuracies)
    full_proximity_metrics = u.load_data(full_proximity_file)
    part_proximity_metrics = u.load_data(part_proximity_file)
    full_curvature_metrics = u.load_data(full_curvature_file)
    part_curvature_metrics = u.load_data(part_curvature_file)
    full_model_metrics = u.load_data(full_model_file)
    part_model_metrics = u.load_data(part_model_file)
    full_training_labels = full_training_dataset.tensors[1].numpy()
    part_training_labels = part_training_dataset.tensors[1].numpy()
    num_classes = len(np.unique(full_training_labels))

    invert_metrics = [False, False, False,  # For metrics measuring intra-class structure
                      True, True, True, False, False,  # For metrics measuring separation from other classes
                      False, False, False,  # For radio-based metrics
                      False,  # For density based-metrics
                      False, False,  # For curvature-based metrics
                      True, False, False, True]  # For model-based metrics

    metric_abbreviations.pop(-2)
    invert_metrics.pop(-2)

    full_hardness_metrics = full_proximity_metrics + full_curvature_metrics + full_model_metrics
    part_hardness_metrics = part_proximity_metrics + part_curvature_metrics + part_model_metrics
    full_class_averages = compute_class_averages_of_metrics(full_hardness_metrics, full_training_labels)

    # Extract the hardest samples for each metric and compute their class distributions
    full_indices, full_distributions = extract_extreme_samples_threshold(full_hardness_metrics, full_training_labels,
                                                                         dataset_name, model_type, invert_metrics,
                                                                         'full', metric_abbreviations)
    part_indices, part_distributions = extract_extreme_samples_threshold(part_hardness_metrics, part_training_labels,
                                                                         dataset_name, model_type, invert_metrics,
                                                                         'part', metric_abbreviations)

    # Compare the indices obtained via data-based approaches and model-based approaches (use the latter as ground truth)
    visualize_evaluation_results(full_indices[0], metric_abbreviations, dataset_name, grayscale, pca,
                                 f"{u.HEATMAP_SAVE_DIR}{model_type}_adaptive_easy_full{dataset_name}_")
    visualize_evaluation_results(part_indices[0], metric_abbreviations, dataset_name, grayscale, pca,
                                 f"{u.HEATMAP_SAVE_DIR}{model_type}_adaptive_easy_part{dataset_name}_")
    visualize_evaluation_results(full_indices[1], metric_abbreviations, dataset_name, grayscale, pca,
                                 f"{u.HEATMAP_SAVE_DIR}{model_type}_adaptive_hard_full{dataset_name}_")
    visualize_evaluation_results(part_indices[1], metric_abbreviations, dataset_name, grayscale, pca,
                                 f"{u.HEATMAP_SAVE_DIR}{model_type}_adaptive_hard_part{dataset_name}_")
    visualize_evaluation_results(full_indices[2], metric_abbreviations, dataset_name, grayscale, pca,
                                 f"{u.HEATMAP_SAVE_DIR}{model_type}_fixed_easy_full{dataset_name}_")
    visualize_evaluation_results(part_indices[2], metric_abbreviations, dataset_name, grayscale, pca,
                                 f"{u.HEATMAP_SAVE_DIR}{model_type}_fixed_easy_part{dataset_name}_")
    visualize_evaluation_results(full_indices[3], metric_abbreviations, dataset_name, grayscale, pca,
                                 f"{u.HEATMAP_SAVE_DIR}{model_type}_fixed_hard_full{dataset_name}_")
    visualize_evaluation_results(part_indices[3], metric_abbreviations, dataset_name, grayscale, pca,
                                 f"{u.HEATMAP_SAVE_DIR}{model_type}_fixed_hard_part{dataset_name}_")

    # Measure the correlation between the distributions of hard samples and class-level accuracies
    for training in ['full', 'part']:
        distributions = [full_distributions, part_distributions][training == 'part']
        accuracies = [full_avg_class_accuracies, part_avg_class_accuracies][training == 'part']
        compare_metrics_to_class_accuracies(distributions[0], accuracies, num_classes, metric_abbreviations,
                                            dataset_name, grayscale, pca,
                                            f'{training}{model_type}{dataset_name}_adaptive_easyPCC',
                                            f'{training}{model_type}{dataset_name}_adaptive_easySRC')
        compare_metrics_to_class_accuracies(distributions[1], accuracies, num_classes, metric_abbreviations,
                                            dataset_name, grayscale, pca,
                                            f'{training}{model_type}{dataset_name}_adaptive_hardPCC',
                                            f'{training}{model_type}{dataset_name}_adaptive_hardSRC')
        compare_metrics_to_class_accuracies(distributions[2], accuracies, num_classes, metric_abbreviations,
                                            dataset_name, grayscale, pca,
                                            f'{training}{model_type}{dataset_name}_fixed_easyPCC',
                                            f'{training}{model_type}{dataset_name}_fixed_easySRC')
        compare_metrics_to_class_accuracies(distributions[3], accuracies, num_classes, metric_abbreviations,
                                            dataset_name, grayscale, pca,
                                            f'{training}{model_type}{dataset_name}_fixed_hardPCC',
                                            f'{training}{model_type}{dataset_name}_fixed_hardSRC')
        compare_metrics_to_class_accuracies(full_class_averages, accuracies, num_classes, metric_abbreviations,
                                            dataset_name, grayscale, pca,
                                            f'{training}{model_type}{dataset_name}_avgPCC',
                                            f'{training}{model_type}{dataset_name}_avgSRC')

    # Measure the ratio of easy:hard samples in the training and test splits proposed by PyTorch
    """print('Computing easy:hard ratios obtained by adaptive threshold in training and test splits proposed by PyTorch.')
    compute_easy_hard_ratios(dataset_name, full_indices[0], full_indices[1])
    print('\nComputing easy:hard ratios obtained by fixed threshold in training and test splits proposed by PyTorch.')
    compute_easy_hard_ratios(dataset_name, full_indices[2], full_indices[3])"""

    """adaptive_iou_values_easy = []
    adaptive_iou_values_hard = []
    adaptive_spearman_values_easy = []
    adaptive_spearman_values_hard = []
    adaptive_kendall_values_easy = []
    adaptive_kendall_values_hard = []

    fixed_iou_values_easy = []
    fixed_iou_values_hard = []
    fixed_spearman_values_easy = []
    fixed_spearman_values_hard = []
    fixed_kendall_values_easy = []
    fixed_kendall_values_hard = []
    # TODO: Modify the below to work with fixes
    for metric_idx in range(len(part_indices[2])):
        adaptive_easy, adaptive_hard = zip(part_indices[0], part_indices[1])
        fixed_easy, fixed_hard = zip(part_indices[2], part_indices[3])

        # Compute IoU
        adaptive_easy_iou = compute_iou(adaptive_easy, full_indices[0][metric_idx])
        adaptive_hard_iou = compute_iou(adaptive_hard, full_indices[1][metric_idx])
        fixed_easy_iou = compute_iou(fixed_easy, full_indices[2][metric_idx])
        fixed_hard_iou = compute_iou(fixed_hard, full_indices[3][metric_idx])
        adaptive_iou_values_easy.append(adaptive_easy_iou)
        adaptive_iou_values_hard.append(adaptive_hard_iou)
        fixed_iou_values_easy.append(fixed_easy_iou)
        fixed_iou_values_hard.append(fixed_hard_iou)

        # Adjust the easy and hard indices obtained via full and part setting to have the same length
        if len(adaptive_easy) > len(full_indices[0][metric_idx]):
            adaptive_easy = adaptive_easy[:len(full_indices[0][metric_idx])]
        else:
            full_indices[0][metric_idx] = full_indices[0][metric_idx][:len(adaptive_easy)]
        if len(adaptive_hard) > len(full_indices[1][metric_idx]):
            adaptive_hard = adaptive_hard[-len(full_indices[1][metric_idx]):]
        else:
            full_indices[1][metric_idx] = full_indices[1][metric_idx][-len(adaptive_hard):]

        # Compute Spearman's
        easy_spearman, _ = spearmanr(adaptive_easy, full_indices[0][metric_idx])
        hard_spearman, _ = spearmanr(adaptive_hard, full_indices[1][metric_idx])
        spearman_values_easy.append(easy_spearman)
        spearman_values_hard.append(hard_spearman)

        # Compute Kendall's Tau
        easy_kendall, _ = kendalltau(adaptive_easy, full_indices[0][metric_idx])
        hard_kendall, _ = kendalltau(adaptive_hard, full_indices[1][metric_idx])
        kendall_values_easy.append(easy_kendall)
        kendall_values_hard.append(hard_kendall)
        # TODO: add p-values and incorporate them into the plot

    # Call the plotting function
    plot_consistency_metrics(metric_abbreviations, iou_values_easy, spearman_values_easy, kendall_values_easy,
                             f'{u.CONSISTENCY_SAVE_DIR}{model_type}{dataset_name}_easy_consistency.pdf')
    plot_consistency_metrics(metric_abbreviations, iou_values_hard, spearman_values_hard, kendall_values_hard,
                             f'{u.CONSISTENCY_SAVE_DIR}{model_type}{dataset_name}_hard_consistency.pdf')"""

    u.save_data(full_indices, full_save_dir)
    u.save_data(part_indices, part_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze hard samples in the official training and test splits using precomputed hardness '
                    'indicators.'
    )
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset (MNIST, CIFAR10, CIFAR100).')
    parser.add_argument('--model_type', type=str, choices=['simple', 'complex'], default='complex',
                        help='Specifies the type of network used for training (MLP vs LeNet or ResNet20 vs ResNet56).')
    parser.add_argument('--ensemble_size', type=str, choices=['small', 'large'], default='small',
                        help='Specifies the size of the ensembles to be used in the experiments.')
    parser.add_argument('--grayscale', action='store_true',
                        help='Raise to use grayscale transformation for CIFAR10 when computing Proximity metrics')
    parser.add_argument('--pca', action='store_true', help='Raise to use PCA for CIFAR10 when computing Proximity '
                                                           'metrics (can be combined with --grayscale).')
    args = parser.parse_args()

    main(**vars(args))
