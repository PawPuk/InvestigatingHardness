import argparse
from collections import defaultdict
import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
    elif is_second_or_third_family_metric(normalized_metric, avg_gradients):
        return is_second_or_third_family_metric(normalized_metric, avg_gradients)
    elif is_fourth_family_metric(second_derivatives):
        return 4
    return 5


def is_first_family_metric(avg_gradients: List[float]) -> bool:
    """Check if the metric belongs to the first family."""
    # Check if the rightmost points have the highest value
    right_most = np.mean(avg_gradients[-100:])
    # Check if the leftmost points are higher than the mean of the middle samples
    left_most = np.mean(avg_gradients[:100])
    middle_mean = np.mean(avg_gradients[len(avg_gradients)//2 - 500: len(avg_gradients)//2 + 500])

    # Conditions: rightmost should be highest, leftmost should be higher than the middle mean
    return right_most > 2 * middle_mean and left_most > 2 * middle_mean


def is_second_or_third_family_metric(normalized_metric: np.ndarray, avg_gradients: List[float]) -> int:
    """Check if the metric belongs to the new family based on distribution and gradient."""
    # Check the normalized distribution (left side low, right side high)
    left_side_distribution = np.mean(normalized_metric[:500])
    right_side_distribution = np.mean(normalized_metric[-20000:])

    # Check the first derivative (left side high, right side low)
    left_side_gradient = np.mean(avg_gradients[:500])
    right_side_gradient = np.mean(avg_gradients[-20000:])
    # Check the condition: distribution (left low, right high) and first derivative (left high, right low)
    if left_side_distribution < right_side_distribution and left_side_gradient > right_side_gradient:
        epsilon = 0.1 * abs(np.mean(avg_gradients[:500]))  # Epsilon is 1% of the leftmost gradient value
        window_size = 500  # Window size to check increases over epsilon

        # Traverse through the gradient and check for increases
        for i in range(1, len(avg_gradients) - window_size):
            current_gradient = np.mean(avg_gradients[i:i + window_size])
            previous_gradient = np.mean(avg_gradients[i - 1:i - 1 + window_size])

            # If we detect an increase over epsilon, it's Subfamily 2 (non-monotonic)
            if current_gradient - previous_gradient > epsilon:
                return 3

        # If no significant increase is found, it's Subfamily 1 (monotonic)
        return 2
    return 0


def is_fourth_family_metric(second_derivatives: List[float]) -> bool:
    """Check if the metric belongs to the fourth family by analyzing second derivatives."""
    # Compute epsilon as a fraction of the right-most value
    right_most_value = np.mean(second_derivatives[-500:])
    epsilon = 0.01 * right_most_value  # 1% of the rightmost value as epsilon

    # Traverse from right to left, checking windows of size 500
    for i in range(len(second_derivatives) - 500, 0, -1):
        window_mean = np.mean(second_derivatives[i:i + 500])

        # If the window mean deviates by more than epsilon from the right-most value, return True
        if abs(window_mean - right_most_value) > epsilon:
            return True

    return False


def find_division_points_for_first_family(second_derivatives: np.ndarray, epsilon_fraction: float = 0.001,
                                          window_size_1: int = 100, window_size_2: int = 500) -> Tuple[int, int]:
    """Find the first and second division points based on the second derivative analysis."""
    max_second_derivative = np.max(second_derivatives)
    epsilon = epsilon_fraction * max_second_derivative

    first_division_point = None
    second_division_point = None

    # Find the first division point (from right to left)
    for i in range(len(second_derivatives) - 1, window_size_1 - 1, -1):
        if np.all(np.abs(second_derivatives[i - window_size_1:i]) < epsilon):
            first_division_point = i
            break

    if first_division_point is not None:
        # Find the second division point (from first division point to the left)
        for i in range(first_division_point, window_size_1 - 1, -1):
            if np.all(np.abs(second_derivatives[i - window_size_1:i]) > 5 * epsilon):
                second_division_point = i
                break

    return first_division_point, second_division_point

def find_division_points_for_second_family(data: np.ndarray, window_size: int = 500,
                                           epsilon_fraction: float = 0.01) -> Tuple[int, int]:
    right_most_value = np.mean(data[-window_size:])
    epsilon = epsilon_fraction * right_most_value

    # Start from the rightmost point and move left
    for i in range(len(data) - window_size, 0, -1):
        window_mean = np.mean(data[i:i + window_size])
        if abs(window_mean - right_most_value) > epsilon:
            return i + 500, i + 500  # Move the point slightly to the right
    raise Exception

def find_division_points_for_third_family(data: np.ndarray, gradients: List[float], window_size: int = 500,
                                          epsilon_fraction: float = 0.01) -> Tuple[int, int]:
    """Find the right and left division points for the third family."""
    # Find the right division point (same as in find_division_points_for_second_family)
    right_most_value = np.mean(data[-window_size:])
    epsilon = epsilon_fraction * right_most_value

    # Start from the rightmost point and move left
    right_division_point = len(data) - 1
    for i in range(len(data) - window_size, 0, -1):
        window_mean = np.mean(data[i:i + window_size])
        if abs(window_mean - right_most_value) > epsilon:
            right_division_point = i + 500
            break

    # Now find the left division point (move from left to right, starting at 0)
    previous_windows_avg = []

    for i in range(0, right_division_point - window_size):
        current_window = np.mean(gradients[i:i + window_size])

        # Track the previous 100 window averages
        if len(previous_windows_avg) >= window_size:
            previous_windows_avg.pop(0)  # Keep only the last 100 window averages
        previous_windows_avg.append(current_window)

        # If the current window's gradient is larger than the average of the last windows, the decreasing trend stopped
        if len(previous_windows_avg) == window_size and current_window > np.mean(previous_windows_avg):
            left_division_point = i
            return left_division_point, right_division_point
    raise Exception

def plot_metric_results(metric_idx: int, sorted_normalized_metric: np.ndarray, avg_gradients: List[float],
                        avg_second_gradients: List[float], first_division_point: int, second_division_point: int,
                        dataset_name: str):
    """Plot the results with division points marked if applicable."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot sorted normalized metric
    axes[0].plot(sorted_normalized_metric, marker='o', linestyle='-')
    if first_division_point is not None:
        axes[0].axvline(x=first_division_point, color='red', linestyle='--', label='First Division')
    if second_division_point is not None:
        axes[0].axvline(x=second_division_point, color='blue', linestyle='--', label='Second Division')
    axes[0].set_title(f'Metric {metric_idx + 1} Normalized Distribution')
    axes[0].set_xlabel('Sample Index (Sorted)')
    axes[0].set_ylabel('Normalized Metric Value')
    axes[0].grid(True)

    # Plot average gradient
    axes[1].plot(avg_gradients, marker='x', linestyle='-', color='r')
    if first_division_point is not None:
        axes[1].axvline(x=first_division_point, color='red', linestyle='--', label='First Division')
    if second_division_point is not None:
        axes[1].axvline(x=second_division_point, color='blue', linestyle='--', label='Second Division')
    axes[1].set_title(f'Metric {metric_idx + 1} Average Gradients (First Derivative)')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Average Gradient')
    axes[1].grid(True)

    # Plot second derivative
    axes[2].plot(avg_second_gradients, marker='x', linestyle='-', color='g')
    if first_division_point is not None:
        axes[2].axvline(x=first_division_point, color='red', linestyle='--', label='First Division')
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
    plt.close()

def extract_extreme_samples(metrics: List[List[float]], labels: List[int], invert: List[bool], dataset_name: str,
                            threshold: float = 0.05) -> Tuple[List[Dict[int, int]], List[List[int]]]:

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

        # Sort normalized metric for gradient computation
        sorted_normalized_metric = np.sort(normalized_metric)

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

        # Compute second derivative averages for plotting
        avg_second_gradients = []
        window_size = 100
        for i in range(window_size, num_samples - window_size):
            start = max(0, i - window_size)
            end = min(num_samples, i + window_size + 1)
            avg_second_gradients.append(np.mean(second_derivatives[start:end]))

        # Only apply division logic if it matches the first family
        first_division_point, second_division_point = None, None
        family = detect_family(sorted_normalized_metric, avg_gradients, avg_second_gradients)
        print(f'Metric {metric_idx + 1} is of family {family}.')
        if family == 1:
            first_division_point, second_division_point = find_division_points_for_first_family(second_derivatives)
        elif family == 2:
            first_division_point, second_division_point = find_division_points_for_second_family(sorted_normalized_metric)
        elif family == 3:
            first_division_point, second_division_point = find_division_points_for_third_family(sorted_normalized_metric,
                                                                                                avg_gradients)
        # Plot results
        plot_metric_results(metric_idx, sorted_normalized_metric, avg_gradients, avg_second_gradients,
                            first_division_point, second_division_point, dataset_name)
    print(len(123))
    return class_distributions, extreme_indices



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

        # Normalize the distribution to [0, 1]
        normalized_distribution = (class_level_distribution - np.min(class_level_distribution)) / \
                                  (np.max(class_level_distribution) - np.min(class_level_distribution))

        # Compute Pearson correlation coefficient
        correlation, _ = pearsonr(avg_class_accuracies, normalized_distribution)
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
        # Handle inf values: replace them with the second largest value
        metric = np.array(metric)
        if np.isinf(metric).any():
            finite_vals = metric[np.isfinite(metric)]
            if len(finite_vals) > 0:
                second_largest = np.partition(finite_vals, -2)[-2]  #  TODO: change to use largest finite value
                metric[np.isinf(metric)] = second_largest

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
    easy_distribution, easy_indices = extract_extreme_samples(all_metrics, labels, invert_metrics, dataset_name,
                                                              threshold)
    hard_distribution, hard_indices = extract_extreme_samples(all_metrics, labels, [not b for b in invert_metrics],
                                                              dataset_name, threshold)

    print(hard_distribution)

    print()
    print('-'*20)
    print()

    print(easy_distribution)

    # Compute and visualize the correlation heatmaps
    compute_correlation_heatmaps(easy_indices, hard_indices)

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
