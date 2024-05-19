import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

import utils


def load_and_analyze_results(filename, total_runs):
    # Load the data containing lists of tensors of hard samples
    hard_samples_indices = utils.load_results(filename)

    # Assuming each tensor in each list is of the same length, e.g., 70,000
    num_samples = hard_samples_indices[0][0].size(0)
    # This will hold counts of how many times each sample was marked as a straggler
    straggler_counts = torch.zeros(num_samples, dtype=torch.int32)

    for run_list in hard_samples_indices:
        for tensor in run_list:
            straggler_counts += tensor.int()

    # Convert to numpy for easy handling with numpy operations
    straggler_counts_np = straggler_counts.numpy()

    # Calculate overlaps for thresholds from 1 to total_runs
    overlaps = [np.sum(straggler_counts_np >= threshold) for threshold in range(1, total_runs + 1)]

    return overlaps


def plot_overlaps(overlaps, total_runs):
    thresholds = range(1, total_runs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, overlaps, marker='o', linestyle='-', color='blue')
    plt.title('Number of Stragglers Appearing in at Least X Runs')
    plt.xlabel('Threshold (Minimum Number of Runs in Which a Straggler Appears)')
    plt.ylabel('Number of Stragglers')
    plt.xticks(thresholds)
    plt.grid(True)
    plt.show()


def main(dataset_name: str, runs: int):
    filename = f'Results/straggler_indices_{dataset_name}_{runs}.pkl'
    overlaps = load_and_analyze_results(filename, runs)
    plot_overlaps(overlaps, runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze overlap of straggler indices across multiple runs.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    main(**vars(args))
