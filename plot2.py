import pickle

import matplotlib.pyplot as plt
import numpy as np

with open('Results/hard_samples_indices.pkl', 'rb') as f:
    all_hard_indices = pickle.load(f)


def calculate_shared_samples(all_hard_indices):
    all_indices_flat = [index for run_indices in all_hard_indices for index in run_indices]
    from collections import Counter
    index_counts = Counter(all_indices_flat)

    num_runs = len(all_hard_indices)
    num_total_hard_samples = len(set(all_indices_flat))  # Unique hard samples across all runs

    thresholds = np.linspace(0, 1, 21)  # Example: 0%, 5%, ..., 100%
    percentages_shared = []
    numbers_shared = []

    for threshold in thresholds:
        min_sets = np.ceil(threshold * num_runs)
        num_shared_samples = sum(1 for count in index_counts.values() if count >= min_sets)

        percent_shared = (num_shared_samples / num_total_hard_samples) * 100
        percentages_shared.append(percent_shared)
        numbers_shared.append(num_shared_samples)

    return thresholds, percentages_shared, numbers_shared


thresholds, percentages_shared, numbers_shared = calculate_shared_samples(all_hard_indices)

# Plot for Percentage of Hard Samples Shared
plt.figure(figsize=(10, 6))
plt.plot(thresholds * 100, percentages_shared, marker='o')  # Thresholds in percentage
plt.xlabel('Threshold Percentage (%)')
plt.ylabel('Percentage of Hard Samples Shared (%)')
plt.title('Percentage of Shared Hard Samples by Threshold')
plt.grid(True)

# Plot for Number of Hard Samples Shared
plt.figure(figsize=(10, 6))
plt.plot(thresholds * 100, numbers_shared, marker='o', color='red')  # Thresholds in percentage
plt.xlabel('Threshold Percentage (%)')
plt.ylabel('Number of Hard Samples Shared')
plt.title('Number of Shared Hard Samples by Threshold')
plt.grid(True)
plt.show()

