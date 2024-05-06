import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_mean_std(data):
    """Compute mean and std for each key across dictionaries in data."""
    aggregated_data, mean_std = {}, {}
    # Aggregate values for each key across runs
    for run in data:
        for key, value in run.items():
            if key in aggregated_data:
                aggregated_data[key].append(value)
            else:
                aggregated_data[key] = [value]
    # Now compute mean and std for each key
    for key, values in aggregated_data.items():
        # Ensure all values are np.arrays for consistent operations
        values = np.array(values)
        mean_std[key] = {'mean': np.mean(values, axis=0), 'std': np.std(values, axis=0)}
    return mean_std


def sort_indices_by_mean(mean_std, total_samples=70000):
    # Initialize a list for all indices with a default mean value (for indices not in mean_std)
    default_mean = float('inf')  # or another value that places these indices at the end of the sorted list
    all_means = [(index, mean_std.get(index, {'mean': default_mean})['mean']) for index in range(total_samples)]

    # Sort all indices based on their mean values
    sorted_indices = sorted(all_means, key=lambda x: x[1])

    # Extract and return the sorted indices
    sorted_indices_only = [index for index, mean in sorted_indices]
    return sorted_indices_only


def aggregate_hard_samples_into_bins(sorted_indices, thresholds, bin_size=5000):
    # Map original indices to their new positions in the sorted order
    sorted_index_positions = {original_index: new_position for new_position, original_index in enumerate(sorted_indices)}
    index_positions = {original_index: original_index for original_index in range(len(sorted_indices))}

    # Initialize bins
    total_bins = (len(sorted_indices) + bin_size - 1) // bin_size  # Calculate total number of bins needed
    hard_sample_counts = [0 for _ in range(total_bins)]  # Initialize a count of hard samples for each bin
    sorted_hard_sample_counts = [0 for _ in range(total_bins)]

    # Count hard samples in each bin
    for index in thresholds.keys():
        if thresholds[index] >= 1.0:  # Consider as hard sample
            position = index_positions[index]
            sorted_position = sorted_index_positions[index]
            bin_index = position // bin_size
            sorted_bin_index = sorted_position // bin_size
            hard_sample_counts[bin_index] += 1
            sorted_hard_sample_counts[sorted_bin_index] += 1

    return hard_sample_counts, sorted_hard_sample_counts


def plot_hard_sample_bins_combined(hard_sample_counts, sorted_hard_sample_counts, bin_size=5000):
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))  # Adjust for 1 row, 2 columns layout

    # X positions for each bin, common for both plots
    bins = range(len(hard_sample_counts))
    bin_labels = [f"{i * bin_size}-{(i + 1) * bin_size - 1}" for i in bins]

    # Plot for Unsorted Dataset Bins
    axs[0].bar(bins, hard_sample_counts, width=0.8, color='blue', edgecolor='black')
    axs[0].set_title("Distribution of Hard Samples in Unsorted Dataset (Binned)")
    axs[0].set_xlabel('Bin (Each containing 100 indices)')
    axs[0].set_ylabel('Count of Hard Samples')
    axs[0].set_xticks(bins)
    axs[0].set_xticklabels(bin_labels, rotation=45, ha="right")
    axs[0].grid(axis='y', linestyle='--', linewidth=0.5)

    # Plot for Sorted Dataset Bins
    axs[1].bar(bins, sorted_hard_sample_counts, width=0.8, color='red', edgecolor='black')
    axs[1].set_title("Distribution of Hard Samples in Sorted Dataset (Binned)")
    axs[1].set_xlabel('Bin (Each containing 100 indices)')
    axs[1].set_ylabel('Count of Hard Samples')
    axs[1].set_xticks(bins)
    axs[1].set_xticklabels(bin_labels, rotation=45, ha="right")
    axs[1].grid(axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def get_hard_sample_thresholds(all_hard_indices, num_runs):
    from collections import Counter
    all_indices_flat = [index for run_indices in all_hard_indices for index in run_indices]
    index_counts = Counter(all_indices_flat)
    thresholds = {index: count / num_runs for index, count in index_counts.items()}
    return thresholds


def main():
    stats_filenames = {
        'Relearned Counts': 'Results/relearned_counters.pkl',
        'First Learned Epoch': 'Results/first_learned_epochs.pkl',
        'Last Learned Epoch': 'Results/last_learned_epochs.pkl'
    }
    with open('Results/hard_samples_indices.pkl', 'rb') as f:
        all_hard_indices = pickle.load(f)

    num_runs = 50  # Or however many runs you've conducted
    thresholds = get_hard_sample_thresholds(all_hard_indices, num_runs)

    # Then, for each non-class-based statistic:
    for title, filename in stats_filenames.items():
        data = load_data(filename)
        mean_std = compute_mean_std(data)
        sorted_indices = sort_indices_by_mean(mean_std)
        hard_sample_counts, sorted_hard_sample_counts = aggregate_hard_samples_into_bins(sorted_indices, thresholds)
        plot_hard_sample_bins_combined(hard_sample_counts, sorted_hard_sample_counts)


if __name__ == '__main__':
    main()
