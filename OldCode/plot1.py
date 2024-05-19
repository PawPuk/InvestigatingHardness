import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_class_mean_std(data, n_classes=10):
    """Compute the mean and std for each class across runs."""
    means, stds = {}, {}
    for class_idx in range(n_classes):
        # Extract data for this class across all runs
        class_data = np.array([run[class_idx] for run in data])
        means[class_idx] = np.mean(class_data, axis=0)
        stds[class_idx] = np.std(class_data, axis=0)
    return means, stds


def compute_mean_std(data):
    """Compute mean and std for each key across dictionaries in data."""
    aggregated_data = {}

    # Aggregate values for each key across runs
    for run in data:
        for key, value in run.items():
            if key in aggregated_data:
                aggregated_data[key].append(value)
            else:
                aggregated_data[key] = [value]

    # Now compute mean and std for each key
    mean_std = {}
    for key, values in aggregated_data.items():
        # Ensure all values are np.arrays for consistent operations
        values = np.array(values)
        mean_std[key] = {
            'mean': np.mean(values, axis=0),
            'std': np.std(values, axis=0)
        }
    return mean_std


def plot_class_statistic(inversion_points_list, means, stds, title):
    """Plots mean and std in GP style for each class."""
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(title)
    for class_idx in range(10):
        ax = axs[class_idx // 5, class_idx % 5]
        epochs = range(1, len(means[class_idx]) + 1)
        ax.plot(epochs, means[class_idx], color='blue')
        ax.fill_between(epochs, means[class_idx] - stds[class_idx], means[class_idx] + stds[class_idx], color='blue',
                        alpha=0.2)

        # Collect all values for the current key
        values = [d[class_idx] for d in inversion_points_list]
        # Calculate mean and standard deviation
        mean_value = np.mean(values)
        std_dev = np.std(values)
        ax.axvline(x=mean_value, color='r', linestyle='-')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])
        ax.fill_betweenx(y=[ymin, ymax], x1=mean_value - std_dev, x2=mean_value + std_dev, alpha=0.1, color='red')

        ax.set_title(f'Class {class_idx}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Count')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_statistic_sorted(mean_std, title):
    # Sort items based on mean values and prepare data for plotting
    sorted_data = sorted(mean_std.items(), key=lambda x: x[1]['mean'])
    indices = np.linspace(0, len(sorted_data) - 1, num=len(sorted_data))  # Smooth x-axis
    sorted_keys, sorted_values = zip(*sorted_data)  # Unpack sorted items
    means = np.array([value['mean'] for value in sorted_values])
    stds = np.array([value['std'] for value in sorted_values])

    # Prepare the figure
    plt.figure(figsize=(12, 6))
    plt.title(title)

    # Plot the mean as a smooth line
    plt.plot(indices, means, color='blue')
    # Add shading for standard deviation in a smoother manner
    plt.fill_between(indices, means - stds, means + stds, color='blue', alpha=0.2)

    # Customizing the plot to better fit GP style
    plt.xlabel('Sample Index (sorted by mean)')
    plt.ylabel('Statistic Value')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_statistic_unsorted(mean_std, title):
    # Prepare data for plotting without sorting
    indices = np.arange(len(mean_std))  # Use original order indices for x-axis
    means = np.array([value['mean'] for value in mean_std.values()])
    stds = np.array([value['std'] for value in mean_std.values()])

    # Prepare the figure
    plt.figure(figsize=(12, 6))
    plt.title(title)

    # Plot the mean as a line
    plt.plot(indices, means, color='blue', label='Mean')

    # Add shading for standard deviation
    plt.fill_between(indices, means - stds, means + stds, color='blue', alpha=0.2)

    # Customizing the plot
    plt.xlabel('Sample Index')
    plt.ylabel('Statistic Value')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def main():
    n_classes = 10
    # Adjust the filenames according to your saved data
    class_stats_filenames = {
        'Learned Counts': 'Results/raw_learned_counts.pkl',
        'Forgotten Counts': 'Results/raw_forgotten_counts.pkl'
    }
    stats_filenames = {
        'Relearned Counts': 'Results/relearned_counters.pkl',
        'First Learned Epoch': 'Results/first_learned_epochs.pkl',
        'Last Learned Epoch': 'Results/last_learned_epochs.pkl'
    }

    with open('Results/inversion_points_list.pkl', 'rb') as f:
        inversion_points_list = pickle.load(f)

    # Plot class-based statistics
    for title, filename in class_stats_filenames.items():
        data = load_data(filename)
        means, stds = compute_class_mean_std(data, n_classes=n_classes)
        plot_class_statistic(inversion_points_list, means, stds, title)

    # Plot non-class-based statistics
    for title, filename in stats_filenames.items():
        data = load_data(filename)
        mean_std = compute_mean_std(data)
        plot_statistic_sorted(mean_std, title)


if __name__ == '__main__':
    main()
