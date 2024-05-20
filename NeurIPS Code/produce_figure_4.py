import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import load_results


def plot_results(data, ax, label_prefix, color, marker, linestyle):
    x_values = sorted(data.keys())
    means = []
    stds = []
    for x in x_values:
        if x == 1.0:  # Exclude 'full' condition (x == 1.0)
            continue
        accuracies = np.array(data[x])
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))

    x_values = np.array(x_values[:-1])  # Exclude the last point (full)
    means = np.array(means)
    stds = np.array(stds)

    # Plot mean with error band (standard deviation)
    ax.plot(x_values, means, label=f'{label_prefix}', color=color, marker=marker, linestyle=linestyle)
    ax.fill_between(x_values, means - stds, means + stds, color=color, alpha=0.3)


def main(dataset_name: str, remove_hard: bool):
    fig, ax = plt.subplots(figsize=(10, 6))

    result_types = ['stragglers', 'confidence', 'energy']
    colors = ['red', 'green', 'blue']  # Example colors for each result type
    markers = ['x', 'o', 's']  # Example markers for each result type
    linestyles = ['-', '--']  # Solid for hard, dashed for easy

    for i, result_type in enumerate(result_types):
        results = load_results(
            f'Results/{dataset_name}_{result_type}_{"True" if remove_hard else "False"}_70000_metrics.pkl')
        plot_results(results['hard'], ax, f'{result_type.capitalize()} (Hard)', colors[i], markers[i], linestyles[0])
        plot_results(results['easy'], ax, f'{result_type.capitalize()} (Easy)', colors[i], markers[i], linestyles[1])

    ax.set_xlabel(f'Percentage of {["easy", "hard"][remove_hard]} samples removed from the training set')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    plt.savefig(f'Figures/{dataset_name}_{remove_hard}.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST, and KMNIST.')
    parser.add_argument('--remove_hard', action='store_true', default=False, help='Flag to remove hard samples or not')
    args = parser.parse_args()
    main(**vars(args))
