import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

from utils import load_results


def plot_results(data, ax, label_prefix, color, marker, linestyle, fillstyle, markersize=15):
    x_values = sorted(data.keys())
    means = []
    stds = []
    for x in x_values:
        accuracies = np.array(data[x])
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))

    x_values = np.array(x_values)  # Use all x_values without excluding any
    means = np.array(means)
    stds = np.array(stds)

    # Plot mean with error band (standard deviation)
    ax.plot(x_values, means, color=color, marker=marker, linestyle=linestyle,  linewidth=2.5,
            markersize=markersize, fillstyle=fillstyle)
    ax.fill_between(x_values, means - stds, means + stds, color=color, alpha=0.3)


def produce_legend(dataset_name: str, remove_hard: bool):
    legend1 = [
        plt.Line2D([0], [0], color='blue', marker='o', lw=0, markersize=15, label='Energy-based'),
        plt.Line2D([0], [0], color='green', marker='s', lw=0, markersize=15, label='Confidence-based'),
        plt.Line2D([0], [0], color='red', marker='X', lw=0, markersize=15, label='Straggler-based')
    ]
    legend1 = plt.legend(handles=legend1, title='Hard Sample Identifier:', loc='lower center', fontsize=16,
                         bbox_to_anchor=(0.185, 0.0), title_fontsize=18)
    plt.gca().add_artist(legend1)

    if True:
        legend2 = [
            plt.Line2D([0], [0], color='black', linestyle='dotted', lw=2, label='Easy samples'),
            plt.Line2D([0], [0], color='black', linestyle='dashed', lw=2, label='Hard samples')
        ]
        legend2 = plt.legend(handles=legend2, title='Accuracy on:', title_fontsize=18,
                             loc='center left', bbox_to_anchor=(0.35, 0.135), fontsize=16)
        plt.gca().add_artist(legend2)

    if not (dataset_name == 'FashionMNIST' and remove_hard) and not (dataset_name == 'KMNIST' and not remove_hard):
        rect = Rectangle((0.01, 0.32), width=0.27, height=0.2, transform=plt.gca().transAxes,
                         linewidth=1, edgecolor='grey', facecolor='none', zorder=3)
        plt.gca().add_patch(rect)
        # Add texts as titles within the rectangle
        plt.text(0.015, 0.5, f'{["Easy", "Hard"][remove_hard]} Training Set', transform=plt.gca().transAxes,
                 fontsize=16, verticalalignment='top', zorder=4)
        plt.text(0.035, 0.44, f'{["Hard", "Easy"][remove_hard]} Test Set', transform=plt.gca().transAxes,
                 fontsize=16, verticalalignment='top', zorder=4)
        line = mlines.Line2D([0.025, 0.26], [0.36, 0.36], transform=plt.gca().transAxes, color='black',
                             linestyle='-', markersize=10, linewidth=2.5)
        plt.gca().add_line(line)
        # Manually add markers
        markers_x = [0.05, 0.13, 0.23]  # Start, middle, end
        for mx in markers_x:
            marker = mlines.Line2D([mx], [0.36], color='black', marker='h', markersize=15,
                                   linestyle='None', transform=plt.gca().transAxes)
            plt.gca().add_line(marker)


def main(dataset_name: str, remove_hard: bool, network: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    result_types = ['stragglers', 'confidence', 'energy']
    colors = ['red', 'green', 'blue']  # Colors for each result type
    markers = ['X', 's', 'o']  # Markers for each result type
    linestyles = ['--', ':']  # Linestyles for each result type
    fillstyles = ['full', 'full', 'none']  # Fill styles for each marker type

    # Load and plot edge results with black solid line and hexagonal marker
    edge_results = load_results(
        f'Old version/Results/Generalizations/{network}/{dataset_name}_{remove_hard}_70000_edge_metrics.pkl')
    plot_results(edge_results, ax, 'Edge Results', 'black', 'H', '-', 'full')  # Hexagonal marker for edge results

    for i, result_type in enumerate(result_types):
        results = load_results(
            f'Old version/Results/Generalizations/{network}/{dataset_name}_{result_type}_{"True" if remove_hard else "False"}_'
            f'70000_metrics.pkl')
        plot_results(results['hard'], ax, f'{result_type.capitalize()} (Hard)', colors[i], markers[i], linestyles[0], fillstyles[i])
        plot_results(results['easy'], ax, f'{result_type.capitalize()} (Easy)', colors[i], markers[i], linestyles[1], fillstyles[i])

    if not (dataset_name == 'FashionMNIST' and not remove_hard) and network == 'LeNet':
        produce_legend(dataset_name, remove_hard)

    ax.set_xlabel(f'Percentage of {["easy", "hard"][remove_hard]} samples removed from the training set', fontsize=20)
    ax.set_ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 100)
    plt.tight_layout()
    ax.legend()
    ax.grid(True)
    plt.savefig(f'Figures/{dataset_name}_{remove_hard}_{network}.pdf')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST, and KMNIST.')
    parser.add_argument('--remove_hard', action='store_true', default=False, help='Flag to remove hard samples or not')
    parser.add_argument('--network', type=str, choices=['LeNet', 'SimpleNN'], default='LeNet')
    args = parser.parse_args()
    main(**vars(args))
