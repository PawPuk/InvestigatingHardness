import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from utils import load_results


def compute_stats(accuracies):
    df = pd.DataFrame(accuracies)
    means = df.mean().to_dict()
    stds = df.std().to_dict()
    return means, stds


def plot_results(means, stds, distributions, labels):
    classes = list(means.keys())
    means_1_minus_x = [1 - means[key] for key in classes]
    stds_values = [stds[key] for key in classes]
    dist_values = {label: [distributions[label][key] for key in classes] for label in labels}

    fig, ax1 = plt.subplots()

    # Bar chart for mean (1-x) with error bars for std
    ax1.bar(classes, means_1_minus_x, yerr=stds_values, capsize=5, alpha=0.6, label='Mean (1-x)', color='blue')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Error rate of LeNet', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for distributions
    ax2 = ax1.twinx()
    colors = ['red', 'green', 'purple']
    for idx, label in enumerate(labels):
        ax2.plot(classes, dist_values[label], label=label, color=colors[idx], marker='o')
    ax2.set_ylabel('Number of hard samples', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add legends
    fig.tight_layout()
    fig.legend(loc='upper left')

    plt.show()

    # Calculate and print Pearson correlations
    for label in labels:
        correlation, _ = pearsonr(means_1_minus_x, dist_values[label])
        print(f'Pearson correlation with {label}: {correlation:.3f}')


def main(dataset_name):
    results = []
    distributions = {}

    if dataset_name == 'CIFAR10':
        result = load_results(f'Results/Distributions/distributions_confidence_{dataset_name}.pkl')
        all_accuracies = result['class_level_accuracies']
        distributions['0.05'] = result['hardness_distribution'][0]
        distributions['0.1'] = result['hardness_distribution'][1]
        distributions['0.2'] = result['hardness_distribution'][2]
        labels = ['0.05', '0.1', '0.2']
    else:
        for strategy in ['stragglers', 'confidence', 'energy']:
            result = load_results(f'Results/Distributions/distributions_{strategy}_{dataset_name}.pkl')
            results.append(result)

        all_accuracies = []
        for result, strategy in zip(results, ['stragglers', 'confidence', 'energy']):
            all_accuracies.extend(result['class_level_accuracies'])
            distributions[strategy] = result['hardness_distribution'][0]
        labels = ['stragglers', 'confidence', 'energy']

    means, stds = compute_stats(all_accuracies)
    plot_results(means, stds, distributions, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST, KMNIST, and CIFAR10.')
    args = parser.parse_args()
    main(**vars(args))
