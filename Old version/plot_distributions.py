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


def plot_results(dataset_name, network, means, stds, distributions, labels):
    classes = list(means.keys())
    means_1_minus_x = [1 - means[key] for key in classes]
    stds_values = [stds[key] for key in classes]
    dist_values = {label: [distributions[label][key] for key in classes] for label in labels}

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Bar chart for mean (1-x) with error bars for std
    ax1.bar(classes, means_1_minus_x, yerr=stds_values, capsize=5, alpha=0.6, label='error rate', color='blue')
    ax1.set_xlabel('Class', fontsize=20)
    ax1.set_ylabel('Error rate of LeNet', color='blue', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=17)
    ax1.tick_params(axis='x', labelsize=17)
    ax1.set_xticks(classes)

    # Create a second y-axis for distributions
    ax2 = ax1.twinx()
    colors = ['red', 'green', 'purple']
    for idx, label in enumerate(labels):
        ax2.plot(classes, dist_values[label], label=label, color=colors[idx], marker='o')
    ax2.set_ylabel('Number of hard samples', color='black', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=17)

    # Add legends inside the plot
    fig.tight_layout()
    if dataset_name == 'CIFAR10':
        bbox = (0.65, 0.999)
    else:
        bbox = (0.677, 0.987)
    fig.legend(loc='upper left', bbox_to_anchor=bbox, fontsize=20)
    plt.savefig(f'Figures/{dataset_name}_distributions_{network}.pdf')

    # Calculate and print Pearson correlations
    for label in labels:
        correlation, _ = pearsonr(means_1_minus_x, dist_values[label])
        print(f'Pearson correlation with {label}: {correlation:.3f}')
    plt.show()


def main(dataset_name, network):
    results = []
    distributions = {}

    if dataset_name == 'CIFAR10':
        result = load_results(f'Old version/Results/Distributions/LeNet/distributions_confidence_{dataset_name}.pkl')
        all_accuracies = result['class_level_accuracies']
        distributions['0.05'] = result['hardness_distribution'][0]
        distributions['0.1'] = result['hardness_distribution'][1]
        distributions['0.2'] = result['hardness_distribution'][2]
        labels = ['0.05', '0.1', '0.2']
    else:
        if network == 'PureLeNet' and dataset_name not in ['MNIST', 'KMNIST']:
            raise ValueError("PureLeNet can only be used with MNIST or KMNIST datasets.")
        net_dir = 'PureLeNet' if network == 'PureLeNet' else 'SimpleNN' if network == 'SimpleNN' else 'LeNet'
        for strategy in ['confidence', 'energy']:
            result = load_results(
                f'Old version/Results/Distributions/{net_dir}/distributions_{strategy}_{dataset_name}.pkl')
            results.append(result)

        # Always load 'stragglers' from LeNet directory
        stragglers_result = load_results(
            f'Old version/Results/Distributions/LeNet/distributions_stragglers_{dataset_name}.pkl')
        results.append(stragglers_result)

        all_accuracies = []
        for result, strategy in zip(results, ['confidence', 'energy', 'stragglers']):
            if not (network == 'SimpleNN' and strategy == 'stragglers'):
                all_accuracies.extend(result['class_level_accuracies'])
            distributions[strategy] = result['hardness_distribution'][0]
        labels = ['stragglers', 'confidence', 'energy']

    means, stds = compute_stats(all_accuracies)
    plot_results(dataset_name, network, means, stds, distributions, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST, KMNIST, and CIFAR10.')
    parser.add_argument('--network', type=str, default='LeNet', choices=['LeNet', 'PureLeNet', 'SimpleNN'],
                        help='Network type. Either LeNet or PureLeNet.')
    args = parser.parse_args()
    main(**vars(args))
