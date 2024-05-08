import pickle

import matplotlib.pyplot as plt
import torch


def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def plot_samples(data, labels, title):
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axes):
        ax.imshow(data[i].permute(1, 2, 0) * torch.tensor([0.247, 0.243, 0.261]) +
                  torch.tensor([0.4914, 0.4822, 0.4465]))
        ax.title.set_text(f'{class_names[labels[i]]}')  # Use class names instead of label numbers
        ax.axis('off')
    plt.savefig(f'../Figures/{title}.png')
    plt.savefig(f'../Figures/{title}.pdf')
    plt.show()


def plot_overlap_over_threshold(thresholds, number_of_models, overlaps):
    plt.figure(figsize=(16, 6))
    for i, key in enumerate(['train', 'test', 'combined']):
        plt.subplot(1, 3, i + 1)
        for c_threshold in number_of_models:
            means = [x[0] for x in overlaps[key][c_threshold]]
            stds = [x[1] for x in overlaps[key][c_threshold]]
            plt.plot(thresholds, means, label=f'Consensus of {c_threshold} Models')
            plt.fill_between(thresholds, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)],
                             alpha=0.2)

        plt.title(f'{key.capitalize()} Set Overlap')
        plt.xlabel('Threshold (%)')
        plt.ylabel('Overlap (%)')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig('../Figures/universality_of_hardness.pdf')
    plt.savefig('../Figures/universality_of_hardness.png')
    plt.show()


def main():
    results = load_results('../Results/results1.pkl')  # Adjust the filename as necessary
    for key in ['train', 'test', 'combined']:
        plot_samples(results['images'][key], results['labels'][key], f'{key}_low_confidence')
    plot_overlap_over_threshold(results['thresholds'], results['number_of_models'], results['overlaps'])


if __name__ == '__main__':
    main()
