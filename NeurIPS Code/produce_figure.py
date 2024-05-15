import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Assuming load_results is properly defined somewhere
from utils import load_results

# Define settings, colors, and markers
settings = ['full', 'hard', 'easy']
colors = {
    0.05: {"full": "#89CFF0", "hard": "#FFB6C1", "easy": "#98FB98"},  # Slightly darker light colors
    0.1: {"full": "#4682B4", "hard": "#DC143C", "easy": "#32CD32"},   # Medium intensity
    0.2: {"full": "#000080", "hard": "#8B0000", "easy": "#006400"}   # Slightly lighter dark colors
}
line_styles = {
    0.05: '-',  # Solid
    0.1: '--',  # Dashed
    0.2: ':'    # Dotted
}
markers = {'full': 'o', 'hard': 's', 'easy': 'D'}


def plot_threshold(threshold_data, threshold_name):
    for setting_name, setting_data in threshold_data.items():
        x_values = list(setting_data.keys())

        for func_idx, function_name in enumerate(['accuracy', 'precision', 'recall', 'f1']):
            if all(len(setting_data[x][function_name]) == 0 for x in x_values):
                continue

            y_means = [np.mean(setting_data[x][function_name]) for x in x_values]
            y_stds = [np.std(setting_data[x][function_name]) for x in x_values]

            plt.plot(x_values, y_means, label=f'{threshold_name} - {setting_name} - {function_name}',
                     color=colors[threshold_name][setting_name], marker=markers[setting_name],
                     linestyle=line_styles[threshold_name])
            plt.fill_between(x_values,
                             [y - s for y, s in zip(y_means, y_stds)],
                             [y + s for y, s in zip(y_means, y_stds)],
                             color=colors[threshold_name][setting_name], alpha=0.2)


def main():
    results = load_results('Results/CIFAR10_True_70000_metrics.pkl')
    plt.figure(figsize=(12, 8))

    for threshold_name, threshold_data in results.items():
        plot_threshold(threshold_data, threshold_name)

    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Mean and Standard Deviation of Functions for All Thresholds')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('Section4_hard.pdf')
    plt.show()


if __name__ == '__main__':
    main()
