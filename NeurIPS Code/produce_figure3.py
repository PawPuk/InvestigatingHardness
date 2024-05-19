import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle

# Assuming load_results is properly defined somewhere
from utils import load_results

# Define settings, colors, and markers
settings = ['full', 'hard', 'easy']
colors = {
    0.05: {"full": "#89CFF0", "hard": "#FFB6C1", "easy": "#98FB98"},
    0.1: {"full": "#4682B4", "hard": "#DC143C", "easy": "#32CD32"},
    0.2: {"full": "#000080", "hard": "#8B0000", "easy": "#006400"}
}
color_list_by_setting = {
        'full': [colors[0.05]['full'], colors[0.1]['full'], colors[0.2]['full']],
        'hard': [colors[0.05]['hard'], colors[0.1]['hard'], colors[0.2]['hard']],
        'easy': [colors[0.05]['easy'], colors[0.1]['easy'], colors[0.2]['easy']]
    }
line_styles = {
    0.05: '-',
    0.1: '--',
    0.2: ':'
}
markers = {'full': 'o', 'hard': 's', 'easy': 'D', 'results2': 'x'}

def plot_threshold(threshold_data, threshold_name):
    for setting_name in settings:
        setting_data = threshold_data.get(setting_name, {})
        x_values = sorted(setting_data.keys())
        y_values = [np.mean(setting_data[x]) for x in x_values]
        y_stds = [np.std(setting_data[x]) for x in x_values]

        plt.plot(x_values, y_values, label=f'{threshold_name} - {setting_name}',
                 color=colors[threshold_name][setting_name], marker=markers[setting_name],
                 linestyle=line_styles[threshold_name])
        plt.fill_between(x_values,
                         [y - s for y, s in zip(y_values, y_stds)],
                         [y + s for y, s in zip(y_values, y_stds)],
                         color=colors[threshold_name][setting_name], alpha=0.2)

def plot_results2(results2):
    x_values = sorted(results2.keys())
    y_values = [np.mean(results2[x]) for x in x_values]
    y_stds = [np.std(results2[x]) for x in x_values]

    plt.plot(x_values, y_values, label='Baseline', color='black', marker='x', linestyle='-')
    plt.fill_between(x_values,
                     [y - s for y, s in zip(y_values, y_stds)],
                     [y + s for y, s in zip(y_values, y_stds)],
                     color='black', alpha=0.2)


def main():
    results1 = load_results('Results/CIFAR10_True_70000_common_metrics.pkl')
    results2 = load_results('Results/CIFAR10_True_70000_edge_metrics.pkl')
    plt.figure(figsize=(12, 8))

    for threshold_name, threshold_data in results1.items():
        plot_threshold(threshold_data, threshold_name)
    plot_results2(results2)

    plt.xlabel('Percentage of hard samples removed trom the training set')
    plt.ylabel('Accuracy')

    legend_items = []
    labels = ['Full test set', 'Hard test samples', 'Easy test samples']
    for setting, color_list in zip(settings, [color_list_by_setting['full'], color_list_by_setting['hard'],
                                              color_list_by_setting['easy']]):
        # Create tuples of Line2D for each setting using its respective marker and colors
        marker_lines = tuple(
            mlines.Line2D([], [], color=color, marker=markers[setting], linestyle='None', markersize=10) for color in
            color_list)
        legend_items.append(marker_lines)

    # Create a single legend with all settings
    legend = plt.legend(legend_items, labels, handler_map={tuple: HandlerTuple(ndivide=None)}, title="Accuracy on:",
                        loc='center left', frameon=True, borderpad=1, bbox_to_anchor=(0.05, 0.08))
    plt.gca().add_artist(legend)

    custom_lines = [
        plt.Line2D([0], [0], color='black', linestyle='solid', lw=2, label='0.05'),
        plt.Line2D([0], [0], color='black', linestyle='dashed', lw=2, label='0.1'),
        plt.Line2D([0], [0], color='black', linestyle='dotted', lw=2, label='0.2')
    ]
    line_marker_legend = plt.legend(handles=custom_lines, title='Confidence threshold (%):',
                                    loc='center left', bbox_to_anchor=(0.225, 0.07))
    plt.gca().add_artist(line_marker_legend)
    rect = Rectangle((0.05, 0.005), 0.355, 0.21, linewidth=1, edgecolor='k', facecolor='white', alpha=0.75,
                     transform=plt.gca().transAxes, zorder=3)
    plt.gca().add_patch(rect)
    plt.text(0.1, 0.17, r"$\frac{\mathrm{hard\_train\_samples}}{\mathrm{easy\_train\_samples}} = \frac{\mathrm{hard\_test\_samples}}{\mathrm{easy\_test\_samples}}$",
             transform=plt.gca().transAxes, fontsize=14, color='black', zorder=4)

    # Create the legend for edge case
    # Create a rectangle as the background for the custom legend
    rect = Rectangle((0.42, 0.005), width=0.16, height=0.1, transform=plt.gca().transAxes,
                     linewidth=1, edgecolor='black', facecolor='none', zorder=3)
    plt.gca().add_patch(rect)

    # Add texts as titles within the rectangle
    plt.text(0.45, 0.09, 'Hard Training Set', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', zorder=4)
    plt.text(0.46, 0.066, 'Easy Test Set', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             zorder=4)

    line = mlines.Line2D([0.44, 0.56], [0.025, 0.025], transform=plt.gca().transAxes, color='black',
                         linestyle='-', markersize=10)
    plt.gca().add_line(line)

    # Manually add markers
    markers_x = [0.44, 0.5, 0.56]  # Start, middle, end
    for mx in markers_x:
        marker = mlines.Line2D([mx], [0.025], color='black', marker='x', markersize=10,
                               linestyle='None', transform=plt.gca().transAxes)
        plt.gca().add_line(marker)

    plt.tight_layout()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig('Figures/Section4_hard.pdf')
    plt.show()

if __name__ == '__main__':
    main()
