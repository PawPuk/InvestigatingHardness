import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def define_neighbourhood_bounds(dataset: int):
    neighborhood_bounds = []
    if dataset == 1:
        # previous dataset definition
        pass
    elif dataset == 2:
        # Updated regions to mimic the complex pattern in the image
        regions = [
            # Blue regions (class 0)
            (0, 7, 0, 2, 0),  # Large blue region on the left
            (0, 2, 2, 6, 0),  # Tall blue region in the middle-left
            (5, 7, 2, 4, 0),  # Narrow blue region on the far right

            # Orange regions (class 1)
            (2.5, 9.5, 4.5, 6.5, 1),  # Top central orange region
            (2.5, 4.5, 2.5, 4.5, 1),  # Lower central orange region
            (7.5, 9.5, 0.5, 4.5, 1)   # Bottom central orange region
        ]
        for x_start, x_end, y_start, y_end, class_id in regions:
            for x in np.arange(x_start, x_end):
                for y in np.arange(y_start, y_end):
                    neighborhood_bounds.append(((x, x + 1, y, y + 1), class_id))

    return neighborhood_bounds


def plot_neighborhoods(bounds):
    fig, ax = plt.subplots(figsize=(9, 6))
    for (x_start, x_end, y_start, y_end), class_id in bounds:
        rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, linewidth=1,
                                 edgecolor='r' if class_id == 0 else 'b', facecolor='lightblue' if class_id == 0 else 'orange')
        ax.add_patch(rect)
        ax.annotate(str(class_id), (x_start + (x_end - x_start) / 2, y_start + (y_end - y_start) / 2),
                    color='white', weight='bold', fontsize=10, ha='center', va='center')

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 7)
    ax.set_aspect('equal')
    plt.show()


# Call the function for dataset 2
bounds = define_neighbourhood_bounds(2)
plot_neighborhoods(bounds)

