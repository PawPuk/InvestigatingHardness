import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def generate_dataset(num_points_per_class, bounds1, bounds2):
    rect1_x_min, rect1_x_max, rect1_y_min, rect1_y_max = bounds1
    rect2_x_min, rect2_x_max, rect2_y_min, rect2_y_max = bounds2

    # Generate points uniformly from each rectangle and assign labels
    rect1_points = torch.rand(num_points_per_class, 2)
    rect1_points[:, 0] = rect1_points[:, 0] * (rect1_x_max - rect1_x_min) + rect1_x_min
    rect1_points[:, 1] = rect1_points[:, 1] * (rect1_y_max - rect1_y_min) + rect1_y_min
    rect1_labels = torch.zeros(num_points_per_class)  # Label for points from rectangle 1

    rect2_points = torch.rand(num_points_per_class, 2)
    rect2_points[:, 0] = rect2_points[:, 0] * (rect2_x_max - rect2_x_min) + rect2_x_min
    rect2_points[:, 1] = rect2_points[:, 1] * (rect2_y_max - rect2_y_min) + rect2_y_min
    rect2_labels = torch.ones(num_points_per_class)  # Label for points from rectangle 2

    # Concatenate the points and labels from both rectangles
    dataset = torch.cat([rect1_points, rect2_points], dim=0)
    labels = torch.cat([rect1_labels, rect2_labels], dim=0)

    return dataset, labels


def generate_local_dataset(num_samples, bounds):
    x_min, x_max, y_min, y_max = bounds
    points = torch.rand(num_samples, 2)
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
    points[:, 1] = points[:, 1] * (y_max - y_min) + y_min
    return points


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_model(model, criterion, optimizer, inputs, labels, num_epochs=10):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()


def divide_into_neighborhoods(dataset, labels, neighborhood_bounds):
    neighborhoods = []
    for bounds, class_label in neighborhood_bounds:
        x_min, x_max, y_min, y_max = bounds
        neighborhood_data = dataset[
            (dataset[:, 0] >= x_min) & (dataset[:, 0] < x_max) &
            (dataset[:, 1] >= y_min) & (dataset[:, 1] < y_max) &
            (labels == class_label)
        ]
        neighborhoods.append({
            'data': neighborhood_data,
            'label': class_label,
            'bounds': (x_min, x_max, y_min, y_max)
        })
    return neighborhoods


def evaluate_model(model, data, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        predictions = outputs > 0.5
        correct = (predictions == labels).sum().item()
    return correct / len(data)


def main():
    # Define the bounds for each class
    bounds1 = (0, 4, 0, 4)  # Bounds for the first rectangle
    bounds2 = (4.1, 8, 0, 4)  # Bounds for the second rectangle

    # Generate the dataset with specified bounds
    dataset, labels = generate_dataset(100, bounds1, bounds2)
    threshold_accuracy = 99

    # Define neighborhoods explicitly, with 16 smaller 1x1 squares
    neighborhood_bounds = []
    # Class 0 neighborhoods
    for x_start in range(0, 4):
        for y_start in range(0, 4):
            neighborhood_bounds.append(((x_start, x_start + 1, y_start, y_start + 1), 0))
    # Class 1 neighborhoods
    for x_start in [4.1, 5.1, 6.1, 7.1]:
        for y_start in range(0, 4):
            neighborhood_bounds.append(((x_start, x_start + 1, y_start, y_start + 1), 1))

    neighborhoods = divide_into_neighborhoods(dataset, labels, neighborhood_bounds)
    additional_samples_needed = []

    fig, ax = plt.subplots()
    # Plot initial dataset
    class0_points = dataset[labels == 0]
    class1_points = dataset[labels == 1]
    ax.scatter(class0_points[:, 0], class0_points[:, 1], marker='o', color='blue', label='Class 0')
    ax.scatter(class1_points[:, 0], class1_points[:, 1], marker='^', color='red', label='Class 1')

    color_map = plt.get_cmap('viridis')

    for neighborhood in neighborhoods:
        neighborhood_data = neighborhood['data']
        class_label = neighborhood['label']
        bounds = neighborhood['bounds']

        # Exclude neighborhood data from the training dataset
        mask = (dataset[:, None] == neighborhood_data).all(-1).any(-1)
        train_data = dataset[~mask]
        train_labels = labels[~mask]
        additional_data_counts = [0 for _ in range(10)]
        criterion = nn.BCELoss()
        for i in range(10):
            while True:
                models = [MLP() for _ in range(10)]
                optimizers = [optim.Adam(model.parameters(), lr=0.01) for model in models]
                for model, optimizer in zip(models, optimizers):
                    train_model(model, criterion, optimizer, train_data, train_labels, num_epochs=100)
                # Generate random evaluation data from the neighborhood bounds
                eval_data = generate_local_dataset(100, bounds)
                eval_labels = torch.full((100,), class_label)

                # Evaluate each model
                accuracies = [evaluate_model(model, eval_data, eval_labels) for model in models]
                if min(accuracies) >= threshold_accuracy:
                    break

                # Increment data for further training if necessary
                additional_data = generate_local_dataset(1, bounds)  # Adjust number of samples as needed
                additional_data_counts[i] += 1
                train_data = torch.cat([train_data, additional_data])
                train_labels = torch.cat([train_labels, torch.full((1,), class_label)])
        additional_data_count = np.mean(additional_data_counts)
        print(f"All models achieved over {threshold_accuracy}% accuracy with {additional_data_count} "
              f"additional samples.")
        additional_samples_needed.append(additional_data_count)
    # Normalize color map based on the maximum additional samples needed
    norm = Normalize(vmin=0, vmax=max(additional_samples_needed))
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])

    for neighborhood, additional_data_count in zip(neighborhoods, additional_samples_needed):
        x_min, x_max, y_min, y_max = neighborhood['bounds']
        rect_color = color_map(norm(additional_data_count))
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                 facecolor=rect_color, alpha=0.5)
        ax.add_patch(rect)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Estimated sample complexity')
    ax.legend()
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.show()


if __name__ == '__main__':
    main()
