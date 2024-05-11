import argparse
from typing import Tuple

import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SampleComplexityEstimator:
    @staticmethod
    def define_neighbourhood_bounds(dataset: int, a: float):
        neighborhood_bounds = []
        if dataset == 1:
            # Class 0 neighborhoods
            for x_start in np.arange(0, 2, a):
                for y_start in np.arange(0, 4, a):
                    neighborhood_bounds.append(((x_start, x_start + a, y_start, y_start + a), 0))
            # Class 1 neighborhoods
            for x_start in np.arange(2.1, 4.1, a):
                for y_start in np.arange(0, 4, a):
                    neighborhood_bounds.append(((x_start, x_start + 0.5, y_start, y_start + a), 1))
        elif dataset == 2:
            regions = [
                (0, 2, 0, 3, 0),  # Class 0 region
                (2, 5, 0, 1, 1),  # Class 1 region
                (5, 8, 0, 3, 0),  # Class 0 region
                (0, 2, 3, 6, 1),  # Class 1 region
                (2, 5, 1, 6, 0),  # Class 0 region
                (5, 8, 3, 6, 1)   # Class 1 region
            ]
            for x_start, x_end, y_start, y_end, class_id in regions:
                for x in np.arange(x_start, x_end, a):
                    for y in np.arange(y_start, y_end, a):
                        neighborhood_bounds.append(((x, x + a, y, y + a), class_id))
        elif dataset == 3:
            regions = [
                # Blue regions (class 0)
                (0, 7, 0, 2, 0),  # Large blue region on the left
                (0, 2, 2, 6, 0),  # Tall blue region in the middle-left
                (5, 7, 2, 4, 0),  # Narrow blue region on the far right

                # Orange regions (class 1)
                (2.5, 9.5, 4.5, 6.5, 1),  # Top central orange region
                (2.5, 4.5, 2.5, 4.5, 1),  # Lower central orange region
                (7.5, 9.5, 0.5, 4.5, 1)  # Bottom central orange region
            ]
            for x_start, x_end, y_start, y_end, class_id in regions:
                for x in np.arange(x_start, x_end, a):
                    for y in np.arange(y_start, y_end, a):
                        neighborhood_bounds.append(((x, x + a, y, y + a), class_id))
        else:
            raise ValueError
        return neighborhood_bounds

    @staticmethod
    def generate_local_dataset(num_samples: int, bounds: Tuple[float, float, float, float]) -> Tensor:
        x_min, x_max, y_min, y_max = bounds
        points = torch.rand(num_samples, 2, device=DEVICE)
        points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
        points[:, 1] = points[:, 1] * (y_max - y_min) + y_min
        return points

    @staticmethod
    def divide_into_neighborhoods(dataset, labels, neighborhood_bounds):
        neighborhoods = []
        for bounds, class_label in neighborhood_bounds:
            x_min, x_max, y_min, y_max = bounds
            # Convert bounds to tensors and ensure they are on the same device as dataset
            x_min_t = torch.tensor(x_min, device=DEVICE)
            x_max_t = torch.tensor(x_max, device=DEVICE)
            y_min_t = torch.tensor(y_min, device=DEVICE)
            y_max_t = torch.tensor(y_max, device=DEVICE)
            class_label_t = torch.tensor([class_label], device=DEVICE)

            # Perform the filtering operation
            neighborhood_data = dataset[
                (dataset[:, 0] >= x_min_t) & (dataset[:, 0] < x_max_t) &
                (dataset[:, 1] >= y_min_t) & (dataset[:, 1] < y_max_t) &
                (labels == class_label_t)
                ]
            neighborhoods.append({
                'data': neighborhood_data,
                'label': class_label,
                'bounds': (x_min, x_max, y_min, y_max)
            })
        return neighborhoods

    class MLP(nn.Module):
        def __init__(self, initialization):
            super().__init__()  # Corrected from super(self.MLP, self).__init__()
            self.fc1 = nn.Linear(2, 4).to(DEVICE)
            self.fc2 = nn.Linear(4, 4).to(DEVICE)
            self.fc3 = nn.Linear(4, 1).to(DEVICE)
            if initialization == 1:
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                nn.init.xavier_uniform_(self.fc3.weight)
                self.initialize_biases()
            elif initialization == 2:
                nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
                self.initialize_biases()
            elif initialization == 3:
                nn.init.uniform_(self.fc1.weight, a=-1.0, b=1.0)
                nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)
                nn.init.uniform_(self.fc3.weight, a=-1.0, b=1.0)
                self.initialize_biases()

        def initialize_biases(self):
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.constant_(self.fc3.bias, 0)

        def forward(self, x):
            x = x.to(DEVICE)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    @staticmethod
    def train_model(model, criterion, optimizer, inputs, labels, num_epochs=15):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

    @staticmethod
    def evaluate_model(model, data, labels):
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            predictions = outputs > 0.5
            correct = (predictions == labels).sum().item()
        return correct / len(data)

    @staticmethod
    def plot_decision_boundary(model, data, labels, ax):
        # Determine the range of the data
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float)

        model.eval()
        with torch.no_grad():
            Z = model(grid).view(xx.shape)
        Z = Z > 0.5

        ax.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', edgecolor='k', s=20)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title('Decision Boundary')

    def main(self, dataset, a, samples_per_neighbourhood, threshold, init, opt, lr, epochs):
        # Define neighborhoods explicitly
        neighborhood_bounds = self.define_neighbourhood_bounds(dataset, a)
        # Generate the dataset
        neighborhood_data_list = []
        neighborhood_labels_list = []
        for bounds, class_label in neighborhood_bounds:
            neighborhood_data = self.generate_local_dataset(samples_per_neighbourhood, bounds)
            neighborhood_labels = torch.full((samples_per_neighbourhood,), class_label)
            neighborhood_data_list.append(neighborhood_data)
            neighborhood_labels_list.append(neighborhood_labels)
        dataset = torch.cat(neighborhood_data_list, dim=0)
        labels = torch.cat(neighborhood_labels_list, dim=0)
        # Generate a dictionary for each neighbourhood
        neighborhoods = self.divide_into_neighborhoods(dataset, labels, neighborhood_bounds)
        # Plot initial dataset
        fig, ax = plt.subplots()
        class0_points = dataset[labels == 0]
        class1_points = dataset[labels == 1]
        ax.scatter(class0_points[:, 0], class0_points[:, 1], marker='o', color='blue')
        ax.scatter(class1_points[:, 0], class1_points[:, 1], marker='^', color='red')
        color_map = plt.get_cmap('viridis')
        # Initialize necessary variables
        final_estimated_sample_complexities = []
        threshold_accuracy = threshold
        criterion = nn.BCELoss()
        # Estimate sample complexity of all neighbourhoods
        for neighborhood in neighborhoods:
            neighborhood_data = neighborhood['data']
            class_label = neighborhood['label']
            bounds = neighborhood['bounds']
            # Rerun the estimations 10 times to make sure the results are statistically significant
            estimated_sample_complexities = [0 for _ in range(10)]
            for i in range(10):
                # Exclude neighborhood data from the training dataset (necessary to estimate sample complexity)
                mask = (dataset[:, None] == neighborhood_data).all(-1).any(-1)
                train_data = dataset[~mask]
                train_labels = labels[~mask]
                while True:
                    # For each run train an ensemble of models
                    models = [self.MLP(init) for _ in range(10)]
                    if opt == 'ADAM':
                        optimizers = [optim.Adam(model.parameters(), lr=lr) for model in models]
                    else:
                        optimizers = [optim.SGD(model.parameters(), lr=lr) for model in models]
                    for model, optimizer in zip(models, optimizers):
                        self.train_model(model, criterion, optimizer, train_data, train_labels, num_epochs=epochs)
                    # Generate random evaluation data from the neighborhood bounds
                    eval_data = self.generate_local_dataset(100, bounds)
                    eval_labels = torch.full((100,), class_label)
                    # Evaluate each model
                    accuracies = [self.evaluate_model(model, eval_data, eval_labels) for model in models]
                    # Stop when min accuracy is above threshold (needed for sample complexity)
                    if min(accuracies) >= threshold_accuracy:
                        break
                    """fig, ax = plt.subplots()
                    plot_decision_boundary(models[np.argmin(accuracies)], train_data.cpu(), train_labels.cpu(), ax)
                    plt.show()"""
                    # Increase the number of samples in neighbourhood if threshold wasn't reached
                    additional_data = self.generate_local_dataset(1, bounds)
                    estimated_sample_complexities[i] += 1
                    train_data = torch.cat([train_data, additional_data])
                    train_labels = torch.cat([train_labels, torch.full((1,), class_label)])
            estimated_sample_complexity = np.mean(estimated_sample_complexities)
            print(f"This neighbourhood required an average of {estimated_sample_complexity} samples (see "
                  f"{estimated_sample_complexities}) to achieve over {threshold_accuracy}% min accuracy (over ensemble of "
                  f"10 networks)on randomly generated data from the neighbourhood.")
            final_estimated_sample_complexities.append(estimated_sample_complexity)
        # Normalize color map based on the maximum additional samples needed
        norm = Normalize(vmin=0, vmax=float(max(final_estimated_sample_complexities)))
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        # Plot the neighbourhoods taking representing the estimated sample complexity using heatmap
        for neighborhood, additional_data_count in zip(neighborhoods, final_estimated_sample_complexities):
            x_min, x_max, y_min, y_max = neighborhood['bounds']
            rect_color = color_map(norm(additional_data_count))
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                     facecolor=rect_color, alpha=0.5)
            ax.add_patch(rect)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Estimated sample complexity')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        plt.savefig(f'Figures/a_{a}_samples_{samples_per_neighbourhood}_t_{threshold}_init_{init}_opt_{opt}_lr_{lr}'
                    f'_epochs_{epochs}.png')
        plt.show()


def main(dataset):
    estimator = SampleComplexityEstimator()
    estimator.main(dataset, a=0.5, samples_per_neighbourhood=5, threshold=95, init=3, opt='ADAM', lr=0.01,
                   epochs=100)
    print(len(123))
    for neighbourhood_width in [0.1, 0.5, 1]:
        for samples in [1, 5, 10]:
            for threshold in [95, 99, 100]:
                for optimizer in ['ADAM', 'SGD']:
                    for learning_rate in [0.1, 0.01, 0.001]:
                        for epochs in [10, 100]:
                            estimator.main(dataset, a=neighbourhood_width, samples_per_neighbourhood=samples,
                                           init=0, threshold=threshold, opt=optimizer, lr=learning_rate, epochs=epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    main(**vars(args))
