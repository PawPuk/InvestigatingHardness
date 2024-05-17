import argparse
from typing import List, Tuple

import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from tqdm import tqdm

import utils


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Neighbourhood:
    def __init__(self, x_start: float, x_end: float, y_start: float, y_end: float, label: int, num_samples: int):
        self.x_bounds = [x_start, x_end]
        self.y_bounds = [y_start, y_end]
        self.label = label
        self.data = self.generate_local_data(num_samples)

    def generate_local_data(self, num_samples: int) -> Tensor:
        points = torch.rand(num_samples, 2, device=DEVICE)
        points[:, 0] = points[:, 0] * (self.x_bounds[1] - self.x_bounds[0]) + self.x_bounds[0]
        points[:, 1] = points[:, 1] * (self.y_bounds[1] - self.y_bounds[0]) + self.y_bounds[0]
        return points


class SampleComplexityEstimator:
    @staticmethod
    def define_neighbourhood_bounds(dataset: int, a: float, num_samples: int) -> List[Neighbourhood]:
        neighborhoods = []
        if dataset == 1:
            # Class 0 neighborhoods
            for x_start in np.arange(0, 2, a):
                for y_start in np.arange(0, 2, a):
                    neighborhoods.append(Neighbourhood(x_start, x_start + a, y_start, y_start + a, 0, num_samples))
            # Class 1 neighborhoods
            for x_start in np.arange(2.1, 4.1, a):
                for y_start in np.arange(0, 2, a):
                    neighborhoods.append(Neighbourhood(x_start, x_start + a, y_start, y_start + a, 1, num_samples))
        elif dataset == 2:
            regions = [
                (0, 1, 0, 1.5, 0),
                (0, 4.5, 1.5, 2.5, 0),
                (3.5, 4.5, 0, 1.5, 0),
                (1.5, 3, 2.5, 4, 0),
                (1.5, 3, 0, 1, 1),
                (0, 1, 3, 4, 1),
                (3.5, 4.5, 3, 4, 1)
            ]
            for x_start, x_end, y_start, y_end, class_id in regions:
                for x in np.arange(x_start, x_end, a):
                    for y in np.arange(y_start, y_end, a):
                        neighborhoods.append(Neighbourhood(x, x + a, y, y + a, class_id, num_samples))
        elif dataset == 3:
            regions = [
                (0, 6, 0, 1, 0),
                (0, 1, 1, 4.5, 0),
                (4, 6, 1, 3, 0),
                (1.5, 3.5, 1.5, 3.5, 1),
                (1.5, 7.5, 3.5, 4.5, 1),
                (6.5, 7.5, 0, 3.5, 1)
            ]
            for x_start, x_end, y_start, y_end, class_id in regions:
                for x in np.arange(x_start, x_end, a):
                    for y in np.arange(y_start, y_end, a):
                        neighborhoods.append(Neighbourhood(x, x + a, y, y + a, class_id, num_samples))
        elif dataset == 4:
            regions = [

            ]
            for x_start, x_end, y_start, y_end, class_id in regions:
                for x in np.arange(x_start, x_end, a):
                    for y in np.arange(y_start, y_end, a):
                        neighborhoods.append(Neighbourhood(x, x + a, y, y + a, class_id, num_samples))
        else:
            raise ValueError("Invalid dataset number")
        return neighborhoods

    @staticmethod
    def create_dataloader_from_neighborhoods(neighborhoods: List[Neighbourhood]) -> Tuple[TensorDataset, List[Tensor]]:
        all_data, all_labels, neighborhood_indices, current_index = [], [], [], 0
        for neighborhood in neighborhoods:
            num_samples = neighborhood.data.size(0)
            all_data.append(neighborhood.data)
            all_labels.append(torch.full((num_samples,), neighborhood.label, dtype=torch.long, device=DEVICE))
            neighborhood_indices.append(torch.arange(current_index, current_index + num_samples, device=DEVICE))
            current_index += num_samples
        all_data, all_labels = torch.cat(all_data, dim=0), torch.cat(all_labels, dim=0)
        dataset = TensorDataset(all_data, all_labels)
        return dataset, neighborhood_indices

    class MLP(nn.Module):
        def __init__(self, initialization):
            super().__init__()  # Corrected from super(self.MLP, self).__init__()
            self.fc1 = nn.Linear(2, 8).to(DEVICE)
            self.fc2 = nn.Linear(8, 8).to(DEVICE)
            self.fc3 = nn.Linear(8, 1).to(DEVICE)
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
            nn.init.constant_(self.fc3.bias, 0)

        def forward(self, x):
            x = x.to(DEVICE)
            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    @staticmethod
    def train_model(model, criterion, optimizer, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                if outputs.size(0) > 1:
                    outputs = outputs.squeeze()
                else:
                    outputs = outputs.view(-1)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

    @staticmethod
    def evaluate_model(model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in dataloader:
                outputs = model(data)
                predictions = outputs > 0.5
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        return correct / total

    @staticmethod
    def plot_data_with_neighborhoods(neighborhoods, train_dataset, sample_complexities):
        # Create a figure and a set of subplots
        fig, ax = plt.subplots()
        # Extract data points and labels from the training dataset
        data_points = train_dataset.tensors[0].cpu().numpy()
        labels = train_dataset.tensors[1].cpu().numpy()
        # Plot data points with different markers for each class
        class_0_indices = labels == 0
        class_1_indices = labels == 1
        ax.scatter(data_points[class_0_indices, 0], data_points[class_0_indices, 1],
                   c='green', marker='o', s=20)  # Blue circles for class 0
        ax.scatter(data_points[class_1_indices, 0], data_points[class_1_indices, 1],
                   c='orange', marker='x', s=20)  # Red crosses for class 1
        # Create a color map for the rectangles
        norm = Normalize(vmin=min(sample_complexities), vmax=max(sample_complexities))
        cmap = plt.get_cmap('viridis')
        # Draw rectangles for each neighborhood
        for idx, neighborhood in enumerate(neighborhoods):
            x_start, x_end = neighborhood.x_bounds
            y_start, y_end = neighborhood.y_bounds
            rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, linewidth=2,
                                     edgecolor='black', facecolor=cmap(norm(sample_complexities[idx])), alpha=0.3)
            ax.add_patch(rect)
        # Adding colorbar based on the sample complexities
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Estimated Sample Complexity')
        # Setting labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

    def main(self, dataset, a, samples_per_neighbourhood, threshold, init, opt, lr, wd, epochs):
        results = []
        # Assuming neighborhoods are already defined:
        neighborhoods = self.define_neighbourhood_bounds(dataset, a, samples_per_neighbourhood)
        train_dataset, indices_list = self.create_dataloader_from_neighborhoods(neighborhoods)
        # Initialize necessary variables
        final_estimated_sample_complexities = []
        criterion = nn.BCELoss()
        # Estimate sample complexity of all neighbourhoods
        for neighbourhood_index, neighbourhood in tqdm(enumerate(neighborhoods)):
            results.append([neighbourhood])
            estimated_sample_complexities = [0 for _ in range(3)]
            for run_index in tqdm(range(3)):
                # Filter out the data of the current neighbourhood from the training dataset
                mask = torch.ones(train_dataset.tensors[0].size(0), dtype=torch.bool, device=DEVICE)
                mask[indices_list[neighbourhood_index]] = False
                filtered_data = train_dataset.tensors[0][mask]
                filtered_labels = train_dataset.tensors[1][mask]
                filtered_dataset = TensorDataset(filtered_data, filtered_labels)
                filtered_dataloader = DataLoader(filtered_dataset, batch_size=int(len(filtered_data) / 10),
                                                 shuffle=True)
                while True:
                    # For each run train an ensemble of models
                    models = [self.MLP(init) for _ in range(5)]
                    if opt == 'ADAM':
                        optimizers = [optim.Adam(model.parameters(), lr=lr, weight_decay=wd) for model in models]
                    else:
                        optimizers = [optim.SGD(model.parameters(), lr=lr, weight_decay=wd) for model in models]
                    for model, optimizer in zip(models, optimizers):
                        self.train_model(model, criterion, optimizer, filtered_dataloader, epochs)
                    # Generate random evaluation data from the current neighbourhood
                    test_data = neighbourhood.generate_local_data(1000)  # Generate 1000 samples for testing
                    test_labels = torch.full((1000,), neighbourhood.label, dtype=torch.long, device=DEVICE)
                    test_dataset = TensorDataset(test_data, test_labels)
                    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
                    # Evaluate each model
                    accuracies = [self.evaluate_model(model, test_dataloader) for model in models]
                    # Check if min accuracy is above threshold
                    if (sum(accuracies) / len(accuracies)) >= threshold:
                        break
                    # Increment sample complexity if not achieved
                    estimated_sample_complexities[run_index] += 1
                    additional_data = neighbourhood.generate_local_data(1)
                    filtered_data = torch.cat([filtered_data, additional_data])
                    filtered_labels = torch.cat([filtered_labels, torch.full((1, ), neighbourhood.label)])
                    filtered_dataset = TensorDataset(filtered_data, filtered_labels)
                    filtered_dataloader = DataLoader(filtered_dataset, batch_size=int(len(filtered_data) / 10),
                                                     shuffle=True)
            estimated_sample_complexity = np.mean(estimated_sample_complexities)
            results[-1].append(estimated_sample_complexity)
            print(f"This neighbourhood required an average of {estimated_sample_complexity} samples (see "
                  f"{estimated_sample_complexities}) to achieve over {threshold}% min accuracy (over ensemble of "
                  f"10 networks)on randomly generated data from the neighbourhood.")
            final_estimated_sample_complexities.append(estimated_sample_complexity)
        utils.save_data(results, f'Results/estimated_sample_complexities{dataset}.pkl')
        # Normalize color map based on the maximum additional samples needed
        self.plot_data_with_neighborhoods(neighborhoods, train_dataset, final_estimated_sample_complexities)
        plt.savefig(f'Figures/{dataset}_a_{a}_samples_{samples_per_neighbourhood}_t_{threshold}_init_{init}_opt_{opt}_'
                    f'lr_{lr}_epochs_{epochs}.pdf')


def main(dataset):
    estimator = SampleComplexityEstimator()
    for i, samples in enumerate([3]):
        estimator.main(dataset, a=0.5, samples_per_neighbourhood=samples, threshold=95, init=2, opt='ADAM', lr=0.01,
                       wd=1e-4, epochs=200)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    main(**vars(args))
