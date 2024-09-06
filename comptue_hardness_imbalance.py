import numpy as np
from typing import List

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import LeNet
import utils as u


class HardnessImbalanceMeasurer:
    def __init__(self, dataset_name: str, models_count: int, training: str):
        self.dataset_name = dataset_name
        self.models_count = models_count
        self.training = training
        self.easy_indices = u.load_data(f'{u.DIVISIONS_SAVE_DIR}/{self.dataset_name}_adaptive_easy_indices.pkl')
        self.hard_indices = u.load_data(f'{u.DIVISIONS_SAVE_DIR}/{self.dataset_name}_adaptive_hard_indices.pkl')
        self.models = []
        for i in range(self.models_count):
            model = LeNet().to(u.DEVICE)
            model_file = f"{u.MODEL_SAVE_DIR}{self.training}{self.dataset_name}_{self.models_count}_ensemble_{i}.pth"
            model.load_state_dict(torch.load(model_file, map_location=u.DEVICE))
            self.models.append(model)
        if self.training == 'full':
            training_dataset = u.load_full_data_and_normalize(self.dataset_name)
            test_dataset = training_dataset
            self.dataset_size = len(training_dataset)
        else:
            training_dataset, test_dataset = u.load_data_and_normalize(self.dataset_name)
            self.dataset_size = len(training_dataset) + len(test_dataset)
            self.test_size = len(test_dataset)
        self.loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    def measure_imbalance_on_one_model(self, model: torch.nn.Module, extreme_indices: List[int]) -> float:
        """Test a single model on a specific set of samples (given by indices) and return accuracy."""
        model.eval()
        all_outputs, all_targets = [], []

        if self.training == 'full':
            test_indices = extreme_indices
        else:  # Filter indices to match only test samples (last portion of the dataset)
            test_start_idx = self.dataset_size - self.test_size
            test_indices = [i - test_start_idx for i in extreme_indices if test_start_idx <= i < self.dataset_size]

        with torch.no_grad():
            for data, target in self.loader:
                data, target = data.to(u.DEVICE), target.to(u.DEVICE)
                outputs = model(data).cpu().numpy()
                all_outputs.append(outputs)
                all_targets.append(target.cpu().numpy())

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        if len(test_indices) > 0:
            selected_outputs = all_outputs[test_indices]
            selected_targets = all_targets[test_indices]
        else:
            raise Exception('No valid test indices found.')

        correct = (np.argmax(selected_outputs, axis=1) == selected_targets).sum()
        total = selected_targets.shape[0]

        accuracy = correct / total if total > 0 else None  # Using None, as this should not occur.
        return accuracy

    def measure_and_visualize_hardness_based_imbalance(self):
        """Test an ensemble of models on the dataset, returning accuracies for each metric."""
        accuracies = []
        metric_abbreviations = [
            'SameCentroidDist', 'OtherCentroidDist', 'CentroidDistRatio', 'Same1NNDist', 'Other1NNDist', '1NNRatioDist',
            'AvgSame40NNDist', 'AvgOther40NNDist', 'AvgAll40NNDist', 'Avg40NNDistRatio', '40NNPercSame',
            '40NNPercOther', 'AvgSame40NNCurv', 'AvgOther40NNCurv', 'AvgAll40NNCurv', 'GaussCurv', 'MeanCurv'
        ]

        for metric_idx in tqdm(range(len(self.easy_indices)), desc='Iterating through metrics'):
            metric_easy_indices = self.easy_indices[metric_idx]
            metric_hard_indices = self.hard_indices[metric_idx]

            all_accuracies, easy_accuracies, hard_accuracies = [], [], []

            for model in self.models[:3]:
                print('start', self.dataset_size)
                all_accuracies.append(self.measure_imbalance_on_one_model(model, list(range(self.dataset_size))))
                easy_accuracies.append(self.measure_imbalance_on_one_model(model, metric_easy_indices))
                hard_accuracies.append(self.measure_imbalance_on_one_model(model, metric_hard_indices))

            accuracies.append({
                'metric': metric_abbreviations[metric_idx],
                'mean_acc_all': np.mean(all_accuracies),
                'std_acc_all': np.std(all_accuracies),
                'mean_acc_easy': np.mean(easy_accuracies),
                'std_acc_easy': np.std(easy_accuracies),
                'mean_acc_hard': np.mean(hard_accuracies),
                'std_acc_hard': np.std(hard_accuracies)
            })
        self.plot_error_rates(accuracies, metric_abbreviations)

    @staticmethod
    def plot_error_rates(accuracies: List[dict], metric_abbreviations: List[str]) -> None:
        """Plot the error rates for all metrics in one figure, with vertical line segments for each error rate type."""

        num_metrics = len(accuracies)
        max_error_rate = 0

        # Calculate epsilon based on the number of metrics and plot width
        epsilon = 0.1  # Adjust epsilon to control the spacing between lines for each metric

        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Loop through each metric and plot the error rates (100 - accuracy) as vertical line segments
        for i, acc in enumerate(accuracies):
            metric_idx = acc['metric_idx']

            # Compute error rates (since error rate = 100% - accuracy)
            error_all = 1 - acc['mean_acc_all']
            error_easy = 1 - acc['mean_acc_easy']
            error_hard = 1 - acc['mean_acc_hard']

            max_error_rate = max(max_error_rate, error_all, error_easy, error_hard)

            # Plot vertical line segments using Line2D for all, easy, and hard samples
            ax.add_line(Line2D([metric_idx - epsilon, metric_idx + epsilon], [error_easy, error_easy],
                               color='green', linewidth=5))  # Green for easy
            ax.add_line(Line2D([metric_idx - epsilon, metric_idx + epsilon], [error_all, error_all],
                               color='black', linewidth=5))  # Black for all
            ax.add_line(Line2D([metric_idx - epsilon, metric_idx + epsilon], [error_hard, error_hard],
                               color='red', linewidth=5))  # Red for hard

        # Calculate ylim based on the max error rate with an epsilon margin (10% of max error rate)
        epsilon_y = 0.1 * max_error_rate
        ax.set_ylim(0, max_error_rate + epsilon_y)
        # Labeling and formatting
        ax.set_xticks(metric_abbreviations)
        ax.set_xticklabels([f'{acc["metric_idx"]}' for acc in accuracies])
        ax.set_xlabel('Metric')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Rates across Different Metrics')
        ax.grid(True)

        # Adjust the x limits to avoid cutting off lines on the right edge
        ax.set_xlim(0.5, num_metrics + 0.5)  # Padding added to the left and right

        # Show the plot
        plt.tight_layout()
        plt.savefig('hardness_based_imbalances.pdf')
        plt.savefig('hardness_based_imbalances.png')
        plt.show()


if __name__ == "__main__":
    import argparse

    # Argument parser for dataset_name and models_count
    parser = argparse.ArgumentParser(description="Load easy/hard indices and pre-trained models.")
    parser.add_argument("--dataset_name", type=str, default='MNIST',
                        help="Name of the dataset (e.g., MNIST, CIFAR10, etc.)")
    parser.add_argument("--models_count", type=int, default='100', help="Number of models in the ensemble")
    parser.add_argument('--training', type=str, choices=['full', 'part'], default='full',
                        help='Indicates which models to choose for evaluations - the ones trained on the entire dataset '
                             '(full), or the ones trained only on the training set (part).')

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    imbalance = HardnessImbalanceMeasurer(**vars(args))
    imbalance.measure_and_visualize_hardness_based_imbalance()
