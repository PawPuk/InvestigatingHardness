import argparse
from glob import glob
from typing import List

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as u


class HardnessImbalanceMeasurer:
    def __init__(self, dataset_name: str, training: str, model_type: str):
        self.dataset_name = dataset_name
        self.training = training
        self.model_type = model_type
        _, _, self.easy_indices, self.hard_indices = u.load_data(
            f'{u.DIVISIONS_SAVE_DIR}{training}{model_type}{dataset_name}_indices.pkl')
        self.models = []
        self.model_paths = self.get_all_trained_model_paths()
        # Load models from memory (similar to the logic in `EnsembleTrainer.train_ensemble`)
        for model_path in self.model_paths:
            model, _ = u.initialize_models(dataset_name, model_type)
            model.load_state_dict(torch.load(model_path, map_location=u.DEVICE))
            self.models.append(model)
        if self.training == 'full':
            training_dataset = u.load_full_data_and_normalize(self.dataset_name)
            test_dataset = training_dataset
            self.dataset_size = len(training_dataset)
        else:
            training_dataset, test_dataset = u.load_data_and_normalize(self.dataset_name)
            self.dataset_size = len(training_dataset) + len(test_dataset)
            self.test_size = len(test_dataset)
        self.loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    def get_all_trained_model_paths(self):
        """Retrieve all trained model paths matching current dataset, training, and model type."""
        pattern = f"{u.MODEL_SAVE_DIR}{self.training}{self.dataset_name}_{self.model_type}ensemble_*.pth"
        model_paths = glob(pattern)
        return sorted(model_paths)

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

    def plot_error_rates(self, accuracies: List[dict], metric_abbreviations: List[str]) -> None:
        """Plot the error rates for all metrics in one figure, with vertical line segments for each error rate type."""

        num_metrics = len(accuracies)
        max_error_rate = 0

        # Calculate epsilon based on the number of metrics and plot width
        epsilon = 0.1  # Adjust epsilon to control the spacing between lines for each metric

        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Loop through each metric and plot the error rates (100 - accuracy) as vertical line segments
        for i, acc in enumerate(accuracies):

            # Compute error rates (since error rate = 100% - accuracy)
            error_all = 1 - acc['mean_acc_all']
            error_easy = 1 - acc['mean_acc_easy']
            error_hard = 1 - acc['mean_acc_hard']

            max_error_rate = max(max_error_rate, error_all, error_easy, error_hard)

            # Plot vertical line segments using Line2D for all, easy, and hard samples
            ax.add_line(Line2D([i - epsilon, i + epsilon], [error_easy, error_easy],
                               color='green', linewidth=5))  # Green for easy
            ax.add_line(Line2D([i - epsilon, i + epsilon], [error_all, error_all],
                               color='black', linewidth=5))  # Black for all
            ax.add_line(Line2D([i - epsilon, i + epsilon], [error_hard, error_hard],
                               color='red', linewidth=5))  # Red for hard

        # Calculate ylim based on the max error rate with an epsilon margin (10% of max error rate)
        epsilon_y = 0.1 * max_error_rate
        ax.set_ylim(0, max_error_rate + epsilon_y)
        # Labeling and formatting
        ax.set_xticks(range(num_metrics))
        ax.set_xticklabels(metric_abbreviations)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Rates across Different Metrics')
        ax.grid(True)
        plt.xticks(rotation=45, ha='right')  # Rotate x-tick labels
        ax.set_xlim(0.5, num_metrics + 0.5)  # Padding added to the left and right

        # Show the plot
        plt.tight_layout()
        plt.savefig(f'{u.HARD_IMBALANCE_DIR}{self.model_type}_{self.training}{self.dataset_name}_imbalances.pdf')
        plt.savefig(f'{u.HARD_IMBALANCE_DIR}{self.model_type}_{self.training}{self.dataset_name}_imbalances.png')
        plt.show()

    def plot_consistency(self, easy_accuracies, hard_accuracies, metric_abbreviations: List[str], num_metrics):
        """Plot how easy and hard accuracies change as we increase the number of models for each metric."""
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        total_models = min(len(easy_accuracies[0]), 500)  # Number of models should be the length of inner lists
        assert len(easy_accuracies) == num_metrics, "Number of metrics should match the length of easy_accuracies"

        # Initialize lists to store running averages and standard deviations for easy and hard accuracies
        running_avg_easy = np.zeros((num_metrics, total_models))
        running_std_easy = np.zeros((num_metrics, total_models))
        running_avg_hard = np.zeros((num_metrics, total_models))
        running_std_hard = np.zeros((num_metrics, total_models))

        # Compute running averages and standard deviations across models for each metric
        for i in range(total_models):
            running_avg_easy[:, i] = np.mean(easy_accuracies[:, :i + 1], axis=1)
            running_std_easy[:, i] = np.std(easy_accuracies[:, :i + 1], axis=1)
            running_avg_hard[:, i] = np.mean(hard_accuracies[:, :i + 1], axis=1)
            running_std_hard[:, i] = np.std(hard_accuracies[:, :i + 1], axis=1)

        # Plotting the consistency for each metric
        for metric_idx in range(num_metrics):
            row, col = divmod(metric_idx, 5)  # Calculate row and column indices for the subplot
            ax = axes[row, col]
            x_vals = range(1, total_models + 1)

            # Plot easy and hard accuracies for the metric
            mean_easy = running_avg_easy[metric_idx, :]
            std_easy = running_std_easy[metric_idx, :]
            mean_hard = running_avg_hard[metric_idx, :]
            std_hard = running_std_hard[metric_idx, :]

            # Plot easy accuracies
            ax.plot(x_vals, mean_easy, label='Easy Accuracy', color='blue')
            ax.fill_between(x_vals, mean_easy - std_easy, mean_easy + std_easy, color='blue', alpha=0.3)

            # Plot hard accuracies
            ax.plot(x_vals, mean_hard, label='Hard Accuracy', color='red')
            ax.fill_between(x_vals, mean_hard - std_hard, mean_hard + std_hard, color='red', alpha=0.3)

            ax.set_title(f'Metric {metric_abbreviations[metric_idx]}')
            ax.set_xlabel('Number of Models')
            ax.set_ylabel('Accuracy')
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'{u.CONSISTENCY_SAVE_DIR}{self.model_type}_{self.training}{self.dataset_name}_imbalance_'
                    f'consistency.png')
        plt.show()

    def measure_and_visualize_hardness_based_imbalance(self):
        """Test an ensemble of models on the dataset, returning accuracies for each metric."""
        accuracies, easy_accuracies, hard_accuracies = [], [], []
        metric_abbreviations = [
            'SameCentroidDist', 'OtherCentroidDist', 'CentroidDistRatio', 'Same1NNDist', 'Other1NNDist', '1NNRatioDist',
            'AvgSame40NNDist', 'AvgOther40NNDist', 'AvgAll40NNDist', 'Avg40NNDistRatio', '40NNPercSame',
            '40NNPercOther', 'N3', 'GaussCurv', 'MeanCurv'
        ]

        for metric_idx in tqdm(range(len(self.easy_indices)), desc='Iterating through metrics'):
            metric_easy_indices = self.easy_indices[metric_idx]
            metric_hard_indices = self.hard_indices[metric_idx]

            all_accuracies, easy_acc, hard_acc = [], [], []

            for model in self.models[:50]:
                all_accuracies.append(self.measure_imbalance_on_one_model(model, list(range(self.dataset_size))))
                easy_acc.append(self.measure_imbalance_on_one_model(model, metric_easy_indices))
                hard_acc.append(self.measure_imbalance_on_one_model(model, metric_hard_indices))

            # Convert to numpy arrays to ensure proper shape and alignment
            easy_acc = np.array(easy_acc)
            hard_acc = np.array(hard_acc)

            easy_accuracies.append(easy_acc)
            hard_accuracies.append(hard_acc)

            accuracies.append({
                'metric': metric_idx,
                'mean_acc_all': np.mean(all_accuracies),
                'std_acc_all': np.std(all_accuracies),
                'mean_acc_easy': np.mean(easy_acc),
                'std_acc_easy': np.std(easy_acc),
                'mean_acc_hard': np.mean(hard_acc),
                'std_acc_hard': np.std(hard_acc)
            })

        # Convert the lists to numpy arrays to ensure alignment across metrics
        easy_accuracies = np.array(easy_accuracies)
        hard_accuracies = np.array(hard_accuracies)

        print(f"Shape of easy_accuracies: {easy_accuracies.shape}")
        print(f"Shape of hard_accuracies: {hard_accuracies.shape}")

        # Plot error rates
        self.plot_error_rates(accuracies, metric_abbreviations)

        # Plot consistency
        self.plot_consistency(easy_accuracies, hard_accuracies, metric_abbreviations, len(self.easy_indices))


if __name__ == "__main__":
    # Argument parser for dataset_name and models_count
    parser = argparse.ArgumentParser(description="Load easy/hard indices and pre-trained models.")
    parser.add_argument("--dataset_name", type=str, default='MNIST',
                        help="Name of the dataset (e.g., MNIST, CIFAR10, etc.)")
    parser.add_argument('--training', type=str, choices=['full', 'part'], default='part',
                        help='Indicates which models to choose for evaluations - the ones trained on the entire dataset'
                             ' (full), or the ones trained only on the training set (part).')
    parser.add_argument('--model_type', type=str, choices=['simple', 'complex'], default='complex',
                        help='Specifies the type of network used for training (MLP vs LeNet or ResNet20 vs ResNet56).')
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    imbalance = HardnessImbalanceMeasurer(**vars(args))
    imbalance.measure_and_visualize_hardness_based_imbalance()
