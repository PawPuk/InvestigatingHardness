import argparse
from glob import glob
import os
from typing import List, Tuple

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as u


class HardnessImbalanceMeasurer:
    def __init__(self, dataset_name: str, training: str, model_type: str, ensemble_size: str, grayscale: bool,
                 pca: bool):
        self.dataset_name = dataset_name
        self.training = training
        self.model_type = model_type
        self.ensemble_size = ensemble_size
        self.grayscale = grayscale
        self.pca = pca
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
            self.training_size = len(training_dataset)
        self.loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.easy_indices, self.medium_indices, self.hard_indices = self.categorize_indices_by_hardness()

    def get_all_trained_model_paths(self):
        """Retrieve all trained model paths matching current dataset, training, and model type."""
        pattern = f"{u.MODEL_SAVE_DIR}{self.training}{self.dataset_name}_{self.model_type}ensemble_*.pth"
        model_paths = glob(pattern)
        return sorted(model_paths)

    def categorize_indices_by_hardness(self) -> Tuple[List[set], List[set], List[set]]:
        # We use full indices as the part indices do not contain information on test samples.
        indices_dir = f'{u.DIVISIONS_SAVE_DIR}{self.ensemble_size}_full{self.model_type}{self.dataset_name}_indices'
        if self.dataset_name == 'CIFAR10':
            if self.grayscale:
                indices_dir += 'gray'
            if self.pca:
                indices_dir += 'pca'

        easy_indices, hard_indices, _, _ = u.load_data(indices_dir + '.pkl')
        easy_test_indices, medium_test_indices, hard_test_indices = [], [], []
        for metric_idx in range(len(easy_indices)):
            metric_easy_indices = easy_indices[metric_idx]
            metric_hard_indices = hard_indices[metric_idx]
            if self.training == 'full':
                metric_medium_indices = set(range(self.dataset_size)).difference(
                    set(metric_easy_indices).union(set(metric_hard_indices)))
                easy_test_indices.append(set(metric_easy_indices))
                medium_test_indices.append(metric_medium_indices)
                hard_test_indices.append(set(metric_hard_indices))
            else:
                part_easy_indices = set(range(self.training_size, self.dataset_size)).intersection(
                    set(metric_easy_indices))
                part_hard_indices = set(range(self.training_size, self.dataset_size)).intersection(
                    set(metric_hard_indices))
                part_medium_indices = set(range(self.dataset_size)).difference(
                    set(part_easy_indices).union(set(part_hard_indices)))
                easy_test_indices.append(part_easy_indices)
                medium_test_indices.append(part_medium_indices)
                hard_test_indices.append(part_hard_indices)
        return easy_test_indices, medium_test_indices, hard_test_indices

    def measure_imbalance_on_one_model(self, model: torch.nn.Module, extreme_indices: set) -> float:
        """Test a single model on a specific set of samples (given by indices) and return accuracy."""
        if len(extreme_indices) == 0 or \
                (self.training == 'part' and max(extreme_indices) < (self.dataset_size - self.test_size)):
            return -1.0

        model.eval()
        all_outputs, all_targets = [], []

        if self.training == 'full':
            test_indices = extreme_indices
        else:  # Filter indices to match only test samples (last portion of the dataset)
            test_indices = {i - self.training_size for i in extreme_indices if
                            self.training_size <= i < self.dataset_size}

        with torch.no_grad():
            for data, target in self.loader:
                data, target = data.to(u.DEVICE), target.to(u.DEVICE)
                outputs = model(data).cpu().numpy()
                all_outputs.append(outputs)
                all_targets.append(target.cpu().numpy())

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        if len(test_indices) > 0:
            selected_outputs = all_outputs[list(test_indices)]
            selected_targets = all_targets[list(test_indices)]
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
            error_medium = 1 - acc['mean_acc_medium']
            error_hard = 1 - acc['mean_acc_hard']

            errors = [error_all, error_easy, error_medium, error_hard]
            current_max = max(errors) if max(errors) != 2 else sorted(errors, reverse=True)[1]
            max_error_rate = max(max_error_rate, current_max)
            if error_all != 2:
                # Plot vertical line segments using Line2D for all, easy, and hard samples
                ax.add_line(Line2D([i - epsilon, i + epsilon], [error_all, error_all],
                                   color='black', linewidth=5))  # Black for all
            if error_easy != 2:
                ax.add_line(Line2D([i - epsilon, i + epsilon], [error_easy, error_easy],
                                   color='green', linewidth=5))  # Green for easy
            if error_medium != 2:
                ax.add_line(Line2D([i - epsilon, i + epsilon], [error_medium, error_medium],
                                   color='blue', linewidth=5))  # Blue for medium
            if error_hard:
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
        ax.set_xlim(-0.5, num_metrics - 0.5)  # Padding added to the left and right

        save_file = f'{u.HARD_IMBALANCE_DIR}{self.ensemble_size}_{self.model_type}_{self.training}' \
                    f'{self.dataset_name}_imbalances'
        if self.dataset_name == 'CIFAR10':
            if self.grayscale:
                save_file += 'gray'
            if self.pca:
                save_file += 'pca'

        # Show the plot
        plt.tight_layout()
        plt.savefig(save_file + '.pdf')
        plt.show()

    def plot_consistency(self, easy_accuracies, medium_accuracies, hard_accuracies, metric_abbreviations: List[str],
                         num_metrics: int):
        """Plot how easy and hard accuracies change as we increase the number of models for each metric."""
        fig, axes = plt.subplots(4, 5, figsize=(20, 15))

        total_models = 10 if self.dataset_name == 'CIFAR10' else 25
        assert len(easy_accuracies) == num_metrics, "Number of metrics should match the length of easy_accuracies"

        # Initialize lists to store running averages and standard deviations for easy and hard accuracies
        running_avg_easy = np.zeros((num_metrics, total_models))
        running_std_easy = np.zeros((num_metrics, total_models))
        running_avg_medium = np.zeros((num_metrics, total_models))
        running_std_medium = np.zeros((num_metrics, total_models))
        running_avg_hard = np.zeros((num_metrics, total_models))
        running_std_hard = np.zeros((num_metrics, total_models))

        # Compute running averages and standard deviations across models for each metric
        for i in range(total_models):
            running_avg_easy[:, i] = np.mean(easy_accuracies[:, :i + 1], axis=1)
            running_std_easy[:, i] = np.std(easy_accuracies[:, :i + 1], axis=1)
            running_avg_medium[:, i] = np.mean(medium_accuracies[:, :i + 1], axis=1)
            running_std_medium[:, i] = np.std(medium_accuracies[:, :i + 1], axis=1)
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
            mean_medium = running_avg_medium[metric_idx, :]
            std_medium = running_std_medium[metric_idx, :]
            mean_hard = running_avg_hard[metric_idx, :]
            std_hard = running_std_hard[metric_idx, :]

            # Plot the accuracies if the samples of the given hardness type exist in the test set
            if -1 not in mean_easy:
                ax.plot(x_vals, mean_easy, label='Easy Accuracy', color='green', linewidth=5)
                ax.fill_between(x_vals, mean_easy - std_easy, mean_easy + std_easy, color='green', alpha=0.1)
            if -1 not in mean_medium:
                ax.plot(x_vals, mean_medium, label='Medium Accuracy', color='blue', linewidth=5)
                ax.fill_between(x_vals, mean_medium - std_medium, mean_medium + std_medium, color='blue', alpha=0.1)
            if -1 not in mean_hard:
                ax.plot(x_vals, mean_hard, label='Hard Accuracy', color='red', linewidth=5)
                ax.fill_between(x_vals, mean_hard - std_hard, mean_hard + std_hard, color='red', alpha=0.1)

            ax.set_title(f'Metric {metric_abbreviations[metric_idx]}')
            ax.set_xlabel('Number of Models')
            ax.set_ylabel('Accuracy')
            ax.legend()

        save_file = f'{u.CONSISTENCY_SAVE_DIR}{self.ensemble_size}_{self.model_type}_{self.training}' \
                    f'{self.dataset_name}_imbalance_consistency'
        if self.dataset_name == 'CIFAR10':
            if self.grayscale:
                save_file += 'gray'
            if self.pca:
                save_file += 'pca'
        plt.tight_layout()
        plt.savefig(save_file + '.pdf')
        plt.show()

    def measure_and_visualize_hardness_based_imbalance(self):
        """Test an ensemble of models on the dataset, returning accuracies for each metric."""
        accuracies, easy_accuracies, medium_accuracies, hard_accuracies = [], [], [], []
        metric_abbreviations = ['DCC', 'MDSC', 'ADSC', 'DNOC', 'MDOC', 'ADOC', 'CP', 'N3', 'CDR', 'MDR', 'ADR', 'GC',
                                'MC', 'Cleanlab', 'EL2N', 'Margins']

        accuracies_file = f'{u.ACCURACIES_SAVE_DIR}{self.ensemble_size}_{self.model_type}_{self.training}' \
                          f'{self.dataset_name}_accuracies.pkl'
        if self.ensemble_size == 'small':
            total_models = 10 if self.dataset_name == 'CIFAR10' else 25
        else:
            total_models = 25 if self.dataset_name == 'CIFAR10' else 100

        if os.path.exists(accuracies_file):
            accuracies, easy_accuracies, medium_accuracies, hard_accuracies = u.load_data(accuracies_file)
        else:
            for metric_idx in tqdm(range(len(self.easy_indices)), desc='Iterating through metrics'):
                metric_easy_indices = self.easy_indices[metric_idx]
                metric_medium_indices = self.medium_indices[metric_idx]
                metric_hard_indices = self.hard_indices[metric_idx]
                all_accuracies, easy_acc, medium_acc, hard_acc = [], [], [], []
                for model in self.models[:total_models]:
                    all_accuracies.append(self.measure_imbalance_on_one_model(model, set(range(self.dataset_size))))
                    easy_acc.append(self.measure_imbalance_on_one_model(model, metric_easy_indices))
                    medium_acc.append(self.measure_imbalance_on_one_model(model, metric_medium_indices))
                    hard_acc.append(self.measure_imbalance_on_one_model(model, metric_hard_indices))
                # Convert to numpy arrays to ensure proper shape and alignment
                easy_acc, medium_acc, hard_acc = np.array(easy_acc), np.array(medium_acc), np.array(hard_acc)

                easy_accuracies.append(easy_acc)
                medium_accuracies.append(medium_acc)
                hard_accuracies.append(hard_acc)
                accuracies.append({
                    'metric': metric_idx,
                    'mean_acc_all': np.mean(all_accuracies),
                    'mean_acc_easy': np.mean(easy_acc),
                    'mean_acc_medium': np.mean(medium_acc),
                    'mean_acc_hard': np.mean(hard_acc),
                })

        # Convert the lists to numpy arrays to ensure alignment across metrics
        easy_accuracies = np.array(easy_accuracies)
        medium_accuracies = np.array(medium_accuracies)
        hard_accuracies = np.array(hard_accuracies)
        u.save_data((accuracies, easy_accuracies, medium_accuracies, hard_accuracies), accuracies_file)

        self.plot_error_rates(accuracies, metric_abbreviations)
        self.plot_consistency(easy_accuracies, medium_accuracies, hard_accuracies, metric_abbreviations,
                              len(self.easy_indices))


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
    parser.add_argument('--ensemble_size', type=str, choices=['small', 'large'], default='large',
                        help='Specifies the size of the ensembles to be used in the experiments.')
    parser.add_argument('--grayscale', action='store_true',
                        help='Raise to use grayscale transformation for CIFAR10 when computing Proximity metrics')
    parser.add_argument('--pca', action='store_true', help='Raise to use PCA for CIFAR10 when computing Proximity '
                                                           'metrics (can be combined with --grayscale).')
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    imbalance = HardnessImbalanceMeasurer(**vars(args))
    imbalance.measure_and_visualize_hardness_based_imbalance()
