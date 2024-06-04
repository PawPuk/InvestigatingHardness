import argparse
import pickle
from typing import List, Tuple

from torch import Tensor

from utils import load_data_and_normalize, load_results, investigate_within_class_imbalance_edge, save_data


def sort_dataset_by_confidence(confidences_and_energies: List[List[Tuple[float, float]]],
                               dataset) -> Tuple[Tensor, Tensor]:
    # Extract confidences and ignore energies
    transposed_confidences = list(zip(*[[ce[0] for ce in run_results] for run_results in confidences_and_energies]))
    # Calculate mean confidence per sample
    average_confidences = [sum(sample_confidences) / len(sample_confidences)
                           for sample_confidences in transposed_confidences]
    # Sort indices by average confidence, ascending (the least confident first)
    sorted_indices = sorted(range(len(average_confidences)), key=lambda i: average_confidences[i])
    # Extract data and targets from dataset using sorted indices
    data, targets = dataset.tensors
    sorted_data = data[sorted_indices]
    sorted_targets = targets[sorted_indices]
    return sorted_data, sorted_targets


def main(dataset_name: str, runs: int, sample_removal_rates: List[float], remove_hard: bool):
    dataset_size = 60000 if dataset_name == 'CIFAR10' else 70000
    dataset = load_data_and_normalize(dataset_name, dataset_size)
    confidences = load_results(f'Results/Confidences/{dataset_name}_5_metrics.pkl')
    results = {sample_removal_rate: [] for sample_removal_rate in sample_removal_rates}
    data, targets = sort_dataset_by_confidence(confidences, dataset)
    investigate_within_class_imbalance_edge(runs, data, targets, remove_hard, sample_removal_rates, dataset_name,
                                            results)
    save_data(results, f"Results/Generalizations/{dataset_name}_{remove_hard}_{runs}_edge_metrics.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--runs', type=int, default=20,
                        help='Specifies how many straggler sets will be computed for the experiment, and how many '
                             'networks will be trained per a straggler set (for every ratio in remaining_train_ratios. '
                             'The larger this value the higher the complexity and the statistical significance.')
    parser.add_argument('--sample_removal_rates', nargs='+', type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95],
                        help='Percentage of train samples on which we train.')
    parser.add_argument('--remove_hard', action='store_true', default=False,
                        help='Flag indicating whether we want to see the effect of changing the number of easy (False) '
                             'or hard (True) samples.')
    args = parser.parse_args()
    main(**vars(args))
