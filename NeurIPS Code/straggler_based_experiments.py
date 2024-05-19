import argparse
from torch.utils.data import TensorDataset
from typing import List, Tuple

import torch

from utils import load_results, load_data_and_normalize, identify_hard_samples_with_confidences_or_energies, \
    investigate_within_class_imbalance_common, save_data


def find_universal_stragglers(dataset: TensorDataset, filename: str,
                              threshold: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Load the results which contain lists of tensors with straggler flags
    hard_samples_indices = load_results(filename)

    # Check the device of the first tensor in the first list (assuming all tensors are on the same device)
    if len(hard_samples_indices) > 0 and len(hard_samples_indices[0]) > 0:
        device = hard_samples_indices[0][0].device
    else:
        raise ValueError("No straggler data found in the results.")

    num_samples = dataset.tensors[0].size(0)
    straggler_counts = torch.zeros(num_samples, dtype=torch.int32, device=device)

    # Aggregate the counts of straggler flags across all runs
    for run_list in hard_samples_indices:
        for tensor in run_list:
            straggler_counts += tensor.to(device).int()  # Ensure tensor is on the correct device

    # Identify indices that meet the threshold
    hard_indices = torch.where(straggler_counts >= threshold)[0].cpu()  # Move indices to CPU
    easy_indices = torch.where(straggler_counts < threshold)[0].cpu()  # Move indices to CPU

    # Extract the hard and easy data and targets from the TensorDataset
    hard_data = dataset.tensors[0][hard_indices]  # Indexing on CPU
    hard_target = dataset.tensors[1][hard_indices]  # Indexing on CPU
    easy_data = dataset.tensors[0][easy_indices]  # Indexing on CPU
    easy_target = dataset.tensors[1][easy_indices]  # Indexing on CPU

    return hard_data, hard_target, easy_data, easy_target


def main(dataset_name: str, strategy: str, runs: int, sample_removal_rates: List[float], remove_hard: bool,
         subset_size: int):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    confidences_and_energy = load_results(f'Results/{dataset_name}_20_metrics.pkl')
    results = {setting: {reduce_train_ratio: [] for reduce_train_ratio in sample_removal_rates}
               for setting in ['full', 'hard', 'easy']}
    filename = f'Results/straggler_indices_{dataset_name}_20.pkl'
    hard_data, hard_target, easy_data, easy_target = find_universal_stragglers(dataset, filename)
    print(len(hard_data))
    if strategy != 'stragglers':
        hard_data, hard_target, easy_data, easy_target = \
            identify_hard_samples_with_confidences_or_energies(confidences_and_energy, dataset, strategy,
                                                               len(hard_data))
    print(f'A total of {len(hard_data)} hard samples and {len(easy_data)} easy samples were found.')
    investigate_within_class_imbalance_common(runs, hard_data, hard_target, easy_data, easy_target, remove_hard,
                                              sample_removal_rates, dataset_name, results)
    save_data(results, f"{dataset_name}_{strategy}_{remove_hard}_{subset_size}_metrics.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--strategy', type=str, choices=['stragglers', 'confidence', 'energy'],
                        default='stragglers', help='Strategy (method) to use for identifying hard samples.')
    parser.add_argument('--runs', type=int, default=20,
                        help='Specifies how many straggler sets will be computed for the experiment, and how many '
                             'networks will be trained per a straggler set (for every ratio in remaining_train_ratios. '
                             'The larger this value the higher the complexity and the statistical significance.')
    parser.add_argument('--sample_removal_rates', nargs='+', type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
                        help='Percentage of easy/hard (depending on the remove_hard flag) of training samples that will'
                             'be removed.')
    parser.add_argument('--remove_hard', action='store_true', default=False,
                        help='flag indicating whether we want to see the effect of changing the number of easy (False) '
                             'or hard (True) samples.')
    parser.add_argument('--subset_size', default=70000, type=int,
                        help='Specifies the subset of the dataset used for the experiments. Later it will be divided '
                             'into train and testing training and test sets based pm the --train_ratios parameter.')
    args = parser.parse_args()
    main(**vars(args))
