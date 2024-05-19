import argparse
from typing import List

import torch
import tqdm

from utils import load_results, load_data_and_normalize, identify_hard_samples, \
    investigate_within_class_imbalance_common, save_data


def main(dataset_name: str, strategy: str, runs: int, sample_removal_rates: List[float], remove_hard: bool,
         subset_size: int):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    confidences_and_energy = load_results(f'Results/{dataset_name}_20_metrics.pkl')
    indices_of_hard_samples = []
    results = {setting: {reduce_train_ratio: [] for reduce_train_ratio in sample_removal_rates}
               for setting in ['full', 'hard', 'easy']}
    for _ in tqdm.tqdm(range(runs), desc='Repeating the experiment for different straggler sets'):
        hard_data, hard_target, easy_data, easy_target, hard_indices = identify_hard_samples(strategy, dataset,
                                                                                             confidences_and_energy)
        print(f'A total of {len(hard_data)} hard samples and {len(easy_data)} easy samples were found.')
        investigate_within_class_imbalance_common(runs, hard_data, hard_target, easy_data, easy_target, remove_hard,
                                                  sample_removal_rates, dataset_name, results)
        hard_indices = [h_index.cpu() for h_index in hard_indices] if torch.cuda.is_available() else hard_indices
        indices_of_hard_samples.append(hard_indices)
    save_data(results, f"{dataset_name}_{strategy}_{remove_hard}_{subset_size}_metrics.pkl")
    save_data(indices_of_hard_samples, f"{dataset_name}_{strategy}_{remove_hard}_{subset_size}_indices.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--strategy', type=str, choices=['stragglers', 'confidence', 'energy'],
                        default='stragglers', help='Strategy (method) to use for identifying hard samples.')
    parser.add_argument('--runs', type=int, default=5,
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
    parser.add_argument('--subset_size', default=20000, type=int,
                        help='Specifies the subset of the dataset used for the experiments. Later it will be divided '
                             'into train and testing training and test sets based pm the --train_ratios parameter.')
    args = parser.parse_args()
    main(**vars(args))
