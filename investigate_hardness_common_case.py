import argparse
import pickle
from typing import List

import tqdm

from utils import load_data_and_normalize, identify_hard_samples_with_confidences_or_energies, \
    investigate_within_class_imbalance_common, load_results, save_data


def main(dataset_name: str, thresholds: List[float], runs: int, sample_removal_rates: List[float], remove_hard: bool):
    dataset_size = 60000 if dataset_name == 'CIFAR10' else 70000
    dataset = load_data_and_normalize(dataset_name, dataset_size)
    confidences = load_results(f'Results/Confidences/{dataset_name}_20_metrics.pkl')
    generalisation_settings = ['full', 'hard', 'easy']
    accuracies = {}
    for idx, threshold in tqdm.tqdm(enumerate(thresholds)):
        current_accuracies = {setting: {sample_removal_rate: [] for sample_removal_rate in sample_removal_rates}
                              for setting in generalisation_settings}
        hard_data, hard_target, easy_data, easy_target = (
            identify_hard_samples_with_confidences_or_energies(confidences, dataset, 'confidence',
                                                               int(dataset_size * threshold)))
        print(f'A total of {len(hard_data)} hard samples and {len(easy_data)} easy samples were found.')
        investigate_within_class_imbalance_common(runs, hard_data, hard_target, easy_data, easy_target, remove_hard,
                                                  sample_removal_rates, dataset_name, current_accuracies)
        # After each train_ratio, add the collected metrics to the all_metrics dictionary
        accuracies[threshold] = current_accuracies
    save_data(accuracies, f"{dataset_name}_{remove_hard}_metrics.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST, KMNIST, and CIFAR10.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.05, 0.1, 0.2],
                        help='This threshold is used to define hard/easy samples in the confidence- and energy-based'
                             'approaches. Samples with lowest confidence (highest energy) will be considered as hard')
    parser.add_argument('--runs', type=int, default=20,
                        help='Specifies how many networks will be trained when measuring generalization. The larger '
                             'this value, the higher the computational complexity and the statistical significance.')
    parser.add_argument('--sample_removal_rates', nargs='+', type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
                        help='Percentage of easy/hard (depending on the remove_hard flag) data samples that will be '
                             'removed from the training set.')
    parser.add_argument('--remove_hard', action='store_true', default=False,
                        help='Flag indicating whether we want to see the effect of changing the number of easy (False) '
                             'or hard (True) samples in the training set on the generalization.')
    args = parser.parse_args()
    main(**vars(args))
