import argparse
import pickle
from typing import List

import tqdm

from utils import load_data_and_normalize, identify_hard_samples_with_confidences_or_energies, \
    investigate_within_class_imbalance_common, load_results


def main(dataset_name: str, thresholds: List[float], runs: int, sample_removal_rates: List[float], remove_hard: bool,
         subset_size: int):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    confidences = load_results(f'Results/Confidences/{dataset_name}_20_metrics.pkl')
    generalisation_settings = ['full', 'hard', 'easy']
    accuracies = {}
    for idx, threshold in tqdm.tqdm(enumerate(thresholds)):
        current_accuracies = {setting: {sample_removal_rate: [] for sample_removal_rate in sample_removal_rates}
                              for setting in generalisation_settings}
        hard_data, hard_target, easy_data, easy_target = (
            identify_hard_samples_with_confidences_or_energies(confidences, dataset, 'confidence',
                                                               int(70000 * threshold)))
        print(f'A total of {len(hard_data)} hard samples and {len(easy_data)} easy samples were found.')
        investigate_within_class_imbalance_common(runs, hard_data, hard_target, easy_data, easy_target, remove_hard,
                                                  sample_removal_rates, dataset_name, current_accuracies)
        # After each train_ratio, add the collected metrics to the all_metrics dictionary
        accuracies[threshold] = current_accuracies
    metrics_filename = f"{dataset_name}_{remove_hard}_{subset_size}_metrics.pkl"
    with open(metrics_filename, 'wb') as f:
        pickle.dump(accuracies, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.05, 0.1, 0.2],
                        help='')
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
