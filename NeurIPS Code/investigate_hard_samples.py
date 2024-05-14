import argparse
import pickle
from typing import List

from torch import Tensor
import tqdm

from utils import load_data_and_normalize, straggler_ratio_vs_generalisation


def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def identify_hard_samples(confidences: List[List[float]], dataset, threshold: float) -> List[Tensor]:
    # Number of samples to include in the least confident subset
    num_least_confident = int(threshold * len(confidences[0]))
    # Sort indices by confidence, ascending (least confident first)
    sorted_indices = sorted(range(len(confidences[0])), key=lambda i: confidences[0][i])
    # Divide indices into least and most confident based on the threshold
    least_confident_indices = sorted_indices[:num_least_confident]
    most_confident_indices = sorted_indices[num_least_confident:]
    # Extract data and targets from dataset
    data, targets = dataset.tensors
    # Extract least and most confident data and targets
    least_confident_data = data[least_confident_indices]
    least_confident_targets = targets[least_confident_indices]
    most_confident_data = data[most_confident_indices]
    most_confident_targets = targets[most_confident_indices]
    return [least_confident_data, least_confident_targets, most_confident_data, most_confident_targets]


def main(dataset_name: str, thresholds: List[float], sample_removal_rates: List[float], remove_hard: bool,
         subset_size: int):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    confidences = load_results('Results/MNIST_10_metrics.pkl')
    generalisation_settings = ['full', 'hard', 'easy']
    all_metrics = {}
    for idx, threshold in tqdm.tqdm(enumerate(thresholds)):
        current_metrics = {setting: {sample_removal_rate: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
                                     for sample_removal_rate in sample_removal_rates}
                           for setting in generalisation_settings}
        hard_data, hard_target, easy_data, easy_target = identify_hard_samples(confidences, dataset, threshold)
        print(f'A total of {len(hard_data)} hard samples and {len(easy_data)} easy samples were found.')
        straggler_ratio_vs_generalisation(hard_data, hard_target, easy_data, easy_target, remove_hard,
                                          sample_removal_rates, dataset_name, current_metrics)
        # After each train_ratio, add the collected metrics to the all_metrics dictionary
        all_metrics[threshold] = current_metrics
    metrics_filename = f"{dataset_name}_{remove_hard}_{subset_size}_metrics.pkl"
    with open(metrics_filename, 'wb') as f:
        pickle.dump(all_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.05, 0.1, 0.2],
                        help='')
    parser.add_argument('--sample_removal_rates', nargs='+', type=float,
                        default=[0.0, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0],
                        help='Percentage of train hard/easy samples on which we train; we only reduce the number of '
                             'hard OR easy samples (depending on --reduce_hard flag). So 0.1 means that 90% of hard '
                             'samples will be removed from the train set before training (when reduce_hard == True).')
    parser.add_argument('--remove_hard', action='store_true', default=False,
                        help='flag indicating whether we want to see the effect of changing the number of easy (False) '
                             'or hard (True) samples.')
    parser.add_argument('--subset_size', default=70000, type=int,
                        help='Specifies the subset of the dataset used for the experiments. Later it will be divided '
                             'into train and testing training and test sets based pm the --train_ratios parameter.')
    args = parser.parse_args()
    main(**vars(args))
