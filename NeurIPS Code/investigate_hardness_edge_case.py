import argparse
import pickle
from typing import List, Tuple

from torch import Tensor
import tqdm

from utils import load_data_and_normalize, investigate_within_class_imbalance2


def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def sort_dataset_by_confidence(confidences: List[List[float]], dataset) -> Tuple[Tensor, Tensor]:
    transposed_confidences = list(zip(*confidences))
    # Calculate mean confidence per sample
    average_confidences = [sum(sample_confidences) / len(sample_confidences) for sample_confidences in
                           transposed_confidences]
    # Sort indices by average confidence, ascending (least confident first)
    sorted_indices = sorted(range(len(average_confidences)), key=lambda i: average_confidences[i])
    # Extract data and targets from dataset using sorted indices
    data, targets = dataset.tensors
    sorted_data = data[sorted_indices]
    sorted_targets = targets[sorted_indices]
    return sorted_data, sorted_targets


def main(dataset_name: str, thresholds: List[float], sample_removal_rates: List[float], remove_hard: bool,
         subset_size: int):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    confidences = load_results('Results/MNIST_10_metrics.pkl')
    all_metrics = {}
    for idx, threshold in tqdm.tqdm(enumerate(thresholds)):
        current_metrics = {sample_removal_rate: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
                           for sample_removal_rate in sample_removal_rates}
        data, targets = sort_dataset_by_confidence(confidences, dataset)
        investigate_within_class_imbalance2(data, targets, remove_hard, sample_removal_rates, dataset_name,
                                            current_metrics)
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
                        default=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
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
