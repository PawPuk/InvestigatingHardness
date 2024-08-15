import argparse
from typing import List, Tuple

import torch
from torch.utils.data import Subset, TensorDataset
import numpy as np

import utils as u

CONFIDENCES_SAVE_DIR = "confidences/"
DATA_SAVE_DIR = "data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def identify_hard_and_easy_data(dataset: TensorDataset, hardness_indicators: List[Tuple[float, float, bool]],
                                threshold: float) -> Tuple[Subset, Subset]:
    """Split the dataset into easy and hard samples based on confidence, margin, and misclassification."""
    confidence_hard_indices = {i for i, (conf, _, _) in enumerate(hardness_indicators) if conf < threshold}
    margin_hard_indices = {i for i, (_, margin, _) in enumerate(hardness_indicators) if margin < threshold}
    misclassified_hard_indices = {i for i, (_, _, misclassified) in enumerate(hardness_indicators) if misclassified}

    # Combine hard indices from all conditions without duplicates
    combined_hard_indices = confidence_hard_indices.union(margin_hard_indices).union(misclassified_hard_indices)
    easy_indices = [i for i in range(len(dataset)) if i not in combined_hard_indices]
    easy_dataset, hard_dataset = Subset(dataset, easy_indices), Subset(dataset, list(combined_hard_indices))

    # Reporting with a symmetric matrix-like structure
    indicators = ['Confidence', 'Margin', 'Misclassified']
    hard_indices_list = [confidence_hard_indices, margin_hard_indices, misclassified_hard_indices]
    overlap_matrix = np.zeros((len(indicators), len(indicators)), dtype=int)

    for i in range(len(indicators)):
        for j in range(len(indicators)):
            overlap_matrix[i, j] = len(hard_indices_list[i].intersection(hard_indices_list[j]))

    print(f"Found {len(confidence_hard_indices)} hard samples based on confidence threshold.")
    print(f"Found {len(margin_hard_indices)} hard samples based on margin threshold.")
    print(f"Found {len(misclassified_hard_indices)} hard samples based on misclassification.")
    print("Overlap Matrix:")
    print(f"{'':>15} {'Confidence':>12} {'Margin':>12} {'Misclassified':>15}")
    for i, indicator in enumerate(indicators):
        print(f"{indicator:>15} {overlap_matrix[i, 0]:>12} {overlap_matrix[i, 1]:>12} {overlap_matrix[i, 2]:>15}")

    print(f"Continuing with {len(easy_dataset)} easy data samples, and {len(hard_dataset)} hard data samples.")

    return easy_dataset, hard_dataset


def main(dataset_name: str, threshold: float):
    dataset = u.load_data_and_normalize(dataset_name)
    hardness_indicators = u.load_data(f"{CONFIDENCES_SAVE_DIR}{dataset_name}_bma_hardness_indicators.pkl")
    # Identify hard and easy samples
    easy_dataset, hard_dataset = identify_hard_and_easy_data(dataset, hardness_indicators, threshold)
    # Combine and split data into train and test sets, and get the corresponding TensorDatasets
    train_loaders, test_loaders = u.combine_and_split_data(hard_dataset, easy_dataset, dataset_name)
    # Save the DataLoaders
    u.save_data(train_loaders, f"{DATA_SAVE_DIR}{dataset_name}_train_loaders.pkl")
    u.save_data(test_loaders, f"{DATA_SAVE_DIR}{dataset_name}_test_loaders.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split dataset into easy and hard samples based on confidence, margin, and misclassification.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Confidence and margin threshold to split easy and hard samples.')
    args = parser.parse_args()
    main(**vars(args))
