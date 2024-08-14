import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset
import matplotlib.pyplot as plt

import utils as u

CONFIDENCES_SAVE_DIR = "confidences/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_dataset_by_confidence(dataset: TensorDataset, confidences: List[float],
                                threshold: float) -> Tuple[Subset, Subset]:
    """Split the dataset into easy and hard samples based on the confidence threshold."""
    # TODO: make it possible to change the meaning of threshold (to mean % of samples)
    easy_indices = [i for i, conf in enumerate(confidences) if conf >= threshold]
    hard_indices = [i for i, conf in enumerate(confidences) if conf < threshold]
    easy_dataset, hard_dataset = Subset(dataset, easy_indices), Subset(dataset, hard_indices)
    return easy_dataset, hard_dataset


def show_samples(dataset: Subset, indices, title: str, n=30):
    """Display n samples from the dataset."""
    plt.figure(figsize=(15, 15))
    for i, index in enumerate(indices[:n]):
        plt.subplot(6, 5, i + 1)
        plt.imshow(dataset[index][0].numpy().squeeze(), cmap='gray')
        true_label = dataset[index][1]
        plt.title(f'Y: {true_label}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def main(dataset_name: str, threshold: float):
    dataset = u.load_data_and_normalize(dataset_name)
    bma_confidences = u.load_data(f"{CONFIDENCES_SAVE_DIR}{dataset_name}_bma_confidences.pkl")
    easy_dataset, hard_dataset = split_dataset_by_confidence(dataset, bma_confidences, threshold)
    # Show samples from easy and hard datasets
    easy_indices = [i for i in range(len(easy_dataset))]
    hard_indices = [i for i in range(len(hard_dataset))]
    print(f'Found {len(easy_indices)} easy data samples, and {len(hard_indices)} hard data samples.')
    show_samples(easy_dataset, easy_indices, title='Easy Samples')
    show_samples(hard_dataset, hard_indices, title='Hard Samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into easy and hard samples based on BMA confidences.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Confidence threshold to split easy and hard samples.')
    args = parser.parse_args()
    main(**vars(args))
