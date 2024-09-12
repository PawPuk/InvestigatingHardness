import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from compute_confidences import compute_curvatures, compute_proximity_metrics
import utils as u

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def main(dataset_name: str, training: str, k2: int):
    # Define file paths for saving and loading cached results
    proximity_file = f"{u.HARD_IMBALANCE_DIR}{training}{dataset_name}_proximity_indicators.pkl"
    curvature_file = f"{u.HARD_IMBALANCE_DIR}{training}{dataset_name}_curvature_indicators.pkl"
    # Load the dataset
    if training == 'full':
        training_dataset = u.load_full_data_and_normalize(dataset_name, to_grayscale=True)
    else:
        training_dataset, _ = u.load_data_and_normalize(dataset_name, to_grayscale=True)

    loader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=False)

    if os.path.exists(proximity_file):
        raise Exception(f'Proximities were already computed for {dataset_name}.')
    else:
        print('Calculating proximities.')
        proximity_metrics = compute_proximity_metrics(loader, k2)
        u.save_data(proximity_metrics, proximity_file)

    if os.path.exists(curvature_file):
        raise Exception(f'Curvatures were already computed for {dataset_name}.')
    else:
        print('Calculating curvatures.')
        curvature_metrics = compute_curvatures(loader, k2)
        u.save_data(curvature_metrics, curvature_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze hard samples in the official training and test splits using precomputed hardness '
                    'indicators.'
    )
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset (MNIST, CIFAR10, CIFAR100).')
    parser.add_argument('--training', type=str, choices=['full', 'part'], default='full',
                        help='Indicates which models to choose for evaluations - the ones trained on the entire dataset'
                             ' (full), or the ones trained only on the training set (part).')
    parser.add_argument('--k2', type=int, default=40, help='k parameter for the kNN in proximity computations.')
    args = parser.parse_args()

    main(**vars(args))
