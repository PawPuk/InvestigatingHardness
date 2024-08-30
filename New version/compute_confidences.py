import argparse
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from geometric_measures import Curvature, Proximity
from neural_networks import LeNet
import utils as u

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def compute_hardness_indicators(loader: DataLoader):
    """Compute the geometric metrics of the data that can be used to identify hard samples, class-wise."""
    proximity = Proximity(loader)
    proximity_metrics = proximity.compute_proximity_ratios()
    return proximity_metrics


def main(dataset_name: str, models_count: int, long_tailed: bool, imbalance_ratio: float):
    train_dataset, _ = u.load_data_and_normalize(dataset_name, long_tailed, imbalance_ratio)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    models = []
    # Load only the specified number of models
    for i in range(models_count):
        model = LeNet().to(u.DEVICE)
        model.load_state_dict(torch.load(
            f"{u.MODEL_SAVE_DIR}{dataset_name}_{models_count}_{imbalance_ratio}_ensemble_{i}.pth"))
        models.append(model)
    # Compute and save Bayesian Model Averaging confidences, margins, and misclassifications
    hardness_indicators = compute_hardness_indicators(models, train_loader, models_count)
    u.save_data(hardness_indicators,
                f"{u.CONFIDENCES_SAVE_DIR}{dataset_name}_{imbalance_ratio}_bma_hardness_indicators.pkl")
    # Show the samples with the lowest BMA confidence
    # labels = [label for _, label in train_dataset]
    # show_lowest_confidence_samples(train_dataset, hardness_indicators, labels, n=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load models and compute Bayesian Model Averaging confidences, margins, and misclassifications.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--long_tailed', type=bool, default=False,
                        help='Flag to indicate if the dataset should be long-tailed.')
    parser.add_argument('--imbalance_ratio', type=float, default=1.0,
                        help='Imbalance ratio for long-tailed dataset.')
    args = parser.parse_args()
    main(**vars(args))
