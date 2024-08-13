import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import LeNet
import utils as u


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
MODEL_SAVE_DIR = "models/"
CONFIDENCES_SAVE_DIR = "confidences/"
os.makedirs(CONFIDENCES_SAVE_DIR, exist_ok=True)


def compute_bma_confidences(models: List[torch.nn.Module], loader: DataLoader, weights: List[float]) -> List[float]:
    """Compute Bayesian averages of models' confidences on the entire dataset."""
    results = []
    with torch.no_grad():
        for data, _ in tqdm(loader, desc='Computing BMA confidences'):
            data = data.to(DEVICE)
            weighted_confidences = []
            for model, weight in zip(models, weights):
                model.eval()
                outputs = model(data)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_probs = torch.max(probabilities, dim=1)[0]
                weighted_confidences.append(weight * max_probs.cpu().numpy())
            # Compute and store Bayesian Model Average of confidences
            avg_confidences = np.sum(weighted_confidences, axis=0)
            results.extend(avg_confidences)
    return results


def show_lowest_confidence_samples(dataset, confidences, labels, n=30):
    """Display the n samples with the lowest BMA confidence."""
    # Find the indices of the n lowest confidence values
    lowest_confidence_indices = np.argsort(confidences)[:n]
    lowest_confidences = np.array(confidences)[lowest_confidence_indices]

    # Plot the samples with the lowest confidences
    plt.figure(figsize=(15, 15))
    for i, index in enumerate(lowest_confidence_indices):
        plt.subplot(6, 5, i + 1)
        plt.imshow(dataset[index][0].numpy().squeeze(), cmap='gray')
        true_label = labels[index]
        plt.title(f'Y: {true_label}, Conf: {lowest_confidences[i]:.4f}')
        plt.axis('off')
    plt.show()


def main(dataset_name, models_count, averaging_type):
    dataset = u.load_data_and_normalize(dataset_name)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    models = []
    model_weights = []
    # Load only the specified number of models
    for _ in range(models_count):
        model = LeNet().to(DEVICE)
        model.load_state_dict(torch.load(f"{MODEL_SAVE_DIR}{dataset_name}_{models_count}_ensemble.pth"))
        models.append(model)
        # Determine weights based on averaging type
        if averaging_type == 'MEAN':
            model_weights = [1.0 / models_count] * models_count
        elif averaging_type == 'BMA':
            # For now, default to uniform weights; you'll implement BMA logic later
            model_weights = [1.0 / models_count] * models_count
        else:
            raise ValueError("Averaging type must be 'BMA' or 'MEAN'.")
    # Compute and save Bayesian Model Averaging confidences
    bma_confidences = compute_bma_confidences(models, loader, model_weights)
    u.save_data(bma_confidences, f"{CONFIDENCES_SAVE_DIR}{dataset_name}_bma_confidences.pkl")
    # Show the samples with the lowest BMA confidence
    labels = [label for _, label in dataset]
    show_lowest_confidence_samples(dataset, bma_confidences, labels, n=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load models and compute Bayesian Model Averaging confidences.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--averaging_type', type=str, default='MEAN', choices=['BMA', 'MEAN'],
                        help='Averaging type for model confidences - either classical mean or Bayesian Model Average.')
    args = parser.parse_args()
    main(**vars(args))
