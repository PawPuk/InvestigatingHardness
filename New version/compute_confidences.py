import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neural_networks import LeNet
import utils as u

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def compute_hardness_indicators(models: List[torch.nn.Module], loader: DataLoader,
                                weights: List[float]) -> List[Tuple[float, float, bool]]:
    """Compute BMA of confidences, margins, and track whether each sample was misclassified."""
    results = []
    with torch.no_grad():
        for data, targets in tqdm(loader, desc='Computing BMA confidences, margins, and misclassifications'):
            data, targets = data.to(u.DEVICE), targets.to(u.DEVICE)
            weighted_confidences = []
            weighted_margins = []
            all_predictions = []
            for model, weight in zip(models, weights):
                model.eval()
                outputs = model(data)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, max_indices = torch.max(probabilities, dim=1)
                second_max_probs = torch.topk(probabilities, 2, dim=1)[0][:, 1]

                weighted_confidences.append(weight * max_probs.cpu().numpy())
                weighted_margins.append(weight * (max_probs - second_max_probs).cpu().numpy())
                all_predictions.append(max_indices)
            # Compute and store Bayesian Model Averages
            avg_confidences = np.sum(weighted_confidences, axis=0)
            avg_margins = np.sum(weighted_margins, axis=0)
            # Majority vote for ensemble prediction
            ensemble_predictions = torch.stack(all_predictions).mode(dim=0)[0]
            misclassifications = ensemble_predictions != targets

            results.extend(zip(avg_confidences, avg_margins, misclassifications.cpu().numpy()))
    return results


def show_lowest_confidence_samples(dataset: TensorDataset,
                                   confidences_margins_misclassifications: List[Tuple[float, float, bool]],
                                   labels: List[Tensor], n=30):
    """Display the n samples with the lowest BMA confidence."""
    confidences = [conf for conf, _, _ in confidences_margins_misclassifications]
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


def main(dataset_name: str, models_count: int, averaging_type: str):
    dataset = u.load_data_and_normalize(dataset_name)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    models = []
    model_weights = []
    # Load only the specified number of models
    for i in range(models_count):
        model = LeNet().to(u.DEVICE)
        model.load_state_dict(torch.load(f"{u.MODEL_SAVE_DIR}{dataset_name}_{models_count}_ensemble_{i}.pth"))
        models.append(model)
        # Determine weights based on averaging type
        if averaging_type == 'MEAN':
            model_weights = [1.0 / models_count] * models_count
        elif averaging_type == 'BMA':
            # For now, default to uniform weights; you'll implement BMA logic later
            model_weights = [1.0 / models_count] * models_count
        else:
            raise ValueError("Averaging type must be 'BMA' or 'MEAN'.")
    # Compute and save Bayesian Model Averaging confidences, margins, and misclassifications
    hardness_indicators = compute_hardness_indicators(models, loader, model_weights)
    u.save_data(hardness_indicators, f"{u.CONFIDENCES_SAVE_DIR}{dataset_name}_bma_hardness_indicators.pkl")
    # Show the samples with the lowest BMA confidence
    labels = [label for _, label in dataset]
    show_lowest_confidence_samples(dataset, hardness_indicators, labels, n=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load models and compute Bayesian Model Averaging confidences, margins, and misclassifications.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--averaging_type', type=str, default='MEAN', choices=['BMA', 'MEAN'],
                        help='Averaging type for model confidences - either classical mean or Bayesian Model Average.')
    args = parser.parse_args()
    main(**vars(args))
