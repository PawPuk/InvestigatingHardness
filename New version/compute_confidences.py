import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neural_networks import LeNet
import utils as u

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def visualize_hardness_indicators(hardness_indicators: List[Tuple[float, float, int]], num_models: int):
    """
    Visualize how the number of hard samples changes as we adjust the threshold for confidence, margin, and misclassification.

    :param hardness_indicators: List of tuples containing (confidence, margin, misclassification count) for each sample.
    :param num_models: The number of models in the ensemble (used as the upper bound for misclassification thresholds).
    """
    # Define the thresholds
    confidence_thresholds = np.linspace(0.8, 1, 100)
    margin_thresholds = np.linspace(0.8, 1, 100)
    misclassification_thresholds = np.arange(1, num_models + 1)
    confidence_hard_counts = []
    margin_hard_counts = []
    misclassification_hard_counts = []
    # Loop over thresholds to calculate hard sample counts
    for threshold in confidence_thresholds:
        confidence_hard = sum(1 for conf, _, _ in hardness_indicators if conf < threshold)
        confidence_hard_counts.append(confidence_hard)
    for threshold in margin_thresholds:
        margin_hard = sum(1 for _, margin, _ in hardness_indicators if margin < threshold)
        margin_hard_counts.append(margin_hard)
    for threshold in misclassification_thresholds:
        misclassification_hard = sum(1 for _, _, misclassified in hardness_indicators if misclassified >= threshold)
        misclassification_hard_counts.append(misclassification_hard)
    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Plot Confidence-based hardness
    axes[0].plot(confidence_thresholds, confidence_hard_counts, color='blue')
    axes[0].set_title("Confidence-based Hardness")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Number of Hard Samples")
    # Plot Margin-based hardness
    axes[1].plot(margin_thresholds, margin_hard_counts, color='green')
    axes[1].set_title("Margin-based Hardness")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Number of Hard Samples")
    # Plot Misclassification-based hardness
    axes[2].plot(misclassification_thresholds, misclassification_hard_counts, color='red')
    axes[2].set_title("Misclassification-based Hardness")
    axes[2].set_xlabel("Number of Misclassifications")
    axes[2].set_ylabel("Number of Hard Samples")
    plt.tight_layout()
    plt.savefig('Hardness_indicators.pdf')
    plt.show()


def compute_hardness_indicators(models: List[torch.nn.Module], loader: DataLoader,
                                weights: List[float]) -> List[Tuple[float, float, int]]:
    """Compute BMA of confidences, margins, and track the number of times each sample was misclassified."""
    # TODO: add other hardness indicators (maybe loss and gradient)
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
            # Compute Bayesian Model Averages
            avg_confidences = np.sum(weighted_confidences, axis=0)
            avg_margins = np.sum(weighted_margins, axis=0)
            # Count how many times the sample was misclassified across all models
            all_predictions = torch.stack(all_predictions)
            misclassification_counts = torch.sum(all_predictions != targets.unsqueeze(0), dim=0).cpu().numpy()
            results.extend(zip(avg_confidences, avg_margins, misclassification_counts))
    return results


def show_lowest_confidence_samples(dataset: Dataset,
                                   confidences_margins_misclassifications: List[Tuple[float, float, int]],
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


def main(dataset_name: str, models_count: int, averaging_type: str, long_tailed: bool, imbalance_ratio: float):
    train_dataset, _ = u.load_data_and_normalize(dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    models, model_weights = [], []
    # Load only the specified number of models
    for i in range(models_count):
        model = LeNet().to(u.DEVICE)
        model.load_state_dict(torch.load(
            f"{u.MODEL_SAVE_DIR}{dataset_name}_{models_count}_{imbalance_ratio}_ensemble_{i}.pth"))
        models.append(model)
        # Determine weights based on averaging type
        if averaging_type == 'MEAN':
            model_weights = [1.0 / models_count] * models_count
        elif averaging_type == 'BMA':
            model_weights = [1.0 / models_count] * models_count  # TODO: implement
        else:
            raise ValueError("Averaging type must be 'BMA' or 'MEAN'.")
    # Compute and save Bayesian Model Averaging confidences, margins, and misclassifications
    hardness_indicators = compute_hardness_indicators(models, train_loader, model_weights)
    u.save_data(hardness_indicators,
                f"{u.CONFIDENCES_SAVE_DIR}{dataset_name}_{imbalance_ratio}_bma_hardness_indicators.pkl")
    visualize_hardness_indicators(hardness_indicators, models_count)
    # Show the samples with the lowest BMA confidence
    labels = [label for _, label in train_dataset]
    show_lowest_confidence_samples(train_dataset, hardness_indicators, labels, n=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load models and compute Bayesian Model Averaging confidences, margins, and misclassifications.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--averaging_type', type=str, default='MEAN', choices=['BMA', 'MEAN'],
                        help='Averaging type for model confidences - either classical mean or Bayesian Model Average.')
    parser.add_argument('--long_tailed', type=bool, default=False,
                        help='Flag to indicate if the dataset should be long-tailed.')
    parser.add_argument('--imbalance_ratio', type=float, default=1.0,
                        help='Imbalance ratio for long-tailed dataset.')
    args = parser.parse_args()
    main(**vars(args))
