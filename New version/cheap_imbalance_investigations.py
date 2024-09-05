import numpy as np
from typing import List

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import LeNet
import utils as u


def test_model_on_samples(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                          indices: List[int], dataset_size: int, test_size: int) -> float:
    """Test a single model on a specific set of samples (given by indices) and return accuracy."""
    model.eval()
    all_outputs = []
    all_targets = []

    # Compute the starting index for the test set based on its size
    test_start_idx = dataset_size - test_size

    # Filter indices to match only test samples (last portion of the dataset)
    test_indices = [i - test_start_idx for i in indices if test_start_idx <= i < dataset_size]

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data).cpu().numpy()
            all_outputs.append(outputs)
            all_targets.append(target.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    if len(test_indices) > 0:
        selected_outputs = all_outputs[test_indices]
        selected_targets = all_targets[test_indices]
    else:
        raise Exception('No valid test indices found.')

    correct = (np.argmax(selected_outputs, axis=1) == selected_targets).sum()
    total = selected_targets.shape[0]

    accuracy = 100 * correct / total if total > 0 else None  # Using None, as this should not occur.
    return accuracy


def test_ensemble(models: List[torch.nn.Module], loader: DataLoader, device: torch.device,
                  easy_indices: List[List[int]], hard_indices: List[List[int]],
                  dataset_size: int, test_size: int) -> List[dict]:
    """Test an ensemble of models on the dataset, returning accuracies for each metric."""

    accuracies = []

    for metric_idx in tqdm(range(len(easy_indices)), desc='Iterating through metrics'):
        metric_easy_indices = easy_indices[metric_idx]
        metric_hard_indices = hard_indices[metric_idx]

        all_accuracies = []
        easy_accuracies = []
        hard_accuracies = []

        for model in models[:3]:
            acc_all = test_model_on_samples(model, loader, device, list(range(dataset_size)), dataset_size, test_size)
            acc_easy = test_model_on_samples(model, loader, device, metric_easy_indices, dataset_size, test_size)
            acc_hard = test_model_on_samples(model, loader, device, metric_hard_indices, dataset_size, test_size)

            all_accuracies.append(acc_all)
            easy_accuracies.append(acc_easy)
            hard_accuracies.append(acc_hard)

        # Compute mean and std for each accuracy type
        accuracies.append({
            'metric_idx': metric_idx + 1,
            'mean_acc_all': np.mean(all_accuracies),
            'std_acc_all': np.std(all_accuracies),
            'mean_acc_easy': np.mean(easy_accuracies),
            'std_acc_easy': np.std(easy_accuracies),
            'mean_acc_hard': np.mean(hard_accuracies),
            'std_acc_hard': np.std(hard_accuracies)
        })

    return accuracies


def plot_error_rates(accuracies: List[dict]) -> None:
    """Plot the error rates for all metrics in one figure, with vertical line segments for each error rate type."""

    num_metrics = len(accuracies)
    indices = np.arange(1, num_metrics + 1)  # Indices for each metric

    # Calculate epsilon based on the number of metrics and plot width
    epsilon = 0.1  # Adjust epsilon to control the spacing between lines for each metric

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through each metric and plot the error rates (100 - accuracy) as vertical line segments
    for i, acc in enumerate(accuracies):
        metric_idx = acc['metric_idx']

        # Compute error rates (since error rate = 100% - accuracy)
        error_all = 100 - acc['mean_acc_all']
        error_easy = 100 - acc['mean_acc_easy']
        error_hard = 100 - acc['mean_acc_hard']

        # Plot vertical line segments using Line2D for all, easy, and hard samples
        ax.add_line(Line2D([metric_idx - epsilon, metric_idx + epsilon], [error_easy, error_easy],
                           color='green', linewidth=5))  # Green for easy
        ax.add_line(Line2D([metric_idx - epsilon, metric_idx + epsilon], [error_all, error_all],
                           color='black', linewidth=5))  # Black for all
        ax.add_line(Line2D([metric_idx - epsilon, metric_idx + epsilon], [error_hard, error_hard],
                           color='red', linewidth=5))  # Red for hard

    # Labeling and formatting
    ax.set_xticks(indices)
    ax.set_xticklabels([f'{acc["metric_idx"]}' for acc in accuracies])
    ax.set_xlabel('Metric')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rates across Different Metrics')
    ax.grid(True)

    # Adjust the x limits to avoid cutting off lines on the right edge
    ax.set_xlim(0.5, num_metrics + 0.5)  # Padding added to the left and right

    # Show the plot
    plt.tight_layout()
    plt.show()



def main(dataset_name: str, models_count: int):
    """Main function to load the hard/easy indices and pre-trained models, and test on dataset."""

    # Load necessary data
    easy_indices = u.load_data(f'{u.DIVISIONS_SAVE_DIR}/{dataset_name}_easy_indices.pkl')
    hard_indices = u.load_data(f'{u.DIVISIONS_SAVE_DIR}/{dataset_name}_hard_indices.pkl')
    models = []

    # Load models
    for i in range(models_count):
        model = LeNet().to(u.DEVICE)
        model_file = f"{u.MODEL_SAVE_DIR}{dataset_name}_{models_count}_ensemble_{i}.pth"
        model.load_state_dict(torch.load(model_file, map_location=u.DEVICE))
        models.append(model)

    # Load the dataset and normalize it
    train_dataset, test_dataset = u.load_data_and_normalize(dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Compute dataset size and test size
    dataset_size = len(train_dataset) + len(test_dataset)
    test_size = len(test_dataset)

    # Test the ensemble on the test set and store accuracies
    accuracies = test_ensemble(models, test_loader, u.DEVICE, easy_indices, hard_indices, dataset_size, test_size)

    # Plot accuracies
    plot_error_rates(accuracies)



if __name__ == "__main__":
    import argparse

    # Argument parser for dataset_name and models_count
    parser = argparse.ArgumentParser(description="Load easy/hard indices and pre-trained models.")
    parser.add_argument("--dataset_name", type=str, default='MNIST',
                        help="Name of the dataset (e.g., MNIST, CIFAR10, etc.)")
    parser.add_argument("--models_count", type=int, default='100', help="Number of models in the ensemble")

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.dataset_name, args.models_count)
