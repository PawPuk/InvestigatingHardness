from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from geometric_measures import Curvature, ModelBasedMetrics, Proximity
import utils as u

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def divide_by_class(loader: DataLoader) -> Tuple[Dict[int, DataLoader], Dict[int, List[int]]]:
    """Divide the data in the loader into separate loaders by class."""
    class_indices = {}
    dataset = loader.dataset
    targets = dataset.tensors[1] if hasattr(dataset, 'tensors') else dataset.targets

    # Group indices by class
    for idx, target in enumerate(targets):
        target = target.item()
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(idx)

    # Create DataLoaders for each class
    class_loaders = {}
    for cls, indices in class_indices.items():
        class_loaders[cls] = DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)

    return class_loaders, class_indices


def compute_curvatures(training_loader: DataLoader, test_loader: DataLoader, k1: int):
    """Compute curvatures for test samples using nearest neighbors from the training set."""
    # Prepare training data for kNN
    train_data_list, train_labels_list = [], []
    for data, targets in training_loader:
        train_data_list.append(data.view(data.size(0), -1))  # Flatten training samples
        train_labels_list.append(targets)

    train_data = torch.cat(train_data_list).cpu().numpy()

    # Initialize final curvature lists for the test data
    total_test_samples = sum(len(data) for data, _ in test_loader)
    gaussian_curvatures = [None] * total_test_samples
    mean_curvatures = [None] * total_test_samples

    # Process the test data in batches
    test_idx_offset = 0
    for data, _ in tqdm(test_loader, desc="Computing curvatures for test samples"):
        data_flat = data.view(data.size(0), -1).cpu().numpy()  # Flatten test samples
        num_test_samples = data_flat.shape[0]

        # Estimate curvatures using training data for neighbors
        Curvature(train_data, np.arange(len(train_data)), k=k1).estimate_curvatures_for_test(
            test_data=data_flat,
            gaussian_curvatures=gaussian_curvatures,
            mean_curvatures=mean_curvatures,
            test_idx_offset=test_idx_offset
        )

        test_idx_offset += num_test_samples

    return gaussian_curvatures, mean_curvatures


def compute_proximity_metrics(training_loader: DataLoader, test_loader: DataLoader, k2: int,
                              class_loaders: List[DataLoader]):
    """Compute the geometric metrics of the data that can be used to identify hard samples, class-wise."""
    proximity = Proximity(training_loader, test_loader, class_loaders, k=k2)
    proximity_metrics = proximity.compute_proximity_metrics()
    return proximity_metrics


def compute_model_based_metrics(dataset_name: str, training: str, dataset: TensorDataset, ensemble_size: str):
    data = dataset.tensors[0]
    labels = dataset.tensors[1].numpy()
    modelBasedMetrics = ModelBasedMetrics(dataset_name, training, data, labels, ensemble_size)
    complex_metrics = modelBasedMetrics.compute_model_based_hardness('complex')
    simple_metrics = modelBasedMetrics.compute_model_based_hardness('simple')
    return simple_metrics, complex_metrics


"""def main(dataset_name: str, models_count: int, long_tailed: bool, imbalance_ratio: float):
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
    main(**vars(args))"""
