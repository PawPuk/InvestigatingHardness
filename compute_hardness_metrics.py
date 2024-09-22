import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from compute_confidences import compute_curvatures, compute_model_based_metrics, compute_proximity_metrics
import utils as u

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def create_class_loaders(tensor_dataset: TensorDataset, batch_size=32, shuffle=False):
    """
    Create a list of DataLoaders where each DataLoader contains only samples of a specific class.
    This function assumes the dataset is a TensorDataset.
    """
    class_loaders = []
    data_tensors, target_tensors = tensor_dataset.tensors

    num_classes = len(torch.unique(target_tensors))  # Get the number of classes

    for cls in range(num_classes):
        # Get indices of the samples that belong to the current class
        class_indices = (target_tensors == cls).nonzero(as_tuple=True)[0]
        # Create a subset of the dataset for the current class
        class_subset = Subset(tensor_dataset, class_indices)
        # Create a DataLoader for the current class subset
        class_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=shuffle)
        class_loaders.append(class_loader)

    return class_loaders


def main(dataset_name: str, training: str, k2: int, ensemble_size: str, grayscale: bool, pca: bool):
    # Define file paths for saving and loading cached results
    proximity_file = f"{u.METRICS_SAVE_DIR}{training}{dataset_name}"
    curvature_file = f"{u.METRICS_SAVE_DIR}{training}{dataset_name}"
    model_based_file = f"{u.METRICS_SAVE_DIR}{ensemble_size}_{training}{dataset_name}"
    if dataset_name == 'CIFAR10':
        if grayscale:
            proximity_file += 'gray'
            curvature_file += 'gray'
        if pca:
            proximity_file += 'pca'
            curvature_file += 'pca'
    proximity_file += "_proximity_indicators.pkl"
    curvature_file += "_curvature_indicators.pkl"
    # Load the dataset
    if training == 'full':
        training_dataset = u.load_full_data_and_normalize(dataset_name, to_grayscale=grayscale, apply_pca=pca)
    else:
        training_dataset, _ = u.load_data_and_normalize(dataset_name, to_grayscale=grayscale, apply_pca=pca)

    class_loaders = create_class_loaders(training_dataset, batch_size=len(training_dataset), shuffle=False)
    loader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=False)

    if os.path.exists(proximity_file):
        print(f'Proximities were already computed for {dataset_name}.')
    else:
        print('Calculating proximities.')
        proximity_metrics = compute_proximity_metrics(loader, k2, class_loaders)
        u.save_data(proximity_metrics, proximity_file)

    if os.path.exists(curvature_file):
        print(f'Curvatures were already computed for {dataset_name}.')
    else:
        print('Calculating curvatures.')
        curvature_metrics = compute_curvatures(loader, k2)
        u.save_data(curvature_metrics, curvature_file)

    if training == 'full':
        training_dataset = u.load_full_data_and_normalize(dataset_name, to_grayscale=False, apply_pca=False)
    else:
        training_dataset, _ = u.load_data_and_normalize(dataset_name, to_grayscale=False, apply_pca=False)

    if os.path.exists(model_based_file):
        print(f'Model-based hardness metrics were already computed for {dataset_name}.')
    else:
        print('Computing model-based hardness metrics')
        simple_model_based_metrics, complex_model_based_metrics = compute_model_based_metrics(dataset_name, training,
                                                                                              training_dataset,
                                                                                              ensemble_size)
        u.save_data(simple_model_based_metrics, model_based_file + '_simple_model_based_indicators.pkl')
        u.save_data(complex_model_based_metrics, model_based_file + '_complex_model_based_indicators.pkl')


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
    parser.add_argument('--ensemble_size', type=str, choices=['small', 'large'], default='small',
                        help='Specifies the size of the ensembles to be used in the experiments.')
    parser.add_argument('--grayscale', action='store_true',
                        help='Raise to use grayscale transformation for CIFAR10 when computing Proximity metrics')
    parser.add_argument('--pca', action='store_true', help='Raise to use PCA for CIFAR10 when computing Proximity '
                                                           'metrics (can be combined with --grayscale).')
    args = parser.parse_args()

    main(**vars(args))
