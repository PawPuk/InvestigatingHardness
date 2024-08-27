import argparse
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

import utils as u
from prepare_dataset import DatasetPreparer
from train_ensembles import EnsembleTrainer


def train_and_evaluate_ensemble(train_loader: DataLoader, test_loaders: List[DataLoader], dataset_name: str,
                                models_count: int) -> Dict[str, Tuple[float, float]]:
    """Train an ensemble of LeNet models and evaluate them on the provided test sets.

    :param train_loader: DataLoader for training data.
    :param test_loaders: List of DataLoaders for hard, easy, and all test data.
    :param dataset_name: Name of the dataset being used (e.g., MNIST, CIFAR10).
    :param models_count: Number of models in the ensemble.
    """
    trainer = EnsembleTrainer(dataset_name, models_count)
    trainer.train_ensemble(train_loader)
    ensemble_accuracies = {
        'hard': [],
        'easy': [],
        'all': []
    }
    # Test each trained model on the test sets
    for _, model in enumerate(trainer.get_trained_models()):
        for test_loader, test_set_name in zip(test_loaders, ['hard', 'easy', 'all']):
            accuracy = u.test(model, test_loader)
            ensemble_accuracies[test_set_name].append(accuracy)
    # Calculate mean and std for each test set
    results = {}
    for test_set_name in ['hard', 'easy', 'all']:
        results[test_set_name] = u.calculate_mean_std(ensemble_accuracies[test_set_name])
    return results


def main(dataset_name: str, models_count: int, threshold: float, oversampling_factor: float, undersampling_ratio: float,
         smote: bool, training_data = str):
    # Generate custom training and test splits and apply measures against hardness-based data imbalance.
    DP = DatasetPreparer(dataset_name, models_count, threshold, oversampling_factor, undersampling_ratio, smote)
    train_loader_all, train_loader_hard, train_loader_easy, test_loaders = DP.load_and_prepare_data()
    train_loader = train_loader_hard if training_data == 'hard' else train_loader_easy if training_data == 'easy' else train_loader_all
    # Train and evaluate on the selected training set
    results = train_and_evaluate_ensemble(train_loader, test_loaders, dataset_name, models_count)
    u.save_data(results, f"{u.ACCURACIES_SAVE_DIR}{training_data}{dataset_name}_osf_{oversampling_factor}_usr_{undersampling_ratio}.pkl")
    # Print results as a formatted string
    print(f"Results for {training_data} training set:")
    for test_set_name in ['hard', 'easy', 'all']:
        mean_accuracy, std_accuracy = results[test_set_name]
        print(f"Test Set: {test_set_name.capitalize()}")
        print(f"Mean Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")
        print("-" * 40)  # Divider for clarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of LeNet models and evaluate on test sets.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset to be used (e.g., MNIST, CIFAR10).')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--threshold', type=float, default=0.99,
                        help='Confidence and margin threshold to split easy and hard samples.')
    parser.add_argument('--oversampling_factor', type=float, default=0.0,
                        help='Factor for oversampling hard samples using random duplication. '
                             '0.0 keeps the size of hard samples the same as the hard dataset size, '
                             '1.0 increases the hard dataset size to match the easy dataset size. '
                             'Values in between allow partial oversampling.')
    parser.add_argument('--undersampling_ratio', type=float, default=1.0,
                        help='Ratio for undersampling easy samples using random removal. '
                             '0.0 reduces the size of easy samples to match the hard dataset size, '
                             '1.0 keeps the size of easy samples the same as the easy dataset size. '
                             'Values in between allow partial undersampling.')
    parser.add_argument('--smote', default=False, action='store_true',
                        help='')  # TODO: finish
    parser.add_argument('--training_data', type=str, choices=['hard, easy, all'], default='all',
                        help='')
    args = parser.parse_args()
    main(**vars(args))
