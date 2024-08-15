import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List

import utils as u
from train_ensembles import EnsembleTrainer  # Import the refactored EnsembleTrainer class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCES_SAVE_DIR = "confidences/"
DATA_SAVE_DIR = "data/"


def train_and_evaluate_ensemble(train_loader: DataLoader, test_loaders: List[DataLoader], dataset_name: str,
                                models_count: int):
    """Train an ensemble of LeNet models and evaluate them on the provided test sets.

    :param train_loader: DataLoader for training data.
    :param test_loaders: List of DataLoaders for hard, easy, and all test data.
    :param dataset_name: Name of the dataset being used (e.g., MNIST, CIFAR10).
    :param models_count: Number of models in the ensemble.
    """
    hard_test_loader, easy_test_loader, test_loader = test_loaders
    # Use the EnsembleTrainer class to train the ensemble
    trainer = EnsembleTrainer(dataset_name, (train_loader, test_loader), models_count)
    trainer.train_ensemble()
    ensemble_accuracies = {
        'hard': [],
        'easy': [],
        'all': []
    }
    # Test each trained model on the test sets
    for i, model in enumerate(trainer.get_trained_models()):
        for loader, test_set_name in zip(test_loaders, ['hard', 'easy', 'all']):
            accuracy = u.test(model, loader)
            ensemble_accuracies[test_set_name].append(accuracy)
    # Calculate and print the mean and standard deviation for each test set
    for test_set_name in ['hard', 'easy', 'all']:
        accuracies = ensemble_accuracies[test_set_name]
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        print(f"\n{test_set_name.capitalize()} test set: Mean accuracy = {mean_accuracy:.2f}%, "
              f"Standard deviation = {std_accuracy:.2f}%\n")


def main(dataset_name: str, models_count: int):
    # Load the saved DataLoaders
    train_loader = u.load_data(f"{DATA_SAVE_DIR}{dataset_name}_train_loader.pkl")
    test_loaders = u.load_data(f"{DATA_SAVE_DIR}{dataset_name}_test_loaders.pkl")
    # Train and evaluate the ensemble of models
    train_and_evaluate_ensemble(train_loader, test_loaders, dataset_name, models_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of LeNet models and evaluate on test sets.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset to be used (e.g., MNIST, CIFAR10).')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    args = parser.parse_args()
    main(**vars(args))
