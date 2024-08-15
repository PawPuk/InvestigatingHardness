import argparse
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader
from prettytable import PrettyTable

import utils as u
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


def main(dataset_name: str, models_count: int):
    # Load the saved DataLoaders
    train_loaders = u.load_data(f"{u.DATA_SAVE_DIR}{dataset_name}_train_loaders.pkl")
    test_loaders = u.load_data(f"{u.DATA_SAVE_DIR}{dataset_name}_test_loaders.pkl")
    # Train and evaluate on all training sets
    results = {
        'Hard': train_and_evaluate_ensemble(train_loaders[0], test_loaders, dataset_name, models_count),
        'Easy': train_and_evaluate_ensemble(train_loaders[1], test_loaders, dataset_name, models_count),
        'All': train_and_evaluate_ensemble(train_loaders[2], test_loaders, dataset_name, models_count)
    }
    # Create a 3x3 table with Training sets as rows and Test sets as columns
    table = PrettyTable()
    table.field_names = ["Training Set / Test Set", "Hard", "Easy", "All"]
    # Populate the table with the results
    for train_set_name in ['Hard', 'Easy', 'All']:
        row = [train_set_name]
        for test_set_name in ['hard', 'easy', 'all']:
            mean_accuracy, std_accuracy = results[train_set_name][test_set_name]
            row.append(f"{mean_accuracy:.2f} ± {std_accuracy:.2f}")
        table.add_row(row)
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of LeNet models and evaluate on test sets.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset to be used (e.g., MNIST, CIFAR10).')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    args = parser.parse_args()
    main(**vars(args))
