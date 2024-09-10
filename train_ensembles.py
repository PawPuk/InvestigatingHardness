import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as u


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class EnsembleTrainer:
    def __init__(self, dataset_name: str, models_count: int, save: bool, training: str):
        self.dataset_name = dataset_name
        self.models_count = models_count
        self.save = save
        self.models = []
        self.training = training

    def train_ensemble(self, train_loader: DataLoader, test_loader: DataLoader):
        """Train an ensemble of models on the full dataset."""
        if self.training == 'full':
            print('Training an ensemble of networks in `full information` scenario')
        else:
            print('Training an ensemble of networks in `partial information` scenario')
        num_classes = len(torch.unique(train_loader.dataset.tensors[1]))
        class_accuracies = np.zeros((self.models_count, num_classes))  # Store class-level accuracies for all models
        epochs = 100 if self.dataset_name == 'CIFAR10' else 10

        for i in tqdm(range(self.models_count)):
            model, optimizer = u.initialize_models(self.dataset_name)
            # Train the model
            u.train(self.dataset_name, model, train_loader, optimizer, epochs)
            self.models.append(model)
            # Save model state
            if self.save:
                torch.save(model.state_dict(), f"{u.MODEL_SAVE_DIR}{self.training}"
                                               f"{self.dataset_name}_{self.models_count}_ensemble_{i}.pth",
                           _use_new_zipfile_serialization=False)  # Ensuring backward compatibility
            # Evaluate on the training set
            accuracy = u.test(model, test_loader)
            class_accuracies[i] = u.class_level_test(model, test_loader, num_classes)
            print(f'Model {i} finished training, achieving accuracy of {accuracy}% on the test set.')

        if self.training == 'full':
            running_avg_class_accuracies = np.cumsum(class_accuracies, axis=0) / np.arange(1, self.models_count + 1)[:, None]
            # Plot the results for each class
            self.plot_class_accuracies(running_avg_class_accuracies, num_classes)
            u.save_data(running_avg_class_accuracies, f"{u.HARD_IMBALANCE_DIR}{self.dataset_name}"
                                                      f"_avg_class_accuracies.pkl")

    def plot_class_accuracies(self, running_avg_class_accuracies, num_classes):
        """Plot how the average accuracy of each class changes as we increase the number of models."""
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 5 columns

        for class_idx in range(num_classes):
            row, col = divmod(class_idx, 5)  # Calculate row and column indices for the subplot
            ax = axes[row, col]
            ax.plot(range(1, self.models_count + 1), running_avg_class_accuracies[:, class_idx])
            ax.set_title(f'Class {class_idx}')
            ax.set_xlabel('Number of models')
            ax.set_ylabel('Avg Accuracy')

        plt.tight_layout()
        plt.savefig(f'{self.training}{self.dataset_name}_class_bias.png')
        plt.show()

    def get_trained_models(self):
        """Return the list of trained models."""
        return self.models


def main(dataset_name: str, models_count: int, training: str):
    trainer = EnsembleTrainer(dataset_name, models_count, True, training)
    if training == 'full':  # 'full information' scenario
        train_dataset = u.load_full_data_and_normalize(dataset_name)
        test_dataset = train_dataset
    else:  # 'limited information' scenario
        train_dataset, test_dataset = u.load_data_and_normalize(dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    trainer.train_ensemble(train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on the full dataset and save parameters.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--training', type=str, choices=['full', 'part'], default='full',
                        help='Indicates which models to choose for evaluations - the ones trained on the entire dataset'
                             ' (full), or the ones trained only on the training set (part).')
    args = parser.parse_args()
    main(**vars(args))
