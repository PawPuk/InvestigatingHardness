import argparse

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import LeNet
import utils as u


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class EnsembleTrainer:
    def __init__(self, dataset_name: str, models_count: int, save: bool):
        self.dataset_name = dataset_name
        self.models_count = models_count
        self.save = save
        self.models = []

    def train_ensemble(self, train_loader: DataLoader, test_loader: DataLoader):
        """Train an ensemble of models on the full dataset."""

        for i in tqdm(range(self.models_count)):
            model = LeNet().to(u.DEVICE)
            optimizer = Adam(model.parameters(), lr=0.001)
            # Train the model
            u.train(self.dataset_name, model, train_loader, optimizer)
            self.models.append(model)
            # Save model state
            if self.save:
                torch.save(model.state_dict(), f"{u.MODEL_SAVE_DIR}{['part', 'full'][train_loader == test_loader]}"
                                               f"{self.dataset_name}_{self.models_count}_ensemble_{i}.pth",
                           _use_new_zipfile_serialization=False)  # Ensuring backward compatibility
            # Evaluate on the training set
            accuracy = u.test(model, test_loader)
            print(f'Model {i} finished training, achieving accuracy of {accuracy}% on the test set.')

        if train_loader == test_loader:
            # Compute and save the average class-level accuracies
            num_classes = len(torch.unique(train_loader.dataset.tensors[1]))
            class_accuracies = np.zeros((self.models_count, num_classes))
            for model_idx, model in enumerate(self.models):
                class_accuracies[model_idx] = u.class_level_test(model, test_loader, num_classes)
            avg_class_accuracies = class_accuracies.mean(axis=0)
            u.save_data(avg_class_accuracies, f"{u.HARD_IMBALANCE_DIR}{self.dataset_name}"
                                              f"_avg_class_accuracies.pkl")

            # Find the hardest and easiest classes, analyze hard sample distribution and visualize results
            hardest_class = np.argmin(avg_class_accuracies)
            easiest_class = np.argmax(avg_class_accuracies)
            print(f"\nHardest class accuracy (class {hardest_class}): {avg_class_accuracies[hardest_class]:.5f}%")
            print(f"Easiest class accuracy (class {easiest_class}): {avg_class_accuracies[easiest_class]:.5f}%")

    def get_trained_models(self):
        """Return the list of trained models."""
        return self.models


def main(dataset_name: str, models_count: int, training: str):
    trainer = EnsembleTrainer(dataset_name, models_count, True)
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
                        help='Indicates which models to choose for evaluations - the ones trained on the entire dataset '
                             '(full), or the ones trained only on the training set (part).')
    args = parser.parse_args()
    main(**vars(args))
