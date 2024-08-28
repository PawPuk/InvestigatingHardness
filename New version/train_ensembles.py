import argparse
from typing import Union

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
    def __init__(self, dataset_name: str, models_count: int = 20, save: bool = False):
        self.dataset_name = dataset_name
        self.models_count = models_count
        self.save = save
        self.models = []

    def train_ensemble(self, train_loader: DataLoader, test_loader: Union[DataLoader, None] = None,
                       imbalanced_ratio: float = 1.0):
        """Train an ensemble of models on the full dataset."""
        for i in tqdm(range(self.models_count)):
            model = LeNet().to(u.DEVICE)
            optimizer = Adam(model.parameters(), lr=0.001)
            # Train the model
            u.train(self.dataset_name, model, train_loader, optimizer)
            self.models.append(model)
            # Save model state
            if self.save:
                torch.save(model.state_dict(),
                           f"{u.MODEL_SAVE_DIR}{self.dataset_name}_{self.models_count}_{imbalanced_ratio}_ensemble_{i}.pth",
                           _use_new_zipfile_serialization=False)  # Ensuring backward compatibility
            # Evaluate on the training set
            if test_loader is not None:
                accuracy = u.test(model, test_loader)
                print(f'Model {i} finished training, achieving accuracy of {accuracy}% on the test set.')

    def get_trained_models(self):
        """Return the list of trained models."""
        return self.models


def main(dataset_name: str, models_count: int, long_tailed: bool, imbalance_ratio: float):
    trainer = EnsembleTrainer(dataset_name, models_count, True)
    train_dataset, test_dataset = u.load_data_and_normalize(dataset_name, long_tailed, imbalance_ratio)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    trainer.train_ensemble(train_loader, test_loader, imbalance_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on the full dataset and save parameters.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    parser.add_argument('--long_tailed', type=bool, default=False,
                        help='Flag to indicate if the dataset should be long-tailed.')
    parser.add_argument('--imbalance_ratio', type=float, default=1.0,
                        help='Imbalance ratio for long-tailed dataset.')
    args = parser.parse_args()
    main(**vars(args))
