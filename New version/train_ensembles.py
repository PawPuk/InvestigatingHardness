import argparse
import os
from typing import Union

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from neural_networks import LeNet
import utils as u


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
MODEL_SAVE_DIR = "models/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


class EnsembleTrainer:
    def __init__(self, dataset_name: str, models_count: int = 20, save: bool = False):
        self.dataset_name = dataset_name
        self.models_count = models_count
        self.save = save
        self.models = []

    def train_ensemble(self, train_loader: DataLoader, test_loader: Union[DataLoader, None] = None):
        """Train an ensemble of models on the full dataset."""
        for i in tqdm(range(self.models_count)):
            model = LeNet().to(DEVICE)
            optimizer = Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-4)
            # Train the model
            u.train(self.dataset_name, model, train_loader, optimizer)
            self.models.append(model)
            # Save model state
            if self.save:
                torch.save(model.state_dict(),
                           f"{MODEL_SAVE_DIR}{self.dataset_name}_{self.models_count}_ensemble_{i}.pth")
            # Evaluate on the training set
            if test_loader is not None:
                accuracy = u.test(model, test_loader)
                print(f'Model {i} finished training, achieving accuracy of {accuracy}% on the training set.')

    def get_trained_models(self):
        """Return the list of trained models."""
        return self.models


def main(dataset_name: str, models_count: int):
    trainer = EnsembleTrainer(dataset_name, models_count, True)
    dataset = u.load_data_and_normalize(dataset_name)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    trainer.train_ensemble(loader, loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on the full dataset and save parameters.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    args = parser.parse_args()
    main(**vars(args))
