import argparse
import os

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


def main(dataset_name, models_count):
    dataset = u.load_data_and_normalize(dataset_name)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    models = []
    for i in tqdm(range(models_count)):
        model = LeNet().to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-4)
        u.train(dataset_name, model, loader, optimizer)
        models.append(model)
        torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}{dataset_name}_{models_count}_ensemble.pth")
        accuracy = u.test(model, loader)
        print(f'Model {i} finished training achieving accuracy of {accuracy} on training set.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on the full dataset and save parameters.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models in the ensemble.')
    args = parser.parse_args()
    main(**vars(args))
