import argparse
import pickle
from typing import List

from numpy import array
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()


def compute_confidences(model: torch.nn.Module, loader: DataLoader) -> List[array]:
    """Compute model confidences on the combined CIFAR10 dataset."""
    confidences = []
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(DEVICE)
            outputs = model(data)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]
            confidences.extend(max_probs.cpu().numpy())
    return confidences


def save_data(data, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def main(dataset_name: str, runs: int):
    dataset = utils.load_data_and_normalize(dataset_name, 70000)
    full_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_confidences = []
    my_models, optimizers = utils.initialize_models(dataset_name)
    for i in tqdm(range(len(my_models))):
        if i == runs:
            break
        utils.train(dataset_name, my_models[i], full_loader, optimizers[i])
        confidences = compute_confidences(my_models[i], full_loader)
        all_confidences.append(confidences)
        current_metrics = utils.test(my_models[i], full_loader)
        print(current_metrics['accuracy'])
    save_data(all_confidences, f"Results/{dataset_name}_{runs}_metrics.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    main(**vars(args))
