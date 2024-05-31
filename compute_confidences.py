import argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()


def compute_confidences_and_energies(model: torch.nn.Module, loader: DataLoader) -> List[Tuple[float, float]]:
    """Compute model confidences and energies on the combined CIFAR10 dataset."""
    results = []
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(DEVICE)
            outputs = model(data)
            # Calculate confidences
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]
            confidences = max_probs.cpu().numpy()
            # Calculate energies
            energies = -torch.logsumexp(outputs, dim=1).cpu().numpy()
            # Store the results as tuples of (confidence, energy)
            results.extend(zip(confidences, energies))
    return results


def main(dataset_name: str, model_instances: int):
    dataset = utils.load_data_and_normalize(dataset_name, 70000)
    full_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    final_results = []
    my_models, optimizers = utils.initialize_models(dataset_name, model_instances)
    for i in tqdm(range(len(my_models))):
        utils.train(dataset_name, my_models[i], full_loader, optimizers[i])
        confidences_and_energies = compute_confidences_and_energies(my_models[i], full_loader)
        final_results.append(confidences_and_energies)
        # This part is just for sanity check (to make sure the training converged).
        current_metrics = utils.test(my_models[i], full_loader)
        print(current_metrics)
    utils.save_data(final_results, f"Results/Confidences/{dataset_name}_{model_instances}_metrics.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        choices=['MNIST', 'KMNIST', 'FashionMNIST', 'CIFAR10'],
                        help='Specifies which dataset to run the experiment on. We limit the amount of available '
                             'datasets to the ones we tested, but the code should generalize to other datasets, with '
                             'similar dimensions.')
    parser.add_argument('--model_instances', type=int, default=5,
                        help='Specifies how many instances of each architecture we want to train in our experiment. '
                             'There are 4 architectures for each dataset type, so the total number of networks is 4 '
                             'times model_instances.')
    args = parser.parse_args()
    main(**vars(args))
