import argparse
import torch
from tqdm import tqdm
import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(dataset_name: str, runs: int):
    dataset = utils.load_data_and_normalize(dataset_name, 70000)
    hard_samples_indices = []
    for _ in tqdm(range(runs)):
        _, _, _, _, stragglers = utils.find_stragglers(dataset)
        hard_samples_indices.append(stragglers)
    utils.save_data(hard_samples_indices, f"Results/hard_samples_indices_{dataset_name}_{runs}.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect indices of hard samples across multiple runs.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    main(**vars(args))
