import argparse
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils as u
from imbalance_measures import ImbalanceMeasures
from train_ensembles import EnsembleTrainer


def apply_resampling_techniques(tensor_datasets, dataset_size):
    # If the size of tensor_dataset (percentage of dataset_size) is lower than this number than apply strict rules
    string_oversampling_threshold = 0.1
    strict_undersampling_threshold = 0.25
    strict_addition_ratio = 2.0
    lenient_addition_ratio = 1.0
    strict_removal_ratio = 1.0
    lenient_removal_ratio = 0.5

    if 'hard' in tensor_datasets:
        hard_dataset = tensor_datasets['hard']
        hard_size_before = len(hard_dataset)
        IM = ImbalanceMeasures(None, hard_dataset)

        if len(hard_dataset) < string_oversampling_threshold * dataset_size:
            hard_dataset_resampled = IM.random_oversampling(2.0)  # Increase hard samples two-fold
        else:
            hard_percentage = len(hard_dataset) / dataset_size
            adaptive_size = (hard_percentage - string_oversampling_threshold) / (0.5 - string_oversampling_threshold)
            adaptive_resampling_rate = strict_addition_ratio - adaptive_size * (strict_addition_ratio -
                                                                                lenient_addition_ratio)
            hard_dataset_resampled = IM.random_oversampling(adaptive_resampling_rate)

        hard_size_after = len(hard_dataset_resampled)
        tensor_datasets['hard'] = hard_dataset_resampled
        print(f"\tAdded {hard_size_after - hard_size_before} hard samples via oversampling.")

    if 'easy' in tensor_datasets:
        easy_dataset = tensor_datasets['easy']
        easy_size_before = len(easy_dataset)
        IM = ImbalanceMeasures(easy_dataset, None)

        if len(easy_dataset) < strict_undersampling_threshold * dataset_size:
            easy_dataset_resampled = IM.random_undersampling(strict_removal_ratio)  # Apply strict rules
        else:
            easy_percentage = len(easy_dataset) / dataset_size
            adaptive_size = (easy_percentage - strict_undersampling_threshold) / (1.0 - strict_undersampling_threshold)
            adaptive_resampling_rate = strict_removal_ratio - adaptive_size * (strict_removal_ratio -
                                                                               lenient_removal_ratio)
            easy_dataset_resampled = IM.random_undersampling(adaptive_resampling_rate)

        easy_size_after = len(easy_dataset_resampled)
        tensor_datasets['easy'] = easy_dataset_resampled
        print(f"\tRemoved {easy_size_before - easy_size_after} easy samples via undersampling.")

    if 'medium' in tensor_datasets:
        medium_dataset = tensor_datasets['medium']
        medium_size_before = len(medium_dataset)
        IM = ImbalanceMeasures(medium_dataset, None)
        medium_dataset_resampled = IM.random_undersampling(lenient_removal_ratio)
        medium_size_after = len(medium_dataset_resampled)
        tensor_datasets['medium'] = medium_dataset_resampled
        print(f"\tRemoved {medium_size_before - medium_size_after} medium samples via undersampling.")


def load_and_prepare_data(model_type, dataset_name) -> List[Tuple[DataLoader, DataLoader]]:
    # Load and normalize dataset
    training_dataset, test_dataset = u.load_data_and_normalize(dataset_name)
    data_tensor, target_tensor = training_dataset.tensors
    easy_indices_list, hard_indices_list, _, _ = u.load_data(
        f'{u.DIVISIONS_SAVE_DIR}part{model_type}{dataset_name}_indices.pkl')

    # Divide training_dataset into easy, medium, and hard based on easy_indices and hard_indices
    metric_datasets = []
    for metric_idx, (easy_indices, hard_indices) in enumerate(zip(easy_indices_list, hard_indices_list)):
        print(f'\nWorking on metric {metric_idx}.')
        easy_indices_set = set(easy_indices)  # Convert to set for fast lookup
        hard_indices_set = set(hard_indices)  # Convert to set for fast lookup

        # Determine medium indices (those not in easy or hard)
        all_indices = set(range(len(training_dataset)))  # All indices of the dataset
        medium_indices_set = all_indices - easy_indices_set - hard_indices_set

        # Create empty lists to hold samples for easy, medium, and hard categories
        easy_data, medium_data, hard_data = [], [], []
        easy_targets, medium_targets, hard_targets = [], [], []

        # Categorize data and targets into easy, medium, and hard lists
        for idx in easy_indices_set:
            easy_data.append(data_tensor[idx])
            easy_targets.append(target_tensor[idx])
        for idx in hard_indices_set:
            hard_data.append(data_tensor[idx])
            hard_targets.append(target_tensor[idx])
        for idx in medium_indices_set:
            medium_data.append(data_tensor[idx])
            medium_targets.append(target_tensor[idx])

        # Convert lists into TensorDatasets if they contain samples
        tensor_datasets = {}  # List to store created TensorDatasets
        if easy_data:
            easy_dataset = TensorDataset(torch.stack(easy_data), torch.tensor(easy_targets))
            tensor_datasets['easy'] = easy_dataset
        if medium_data:
            medium_dataset = TensorDataset(torch.stack(medium_data), torch.tensor(medium_targets))
            tensor_datasets['medium'] = medium_dataset
        if hard_data:
            hard_dataset = TensorDataset(torch.stack(hard_data), torch.tensor(hard_targets))
            tensor_datasets['hard'] = hard_dataset

        apply_resampling_techniques(tensor_datasets, len(training_dataset) + len(test_dataset))

        # Now combine the tensors from all categories (easy, medium, hard) into one TensorDataset
        combined_data, combined_targets = [], []

        for dataset in tensor_datasets.values():
            combined_data.append(dataset.tensors[0])
            combined_targets.append(dataset.tensors[1])

        # Concatenate the tensors to create a single TensorDataset
        final_data = torch.cat(combined_data, dim=0)
        final_targets = torch.cat(combined_targets, dim=0)
        combined_dataset = TensorDataset(final_data, final_targets)

        # Check the size difference and print the adaptive message
        final_size_before = len(training_dataset)
        final_size_after = len(combined_dataset)
        change_in_size = final_size_after - final_size_before
        percentage_of_original = (final_size_after / final_size_before) * 100

        if change_in_size < 0:
            print(
                f"Removed {-change_in_size} samples. The new training set is {percentage_of_original:.2f}% of the "
                f"original size (new size: {final_size_after}).")
        elif change_in_size > 0:
            print(
                f"Added {change_in_size} samples. The new training set is {percentage_of_original:.2f}% of the "
                f"original size (new size: {final_size_after}).")
        else:
            print(f"No change in the training set size. It remains the same with {final_size_after} samples.")
        metric_datasets.append(combined_dataset)

    return [(DataLoader(dataset, batch_size=32, shuffle=True),
             DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)) for dataset in metric_datasets]


def main(dataset_name: str, models_count: int, model_type: str):
    dataloaders = load_and_prepare_data(model_type, dataset_name)
    for metric_idx, (training_loader, test_loader) in tqdm(enumerate(dataloaders)):
        save_dir = f'{u.RESAMPLED_SAVE_DIR}/metric{metric_idx}/'
        os.makedirs(save_dir, exist_ok=True)
        if dataset_name == 'CIFAR10' and model_type == 'complex' and metric_idx not in [1, 2, 4, 7, 12]:
            continue
        trainer = EnsembleTrainer(dataset_name, models_count, True, 'part', model_type, save_dir)
        trainer.train_ensemble(training_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on the full dataset and save parameters.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=50, help='Number of models to train in this run.')
    parser.add_argument('--model_type', type=str, choices=['simple', 'complex'],
                        help='Specifies the type of network used for training (MLP vs LeNet or ResNet20 vs ResNet56).')
    args = parser.parse_args()
    main(**vars(args))
