from typing import Dict, List, Tuple, Union
import pickle

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from neural_networks import SimpleNN

EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
CRITERION = torch.nn.CrossEntropyLoss()
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_data(data, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def transform_datasets_to_dataloaders(dataset: TensorDataset, shuffle=False) -> DataLoader:
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data, target in loader:
        loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=shuffle)
    return loader


def load_data_and_normalize(dataset_name: str, subset_size: int) -> TensorDataset:
    """ Used to load the data from common datasets available in torchvision, and normalize them. The normalization
    is based on the mean and std of a random subset of the dataset of the size subset_size.

    :param dataset_name: name of the dataset to load. It has to be available in `torchvision.datasets`
    :param subset_size: used when not working on the full dataset - the results will be less reliable, but the
    complexity will be lowered
    :return: random, normalized subset of dataset_name of size subset_size with (noise_rate*subset_size) labels changed
    to introduce label noise
    """
    # Load the train and test datasets based on the 'dataset_name' parameter
    train_dataset = getattr(datasets, dataset_name)(root="./data", train=True, download=True,
                                                    transform=transforms.ToTensor())
    test_dataset = getattr(datasets, dataset_name)(root="./data", train=False, download=True,
                                                   transform=transforms.ToTensor())
    if dataset_name == 'CIFAR10':
        train_dataset.data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2)
        test_dataset.data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2)
    # Concatenate train and test datasets
    full_data = torch.cat([train_dataset.data.unsqueeze(1).float(), test_dataset.data.unsqueeze(1).float()])
    if dataset_name == 'CIFAR10':
        full_data = full_data.squeeze(1)
    full_targets = torch.cat([torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)])
    # Shuffle the combined dataset
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(full_data))
    full_data, full_targets = full_data[torch.tensor(shuffled_indices)], full_targets[torch.tensor(shuffled_indices)]
    # Select a subset based on the 'subset_size' parameter
    subset_data, subset_targets = full_data[:subset_size], full_targets[:subset_size]
    # Normalize the data
    data_means = torch.mean(subset_data, dim=(0, 2, 3)) / 255.0
    data_vars = torch.sqrt(torch.var(subset_data, dim=(0, 2, 3)) / 255.0 ** 2 + EPSILON)
    # Apply the calculated normalization to the subset
    normalize_transform = transforms.Normalize(mean=data_means, std=data_vars)
    normalized_subset_data = normalize_transform(subset_data / 255.0)
    print(torch.mean(normalized_subset_data, dim=(0, 2, 3)))
    print(torch.std(normalized_subset_data, dim=(0, 2, 3)))
    return TensorDataset(normalized_subset_data, subset_targets)


def initialize_models(dataset_name: str) -> Tuple[List[torch.nn.Module], List[Union[Adam, SGD]]]:
    if dataset_name == 'CIFAR10':
        print('Loading ResNet56, VGG19_bn, MobileNetV2_x1_4, ShuffleNetV2_x2_0 and RepVGG_A2.')
        model_names = [
            "cifar10_resnet56",
            "cifar10_mobilenetv2_x1_4",
            "cifar10_shufflenetv2_x2_0",
            "cifar10_repvgg_a2"
        ]
        # Create three instances of each model type with fresh initializations
        model_list = [torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=False).to(DEVICE)
                      for model_name in model_names for _ in range(5)]
    else:
        print('Loading SimpleNN.')
        model_list = [SimpleNN(28 * 28, 2, 20, 1).to(DEVICE) for _ in range(10)]
    if dataset_name == 'CIFAR10':
        optimizer_list = [Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-4) for model in
                          model_list]
    else:
        optimizer_list = [SGD(model.parameters(), lr=0.1) for model in model_list]
    return model_list, optimizer_list


def test(model: torch.nn.Module, loader: DataLoader) -> float:
    """Measures the accuracy of the 'model' on the test set.

    :param model: The model to evaluate.
    :param loader: DataLoader containing test data.
    :return: Dictionary with accuracy on the test set rounded to 2 decimal places.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            total += target.size(0)  # Increment the total count
            correct += (predicted == target).sum().item()  # Increment the correct count

    accuracy = 100 * correct / total
    return round(accuracy, 2)


def combine_and_split_data(hard_data: Tensor, easy_data: Tensor, hard_target: Tensor, easy_target: Tensor,
                           remove_hard: bool, sample_removal_rate: float) -> Tuple[DataLoader, List[DataLoader]]:
    """ This function divides easy and hard data samples into train and test sets.

    :param hard_data: identifies hard samples (data)
    :param hard_target: identified hard samples (target)
    :param easy_data: identified easy samples (data)
    :param easy_target: identified easy samples (target)
    :param remove_hard: flag indicating whether we want to see the effect of changing the number of easy (False) or
    hard (True) samples
    :param sample_removal_rate: ratio of easy/hard samples remaining in the train set (0.1 means that 90% of hard
    samples will be removed from the pool of hard samples when generating train set, when reduce_hard == True)
    :return: returns train loader and 3 test loaders - 1) with all data samples; 2) with only hard data samples; and 3)
    with only easy data samples.
    """
    # Randomly shuffle hard and easy samples
    hard_perm, easy_perm = torch.randperm(hard_data.size(0)), torch.randperm(easy_data.size(0))
    hard_data, hard_target = hard_data[hard_perm], hard_target[hard_perm]
    easy_data, easy_target = easy_data[easy_perm], easy_target[easy_perm]
    # Split data into initial train/test sets (use 10k test samples)
    train_size_hard = int(len(hard_data) * (1 - (10000 / (len(hard_data) + len(easy_data)))))
    train_size_easy = int(len(easy_data) * (1 - (10000 / (len(hard_data) + len(easy_data)))))
    hard_train_data, hard_test_data = hard_data[:train_size_hard], hard_data[train_size_hard:]
    hard_train_target, hard_test_target = hard_target[:train_size_hard], hard_target[train_size_hard:]
    easy_train_data, easy_test_data = easy_data[:train_size_easy], easy_data[train_size_easy:]
    easy_train_target, easy_test_target = easy_target[:train_size_easy], easy_target[train_size_easy:]
    # Reduce the number of train samples by remaining_train_ratio
    if not 0 <= sample_removal_rate <= 1:
        raise ValueError(f'The parameter remaining_train_ratio must be in [0, 1]; {sample_removal_rate} not allowed.')
    if remove_hard:
        reduced_hard_train_size = int(train_size_hard * (1 - sample_removal_rate))
        reduced_easy_train_size = train_size_easy
    else:
        reduced_hard_train_size = train_size_hard
        reduced_easy_train_size = int(train_size_easy * (1 - sample_removal_rate))
    print(f'Proceeding with {reduced_hard_train_size} hard samples, and {reduced_easy_train_size} easy samples.')
    # Combine easy and hard samples into train and test data
    train_data = torch.cat((hard_train_data[:reduced_hard_train_size],
                            easy_train_data[:reduced_easy_train_size]), dim=0)
    train_targets = torch.cat((hard_train_target[:reduced_hard_train_size],
                               easy_train_target[:reduced_easy_train_size]), dim=0)
    # Shuffle the final train dataset (important when working in full-batch setting)
    train_permutation = torch.randperm(train_data.size(0))
    train_data, train_targets = train_data[train_permutation], train_targets[train_permutation]
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=128, shuffle=True)
    # Create two test sets - one containing only hard samples, and the other only easy samples
    hard_and_easy_test_sets = [(hard_test_data, hard_test_target), (easy_test_data, easy_test_target)]
    full_test_data = torch.cat((hard_and_easy_test_sets[0][0], hard_and_easy_test_sets[1][0]), dim=0)
    full_test_targets = torch.cat((hard_and_easy_test_sets[0][1], hard_and_easy_test_sets[1][1]), dim=0)
    # Create 3 test loaders: 1) with all data samples; 2) with only hard data samples; 3) with only easy data samples
    test_loaders = []
    for data, target in [(full_test_data, full_test_targets)] + hard_and_easy_test_sets:
        test_loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
        test_loaders.append(test_loader)
    return train_loader, test_loaders


def split_data(data: Tensor, targets: Tensor, remove_hard: bool,
               sample_removal_rate: float) -> Tuple[DataLoader, DataLoader]:
    """ This function divides easy and hard data samples into train and test sets.

    :param data: data samples sorted by their average confidence (over models computed with compute_confidences.py)
    :param targets: labels corresponding to the sorted data samples
    :param remove_hard: flag indicating whether we want to see the effect of changing the number of easy (False) or
    hard (True) samples
    :param sample_removal_rate: ratio of training samples remaining in the train set (0.1 means that 90% of hard
    samples will be removed from the pool of hard samples when generating train set, when reduce_hard == True)
    :return: returns train loader and test loader
    """
    if not remove_hard:
        data, targets = torch.flip(data, dims=[0]), torch.flip(targets, dims=[0])
    # Split data into initial train/test sets (use 10k test samples)
    training_set_size = int(len(data) * (1 - (10000 / len(data))))
    training_data, test_data = data[:training_set_size], data[training_set_size:]
    training_targets, test_targets = targets[:training_set_size], targets[training_set_size:]
    # Reduce the number of train samples by remaining_train_ratio
    if not 0 <= sample_removal_rate <= 1:
        raise ValueError(f'The parameter remaining_train_ratio must be in [0, 1]; {sample_removal_rate} not allowed.')
    reduced_training_set_size = int(training_set_size * (1 - sample_removal_rate))
    training_data = training_data[:reduced_training_set_size]
    training_targets = training_targets[:reduced_training_set_size]
    print(f'Proceeding with {reduced_training_set_size} train samples.')
    # Shuffle the final train dataset (important when working in full-batch setting)
    training_permutation = torch.randperm(training_data.size(0))
    training_data, training_targets = training_data[training_permutation], training_targets[training_permutation]
    train_loader = DataLoader(TensorDataset(training_data, training_targets), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_targets), batch_size=len(test_data), shuffle=False)
    return train_loader, test_loader


def train(dataset: str, model: Union[SimpleNN, torch.nn.Module], loader: DataLoader, optimizer: Adam,
          compute_radii: bool = False, epochs=EPOCHS) -> List[Tuple[int, Dict[int, torch.Tensor]]]:
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    epoch_radii = []
    for epoch in range(epochs):
        model.train()
        for data, target in loader:
            inputs, labels = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()
        if dataset == 'CIFAR10':
            scheduler.step()
        # Do not compute the radii for the first 20 epochs, as those can be unstable. The number 20 was taken from
        # https://github.com/marco-gherardi/stragglers
        if compute_radii and epoch > 20:
            current_radii = model.radii(loader, set())
            epoch_radii.append((epoch, current_radii))
    return epoch_radii


def investigate_within_class_imbalance1(hard_data: Tensor, hard_target: Tensor, easy_data: Tensor, easy_target: Tensor,
                                        remove_hard: bool, sample_removal_rates: List[float], dataset_name: str,
                                        current_metrics: Dict[str, Dict[float, List]]):
    """ In this function we want to measure the effect of changing the number of easy/hard samples on the accuracy on
    the test set for distinct train:test ratio (where train:test ratio is passed as a parameter). The experiments are
    repeated multiple times to ensure that they are initialization-invariant.

    :param hard_data: identifies hard samples (data)
    :param hard_target: identified hard samples (target)
    :param easy_data: identified easy samples (data)
    :param easy_target: identified easy samples (target)
    :param remove_hard: flag indicating whether we want to see the effect of changing the number of easy (False) or
    hard (True) samples
    :param sample_removal_rates: list of ratios of easy/hard samples remaining in the train set (0.1 means that 90% of
    hard samples were removed from the train set before training, when reduce_hard == True)
    :param dataset_name: name of the dataset
    :param current_metrics: used to save accuracies, precision, recall and f1-score to the outer scope
    """
    generalisation_settings = ['full', 'hard', 'easy']
    for sample_removal_rate in tqdm(sample_removal_rates, desc='Sample removal rates'):
        train_loader, test_loaders = combine_and_split_data(hard_data, easy_data, hard_target, easy_target, remove_hard,
                                                            sample_removal_rate)
        # We train multiple times to make sure that the performance is initialization-invariant
        for _ in range(10):
            models, optimizers = initialize_models(dataset_name)
            train(dataset_name, models[0], train_loader, optimizers[0])
            print(f'Accuracies for {sample_removal_rate} % of {["easy", "hard"][remove_hard]} samples removed from '
                  f'training set.')
            # Evaluate the model on test set
            for i in range(3):
                accuracy = test(models[0], test_loaders[i])
                print(f'    {generalisation_settings[i]} - {accuracy}%')
                current_metrics[generalisation_settings[i]][sample_removal_rate].append(accuracy)
        print()


def investigate_within_class_imbalance2(data: Tensor, targets: Tensor, remove_hard: bool,
                                        sample_removal_rates: List[float], dataset_name: str,
                                        current_metrics: Dict[float, List]):
    """ In this function we want to measure the effect of changing the number of easy/hard samples on the accuracy on
    the test set for distinct train:test ratio (where train:test ratio is passed as a parameter). The experiments are
    repeated multiple times to ensure that they are initialization-invariant.
    :param data: data samples sorted by their average confidence (over models computed with compute_confidences.py)
    :param targets: labels corresponding to the sorted data samples
    :param remove_hard: flag indicating whether we want to see the effect of changing the number of easy (False) or
    hard (True) samples
    :param sample_removal_rates: list of ratios of easy/hard samples remaining in the train set (0.1 means that 90% of
    hard samples were removed from the train set before training, when reduce_hard == True)
    :param dataset_name: name of the dataset
    :param current_metrics: used to save accuracies, precision, recall and f1-score to the outer scope
    """
    for sample_removal_rate in tqdm(sample_removal_rates, desc='Sample removal rates'):
        train_loader, test_loader = split_data(data, targets, remove_hard, sample_removal_rate)
        # We train multiple times to make sure that the performance is initialization-invariant
        for _ in range(10):
            models, optimizers = initialize_models(dataset_name)
            train(dataset_name, models[0], train_loader, optimizers[0])
            print(f'Accuracies for {sample_removal_rate} % of {["easy", "hard"][remove_hard]} samples removed from '
                  f'training set.')
            # Evaluate the model on test set
            accuracy = test(models[0], test_loader)
            print(f'    Achieved {accuracy}% accuracy on the test set.')
            current_metrics[sample_removal_rate].append(accuracy)
        print()
