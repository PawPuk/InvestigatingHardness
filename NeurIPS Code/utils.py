from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchvision import datasets, transforms

from neural_networks import BasicBlock, SimpleNN, ResNet

EPSILON = 0.000000001  # cutoff for the computation of the variance in the standardisation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500
CRITERION = torch.nn.CrossEntropyLoss()
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


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
    # Concatenate train and test datasets
    full_data = torch.cat([train_dataset.data.unsqueeze(1).float(), test_dataset.data.unsqueeze(1).float()])
    full_targets = torch.cat([train_dataset.targets, test_dataset.targets])
    # Shuffle the combined dataset
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(full_data))
    full_data, full_targets = full_data[torch.tensor(shuffled_indices)], full_targets[torch.tensor(shuffled_indices)]
    # Select a subset based on the 'subset_size' parameter
    subset_data = full_data[:subset_size]
    subset_targets = full_targets[:subset_size]
    # Calculate mean and variance for the subset
    data_means = torch.mean(subset_data, dim=(0, 2, 3)) / 255.0
    data_vars = torch.sqrt(torch.var(subset_data, dim=(0, 2, 3)) / 255.0 ** 2 + EPSILON)
    # Apply the calculated normalization to the subset
    normalize_transform = transforms.Normalize(mean=data_means, std=data_vars)
    normalized_subset_data = normalize_transform(subset_data / 255.0)
    # Create a TensorDataset from the normalized subset. This will make the code significantly faster than passing the
    # normalization transform to the DataLoader (as it's usually done).
    normalized_subset = TensorDataset(normalized_subset_data, subset_targets)
    return normalized_subset


def transform_datasets_to_dataloaders(dataset: TensorDataset) -> DataLoader:
    """ Transforms TensorDataset to DataLoader for bull-batch training. The below implementation makes full-batch
    training faster than it would usually be.

    :param dataset: TensorDataset to be transformed
    :return: DataLoader version of dataset ready for full-batch training
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data, target in loader:
        loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
    return loader


def initialize_model(latent: int = 1, evaluation_network: str = 'SimpleNN',
                     dataset: str = 'MNIST') -> Tuple[SimpleNN, SGD]:
    """ Used to initialize the model and optimizer.

    :param latent: the index of the hidden layer used to extract the latent representation for radii computations
    :param evaluation_network: specifies the type of network to initialize
    :param dataset: specifies which dataset will the model be trained on (needed to set input & output layer)
    :return: initialized SimpleNN model and SGD optimizer
    """
    if evaluation_network == 'SimpleNN':
        if dataset in ['MNIST', 'FashionMNIST', 'KMNIST']:
            model = SimpleNN(28 * 28, 2, 20, latent)
        else:
            model = SimpleNN(3*32*32, 8, 30, latent)
        optimizer = SGD(model.parameters(), lr=0.1)
    else:
        if dataset in ['MNIST', 'FashionMNIST', 'KMNIST']:
            model = ResNet(img_channels=1, num_layers=18, block=BasicBlock, num_classes=10)
        else:
            model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    model.to(DEVICE)
    return model, optimizer


def train_model(model: Union[SimpleNN, ResNet], loader: DataLoader, optimizer: Union[SGD, Adam],
                compute_radii: bool = True) -> List[Tuple[int, Dict[int, torch.Tensor]]]:
    """

    :param model: model to be trained
    :param loader: DataLoader to be used for training
    :param optimizer: optimizer to be used for training
    :param compute_radii: flag specifying if the user wants to compute the radii of class manifolds during training
    :return: list of tuples of the form (epoch_index, radii_of_class_manifolds), where the radii are stored in a
    dictionary of the form {class_index: [torch.Tensor]}
    """
    epoch_radii = []
    for epoch in range(EPOCHS):
        model.train()
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = CRITERION(output, target)
            loss.backward()
            optimizer.step()
        # Do not compute the radii for the first 20 epochs, as those can be unstable. The number 20 was taken from
        # https://github.com/marco-gherardi/stragglers
        if compute_radii and epoch > 20:
            current_radii = model.radii(loader, set())
            epoch_radii.append((epoch, current_radii))
    return epoch_radii


def load_results(filename):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)


def identify_hard_samples(strategy: str, dataset: TensorDataset, level: str, noise_ratio: float) -> List[Tensor]:
    """ This function divides 'loader' (or 'dataset', depending on the used 'strategy') into hard and easy samples.

    :param strategy: specifies the strategy used for identifying hard samples; only 'stragglers', 'confidence' and
    'energy' allowed
    :param dataset: TensorDataset that contains the data to be divided into easy and hard samples
    :param level: specifies the level at which the energy is computed and how the hard samples are chosen; only
    'dataset' and 'class' allowed
    :param noise_ratio: used when adding label noise to the dataset. Make sure that noise_ratio is in range of [0, 1)
    :return: list containing the identified hard and easy samples with the indices of hard samples
    """

    """confidences = []
    model, optimizer = initialize_model()
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data, target in loader:  # This is done to increase the speed (works due to full-batch setting)
        loader = DataLoader(TensorDataset(data, target), batch_size=len(data), shuffle=False)
    train_model(model, loader, optimizer, False)
    model.eval()
    # Iterate through the data samples in 'loader'; compute and save their confidence/energy (depending on 'strategy')
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            confidence = output.max(1)[0].cpu().numpy()
            confidences.extend(list(zip(range(len(dataset)), confidence)))"""

    confidences = load_results(f'MNIST_1_metrics.pkl')[0]
    # Corrected list comprehension to directly unpack index and confidence
    confidence_indices = [(confidence, index) for index, confidence in confidences]

    # Sort the list by confidence in ascending order to find the most uncertain samples
    sorted_confidence_indices = sorted(confidence_indices, key=lambda x: x[0], reverse=False)

    # Select the indices of the hard and easy samples based on the threshold
    num_hard_samples = int(0.05 * len(confidences))
    hard_sample_indices = list({index for _, index in sorted_confidence_indices[:num_hard_samples]})
    easy_sample_indices = list({index for _, index in sorted_confidence_indices[num_hard_samples:]})
    hard_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    hard_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    easy_data = torch.tensor([], dtype=torch.float32).to(DEVICE)
    easy_target = torch.tensor([], dtype=torch.long).to(DEVICE)
    loader = transform_datasets_to_dataloaders(dataset)
    for data, target in loader:
        hard_data = torch.cat((hard_data, data[hard_sample_indices]), dim=0)
        hard_target = torch.cat((hard_target, target[hard_sample_indices]), dim=0)
        easy_data = torch.cat((easy_data, data[easy_sample_indices]), dim=0)
        easy_target = torch.cat((easy_target, target[easy_sample_indices]), dim=0)

    return [hard_data, hard_target, easy_data, easy_target]


def create_dataloaders_with_straggler_ratio(hard_data: Tensor, easy_data: Tensor, hard_target: Tensor,
                                            easy_target: Tensor, reduce_hard: bool,
                                            remaining_train_ratio: float) -> Tuple[DataLoader, List[DataLoader]]:
    """ This function divides easy and hard data samples into train and test sets.

    :param hard_data: identifies hard samples (data)
    :param hard_target: identified hard samples (target)
    :param easy_data: identified easy samples (data)
    :param easy_target: identified easy samples (target)
    :param train_ratio: percentage of train set to whole dataset
    :param reduce_hard: flag indicating whether we want to see the effect of changing the number of easy (False) or
    hard (True) samples
    :param remaining_train_ratio: ratio of easy/hard samples remaining in the train set (0.1 means that 90% of hard
    samples will be removed from the pool of hard samples when generating train set, when reduce_hard == True)
    :return: returns train loader and 3 test loaders - 1) with all data samples; 2) with only hard data samples; and 3)
    with only easy data samples.
    """
    # Randomly shuffle hard and easy samples
    hard_perm, easy_perm = torch.randperm(hard_data.size(0)), torch.randperm(easy_data.size(0))
    hard_data, hard_target = hard_data[hard_perm], hard_target[hard_perm]
    easy_data, easy_target = easy_data[easy_perm], easy_target[easy_perm]
    # Split data into initial train/test sets based on the train_ratio and make sure that train_ratio is correct
    train_size_hard = int(len(hard_data) * 0.8)
    train_size_easy = int(len(easy_data) * 0.8)
    hard_train_data, hard_test_data = hard_data[:train_size_hard], hard_data[train_size_hard:]
    hard_train_target, hard_test_target = hard_target[:train_size_hard], hard_target[train_size_hard:]
    easy_train_data, easy_test_data = easy_data[:train_size_easy], easy_data[train_size_easy:]
    easy_train_target, easy_test_target = easy_target[:train_size_easy], easy_target[train_size_easy:]
    # Reduce the number of train samples by remaining_train_ratio
    if not 0 <= remaining_train_ratio <= 1:
        raise ValueError(f'The parameter remaining_train_ratio must be in [0, 1]; {remaining_train_ratio} not allowed.')
    if reduce_hard:
        reduced_hard_train_size = int(train_size_hard * remaining_train_ratio)
        reduced_easy_train_size = train_size_easy
    else:
        reduced_hard_train_size = train_size_hard
        reduced_easy_train_size = int(train_size_easy * remaining_train_ratio)
    # Combine easy and hard samples into train and test data
    train_data = torch.cat((hard_train_data[:reduced_hard_train_size],
                            easy_train_data[:reduced_easy_train_size]), dim=0)
    train_targets = torch.cat((hard_train_target[:reduced_hard_train_size],
                               easy_train_target[:reduced_easy_train_size]), dim=0)
    # Shuffle the final train dataset
    train_permutation = torch.randperm(train_data.size(0))
    train_data, train_targets = train_data[train_permutation], train_targets[train_permutation]
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=len(train_data))
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


def test(model: SimpleNN, loader: DataLoader) -> dict[str, float]:
    """ Measures the accuracy of the 'model' on the test set.

    :param model: model, which performance we want to evaluate
    :param loader: DataLoader containing test data
    :return: accuracy on the test set rounded to 2 decimal places
    """
    model.eval()
    num_classes = 10
    # Initialize metrics
    accuracy = Accuracy(task="multiclass", num_classes=10).to(DEVICE)
    precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(DEVICE)
    recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(DEVICE)
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(DEVICE)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            # Update metrics
            accuracy.update(outputs, target)
            precision.update(outputs, target)
            recall.update(outputs, target)
            f1_score.update(outputs, target)
    # Compute final results
    accuracy_result = round(accuracy.compute().item() * 100, 2)
    precision_result = round(precision.compute().item() * 100, 2)
    recall_result = round(recall.compute().item() * 100, 2)
    f1_result = round(f1_score.compute().item() * 100, 2)
    return {'accuracy': accuracy_result, 'precision': precision_result, 'recall': recall_result, 'f1': f1_result}


def straggler_ratio_vs_generalisation(hard_data: Tensor, hard_target: Tensor, easy_data: Tensor, easy_target: Tensor,
                                      reduce_hard: bool, remaining_train_ratios: List[float],
                                      current_metrics: Dict[str, Dict[float, Dict[str, List]]],
                                      evaluation_network: str):
    """ In this function we want to measure the effect of changing the number of easy/hard samples on the accuracy on
    the test set for distinct train:test ratio (where train:test ratio is passed as a parameter). The experiments are
    repeated multiple times to ensure that they are initialization-invariant.

    :param hard_data: identifies hard samples (data)
    :param hard_target: identified hard samples (target)
    :param easy_data: identified easy samples (data)
    :param easy_target: identified easy samples (target)
    :param train_ratio: percentage of train set to whole dataset
    :param reduce_hard: flag indicating whether we want to see the effect of changing the number of easy (False) or
    hard (True) samples
    :param remaining_train_ratios: list of ratios of easy/hard samples remaining in the train set (0.1 means that 90% of
    hard samples were removed from the train set before training, when reduce_hard == True)
    :param current_metrics: used to save accuracies, precision, recall and f1-score to the outer scope
    :param evaluation_network: this network will be used to measure the performance on hard/easy data
    """
    generalisation_settings = ['full', 'hard', 'easy']
    for remaining_train_ratio in remaining_train_ratios:
        metrics_for_ratio = {metric: [[], [], []] for metric in ['accuracy', 'precision', 'recall', 'f1']}
        train_loader, test_loaders = create_dataloaders_with_straggler_ratio(hard_data, easy_data, hard_target,
                                                                             easy_target,
                                                                             reduce_hard, remaining_train_ratio)
        # We train multiple times to make sure that the performance is initialization-invariant
        for _ in range(1):
            model, optimizer = initialize_model(evaluation_network=evaluation_network)
            train_model(model, train_loader, optimizer, False)
            # Evaluate the model on test set
            for i in range(3):
                metrics = test(model, test_loaders[i])
                print(metrics['accuracy'])
                for metric_name, metric_values in metrics.items():
                    metrics_for_ratio[metric_name][i].append(metric_values)
        # Save the accuracies to the outer scope (outside of this function)
        for i, setting in enumerate(generalisation_settings):
            for metric_name in metrics_for_ratio:
                if setting not in current_metrics:
                    current_metrics[setting] = {}
                if remaining_train_ratio not in current_metrics[setting]:
                    current_metrics[setting][remaining_train_ratio] = {metric: [] for metric in metrics_for_ratio}
                current_metrics[setting][remaining_train_ratio][metric_name].extend(metrics_for_ratio[metric_name][i])


def plot_radii(all_radii: List[List[Tuple[int, Dict[int, torch.Tensor]]]], dataset_name: str, save: bool = False):
    """ This function plots the radii of class manifolds (generates Figure 2 from our paper).

    :param all_radii: list containing 10 lists - each showing the development of the radii of class manifolds during
    training
    :param dataset_name: used when saving to indicate on which dataset the results come from
    :param save: flag indicating whether to save or not
    """
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(all_radii)))  # Darker to lighter blues
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for run_index in range(len(all_radii)):
        radii = all_radii[run_index]
        for i, ax in enumerate(axes.flatten()):
            y = [radii[j][1][i].cpu() for j in range(len(radii))]
            x = [radii[j][0] for j in range(len(radii))]
            ax.plot(x, y, color=colors[run_index], linewidth=3)
            if run_index == 0:
                ax.set_title(f'Class {i} Radii Over Epoch')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Radius')
                ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(f'Figures/radii_on_{dataset_name}.png')
        plt.savefig(f'Figures/radii_on_{dataset_name}.pdf')


def plot_generalisation(train_ratios: list[float], remaining_train_ratios: list[float], reduce_hard: bool,
                        avg_accuracies: Dict[str, Dict[float, List[float]]], strategy: str, level: str,
                        std_accuracies: Dict[str, Dict[float, List[float]]], noise_ratio: float, dataset_name: str):
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(train_ratios)))
    for setting in ['full', 'hard', 'easy']:
        for idx in range(len(train_ratios)):
            ratio = train_ratios[idx]
            plt.errorbar(remaining_train_ratios, avg_accuracies[setting][ratio], marker='o', markersize=5,
                         yerr=std_accuracies[setting][ratio], capsize=5, linewidth=2, color=colors[idx],
                         label=f'Training:Test={int(100 * ratio)}:{100 - int(100 * ratio)}')
        plt.xlabel(f'Proportion of {["Easy", "Hard"][reduce_hard]} Samples Remaining in Training Set', fontsize=14)
        plt.ylabel(f'Accuracy on {setting.capitalize()} Test Set (%)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.legend(title='Training:Test Ratio')
        plt.tight_layout()
        # Generate a title for the figure
        s = ''
        if strategy != 'stragglers':
            s = f'{level}_'
        plt.savefig(
            f'Figures/generalisation_from_{["easy", "hard"][reduce_hard]}_to_{setting}_on_{dataset_name}_using_{s}'
            f'{strategy}_{noise_ratio}noise.png')
        plt.savefig(
            f'Figures/generalisation_from_{["easy", "hard"][reduce_hard]}_to_{setting}_on_{dataset_name}_using_{s}'
            f'{strategy}_{noise_ratio}noise.pdf')
        plt.clf()
