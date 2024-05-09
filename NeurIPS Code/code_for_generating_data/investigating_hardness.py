import argparse
import pickle
from typing import List, Set, Tuple

import numpy as np
from numpy import array
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torchvision
from torchvision import transforms
from tqdm import tqdm

from neural_network import BasicBlock, ResNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()
EPOCHS = 100
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def identify_hard_samples(confidences: List[array], threshold: float) -> Tuple[Set[int], Set[int]]:
    """Divide the CIFAR10 dataset into the easy and hard ('threshold' percent of the dataset) samples."""
    all_confidences = np.stack(confidences)
    average_confidences = np.mean(all_confidences, axis=0)
    sorted_indices = np.argsort(average_confidences)  # This sorts from lowest to highest
    hard_sample_indices = set(sorted_indices[:int(threshold*len(confidences[0]))])
    easy_sample_indices = set([x for x in range(len(confidences[0]))]).difference(hard_sample_indices)
    return easy_sample_indices, hard_sample_indices


def load_CIFAR10() -> TensorDataset:
    """Load the CIFAR10 dataset and output the TensorDataset."""
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                transform=transforms.ToTensor())
    # Concatenate train and test datasets
    full_data = torch.cat([torch.tensor(train_dataset.data).unsqueeze(1).float(),
                           torch.tensor(test_dataset.data).unsqueeze(1).float()])
    full_targets = torch.cat([torch.tensor(train_dataset.targets), torch.tensor(test_dataset.targets)])
    # Shuffle the combined dataset
    shuffled_indices = torch.randperm(len(full_data))
    full_data, full_targets = full_data[shuffled_indices], full_targets[shuffled_indices]
    # Calculate mean and variance for the subset
    data_mean = full_data.mean(dim=(0, 2, 3))
    data_std = full_data.std(dim=(0, 2, 3))
    normalize_transform = transforms.Normalize(mean=data_mean.tolist(), std=data_std.tolist())
    full_data = normalize_transform(full_data)
    return TensorDataset(full_data, full_targets)


def split_data(dataset: Dataset, easy_samples: Set[int], hard_samples: Set[int], sample_removal_rate: float,
               remove_hard: bool) -> Tuple[DataLoader, List[DataLoader]]:
    """Use the indices from 'easy_samples' and 'hard_samples' to divide the dataset into training and test set. Before
    creating a DataLoader remove sample_removal_rate hard/easy (depending on the 'remove_hard' flag) samples from the
    training set. Note that for test set we give 3 DataLoaders - one for easy samples, one for hard, and one for all."""
    # Randomly shuffle hard and easy samples
    hard_perm, easy_perm = torch.randperm(torch.tensor(list(hard_samples)).size(0)), \
        torch.randperm(torch.tensor(list(easy_samples)).size(0))
    hard_data, hard_target = dataset[:][0][hard_perm].squeeze(1), dataset[:][1][hard_perm]
    easy_data, easy_target = dataset[:][0][easy_perm].squeeze(1), dataset[:][1][easy_perm]
    # Split hard and easy samples into train and test sets (use 80:20 training:test ratio)
    train_size_hard, train_size_easy = int(len(hard_data) * 0.8), int(len(easy_data) * 0.8)
    hard_train_data, hard_test_data = hard_data[:train_size_hard], hard_data[train_size_hard:]
    hard_train_target, hard_test_target = hard_target[:train_size_hard], hard_target[train_size_hard:]
    easy_train_data, easy_test_data = easy_data[:train_size_easy], easy_data[train_size_easy:]
    easy_train_target, easy_test_target = easy_target[:train_size_easy], easy_target[train_size_easy:]
    # Reduce the number of train samples by sample_removal_rate (and check if it's valid)
    if not 0 <= sample_removal_rate <= 1:
        raise ValueError(f'The parameter remaining_train_ratio must be in [0, 1]; {sample_removal_rate} not allowed.')
    if remove_hard:
        reduced_hard_train_size = int(train_size_hard * (1 - sample_removal_rate))
        reduced_easy_train_size = train_size_easy
    else:
        reduced_hard_train_size = train_size_hard
        reduced_easy_train_size = int(train_size_easy * (1 - sample_removal_rate))
    # Combine easy and hard samples into train and test data
    train_data = torch.cat((hard_train_data[:reduced_hard_train_size],
                            easy_train_data[:reduced_easy_train_size]), dim=0)
    train_targets = torch.cat((hard_train_target[:reduced_hard_train_size],
                               easy_train_target[:reduced_easy_train_size]), dim=0)
    # Shuffle the final train dataset
    train_permutation = torch.randperm(train_data.size(0))
    train_data, train_targets = train_data[train_permutation], train_targets[train_permutation]
    train_data = train_data.permute(0, 3, 1, 2)
    train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=128)
    # Create two test sets - one containing only hard samples, and the other only easy samples
    hard_test_data, easy_test_data = hard_test_data.permute(0, 3, 1, 2), easy_test_data.permute(0, 3, 1, 2)
    hard_and_easy_test_sets = [(hard_test_data, hard_test_target), (easy_test_data, easy_test_target)]
    full_test_data = torch.cat((hard_and_easy_test_sets[0][0], hard_and_easy_test_sets[1][0]), dim=0)
    full_test_targets = torch.cat((hard_and_easy_test_sets[0][1], hard_and_easy_test_sets[1][1]), dim=0)
    # Create 3 test loaders: 1) with all data samples; 2) with only hard data samples; 3) with only easy data samples
    test_loaders = []
    for data, target in [(full_test_data, full_test_targets)] + hard_and_easy_test_sets:
        test_loader = DataLoader(TensorDataset(data, target), batch_size=1000, shuffle=False)
        test_loaders.append(test_loader)
    return train_loader, test_loaders


def initialize_model() -> Tuple[ResNet, Adam]:
    model = ResNet(BasicBlock, [3, 3, 3, 3]).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    return model, optimizer


def train(model: ResNet, loader: DataLoader, optimizer: Adam):
    for epoch in tqdm(range(100)):
        # Adjust the learning rate
        lr = 0.001 * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Proceed with training
        model.train()
        running_loss = 0.0
        for i, data in enumerate(loader):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


def test(model: ResNet, loader: DataLoader) -> dict[str, float]:
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


def save_data(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def main(threshold: float, sample_removal_rates: List[float], remove_hard: bool, runs: int):
    results = load_results('../Results/results1.pkl')
    easy_samples, hard_samples = identify_hard_samples(results['confidences']['combined'], threshold)
    dataset = load_CIFAR10()
    metrics = {setting: {sample_removal_rate: {metric: []
                                               for metric in ['accuracy', 'precision', 'recall', 'f1']}
                         for sample_removal_rate in sample_removal_rates}
               for setting in ['full', 'hard', 'easy']}
    for sample_removal_rate in sample_removal_rates:
        train_loader, test_loaders = split_data(dataset, easy_samples, hard_samples, sample_removal_rate, remove_hard)
        for _ in range(runs):
            model, optimizer = initialize_model()
            train(model, train_loader, optimizer)
            # Evaluate the model on test set
            for i, setting in enumerate(['full', 'hard', 'easy']):
                current_metrics = test(model, test_loaders[i])
                # Save the obtained metrics
                for metric_name, metric_values in current_metrics.items():
                    metrics[setting][sample_removal_rate][metric_name].append(metric_values)
    print(metrics)
    save_data(metrics, f"../Results/{remove_hard}_{threshold}_metrics.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='')
    parser.add_argument('--sample_removal_rates', nargs='+', type=float,
                        default=[0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0],
                        help='')
    parser.add_argument('--remove_hard', action='store_true', default=False,
                        help='Flag indicating whether we want to see the effect of changing the number of easy (False) '
                             'or hard (True) samples in the training set on the generalization.')
    parser.add_argument('--runs', type=int, default=3,
                        help='')
    args = parser.parse_args()
    main(**vars(args))
