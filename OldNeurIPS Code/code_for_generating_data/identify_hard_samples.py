import argparse
import pickle
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchvision import datasets, transforms
from tqdm import tqdm

from neural_network import BasicBlock, ResNet, OldSimpleNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()
EPOCHS = 100
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_dataset(flag: str) -> DataLoader:
    """Load either CIFAR10 or MNIST dataset based on the 'flag'."""
    if flag == 'CIFAR':
        dataset_class = datasets.CIFAR10
    elif flag == 'MNIST':
        dataset_class = datasets.MNIST
    else:
        raise ValueError("Unsupported dataset flag!")

    # Load the datasets
    train_dataset = dataset_class(root='./data', train=True, download=True,
                                  transform=transforms.ToTensor())
    test_dataset = dataset_class(root='./data', train=False, download=True,
                                 transform=transforms.ToTensor())

    # Concatenate train and test datasets
    full_data = torch.cat([train_dataset.data.unsqueeze(1).float(), test_dataset.data.unsqueeze(1).float()])
    full_targets = torch.cat([train_dataset.targets, test_dataset.targets])

    """ # Shuffle the combined dataset
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(full_data))
    full_data, full_targets = full_data[torch.tensor(shuffled_indices)], full_targets[torch.tensor(shuffled_indices)]"""

    # Normalize the data
    data_mean = full_data.mean(dim=(0, 2, 3)) / 255.0
    # data_std = full_data.std(dim=(0, 2, 3)) / 255.0
    data_std = torch.sqrt(torch.var(full_data, dim=(0, 2, 3)) / 255.0 ** 2 + 0.0000001)
    normalize_transform = transforms.Normalize(mean=data_mean.tolist(), std=data_std.tolist())
    full_data = normalize_transform(full_data / 255.0)  # Ensure scaling to [0, 1] before normalization
    return DataLoader(TensorDataset(full_data, full_targets), batch_size=len(full_data), shuffle=False)


def initialize_model(dataset_name: str) -> Tuple[torch.nn.Module, Adam]:
    if dataset_name == 'CIFAR':
        model = ResNet(BasicBlock, [3, 3, 3, 3]).to(DEVICE)
    else:
        model = OldSimpleNN(20, 2).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    return model, optimizer


def train(dataset: str, model: torch.nn.Module, loader: DataLoader, optimizer: Adam):
    for epoch in tqdm(range(EPOCHS), desc='Epochs'):
        if dataset == 'CIFAR':
            # Adjust the learning rate
            lr = 0.001 * (0.1 ** (epoch // 30))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # Proceed with training
        model.train()
        for data, target in loader:
            inputs, labels = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()


def compute_confidences(model: torch.nn.Module, loader: DataLoader):
    """Compute model confidences on the combined CIFAR10 dataset."""
    confidences = []
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            print(data.shape)
            data = data.to(DEVICE)
            outputs = model(data)
            confidence = outputs.max(1)[0].cpu().numpy()
            confidences.extend(list(zip(range(data.shape[0]), confidence)))
            # probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # confidences.extend(probabilities.cpu().numpy())  # Store probabilities
    return confidences


def test(model: torch.nn.Module, loader: DataLoader) -> dict[str, float]:
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


def save_data(data, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def main(dataset: str, runs: int):
    full_loader = load_dataset(dataset)
    all_confidences = []
    for _ in tqdm(range(runs)):
        model, optimizer = initialize_model(dataset)
        train(dataset, model, full_loader, optimizer)
        confidences = compute_confidences(model, full_loader)
        all_confidences.append(confidences)
        current_metrics = test(model, full_loader)
        print(current_metrics['accuracy'])
    save_data(all_confidences, f"../Results/{dataset}_{runs}_metrics.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--runs', type=int, default=1,
                        help='')
    args = parser.parse_args()
    main(**vars(args))
