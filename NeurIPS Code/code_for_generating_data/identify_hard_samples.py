import argparse
import pickle
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
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


def load_CIFAR10() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and normalize the CIFAR-10 dataset."""
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load the CIFAR10 training and test datasets with the specified transforms
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    combined_dataset = ConcatDataset([train_dataset, test_dataset])

    # Concatenate train and test datasets if needed or handle separately
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, combined_loader


def initialize_model() -> Tuple[ResNet, Adam]:
    model = ResNet(BasicBlock, [3, 3, 3, 3]).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    return model, optimizer


def train(model: ResNet, loader: DataLoader, optimizer: Adam):
    for epoch in range(100):
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


def compute_confidences(model, loader):
    """Compute model confidences on the combined CIFAR10 dataset."""
    model.eval()
    confidences = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences.extend(probabilities.cpu().numpy())  # Store probabilities
    return confidences


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


def main(runs: int):
    train_loader, test_loader, combined_loader = load_CIFAR10()
    all_confidences = []
    for _ in tqdm(range(runs)):
        model, optimizer = initialize_model()
        train(model, train_loader, optimizer)
        confidences = compute_confidences(model, combined_loader)
        all_confidences.append(confidences)
        # Evaluate the model on test set
        current_metrics = test(model, test_loader)
        print(current_metrics['accuracy'])
    save_data(all_confidences, f"../Results/{runs}_metrics.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--runs', type=int, default=10,
                        help='')
    args = parser.parse_args()
    main(**vars(args))
