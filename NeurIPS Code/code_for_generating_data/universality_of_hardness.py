import pickle
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


def load_cifar10(batch_size=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainset, testset, trainloader, testloader


def get_confidence_scores(model, dataloader, device) -> List:
    model.eval()
    confidences = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs = probs.max(dim=1).values
            confidences.extend(top_probs.cpu().numpy())
    return confidences


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = [
        "resnet20", "resnet32", "resnet44", "resnet56", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
        "mobilenetv2_x0_5", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "mobilenetv2_x1_4",
        "shufflenetv2_x0_5", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "shufflenetv2_x2_0",
        "repvgg_a0", "repvgg_a1", "repvgg_a2"
    ]
    trainset, testset, trainloader, testloader = load_cifar10()

    model_confidences = {
        'train': [],
        'test': [],
        'combined': []
    }
    results = {
        'confidences': {},
        'images': {},
        'labels': {},
        'thresholds': {},
        'number_of_models': {},
        'overlaps': {}
    }

    for model_name in models:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{model_name}", pretrained=True)
        model.to(device)
        train_confidences = get_confidence_scores(model, trainloader, device)
        test_confidences = get_confidence_scores(model, testloader, device)
        combined_loader = DataLoader(trainset + testset, batch_size=1000, shuffle=False)
        combined_confidences = get_confidence_scores(model, combined_loader, device)

        # Convert confidences to numpy arrays
        model_confidences['train'].append(np.array(train_confidences))
        model_confidences['test'].append(np.array(test_confidences))
        model_confidences['combined'].append(np.array(combined_confidences))
    # Find the hardest samples and plot them
    for key in ['train', 'test', 'combined']:
        results['confidences'] = model_confidences
        # Average confidences across all models for each sample
        all_confidences = np.stack([confidences for confidences in model_confidences[key]])
        average_confidences = np.mean(all_confidences, axis=0)  # Calculate mean across models (column-wise)

        # Get indices of the lowest average confidences
        sorted_indices = np.argsort(average_confidences)  # This sorts from lowest to highest
        selected_samples = sorted_indices[:10]  # Select the indices of the 10 lowest average confidences

        # Collect the images and labels of the selected indices
        if key != 'combined':
            images = [trainset[i][0] if key == 'train' else testset[i][0] for i in selected_samples]
            labels = [trainset[i][1] if key == 'train' else testset[i][1] for i in selected_samples]
        else:
            # Correct handling for combined set
            images = [trainset[i][0] if i < len(trainset) else testset[i - len(trainset)][0] for i in selected_samples]
            labels = [trainset[i][1] if i < len(trainset) else testset[i - len(trainset)][1] for i in selected_samples]
        results['images'][key] = images
        results['labels'][key] = labels

    thresholds = range(1, 101, 1)
    number_of_models = range(3, 23, 4)  # Starting at 3 and increasing in steps of 4 up to 19
    overlaps = {'train': {n: [] for n in number_of_models},
                'test': {n: [] for n in number_of_models},
                'combined': {n: [] for n in number_of_models}}
    results['thresholds'] = thresholds
    results['number_of_models'] = number_of_models
    for pct in tqdm(thresholds):
        num_train = int(len(trainset) * pct / 100)
        num_test = int(len(testset) * pct / 100)
        num_combined = int((len(trainset) + len(testset)) * pct / 100)

        for key in ['train', 'test', 'combined']:
            for c_threshold in number_of_models:
                overlap_measurements = []  # List to store multiple overlap measurements
                for _ in range(100):
                    # Randomly choose c_threshold models
                    selected_model_indices = random.sample(range(len(model_confidences[key])), c_threshold)
                    low_conf_sets = []
                    # Collect lowest confidence indices from selected models
                    for model_index in selected_model_indices:
                        confidences = model_confidences[key][model_index]
                        indices = torch.topk(torch.tensor(confidences),
                                             num_train if key == 'train' else num_test if key == 'test'
                                             else num_combined, largest=False).indices
                        low_conf_sets.append(set(indices.numpy()))
                    # Calculate the intersection of all low confidence sets
                    common_indices = set.intersection(*low_conf_sets)
                    overlap_percentage = (len(common_indices) / (
                        num_train if key == 'train' else num_test if key == 'test' else num_combined)) * 100
                    overlap_measurements.append(overlap_percentage)
                # Store the mean and standard deviation of the overlap measurements
                overlaps[key][c_threshold].append((np.mean(overlap_measurements), np.std(overlap_measurements)))
    results['overlaps'] = overlaps
    with open('../Results/results1.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
