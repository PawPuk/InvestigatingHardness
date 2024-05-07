from typing import List

import matplotlib.pyplot as plt
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


def plot_samples(data, labels, title):
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axes):
        ax.imshow(data[i].permute(1, 2, 0) * torch.tensor([0.247, 0.243, 0.261]) +
                  torch.tensor([0.4914, 0.4822, 0.4465]))
        ax.title.set_text(f'{class_names[labels[i]]}')  # Use class names instead of label numbers
        ax.axis('off')
    plt.savefig(f'{title}.png')
    plt.savefig(f'{title}.pdf')
    plt.show()


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
        num_samples = int(len(trainset) * 0.03) if key == 'train' else int(len(testset) * 0.03) if key == 'test' \
            else int((len(trainset) + len(testset)) * 0.03)
        low_conf_sets = [set(torch.topk(torch.tensor(confidences), num_samples, largest=False).indices.numpy()) for
                         confidences in model_confidences[key]]
        common_indices = list(set.intersection(*low_conf_sets))
        common_indices_np = np.array(common_indices)
        # Calculate average confidences for common indices
        average_confidences = np.mean([model_confidences[key][i][common_indices_np] for i in range(len(models))],
                                      axis=0)
        # Get indices of the lowest average confidences
        sorted_indices = np.argsort(average_confidences)  # This sorts from lowest to highest
        selected_samples = sorted_indices[:10]  # Select the indices of the 10 lowest average confidences
        selected_indices = [common_indices[i] for i in selected_samples]  # Map back to the original dataset indices

        # Collect the images and labels of the selected indices
        if key != 'combined':
            images = [trainset[i][0] if key == 'train' else testset[i][0] for i in selected_indices]
            labels = [trainset[i][1] if key == 'train' else testset[i][1] for i in selected_indices]
        else:
            # Correct handling for combined set
            images = [trainset[i][0] if i < len(trainset) else testset[i - len(trainset)][0] for i in selected_indices]
            labels = [trainset[i][1] if i < len(trainset) else testset[i - len(trainset)][1] for i in selected_indices]
        plot_samples(images, labels, f'{key}_low_confidence')

    thresholds = range(1, 101, 1)
    overlaps = {'train': [], 'test': [], 'combined': []}

    for pct in tqdm(thresholds):
        num_train = int(len(trainset) * pct / 100)
        num_test = int(len(testset) * pct / 100)
        num_combined = int((len(trainset) + len(testset)) * pct / 100)

        for key in ['train', 'test', 'combined']:
            low_conf_sets = []
            for confidences in model_confidences[key]:
                indices = torch.topk(torch.tensor(confidences),
                                     num_train if key == 'train' else num_test if key == 'test' else num_combined,
                                     largest=False).indices
                low_conf_sets.append(set(indices.numpy()))
            overlap = len(set.intersection(*low_conf_sets)) / (
                num_train if key == 'train' else num_test if key == 'test' else num_combined) * 100
            overlaps[key].append(overlap)

    plt.figure(figsize=(12, 8))
    for i, key in enumerate(['train', 'test', 'combined'], 1):
        plt.subplot(1, 3, i)
        plt.plot(thresholds, overlaps[key])
        plt.title(f'{key.capitalize()} Set Overlap')
        plt.xlabel('Threshold (%)')
        plt.ylabel('Overlap (%)')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('universality_of_hardness.pdf')
    plt.savefig('universality_of_hardness.png')
    plt.show()


if __name__ == "__main__":
    main()
