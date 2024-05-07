import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def get_confidence_scores(model, dataloader, device):
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

    for model_name in models:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{model_name}", pretrained=True)
        model.to(device)
        model_confidences['train'].append(get_confidence_scores(model, trainloader, device))
        model_confidences['test'].append(get_confidence_scores(model, testloader, device))
        combined_loader = DataLoader(trainset + testset, batch_size=128, shuffle=False)
        model_confidences['combined'].append(get_confidence_scores(model, combined_loader, device))

    thresholds = range(1, 101, 1)  # from 2% to 20% in steps of 2%
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
        plt.plot(thresholds, overlaps[key], marker='o')
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
