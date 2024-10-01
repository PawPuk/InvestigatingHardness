import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

from utils import initialize_models, load_data_and_normalize, test, train

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset, test_dataset = load_data_and_normalize('MNIST')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model, optimizer = initialize_models('MNIST', 'complex')

# Train for 5 epochs
for epoch in range(1, 6):
    train_loss, train_acc = train('MNIST', model, train_loader, optimizer, 1)
    test_loss, test_acc = test(model, test_loader)

    print(f'Epoch {epoch}:')
    print(f'Train Loss: {train_loss[0]:.4f}, Train Accuracy: {train_acc[0]:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')


