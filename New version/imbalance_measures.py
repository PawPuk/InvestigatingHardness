import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import utils as u

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class ImbalanceMeasures:
    def __init__(self, easy_dataset: Subset, hard_dataset: Subset):
        self.easy_data = easy_dataset
        self.hard_data = hard_dataset

    def random_oversampling(self, multiplication_factor) -> Subset:
        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * multiplication_factor)
        # Randomly oversample hard data until it matches the size of easy data
        indices = np.random.choice(len(self.hard_data), size=new_size, replace=True)
        # Return a new Subset
        return Subset(self.hard_data.dataset, indices)

    def random_undersampling(self, removal_ratio):
        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * removal_ratio)
        # Randomly undersample easy data to match the size of hard data
        indices = np.random.choice(len(self.easy_data), size=new_size, replace=False)
        # Return a new Subset
        return Subset(self.easy_data.dataset, indices)

    # TODO: verify correctness of SMOTE!!!
    def SMOTE(self, k_neighbors=5):
        # Extract the features (assuming they are tensors) from the hard dataset
        hard_loader = DataLoader(self.hard_data, batch_size=len(self.hard_data))
        hard_features, hard_labels = next(iter(hard_loader))
        hard_features, hard_labels = hard_features.to(u.DEVICE), hard_labels.to(u.DEVICE)

        # Fit nearest neighbors model to the hard data
        nn = NearestNeighbors(n_neighbors=k_neighbors)
        nn.fit(hard_features.cpu().numpy())

        # For each point, generate synthetic samples
        synthetic_samples = []
        for i in range(len(hard_features)):
            neighbors = nn.kneighbors(hard_features[i].cpu().unsqueeze(0).numpy(), return_distance=False)
            neighbor_idx = np.random.choice(neighbors[0][1:])  # Choose one neighbor randomly
            neighbor = hard_features[neighbor_idx]

            # Interpolate to create a new synthetic sample
            diff = neighbor - hard_features[i]
            synthetic_sample = hard_features[i] + torch.rand(1).to(u.DEVICE) * diff
            synthetic_samples.append(synthetic_sample)

        synthetic_samples = torch.stack(synthetic_samples)

        # Concatenate original hard data with synthetic data
        augmented_features = torch.cat((hard_features, synthetic_samples), dim=0)
        augmented_labels = torch.cat((hard_labels, hard_labels[:len(synthetic_samples)]), dim=0)

        # Convert to TensorDataset and return
        return TensorDataset(augmented_features, augmented_labels)
