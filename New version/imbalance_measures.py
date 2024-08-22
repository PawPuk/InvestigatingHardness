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

    def random_oversampling(self, multiplication_factor: float) -> Subset:
        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * multiplication_factor)
        # Randomly oversample hard data until it matches the size of easy data
        indices = np.random.choice(len(self.hard_data), size=new_size, replace=True)
        # Return a new Subset
        return Subset(self.hard_data.dataset, indices)

    def random_undersampling(self, removal_ratio: float):
        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * removal_ratio)
        # Randomly undersample easy data to match the size of hard data
        indices = np.random.choice(len(self.easy_data), size=new_size, replace=False)
        # Return a new Subset
        return Subset(self.easy_data.dataset, indices)

    def SMOTE(self, multiplication_factor: float, k_neighbors: int = 5):
        """
        Apply SMOTE to balance the dataset by generating synthetic samples.

        :param multiplication_factor: The oversampling factor
        :param k_neighbors: The number of nearest neighbors to consider when generating synthetic samples.
        """
        # Extract features and labels from the hard dataset.
        hard_loader = DataLoader(self.hard_data, batch_size=len(self.hard_data))
        hard_features, hard_labels = next(iter(hard_loader))
        hard_features, hard_labels = hard_features.to(u.DEVICE), hard_labels.to(u.DEVICE)

        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * multiplication_factor)

        # Calculate target samples per class
        num_classes = torch.unique(hard_labels).size(0)
        target_samples_per_class = new_size // num_classes

        # Store synthetic samples and labels
        all_synthetic_samples = []
        all_synthetic_labels = []

        for class_label in torch.unique(hard_labels):
            # Select samples belonging to the current class
            class_mask = (hard_labels == class_label)
            class_features = hard_features[class_mask]
            class_size = len(class_features)
            if multiplication_factor > 0.0:
                nn = NearestNeighbors(n_neighbors=k_neighbors)
                nn.fit(class_features.cpu().numpy())
                num_synthetic_samples = target_samples_per_class - class_size
                synthetic_samples = []
                # Generate synthetic samples
                for i in range(num_synthetic_samples):
                    idx = np.random.randint(0, class_size)
                    neighbors = nn.kneighbors(class_features[idx].cpu().unsqueeze(0).numpy(), return_distance=False)
                    neighbor_idx = np.random.choice(neighbors[0][1:])  # Choose one neighbor randomly
                    neighbor = class_features[neighbor_idx]
                    # Interpolate between the chosen sample and its neighbor
                    diff = neighbor - class_features[idx]
                    synthetic_sample = class_features[idx] + torch.rand(1).to(u.DEVICE) * diff
                    synthetic_samples.append(synthetic_sample)
                # Stack synthetic samples and append them to the list
                synthetic_samples = torch.stack(synthetic_samples)
                synthetic_labels = torch.full((num_synthetic_samples,), class_label, dtype=torch.long).to(u.DEVICE)
                all_synthetic_samples.append(synthetic_samples)
                all_synthetic_labels.append(synthetic_labels)

        # Concatenate all synthetic samples and original data
        if all_synthetic_samples:
            synthetic_samples = torch.cat(all_synthetic_samples, dim=0)
            synthetic_labels = torch.cat(all_synthetic_labels, dim=0)
            augmented_features = torch.cat((hard_features, synthetic_samples), dim=0)
            augmented_labels = torch.cat((hard_labels, synthetic_labels), dim=0)
        else:
            augmented_features = hard_features
            augmented_labels = hard_labels

        return TensorDataset(augmented_features, augmented_labels)