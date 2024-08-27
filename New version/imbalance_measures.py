from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import TensorDataset

from neural_networks import LeNet
import utils as u

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class ImbalanceMeasures:
    def __init__(self, easy_dataset: TensorDataset, hard_dataset: TensorDataset, dataset_name: str, models_count: int,
                 threshold: float):
        self.easy_data = easy_dataset
        self.hard_data = hard_dataset
        self.dataset_name = dataset_name
        self.models_count = models_count
        self.threshold = threshold

    def random_oversampling(self, multiplication_factor: float) -> TensorDataset:
        hard_features, hard_labels = self.hard_data.tensors
        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * multiplication_factor)
        # Randomly oversample hard data
        indices = np.random.choice(len(hard_features), size=new_size, replace=True)
        return TensorDataset(hard_features[indices], hard_labels[indices])

    def random_undersampling(self, removal_ratio: float) -> TensorDataset:
        easy_features, easy_labels = self.easy_data.tensors
        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * removal_ratio)
        # Randomly undersample easy data
        indices = np.random.choice(len(easy_features), size=new_size, replace=False)
        return TensorDataset(easy_features[indices], easy_labels[indices])

    def is_hard(self, models: List[LeNet], synthetic_sample: torch.Tensor) -> Tuple[bool, bool, bool]:
        """
        Determine if a synthetic sample is hard based on the average confidence across an ensemble of models.

        :param models: List of LeNet models (the ensemble).
        :param synthetic_sample: A single synthetic sample to evaluate.
        :return: True if the sample is considered hard (average confidence < self.threshold), otherwise False.
        """
        synthetic_sample = synthetic_sample.unsqueeze(0)  # Add batch dimension
        total_confidence, total_margin, misclassification_count, all_predictions = 0, 0, 0, []
        with torch.no_grad():
            for model in models:
                model.eval()
                # Forward pass
                output = model(synthetic_sample.to(u.DEVICE))
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(output, dim=1)
                # Get the maximum probability (confidence) and the top two probabilities (for margin calculation)
                max_probs, max_indices = torch.max(probabilities, dim=1)
                second_max_probs = torch.topk(probabilities, 2, dim=1)[0][:, 1]
                # Update confidence and margin
                total_confidence += max_probs.item()
                total_margin += (max_probs - second_max_probs).item()
                all_predictions.append(max_indices)
        # Calculate average confidence and margin
        average_confidence = total_confidence / len(models)
        average_margin = total_margin / len(models)
        # Calculate the number of misclassifications (where predictions differ from each other)
        ensemble_predictions = torch.stack(all_predictions).mode(dim=0)[0]  # Majority vote for ensemble prediction
        misclassification_count = torch.sum(ensemble_predictions != synthetic_sample.argmax(dim=1)).item()
        # Check if the sample is hard according to each metric
        confidence_hard = average_confidence < self.threshold
        margin_hard = average_margin < self.threshold
        misclassified_hard = misclassification_count > 5
        return confidence_hard, margin_hard, misclassified_hard

    def SMOTE(self, multiplication_factor: float, k_neighbors: int = 5) -> TensorDataset:
        """
        Apply SMOTE to balance the dataset by generating synthetic samples.

        :param multiplication_factor: The oversampling factor
        :param k_neighbors: The number of nearest neighbors to consider when generating synthetic samples.
        """
        hard_features, hard_labels = self.hard_data.tensors
        hard_features, hard_labels = hard_features.to(u.DEVICE), hard_labels.to(u.DEVICE)
        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * multiplication_factor)
        all_synthetic_samples, all_synthetic_labels = [], []
        # Calculate target samples per class
        num_classes = torch.unique(hard_labels).size(0)
        target_samples_per_class = new_size // num_classes
        # This is necessary to count how many of the generated samples are hard
        confidence_hard_count, margin_hard_count, misclassified_hard_count = 0, 0, 0
        models = [LeNet().to(u.DEVICE) for _ in range(self.models_count)]
        for i in range(self.models_count):
            models[i].load_state_dict(
                torch.load(f"{u.MODEL_SAVE_DIR}{self.dataset_name}_{self.models_count}_ensemble_{i}.pth"))

        for class_label in torch.unique(hard_labels):
            # Select samples belonging to the current class
            class_mask = (hard_labels == class_label)
            class_features = hard_features[class_mask]
            class_size = len(class_features)
            if multiplication_factor > 0.0:  # omit 'class_size > 0' to see if it ever happens (would throw error)
                nn = NearestNeighbors(n_neighbors=k_neighbors)
                nn.fit(class_features.cpu().numpy())
                num_synthetic_samples = target_samples_per_class - class_size
                synthetic_samples = []
                # Generate synthetic samples
                for _ in range(num_synthetic_samples):
                    # Choose one random hard sample
                    idx = np.random.randint(0, class_size)
                    neighbors = nn.kneighbors(class_features[idx].cpu().unsqueeze(0).numpy(), return_distance=False)
                    # Choose one neighbor randomly
                    neighbor_idx = np.random.choice(neighbors[0][1:])
                    neighbor = class_features[neighbor_idx]
                    # Interpolate between the chosen sample and its neighbor
                    diff = neighbor - class_features[idx]
                    synthetic_sample = class_features[idx] + torch.rand(1).to(u.DEVICE) * diff
                    # Check if the synthetic sample is hard according to each metric
                    confidence_hard, margin_hard, misclassified_hard = self.is_hard(models, synthetic_sample)
                    if confidence_hard:
                        confidence_hard_count += 1
                    if margin_hard:
                        margin_hard_count += 1
                    if misclassified_hard:
                        misclassified_hard_count += 1
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
            # Print the summary table
            print(f"Found {confidence_hard_count} hard samples based on confidence threshold.")
            print(f"Found {margin_hard_count} hard samples based on margin threshold.")
            print(f"Found {misclassified_hard_count} hard samples based on misclassification.")
            # TODO: add overlap matrix
        else:
            augmented_features = hard_features
            augmented_labels = hard_labels
        # TODO: give option to rerun until only hard are generated
        # TODO: Run SMOTE on easy samples to show that they take larger part of class submanifolds

        return TensorDataset(augmented_features, augmented_labels)