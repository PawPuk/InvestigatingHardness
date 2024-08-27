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
        Determine if a synthetic sample is hard based on three criteria: confidence, margin, and misclassification.
        :param models: List of LeNet models (the ensemble).
        :param synthetic_sample: A single synthetic sample to evaluate.
        :return: Tuple of three booleans indicating if the sample is hard according to confidence, margin, and misclassification.
        """
        synthetic_sample = synthetic_sample.unsqueeze(0)  # Add batch dimension
        total_confidence, total_margin, misclassification_count, all_predictions = 0, 0, 0, []
        with torch.no_grad():
            for model in models:
                model.eval()
                output = model(synthetic_sample.to(u.DEVICE))
                probabilities = torch.nn.functional.softmax(output, dim=1)
                max_probs, max_indices = torch.max(probabilities, dim=1)
                second_max_probs = torch.topk(probabilities, 2, dim=1)[0][:, 1]
                total_confidence += max_probs.item()
                total_margin += (max_probs - second_max_probs).item()
                all_predictions.append(max_indices)
        average_confidence = total_confidence / len(models)
        average_margin = total_margin / len(models)
        ensemble_predictions = torch.stack(all_predictions).mode(dim=0)[0]  # Majority vote for ensemble prediction
        misclassification_count = torch.sum(ensemble_predictions != synthetic_sample.argmax(dim=1)).item()
        confidence_hard = average_confidence < self.threshold
        margin_hard = average_margin < self.threshold
        misclassified_hard = misclassification_count > 5
        return confidence_hard, margin_hard, misclassified_hard

    def MixUp(self, multiplication_factor: float) -> TensorDataset:
        """
        Apply MixUp to generate synthetic samples by interpolating between two random samples.
        :param multiplication_factor: The oversampling factor.
        """
        hard_features, hard_labels = self.hard_data.tensors
        hard_features, hard_labels = hard_features.to(u.DEVICE), hard_labels.to(u.DEVICE)
        easy_size = len(self.easy_data)
        hard_size = len(self.hard_data)
        new_size = hard_size + int((easy_size - hard_size) * multiplication_factor)
        all_synthetic_samples, all_synthetic_labels = [], []
        confidence_hard_indices, margin_hard_indices, misclassified_hard_indices = set(), set(), set()
        models = [LeNet().to(u.DEVICE) for _ in range(self.models_count)]
        for i in range(self.models_count):
            models[i].load_state_dict(
                torch.load(f"{u.MODEL_SAVE_DIR}{self.dataset_name}_{self.models_count}_ensemble_{i}.pth"))
        # Generate synthetic samples using MixUp
        for _ in range(new_size - hard_size):
            # Randomly select two different hard samples from different classes
            while True:
                idx1, idx2 = np.random.choice(len(hard_features), 2, replace=False)
                sample1, sample2 = hard_features[idx1], hard_features[idx2]
                label1, label2 = hard_labels[idx1], hard_labels[idx2]
                if label1 != label2:
                    break
            # Interpolate between the two samples
            alpha = np.random.beta(0.2, 0.2)
            synthetic_sample = alpha * sample1 + (1 - alpha) * sample2
            synthetic_label = alpha * label1 + (1 - alpha) * label2
            # Check if the synthetic sample is hard according to each metric
            confidence_hard, margin_hard, misclassified_hard = self.is_hard(models, synthetic_sample)
            if confidence_hard:
                confidence_hard_indices.add(len(all_synthetic_samples))
            if margin_hard:
                margin_hard_indices.add(len(all_synthetic_samples))
            if misclassified_hard:
                misclassified_hard_indices.add(len(all_synthetic_samples))
            all_synthetic_samples.append(synthetic_sample)
            all_synthetic_labels.append(synthetic_label)
        # Convert lists to tensors
        synthetic_samples = torch.stack(all_synthetic_samples)
        synthetic_labels = torch.stack(all_synthetic_labels)
        # Concatenate all synthetic samples and original data
        augmented_features = torch.cat((hard_features, synthetic_samples), dim=0)
        augmented_labels = torch.cat((hard_labels, synthetic_labels), dim=0)
        # Print the summary table
        print(f"Found {len(confidence_hard_indices)} hard samples based on confidence threshold.")
        print(f"Found {len(margin_hard_indices)} hard samples based on margin threshold.")
        print(f"Found {len(misclassified_hard_indices)} hard samples based on misclassification.")
        # Calculate overlap matrix
        indicators = ['Confidence', 'Margin', 'Misclassified']
        overlap_matrix = np.zeros((3, 3), dtype=int)
        hard_sets = [confidence_hard_indices, margin_hard_indices, misclassified_hard_indices]
        for i in range(3):
            for j in range(3):
                overlap_matrix[i, j] = len(hard_sets[i].intersection(hard_sets[j]))
        print("Overlap Matrix:")
        print(f"{'':>15} {'Confidence':>12} {'Margin':>12} {'Misclassified':>15}")
        for i, indicator in enumerate(indicators):
            print(f"{indicator:>15} {overlap_matrix[i, 0]:>12} {overlap_matrix[i, 1]:>12} {overlap_matrix[i, 2]:>15}")
        return TensorDataset(augmented_features, augmented_labels)
        # TODO: give option to rerun until only hard are generated
        # TODO: Run SMOTE on easy samples to show that they take larger part of class submanifolds