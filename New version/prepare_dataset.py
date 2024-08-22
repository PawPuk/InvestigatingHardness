from typing import List, Tuple

from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np

import utils as u
from imbalance_measures import ImbalanceMeasures


class DatasetPreparer:
    def __init__(self, dataset_name: str, threshold: float, oversampling_factor: float, undersampling_ratio: float,
                 smote: bool = False):
        self.dataset_name = dataset_name
        self.threshold = threshold
        self.osf = oversampling_factor
        self.usr = undersampling_ratio
        self.smote = smote

    def load_and_prepare_data(self) -> Tuple[List[DataLoader], List[DataLoader]]:
        # Load and normalize dataset
        dataset = u.load_data_and_normalize(self.dataset_name)
        hardness_indicators = u.load_data(f"{u.CONFIDENCES_SAVE_DIR}{self.dataset_name}_bma_hardness_indicators.pkl")
        # Split into easy and hard datasets
        easy_dataset, hard_dataset = self.identify_hard_and_easy_data(dataset, hardness_indicators)
        # Mitigate imbalance if required
        old_easy_size, old_hard_size = len(easy_dataset), len(hard_dataset)
        IM = ImbalanceMeasures(easy_dataset, hard_dataset)
        hard_dataset = IM.SMOTE(self.osf) if self.smote else IM.random_oversampling(self.osf)
        easy_dataset = IM.random_undersampling(self.usr)
        print(f'Added {len(hard_dataset) - old_hard_size} hard samples via oversampling, and removed '
              f'{len(easy_dataset) - old_easy_size} easy samples via undersampling. Continuing with {len(easy_dataset)} '
              f'easy data samples, and {len(hard_dataset)} hard data samples.')
        return u.combine_and_split_data(hard_dataset, easy_dataset, self.dataset_name)

    def identify_hard_and_easy_data(self, dataset: TensorDataset,
                                    hardness_indicators: List[Tuple[float, float, bool]]) -> Tuple[Subset, Subset]:
        """Split the dataset into easy and hard samples based on confidence, margin, and misclassification."""
        confidence_hard_indices = {i for i, (conf, _, _) in enumerate(hardness_indicators) if conf < self.threshold}
        margin_hard_indices = {i for i, (_, margin, _) in enumerate(hardness_indicators) if margin < self.threshold}
        misclassified_hard_indices = {i for i, (_, _, misclassified) in enumerate(hardness_indicators) if misclassified}

        # Combine hard indices from all conditions without duplicates
        combined_hard_indices = confidence_hard_indices.union(margin_hard_indices).union(misclassified_hard_indices)
        easy_indices = [i for i in range(len(dataset)) if i not in combined_hard_indices]
        easy_dataset, hard_dataset = Subset(dataset, easy_indices), Subset(dataset, list(combined_hard_indices))

        # Reporting with a symmetric matrix-like structure
        indicators = ['Confidence', 'Margin', 'Misclassified']
        hard_indices_list = [confidence_hard_indices, margin_hard_indices, misclassified_hard_indices]
        overlap_matrix = np.zeros((len(indicators), len(indicators)), dtype=int)

        for i in range(len(indicators)):
            for j in range(len(indicators)):
                overlap_matrix[i, j] = len(hard_indices_list[i].intersection(hard_indices_list[j]))

        print(f"Found {len(confidence_hard_indices)} hard samples based on confidence threshold.")
        print(f"Found {len(margin_hard_indices)} hard samples based on margin threshold.")
        print(f"Found {len(misclassified_hard_indices)} hard samples based on misclassification.")
        print("Overlap Matrix:")
        print(f"{'':>15} {'Confidence':>12} {'Margin':>12} {'Misclassified':>15}")
        for i, indicator in enumerate(indicators):
            print(f"{indicator:>15} {overlap_matrix[i, 0]:>12} {overlap_matrix[i, 1]:>12} {overlap_matrix[i, 2]:>15}")

        print(f"Continuing with {len(easy_dataset)} easy data samples, and {len(hard_dataset)} hard data samples.")

        return easy_dataset, hard_dataset
