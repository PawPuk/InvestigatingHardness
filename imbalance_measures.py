from typing import Union

import numpy as np
import torch
from torch.utils.data import TensorDataset

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class ImbalanceMeasures:
    def __init__(self, easy_dataset: Union[TensorDataset, None], hard_dataset: Union[TensorDataset, None]):
        self.easy_data = easy_dataset
        self.hard_data = hard_dataset

    def random_oversampling(self, multiplication_factor: float) -> TensorDataset:
        hard_features, hard_labels = self.hard_data.tensors
        multiplication_factor = max(1.0, multiplication_factor)
        new_size = int(len(self.hard_data) * multiplication_factor)
        indices = np.random.choice(len(hard_features), size=new_size, replace=True)
        return TensorDataset(hard_features[indices], hard_labels[indices])

    def random_undersampling(self, removal_ratio: float) -> TensorDataset:
        easy_features, easy_labels = self.easy_data.tensors
        removal_ratio = min(1.0, removal_ratio)
        new_size = int(len(self.easy_data) * (1 - removal_ratio))
        indices = np.random.choice(len(easy_features), size=new_size, replace=False)
        return TensorDataset(easy_features[indices], easy_labels[indices])
