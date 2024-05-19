import pickle

import torch
from tqdm import tqdm

from utils import load_data_and_normalize, transform_datasets_to_dataloaders, initialize_model, train_stop_at_inversion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_data_and_normalize('MNIST', 70000)
loader = transform_datasets_to_dataloaders(dataset)
all_hard_indices, all_easy_indices = [], []

for run in tqdm(range(50), desc='Generating different straggler sets'):
    models = {}
    run_hard_indices = []
    run_easy_indices = []
    while set(models.keys()) != set(range(10)):
        model, optimizer = initialize_model()
        model.to(DEVICE)
        models, _ = train_stop_at_inversion(model, loader, optimizer)
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        indices = torch.arange(0, data.size(0), device=DEVICE)  # Direct index range for the full batch
        hard_indices_run = []
        easy_indices_run = []
        for class_idx in range(10):
            # Model prediction and comparison to identify hard and easy samples
            predictions = torch.argmax(models[class_idx](data), dim=1)
            is_hard = (predictions != target) & (target == class_idx)
            is_easy = (predictions == target) & (target == class_idx)
            # Collect indices for hard and easy samples
            hard_indices = indices[is_hard]
            easy_indices = indices[is_easy]
            hard_indices_run.extend(hard_indices.cpu().numpy().tolist())
            easy_indices_run.extend(easy_indices.cpu().numpy().tolist())
        all_hard_indices.append(hard_indices_run)
        all_easy_indices.append(easy_indices_run)

# Save the collected indices for later analysis
with open('Results/hard_samples_indices.pkl', 'wb') as f:
    pickle.dump(all_hard_indices, f)

with open('Results/easy_samples_indices.pkl', 'wb') as f:
    pickle.dump(all_easy_indices, f)
