import pickle

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import initialize_model, load_data_and_normalize, transform_datasets_to_dataloaders


def train_and_evaluate(model, device, train_loader, optimizer, criterion, epochs=20, n_classes=10) -> \
        Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int], Dict[int, int], Dict[int, int]]:
    learned_samples = {c: [set() for _ in range(epochs)] for c in range(n_classes)}
    forgotten_samples = {c: [set() for _ in range(epochs)] for c in range(n_classes)}
    correctly_classified = {c: set() for c in range(n_classes)}
    relearned_counter = {}
    first_learned_epoch = {}
    last_learned_epoch = {}

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=False)
            for idx, (p, t) in enumerate(zip(pred, target)):
                global_idx = batch_idx * train_loader.batch_size + idx
                if p == t:
                    if global_idx not in correctly_classified[t.item()]:
                        if global_idx in relearned_counter:
                            relearned_counter[global_idx] += 1
                        else:
                            relearned_counter[global_idx] = 1
                            first_learned_epoch[global_idx] = epoch  # First time learned
                        learned_samples[t.item()][epoch].add(global_idx)
                        correctly_classified[t.item()].add(global_idx)
                        last_learned_epoch[global_idx] = epoch  # Update last learned epoch
                else:
                    if global_idx in correctly_classified[t.item()]:
                        forgotten_samples[t.item()][epoch].add(global_idx)
                        correctly_classified[t.item()].remove(global_idx)

    print(len(relearned_counter), len(first_learned_epoch), len(last_learned_epoch))
    learned_counts = {c: [len(epoch_set) for epoch_set in learned_samples[c]] for c in range(n_classes)}
    forgotten_counts = {c: [len(epoch_set) for epoch_set in forgotten_samples[c]] for c in range(n_classes)}
    return learned_counts, forgotten_counts, relearned_counter, first_learned_epoch, last_learned_epoch


def main():
    n_runs = 50
    epochs = 500

    dataset = load_data_and_normalize('MNIST', 70000)
    loader = transform_datasets_to_dataloaders(dataset)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_learned_counts = []
    all_forgotten_counts = []
    all_relearned_counters = []
    all_first_learned_epochs = []
    all_last_learned_epochs = []

    for _ in tqdm(range(n_runs), desc='Trying different initializations'):
        model, optimizer = initialize_model()
        model.to(device)

        learned_counts, forgotten_counts, relearned_counter, first_learned_epoch, last_learned_epoch = \
            train_and_evaluate(model, device, loader, optimizer, criterion, epochs=epochs)
        all_learned_counts.append(learned_counts)
        all_forgotten_counts.append(forgotten_counts)
        all_relearned_counters.append(relearned_counter)
        all_first_learned_epochs.append(first_learned_epoch)
        all_last_learned_epochs.append(last_learned_epoch)

    # Save the raw learned and forgotten counts for each run
    with open('Results/raw_learned_counts.pkl', 'wb') as f:
        pickle.dump(all_learned_counts, f)
    with open('Results/raw_forgotten_counts.pkl', 'wb') as f:
        pickle.dump(all_forgotten_counts, f)
    with open('Results/relearned_counters.pkl', 'wb') as f:
        pickle.dump(all_relearned_counters, f)
    with open('Results/first_learned_epochs.pkl', 'wb') as f:
        pickle.dump(all_first_learned_epochs, f)
    with open('Results/last_learned_epochs.pkl', 'wb') as f:
        pickle.dump(all_last_learned_epochs, f)

    print("All statistics saved.")


if __name__ == '__main__':
    main()
