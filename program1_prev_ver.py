import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from utils import initialize_model, load_data_and_normalize, train_stop_at_inversion, transform_datasets_to_dataloaders


def train_and_evaluate_class_level(model, device, train_loader, optimizer, criterion, epochs=20, n_classes=10):
    # Initialize tracking structures
    class_epoch_accuracy = {c: [] for c in range(n_classes)}
    learned_samples = {c: [set() for _ in range(epochs)] for c in range(n_classes)}
    unlearned_samples = {c: [set() for _ in range(epochs)] for c in range(n_classes)}
    relearned_samples = {c: torch.zeros(len(train_loader.dataset)) for c in range(n_classes)}
    prev_correctly_classified = {c: set() for c in range(n_classes)}
    for epoch in tqdm(range(epochs)):
        model.train()
        current_correctly_classified = {c: set() for c in range(n_classes)}
        class_correct = {c: 0 for c in range(n_classes)}
        class_total = {c: 0 for c in range(n_classes)}
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=False)
            for idx, (p, t) in enumerate(zip(pred, target)):
                global_idx = batch_idx * train_loader.batch_size + idx  # Global index in the dataset
                class_total[t.item()] += 1
                if p == t:
                    class_correct[t.item()] += 1
                    current_correctly_classified[t.item()].add(global_idx)
                    if global_idx not in prev_correctly_classified[t.item()]:
                        learned_samples[t.item()][epoch].add(global_idx)
                        relearned_samples[t.item()][global_idx] += 1
                else:
                    if global_idx in prev_correctly_classified[t.item()]:
                        unlearned_samples[t.item()][epoch].add(global_idx)
        # Update class-level accuracy
        for c in range(n_classes):
            if class_total[c] > 0:
                class_epoch_accuracy[c].append(100 * class_correct[c] / class_total[c])
            else:
                class_epoch_accuracy[c].append(0)
        prev_correctly_classified = current_correctly_classified.copy()
    # Process sets to counts for learned and unlearned for easier plotting
    learned_counts = {c: [len(epoch_set) for epoch_set in learned_samples[c]] for c in range(n_classes)}
    unlearned_counts = {c: [len(epoch_set) for epoch_set in unlearned_samples[c]] for c in range(n_classes)}
    return class_epoch_accuracy, learned_counts, unlearned_counts, relearned_samples


def evaluate_accuracy(model, device, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy


def plot_class_accuracy_shaded(mean_class_epoch_accuracy, std_class_epoch_accuracy, inversion_points, epochs,
                               n_classes=10):
    fig, axs = plt.subplots(2, n_classes // 2, figsize=(15, 8))
    for c in range(n_classes):
        row, col = divmod(c, n_classes // 2)
        x = np.arange(epochs)
        mean = mean_class_epoch_accuracy[c]
        std = std_class_epoch_accuracy[c]
        axs[row, col].plot(x, mean, label='Mean Accuracy')
        axs[row, col].fill_between(x, mean-std, mean+std, color='blue', alpha=0.2)
        axs[row, col].axvline(x=inversion_points[c], color='r', linestyle='--', label='Inversion Point')
        axs[row, col].set_title(f'Class {c}')
    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='Accuracy (%)')
    fig.suptitle('Class-level Accuracy Over Epochs with Std Dev')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def aggregate_into_bins(data, bin_size=25, skip_first=False):
    if skip_first:
        data = data[1:]  # Remove the first epoch's data if needed
    # Sum data points within each bin
    binned_data = [sum(data[i:i + bin_size]) for i in range(0, len(data), bin_size)]
    return binned_data


def plot_learned_samples_shaded(mean_learned_counts, std_learned_counts, inversion_points, epochs, n_classes=10,
                                bin_size=25):
    fig, axs = plt.subplots(2, n_classes // 2, figsize=(15, 8))
    epoch_bins = np.arange(0, epochs, bin_size)

    for c in range(n_classes):
        row, col = divmod(c, n_classes // 2)
        # Aggregate mean and std into bins, skip first epoch for learned samples if desired
        mean_binned = aggregate_into_bins(mean_learned_counts[c], bin_size, skip_first=True)
        std_binned = aggregate_into_bins(std_learned_counts[c], bin_size, skip_first=True)

        axs[row, col].plot(epoch_bins[:len(mean_binned)], mean_binned, label='Mean Learned')
        axs[row, col].fill_between(epoch_bins[:len(mean_binned)], np.array(mean_binned) - np.array(std_binned),
                                   np.array(mean_binned) + np.array(std_binned), alpha=0.2)
        axs[row, col].axvline(x=inversion_points[c] // bin_size * bin_size, color='r', linestyle='--',
                              label='Inversion Point')  # Adjusted inversion point for binned x-axis
        axs[row, col].set_title(f'Class {c}')
        axs[row, col].set_xlabel('Epoch Bins')
    axs[0, 0].set_ylabel('Learned Samples')
    fig.suptitle('Learned Samples Per Epoch for Each Class (Binned with Std Dev)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_unlearned_samples_shaded(mean_unlearned_counts, std_unlearned_counts, inversion_points, epochs, n_classes=10,
                                  bin_size=25):
    fig, axs = plt.subplots(2, n_classes // 2, figsize=(15, 8))
    epoch_bins = np.arange(0, epochs, bin_size)

    for c in range(n_classes):
        row, col = divmod(c, n_classes // 2)
        # Aggregate mean and std into bins
        mean_binned = aggregate_into_bins(mean_unlearned_counts[c], bin_size)
        std_binned = aggregate_into_bins(std_unlearned_counts[c], bin_size)

        axs[row, col].plot(epoch_bins[:len(mean_binned)], mean_binned, label='Mean Unlearned')
        axs[row, col].fill_between(epoch_bins[:len(mean_binned)], np.array(mean_binned) - np.array(std_binned),
                                   np.array(mean_binned) + np.array(std_binned), alpha=0.2)
        axs[row, col].axvline(x=inversion_points[c] // bin_size * bin_size, color='r', linestyle='--',
                              label='Inversion Point')  # Adjusted inversion point for binned x-axis
        axs[row, col].set_title(f'Class {c}')
        axs[row, col].set_xlabel('Epoch Bins')
    axs[0, 0].set_ylabel('Unlearned Samples')
    fig.suptitle('Unlearned Samples Per Epoch for Each Class (Binned with Std Dev)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_relearned_frequencies_shaded(mean_relearned_samples, std_relearned_samples, inversion_points, n_classes=10):
    fig, axs = plt.subplots(2, n_classes // 2, figsize=(15, 8))
    for c in range(n_classes):
        row, col = divmod(c, n_classes // 2)
        relearned_freq = np.bincount(mean_relearned_samples[c].round().astype(int))
        x = np.arange(len(relearned_freq))
        std_area = std_relearned_samples[c][:len(relearned_freq)]  # Assume std_relearned_samples is prepared similarly
        axs[row, col].bar(x, relearned_freq, color='purple', label='Relearned Frequency')
        axs[row, col].fill_between(x, relearned_freq-std_area, relearned_freq+std_area, color='purple', alpha=0.2)
        axs[row, col].axvline(x=inversion_points[c], color='r', linestyle='--', label='Inversion Point')
        axs[row, col].set_title(f'Class {c}')
    for ax in axs.flat:
        ax.set(xlabel='Relearned Count', ylabel='Sample Frequency')
    fig.suptitle('Relearned Frequencies for Each Class with Std Dev')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def compute_mean_std(all_stats, n_runs, epochs, n_classes=10):
    mean_stats = {c: np.zeros(epochs) for c in range(n_classes)}
    std_stats = {c: np.zeros(epochs) for c in range(n_classes)}
    for c in range(n_classes):
        for epoch in range(epochs):
            epoch_values = np.array([all_stats[c][run][epoch] for run in range(n_runs)])
            mean_stats[c][epoch] = np.mean(epoch_values)
            std_stats[c][epoch] = np.std(epoch_values)
    return mean_stats, std_stats


def main():
    n_runs = 1
    epochs = 500
    n_classes = 10
    # Initialize storage for stats across runs
    all_class_epoch_accuracy = {c: [] for c in range(n_classes)}
    all_learned_counts = {c: [] for c in range(n_classes)}
    all_unlearned_counts = {c: [] for c in range(n_classes)}
    all_relearned_samples = {c: [] for c in range(n_classes)}
    for run in range(n_runs):
        print(f"Starting run {run + 1}/{n_runs}")
        dataset = load_data_and_normalize('MNIST', 35000)
        loader = transform_datasets_to_dataloaders(dataset)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure model is re-initialized and a fresh inversion_points calculation
        inversion_points, initial_state_dict = {}, {}
        while set(inversion_points.keys()) != set(range(10)):
            if len(inversion_points.keys()) > 0:
                print(f'Restarting due to incomplete straggler identification (found {inversion_points.keys()})')
            model, optimizer = initialize_model()
            model.to(device)
            initial_state_dict = model.state_dict()
            _, inversion_points = train_stop_at_inversion(model, loader, optimizer)
        model, optimizer = initialize_model()
        model.load_state_dict(initial_state_dict)
        class_epoch_accuracy, learned_counts, unlearned_counts, relearned_samples = (
            train_and_evaluate_class_level(model, device, loader, optimizer, criterion, epochs=epochs))
        accuracy = evaluate_accuracy(model, device, loader)
        print(f'Run {run + 1} Training Accuracy: {accuracy:.2f}%')
        # Accumulate statistics
        for c in range(n_classes):
            all_class_epoch_accuracy[c].append(class_epoch_accuracy[c])
            all_learned_counts[c].append(learned_counts[c])
            all_unlearned_counts[c].append(unlearned_counts[c])
            all_relearned_samples[c].append(relearned_samples[c])
    # Compute mean and std of statistics across runs
    mean_class_epoch_accuracy, std_class_epoch_accuracy = compute_mean_std(all_class_epoch_accuracy, n_runs, epochs)
    mean_learned_counts, std_learned_counts = compute_mean_std(all_learned_counts, n_runs, epochs)
    mean_unlearned_counts, std_unlearned_counts = compute_mean_std(all_unlearned_counts, n_runs, epochs)
    mean_relearned_samples, std_relearned_samples = compute_mean_std(all_relearned_samples, n_runs, epochs)

    # Plotting with shaded regions for std
    plot_class_accuracy_shaded(mean_class_epoch_accuracy, std_class_epoch_accuracy, inversion_points, epochs)
    plot_learned_samples_shaded(mean_learned_counts, std_learned_counts, inversion_points, epochs)
    plot_unlearned_samples_shaded(mean_unlearned_counts, std_unlearned_counts, inversion_points, epochs)
    plot_relearned_frequencies_shaded(mean_relearned_samples, std_relearned_samples, inversion_points)
    plt.show()


if __name__ == '__main__':
    main()
