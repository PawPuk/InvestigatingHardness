import torch
import torch.nn as nn
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


if __name__ == '__main__':
    main()
