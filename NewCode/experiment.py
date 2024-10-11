import os
import torch
import pickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils as u

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Hyperparameters
hidden_sizes = [4, 6, 8, 10, 15, 20, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1250, 1500, 1750, 2000, 3000]
num_epochs = 50
batch_size = 64
dataset_name = 'MNIST'
results_file = 'accuracy_loss_results.pkl'

# Check if data has been computed previously
if os.path.exists(results_file):
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    num_params_list = results['num_params_list']
    train_losses_all = results['train_losses_all']
    train_accuracies_all = results['train_accuracies_all']
    test_losses_all = results['test_losses_all']
    test_accuracies_all = results['test_accuracies_all']
    class_losses_all = results['class_losses_all']
    class_accuracies_all = results['class_accuracies_all']
else:
    # Load dataset and add 10% label noise
    training_dataset, test_dataset = u.load_data_and_normalize(dataset_name, label_noise=0.1)
    training_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Prepare lists to store results
    num_params_list = []
    train_losses_all, train_accuracies_all = [], []
    test_losses_all, test_accuracies_all = [], []
    class_losses_all, class_accuracies_all = [], []

    # Loop over different hidden layer sizes
    for hidden_size in tqdm(hidden_sizes):
        # Initialize model and optimizer for each hidden size
        model, optimizer = u.initialize_models(dataset_name, 'simple', hidden_size)

        # Train the model
        train_losses, train_accuracies, test_losses, test_accuracies = u.train(dataset_name, model, training_loader,
                                                                               test_loader, optimizer, num_epochs)
        # Calculate the number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_params_list.append(num_params)

        # Store the final train/test accuracies and losses
        train_losses_all.append(train_losses)
        train_accuracies_all.append(train_accuracies)
        test_losses_all.append(test_losses)
        test_accuracies_all.append(test_accuracies)

        # Evaluate model to get class-level metrics
        _, _, class_losses, class_accuracies = u.test(model, test_loader)

        # Store test class-level losses and accuracies
        class_losses_all.append(class_losses)
        class_accuracies_all.append([100 - acc for acc in class_accuracies])  # Store class accuracies as errors

    # Save computed results
    results = {
        'num_params_list': num_params_list,
        'train_losses_all': train_losses_all,
        'train_accuracies_all': train_accuracies_all,
        'test_losses_all': test_losses_all,
        'test_accuracies_all': test_accuracies_all,
        'class_losses_all': class_losses_all,
        'class_accuracies_all': class_accuracies_all,
    }

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

# Convert results to numpy arrays for easy plotting
class_accuracies_all = torch.tensor(class_accuracies_all).numpy()
test_accuracies_all = torch.tensor(test_accuracies_all).numpy()

# Plotting Class-Level Errors (10 subplots)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

for i in range(10):  # 10 classes
    ax = axes[i]
    ax.plot(num_params_list, class_accuracies_all[:, i], label=f'Class {i} Error', color='tab:blue', marker='o')
    ax.plot(num_params_list, [100 - acc[-1] for acc in test_accuracies_all], label='Overall Error', color='black', linestyle='--')
    ax.set_xscale('log')  # Logarithmic scale for the number of parameters
    ax.set_xlabel('Number of Parameters (log scale)')
    ax.set_ylabel('Error (%)')
    ax.set_title(f'Class {i} vs Overall Error')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('class_level_errors.pdf')
plt.show()
