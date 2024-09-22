import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as u

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class EnsembleTrainer:
    def __init__(self, dataset_name: str, models_count: int, save: bool, training: str, model_type: str,
                 save_dir: str = u.MODEL_SAVE_DIR):
        self.dataset_name = dataset_name
        self.models_count = models_count
        self.save = save
        self.training = training
        self.model_type = model_type
        self.save_dir = save_dir

    def get_next_model_index(self):
        """Determine the next available model index for saving new models."""
        model_paths = self.get_all_trained_model_paths()
        if not model_paths:
            return 0
        else:
            indices = []
            for path in model_paths:
                filename = os.path.basename(path)
                index_str = filename.split('_')[-1].split('.')[0]
                indices.append(int(index_str))
            return max(indices) + 1

    def get_all_trained_model_paths(self):
        """Retrieve all trained model paths matching current dataset, training, and model type."""
        pattern = f"{u.MODEL_SAVE_DIR}{self.training}{self.dataset_name}_{self.model_type}ensemble_*.pth"
        model_paths = glob(pattern)
        return sorted(model_paths)

    def train_ensemble(self, train_loader: DataLoader, test_loader: DataLoader):
        """Train specified number of models and then evaluate all models (including previously trained ones)."""
        if self.training == 'full':
            print('Training an ensemble of networks in full information scenario')
        else:
            print('Training an ensemble of networks in partial information scenario')
        num_classes = len(torch.unique(train_loader.dataset.tensors[1]))
        epochs = 100 if self.dataset_name == 'CIFAR10' else 10

        # Determine where to start saving model indices
        start_index = self.get_next_model_index()

        # Train specified number of models
        for i in tqdm(range(self.models_count)):
            model, optimizer = u.initialize_models(self.dataset_name, self.model_type)
            u.train(self.dataset_name, model, train_loader, optimizer, epochs)
            # Save model state
            if self.save:
                model_index = start_index + i
                torch.save(model.state_dict(), f"{self.save_dir}{self.training}{self.dataset_name}"
                                               f"_{self.model_type}ensemble_{model_index}.pth",
                           _use_new_zipfile_serialization=False)  # Ensuring backward compatibility
        if self.save_dir == u.MODEL_SAVE_DIR:
            # Collect all trained models
            total_models = 10 if self.dataset_name == 'CIFAR10' else 25
            model_paths = self.get_all_trained_model_paths()[:total_models]
            class_accuracies = np.zeros((total_models, num_classes))  # Store class-level accuracies for all models
            total_accuracies = np.zeros(total_models)

            # Evaluate all models
            for idx, model_path in enumerate(model_paths):
                model, _ = u.initialize_models(self.dataset_name, self.model_type)
                model.load_state_dict(torch.load(model_path))
                # Evaluate the model on the test set
                class_accuracies[idx] = u.class_level_test(model, test_loader, num_classes)
                total_accuracies[idx] = u.test(model, test_loader) / 100

            # Save class accuracies
            class_accuracies_file = f"{u.METRICS_SAVE_DIR}{self.training}{self.dataset_name}" \
                                    f"_avg_class_accuracies_on_{self.model_type}ensemble.pkl"
            u.save_data(class_accuracies, class_accuracies_file)

            # Measure the consistency of class bias (but only for untouched dataset; don't repeat for resampled case)
            running_avg_class_accuracies = np.array([class_accuracies[:i+1].mean(axis=0) for i in range(total_models)])
            running_std_class_accuracies = np.array([class_accuracies[:i+1].std(axis=0) for i in range(total_models)])
            self.plot_class_accuracies(running_avg_class_accuracies, running_std_class_accuracies, total_accuracies,
                                       num_classes)

    def plot_class_accuracies(self, running_avg_class_accuracies, running_std_class_accuracies,
                              total_accuracies, num_classes):
        """Plot how the average accuracy of each class changes as we increase the number of models."""

        fig, ax = plt.subplots(figsize=(10, 6))  # Single figure for all class accuracies

        x_vals = range(1, running_avg_class_accuracies.shape[0] + 1)
        std_accs = []
        for class_idx in range(num_classes):
            mean_acc = running_avg_class_accuracies[:, class_idx]
            std_acc = running_std_class_accuracies[:, class_idx]
            std_accs.append(std_acc)

            # Plot accuracy for each class
            ax.plot(x_vals, mean_acc, label=f'Class {class_idx} Mean', linewidth=6)
            # ax.fill_between(x_vals, mean_acc - std_acc, mean_acc + std_acc, alpha=0.1)

            # Print the final running standard deviation for each class
            print(f"Final running standard deviation for Class {class_idx}: {std_acc[-1]:.4f}")
        print(f'Standard deviation of class-level standard deviations: {np.std(std_accs):.4f}')

        # Customize the plot
        ax.set_xlabel('Number of Models in the Ensemble', fontsize=20)
        ax.set_ylabel('Avg Accuracy', fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim(1, running_avg_class_accuracies.shape[0])

        plt.tight_layout()

        # Save and show the plot
        plt.savefig(f'{u.CLASS_BIAS_SAVE_DIR}{self.training}{self.dataset_name}_class_bias_on_'
                    f'{self.model_type}ensemble.pdf')
        plt.show()

        # Calculate and print the final running standard deviation for overall accuracy (total_accuracies)
        total_accuracies_final_std = total_accuracies.std()
        print(f"Final running standard deviation for total accuracy: {total_accuracies_final_std:.4f}")


def main(dataset_name: str, models_count: int, training: str, model_type: str):
    trainer = EnsembleTrainer(dataset_name, models_count, True, training, model_type)
    if training == 'full':  # 'full information' scenario
        train_dataset = u.load_full_data_and_normalize(dataset_name)
        test_dataset = train_dataset
    else:  # 'limited information' scenario
        train_dataset, test_dataset = u.load_data_and_normalize(dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    trainer.train_ensemble(train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an ensemble of models on the full dataset and save parameters.')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--models_count', type=int, default=20, help='Number of models to train in this run.')
    parser.add_argument('--training', type=str, choices=['full', 'part'], default='full',
                        help='Indicates which models to choose for evaluations - the ones trained on the entire dataset'
                             ' (full), or the ones trained only on the training set (part).')
    parser.add_argument('--model_type', type=str, choices=['simple', 'complex'],
                        help='Specifies the type of network used for training (MLP vs LeNet or ResNet20 vs ResNet56).')
    args = parser.parse_args()
    main(**vars(args))
