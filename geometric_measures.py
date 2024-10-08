from glob import glob
from typing import List

from cleanlab.rank import get_label_quality_scores
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm

import utils as u


class Volume:
    def __init__(self, data):
        self.data = data

    def calculate_volume(self, d=1.0):
        # Reshape the data to (N, C * H * W), where each image is flattened into a vector
        reshaped_data = self.data.reshape(self.data.shape[0], -1)
        # Calculate the mean of Z (Z_mean), across the dataset
        Z_mean = np.mean(reshaped_data, axis=0, keepdims=True)
        # Calculate (Z - Z_mean)
        diff = reshaped_data - Z_mean
        # Calculate (Z - Z_mean)(Z - Z_mean)^T, where each image vector is flattened
        outer_product = np.dot(diff.T, diff)
        # Scale the outer product
        scaled_outer_product = (d / reshaped_data.shape[0]) * outer_product
        # Calculate I + \frac{d}{m}(Z - Z_mean)(Z - Z_mean)^T
        matrix_sum = np.eye(reshaped_data.shape[1]) + scaled_outer_product
        # Calculate the volume: \frac{1}{2} \log_2 \det \left( I + \frac{d}{m}(Z - Z_mean)(Z - Z_mean)^T \right)
        volume = 0.5 * np.log2(np.linalg.det(matrix_sum))
        density = volume / len(self.data)
        return volume, density

    def sum_of_eigenvalues(self):
        reshaped_data = self.data.reshape(self.data.shape[0], -1)
        # Compute covariance matrix
        covariance_matrix = np.cov(reshaped_data, rowvar=False)
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        # Sum of eigenvalues
        sum_eigenvalues = np.sum(eigenvalues)
        return sum_eigenvalues

    def max_eigenvalue(self):
        reshaped_data = self.data.reshape(self.data.shape[0], -1)
        # Compute covariance matrix
        covariance_matrix = np.cov(reshaped_data, rowvar=False)
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        # Max eigenvalue
        max_eigenvalue = np.max(eigenvalues)
        return max_eigenvalue


class Curvature:
    def __init__(self, data, data_indices, k, pca_components=8):
        self.data = data.reshape(data.shape[0], -1)  # Flatten the images (required for image datasets)
        self.data_indices = data_indices
        self.k = k
        self.pca_components = pca_components

    @staticmethod
    def compute_hessian(coords):
        n = coords.shape[1]
        G = np.dot(coords.T, coords)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i, j] = np.mean((G[:, i] - np.mean(G[:, i])) ** 2)
                else:
                    H[i, j] = np.mean((G[:, i] - np.mean(G[:, i])) * (G[:, j] - np.mean(G[:, j])))
        return H

    def estimate_curvatures(self, gaussian_curvatures, mean_curvatures):
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(self.data)
        distances, indices = nn.kneighbors(self.data)

        for i, point_neighbors in enumerate(indices):
            point = self.data[i]
            neighbors = self.data[point_neighbors[1:]]  # Exclude the point itself
            pca = PCA(n_components=min(self.pca_components, self.data.shape[1]))  # Reduce to specified dimensions
            pca.fit(neighbors - point)
            coords = pca.transform(neighbors - point)
            H = self.compute_hessian(coords)
            eigenvalues = np.linalg.eigvals(H)
            if len(eigenvalues) >= 2:
                k1, k2 = eigenvalues[:2]
                gaussian_curvature = k1 * k2  # Gaussian curvature is the product of the principal curvatures
                mean_curvature = (k1 + k2) / 2  # Mean curvature is the average of the principal curvatures
                # The below is to make sure we do not override the data (sanity check)
                if gaussian_curvatures[self.data_indices[i]] is not None:
                    raise Exception
                if mean_curvatures[self.data_indices[i]] is not None:
                    raise Exception
                gaussian_curvatures[self.data_indices[i]] = gaussian_curvature
                mean_curvatures[self.data_indices[i]] = mean_curvature


class Proximity:
    def __init__(self, loader: DataLoader, class_loaders: List[DataLoader], k: int):
        """Initialize with the data loader and set K for KNN."""
        self.loader = loader
        self.k = k
        self.class_loaders = class_loaders
        self.centroids = self.compute_centroids()
        self.samples, self.labels = self.collect_samples()

    def compute_centroids(self):
        """Compute the centroids for each class."""
        centroids, class_counts = {}, {}

        for data, targets in self.loader:
            data, targets = data.to(u.DEVICE), targets.to(u.DEVICE)
            unique_classes = torch.unique(targets)

            for cls in unique_classes:
                class_mask = (targets == cls)
                class_data = data[class_mask]

                if cls.item() not in centroids:
                    centroids[cls.item()] = torch.zeros_like(class_data[0])
                    class_counts[cls.item()] = 0

                centroids[cls.item()] += class_data.sum(dim=0)
                class_counts[cls.item()] += class_data.size(0)

        # Finalize the centroids by dividing by the number of samples per class
        for cls in centroids:
            centroids[cls] /= class_counts[cls]

        return centroids

    def collect_samples(self):
        """Collect all samples and their corresponding labels from the loader."""
        samples = []
        labels = []
        for data, targets in self.loader:
            samples.append(data.to(u.DEVICE))
            labels.append(targets.to(u.DEVICE))
        samples = torch.cat(samples)
        labels = torch.cat(labels)
        return samples, labels

    def compute_proximity_metrics(self):
        """Compute proximity metrics for each sample, processing data in batches to match Code B."""

        same_centroid_dists, closest_same_class_distances, avg_same_class_distances = [], [], []

        other_centroid_dists = []
        closest_other_class_distances = []
        avg_other_class_distances = []
        percentage_other_class_knn = []
        adapted_N3 = []

        centroid_ratios, closest_distance_ratios, avg_distance_ratios = [], [], []

        # Prepare KNN classifier
        flattened_samples = self.samples.view(self.samples.size(0), -1).cpu().numpy()
        labels_np = self.labels.cpu().numpy()
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(flattened_samples)

        num_samples = self.samples.size(0)
        batch_size = 1000  # Adjust batch size as needed

        for start_idx in tqdm(range(0, num_samples, batch_size),
                              desc='Computing sample-level proximity metrics'):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = self.samples[start_idx:end_idx]
            batch_targets = self.labels[start_idx:end_idx]
            batch_samples_flat = batch_samples.view(batch_samples.size(0), -1).cpu().numpy()
            batch_targets_np = batch_targets.cpu().numpy()

            # Centroid distances for the batch
            for sample, target in zip(batch_samples, batch_targets):
                same_class_centroid = self.centroids[target.item()]
                min_other_class_dist = float('inf')

                # Compute distance to the same/other (closest) class centroid + ratio
                same_centroid_dist = torch.norm(sample - same_class_centroid).item()
                for cls, centroid in self.centroids.items():
                    if cls != target.item():
                        dist = torch.norm(sample - centroid).item()
                        if dist < min_other_class_dist:
                            min_other_class_dist = dist
                centroid_ratio = same_centroid_dist / min_other_class_dist
                centroid_ratios.append(centroid_ratio)
                other_centroid_dists.append(min_other_class_dist)
                same_centroid_dists.append(same_centroid_dist)

            # KNN computations for the batch
            distances, indices = knn.kneighbors(batch_samples_flat)
            distances = distances[:, 1:]  # Exclude the sample itself
            indices = indices[:, 1:]

            knn_distances = distances
            knn_labels = labels_np[indices]
            effective_k = self.k - 1  # Since we excluded the sample itself

            for i in range(len(batch_samples)):
                knn_dist = knn_distances[i]
                knn_label = knn_labels[i]
                target = batch_targets_np[i]

                knn_same_class_indices = knn_label == target
                knn_other_class_indices = knn_label != target

                same_class_dists = knn_dist[knn_same_class_indices]
                other_class_dists = knn_dist[knn_other_class_indices]

                # Closest distances to the same and other class samples + ratio
                if same_class_dists.size > 0:
                    min_same_class_dist = np.min(same_class_dists)
                else:
                    min_same_class_dist = np.inf

                if other_class_dists.size > 0:
                    min_other_class_dist = np.min(other_class_dists)
                else:
                    min_other_class_dist = np.inf

                closest_same_class_distances.append(min_same_class_dist)
                closest_other_class_distances.append(min_other_class_dist)
                closest_distance_ratios.append(min_same_class_dist / min_other_class_dist)

                # Average distances to same, other, and all samples in kNN
                if same_class_dists.size > 0:
                    avg_same_dist = np.mean(same_class_dists)
                else:
                    avg_same_dist = np.inf

                if other_class_dists.size > 0:
                    avg_other_dist = np.mean(other_class_dists)
                else:
                    avg_other_dist = np.inf

                avg_same_class_distances.append(avg_same_dist)
                avg_other_class_distances.append(avg_other_dist)
                avg_distance_ratios.append(avg_same_dist / avg_other_dist)

                # Compute the percentage of kNN samples from same and other classes
                num_other_class = knn_other_class_indices.sum()
                if effective_k > 0:
                    percentage_other_class_knn.append(num_other_class / effective_k)
                else:
                    percentage_other_class_knn.append(0)

                # Adapted N3 computation per sample
                if effective_k > 0:
                    nearest_neighbor_label = knn_label[0]
                    n3_different_class = int(nearest_neighbor_label != target)
                else:
                    n3_different_class = 0  # Or np.nan
                adapted_N3.append(n3_different_class)

        return (
            same_centroid_dists,  # Type 1
            closest_same_class_distances, avg_same_class_distances,  # Type 2
            other_centroid_dists, closest_other_class_distances, avg_other_class_distances, percentage_other_class_knn,
            adapted_N3, centroid_ratios, closest_distance_ratios, avg_distance_ratios,  # Type 4
        )

    """def find_neighbors_with_expansion(self, knn, labels_np, sample_flat, target_label, max_neighbors=10000):
        current_k = self.k
        found_same_class = False
        found_other_class = False
        min_same_class_dist = float('inf')
        min_other_class_dist = float('inf')

        while current_k <= max_neighbors and (not found_same_class or not found_other_class):
            # Dynamically expand kNN search
            knn.set_params(n_neighbors=current_k)
            distances, indices = knn.kneighbors([sample_flat])

            knn_dist = distances[0][1:]  # Exclude the sample itself
            knn_indices = indices[0][1:]
            knn_labels = labels_np[knn_indices]

            # Check for same and other class distances
            same_class_indices = knn_labels == target_label
            other_class_indices = knn_labels != target_label

            same_class_dists = knn_dist[same_class_indices]
            other_class_dists = knn_dist[other_class_indices]

            if same_class_dists.size > 0:
                found_same_class = True
                min_same_class_dist = np.min(same_class_dists)

            if other_class_dists.size > 0:
                found_other_class = True
                min_other_class_dist = np.min(other_class_dists)

            current_k += 100  # Expand by 100 neighbors at a time
        knn.set_params(n_neighbors=self.k)
        return min_same_class_dist, min_other_class_dist

    def compute_proximity_metrics(self):

        num_samples = self.samples.size(0)
        batch_size = 1000  # Adjust batch size as needed

        # Pre-allocate result arrays with the same size as the number of samples
        same_centroid_dists = np.zeros(num_samples)
        closest_same_class_distances = np.zeros(num_samples)
        avg_same_class_distances = np.zeros(num_samples)
        other_centroid_dists = np.zeros(num_samples)
        closest_other_class_distances = np.zeros(num_samples)
        avg_other_class_distances = np.zeros(num_samples)
        percentage_other_class_knn = np.zeros(num_samples)
        adapted_N3 = np.zeros(num_samples)
        centroid_ratios = np.zeros(num_samples)
        closest_distance_ratios = np.zeros(num_samples)
        avg_distance_ratios = np.zeros(num_samples)
        avg_all_class_distances = np.zeros(num_samples)

        # Prepare KNN classifier for the entire dataset
        flattened_samples = self.samples.view(self.samples.size(0), -1).cpu().numpy()
        labels_np = self.labels.cpu().numpy()
        knn_full = NearestNeighbors(n_neighbors=self.k)
        knn_full.fit(flattened_samples)

        # Create KNN classifiers for each class using self.class_loaders
        knn_class_models = {}
        for class_idx, class_loader in enumerate(self.class_loaders):
            class_samples, _ = next(iter(class_loader))
            class_samples_flat = class_samples.view(class_samples.size(0), -1).cpu().numpy()
            knn_class_models[class_idx] = NearestNeighbors(n_neighbors=self.k)
            knn_class_models[class_idx].fit(class_samples_flat)

        for start_idx in tqdm(range(0, num_samples, batch_size),
                              desc='Computing sample-level proximity metrics'):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = self.samples[start_idx:end_idx]
            batch_targets = self.labels[start_idx:end_idx]
            batch_samples_flat = batch_samples.view(batch_samples.size(0), -1).cpu().numpy()
            batch_targets_np = batch_targets.cpu().numpy()

            # Process Same-Class Neighbors in Batches
            for class_idx in np.unique(batch_targets_np):
                class_mask = batch_targets_np == class_idx
                class_samples = batch_samples_flat[class_mask]
                batch_indices = np.where(class_mask)[0]  # Get the indices of samples in the current batch
                if len(class_samples) > 0:
                    # Get the k-NN for the batch of same-class samples
                    same_class_knn = knn_class_models[class_idx]
                    same_class_distances, _ = same_class_knn.kneighbors(class_samples)
                    min_same_class_dists = np.min(same_class_distances[:, 1:], axis=1)  # Exclude the sample itself
                    avg_same_class_dists = np.mean(same_class_distances[:, 1:], axis=1)
                    # Store results in the correct indices for same-class metrics
                    closest_same_class_distances[start_idx + batch_indices] = min_same_class_dists
                    avg_same_class_distances[start_idx + batch_indices] = avg_same_class_dists

            # Process Other-Class Neighbors in Batches
            for class_idx in np.unique(batch_targets_np):
                class_mask = batch_targets_np == class_idx
                class_samples = batch_samples_flat[class_mask]
                batch_indices = np.where(class_mask)[0]  # Get the indices of samples in the current batch
                other_class_dists = []
                if len(class_samples) > 0:
                    for other_class_idx in knn_class_models:
                        if other_class_idx != class_idx:
                            other_class_knn = knn_class_models[other_class_idx]
                            other_class_distances, _ = other_class_knn.kneighbors(class_samples)
                            other_class_dists.append(other_class_distances)
                    # Stack distances from other classes and compute the closest
                    other_class_dists = np.hstack(other_class_dists)
                    min_other_class_dists = np.min(other_class_dists, axis=1)
                    avg_other_class_dists = np.mean(other_class_dists, axis=1)
                    # Store results in the correct indices for other-class metrics
                    closest_other_class_distances[start_idx + batch_indices] = min_other_class_dists
                    avg_other_class_distances[start_idx + batch_indices] = avg_other_class_dists

            # Compute the Purity (percentage of other-class kNN neighbors)
            distances, indices = knn_full.kneighbors(batch_samples_flat)
            for i in range(len(batch_samples)):
                target = batch_targets_np[i]
                knn_labels = labels_np[indices[i][1:]]  # Exclude the sample itself
                # Count how many neighbors belong to a different class
                other_class_count = np.sum(knn_labels != target)
                # Compute the percentage of other-class neighbors
                percentage_other_class_knn[start_idx + i] = other_class_count / (
                            self.k - 1)  # Exclude the sample itself

            # Centroid distances and ratios for the batch
            for i in range(len(batch_samples)):
                target = batch_targets_np[i]
                sample = batch_samples[i]

                same_class_centroid = self.centroids[target]
                min_other_class_dist_centroid = float('inf')
                same_centroid_dist = torch.norm(sample - same_class_centroid).item()

                for cls, centroid in self.centroids.items():
                    if cls != target:
                        dist = torch.norm(sample - centroid).item()
                        if dist < min_other_class_dist_centroid:
                            min_other_class_dist_centroid = dist

                centroid_ratio = same_centroid_dist / min_other_class_dist_centroid
                centroid_ratios[start_idx + i] = centroid_ratio
                same_centroid_dists[start_idx + i] = same_centroid_dist
                other_centroid_dists[start_idx + i] = min_other_class_dist_centroid

                # Proximity metrics
                closest_distance_ratios[start_idx + i] = closest_same_class_distances[start_idx + i] / \
                                                         closest_other_class_distances[start_idx + i]
                avg_distance_ratios[start_idx + i] = avg_same_class_distances[start_idx + i] / \
                                                     avg_other_class_distances[start_idx + i]
                avg_all_class_distances[start_idx + i] = (avg_same_class_distances[start_idx + i] +
                                                          avg_other_class_distances[start_idx + i]) / 2

                # Adapted N3 computation per sample
                adapted_N3[start_idx + i] = int(
                    closest_other_class_distances[start_idx + i] < closest_same_class_distances[start_idx + i])

        return (
            same_centroid_dists, closest_same_class_distances, avg_same_class_distances,  # Type 1
            other_centroid_dists, closest_other_class_distances, avg_other_class_distances, percentage_other_class_knn,
            adapted_N3,  # Type 2
            centroid_ratios, closest_distance_ratios, avg_distance_ratios,  # Type 3
            avg_all_class_distances  # Type 4
        )"""


class ModelBasedMetrics:
    def __init__(self, dataset_name, training, data, labels, ensemble_size):
        self.dataset_name = dataset_name
        self.training = training
        self.data = data.to(u.DEVICE)
        self.labels = labels
        self.ensemble_size = ensemble_size

    def compute_model_based_hardness(self, model_type):
        """Compute hardness metrics (Confident Learning, EL2N, VoG, and Margin) using pretrained models."""

        # Load all pretrained models
        model_paths = glob(f"{u.MODEL_SAVE_DIR}/{self.training}{self.dataset_name}_{model_type}ensemble_*.pth")
        models = []
        for model_path in model_paths:
            model, _ = u.initialize_models(self.dataset_name, model_type)
            model.load_state_dict(torch.load(model_path, map_location=u.DEVICE))
            model.eval()
            models.append(model)
        if self.ensemble_size == 'small':
            models = models[:10] if self.dataset_name == 'CIFAR10' else models[:25]
        else:
            models = models[:25] if self.dataset_name == 'CIFAR10' else models[:100]
        print(f'Extracting hard and easy samples with model-based approaches over {len(models)} models.')

        # Prepare to store results for each sample
        el2n_scores, vog_scores, margin_scores = [], [], []

        # Accumulate predictions and logits for each model
        all_outputs, all_logits = [], []
        for model in tqdm(models):
            with torch.no_grad():
                outputs = model(self.data).cpu().numpy()  # Raw logits (not used for EL2N)
                logits = torch.softmax(model(self.data), dim=1).cpu().numpy()  # Softmax probabilities
                all_outputs.append(outputs)
                all_logits.append(logits)

        # Average predictions and logits across the ensemble
        avg_logits = np.mean(all_logits, axis=0)  # Averaged softmax probabilities

        # Use Cleanlab for Confident Learning Scores
        cl_scores = get_label_quality_scores(
            labels=np.array(self.labels),
            pred_probs=avg_logits,  # Use softmax probabilities
        )

        # Compute EL2N (Error L2-Norm) Scores
        for idx, (logits, label) in enumerate(zip(avg_logits, self.labels)):
            # logits are now the softmax probabilities
            true_label_vec = np.zeros_like(logits)
            true_label_vec[label] = 1
            el2n_scores.append(np.linalg.norm(logits - true_label_vec))  # L2 norm in probability space

        # Compute Margin (Similar to AUM but for a single model)
        for idx, logits in enumerate(avg_logits):
            correct_logit = logits[self.labels[idx]]
            sorted_logits = np.sort(logits)
            if correct_logit == sorted_logits[-1]:
                margin = correct_logit - sorted_logits[-2]  # Margin between correct and second-highest logit
            else:
                margin = correct_logit - sorted_logits[-1]  # Margin between correct and highest logit
            margin_scores.append(margin)

        model_based_hardness_metrics = (cl_scores, el2n_scores, margin_scores)
        # Return all computed metrics
        return model_based_hardness_metrics
