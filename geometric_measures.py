from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm

import utils as u


class Curvature:
    def __init__(self, data, data_indices, k, pca_components=8):
        self.data = data.reshape(data.shape[0], -1)  # Flatten the images (required for image datasets)
        self.data_indices = data_indices
        self.k = k
        self.pca_components = pca_components
        # TODO: What pca_components do Ma et al. use? Rerun for latent spaces to verify correctness (same results)

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
    def __init__(self, loader: DataLoader, k: int):
        """Initialize with the data loader and set K for KNN."""
        self.loader = loader
        self.k = k
        self.centroids = self.compute_centroids()
        self.samples, self.labels = self.collect_samples()
        # TODO: For CIFAR10 change to greyscale and see if it has positive effect on correlations

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

        same_centroid_dists = []
        other_centroid_dists = []
        centroid_ratios = []

        closest_same_class_distances = []
        closest_other_class_distances = []
        closest_distance_ratios = []

        avg_same_class_distances = []
        avg_other_class_distances = []
        avg_all_class_distances = []
        avg_distance_ratios = []

        percentage_same_class_knn = []
        percentage_other_class_knn = []

        adapted_N3 = []

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

                avg_all_dist = np.mean(knn_dist)

                avg_same_class_distances.append(avg_same_dist)
                avg_other_class_distances.append(avg_other_dist)
                avg_all_class_distances.append(avg_all_dist)
                avg_distance_ratios.append(avg_same_dist / avg_other_dist)

                # Compute the percentage of kNN samples from same and other classes
                num_same_class = knn_same_class_indices.sum()
                num_other_class = knn_other_class_indices.sum()
                if effective_k > 0:
                    percentage_same_class_knn.append(num_same_class / effective_k)
                    percentage_other_class_knn.append(num_other_class / effective_k)
                else:
                    percentage_same_class_knn.append(0)
                    percentage_other_class_knn.append(0)

                # Adapted N3 computation per sample
                if effective_k > 0:
                    nearest_neighbor_label = knn_label[0]
                    n3_different_class = int(nearest_neighbor_label != target)
                else:
                    n3_different_class = 0  # Or np.nan
                adapted_N3.append(n3_different_class)

        return (
            same_centroid_dists,
            other_centroid_dists,
            centroid_ratios,
            closest_same_class_distances,
            closest_other_class_distances,
            closest_distance_ratios,
            avg_same_class_distances,
            avg_other_class_distances,
            avg_all_class_distances,
            avg_distance_ratios,
            percentage_same_class_knn,
            percentage_other_class_knn,
            adapted_N3
        )
