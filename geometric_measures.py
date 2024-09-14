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
        """Compute proximity metrics for each sample, including adapted N3, processing data in batches."""
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

        n3_different_class = []

        # Prepare KNN classifier
        flattened_samples = self.samples.view(self.samples.size(0), -1).cpu().numpy()
        labels_np = self.labels.cpu().numpy()
        knn = NearestNeighbors(n_neighbors=self.k + 1)  # +1 to include the sample itself
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
            distances, indices = knn.kneighbors(batch_samples_flat, n_neighbors=self.k + 1)
            distances = distances[:, 1:]  # Exclude the sample itself
            indices = indices[:, 1:]

            knn_distances = distances
            knn_labels = labels_np[indices]

            knn_same_class_indices = knn_labels == batch_targets_np[:, None]
            knn_other_class_indices = ~knn_same_class_indices

            # Closest distances to the same and other class samples + ratio
            knn_distances_same_class = np.where(knn_same_class_indices, knn_distances, np.inf)
            min_same_class_dist_batch = knn_distances_same_class.min(axis=1)

            knn_distances_other_class = np.where(knn_other_class_indices, knn_distances, np.inf)
            min_other_class_dist_batch = knn_distances_other_class.min(axis=1)

            closest_distance_ratios_batch = min_same_class_dist_batch / min_other_class_dist_batch

            closest_same_class_distances.extend(min_same_class_dist_batch.tolist())
            closest_other_class_distances.extend(min_other_class_dist_batch.tolist())
            closest_distance_ratios.extend(closest_distance_ratios_batch.tolist())

            # Average distances to same, other, and all samples in kNN
            avg_same_dist_batch = np.nanmean(np.where(knn_same_class_indices, knn_distances, np.nan), axis=1)
            avg_other_dist_batch = np.nanmean(np.where(knn_other_class_indices, knn_distances, np.nan), axis=1)
            avg_all_dist_batch = knn_distances.mean(axis=1)
            avg_distance_ratios_batch = avg_same_dist_batch / avg_other_dist_batch

            avg_same_class_distances.extend(avg_same_dist_batch.tolist())
            avg_other_class_distances.extend(avg_other_dist_batch.tolist())
            avg_all_class_distances.extend(avg_all_dist_batch.tolist())
            avg_distance_ratios.extend(avg_distance_ratios_batch.tolist())

            # Compute the percentage of kNN samples from same and other classes
            percentage_same_class_knn_batch = knn_same_class_indices.sum(axis=1) / self.k
            percentage_other_class_knn_batch = knn_other_class_indices.sum(axis=1) / self.k

            percentage_same_class_knn.extend(percentage_same_class_knn_batch.tolist())
            percentage_other_class_knn.extend(percentage_other_class_knn_batch.tolist())

            # Adapted N3 computation
            nearest_neighbor_labels = knn_labels[:, 0]
            n3_different_class_batch = (nearest_neighbor_labels != batch_targets_np).astype(int)
            n3_different_class.extend(n3_different_class_batch.tolist())

        # Compute adapted N3 per class
        unique_classes = np.unique(labels_np)
        adapted_N3 = {}

        n3_array = np.array(n3_different_class)

        for cls in unique_classes:
            class_indices = np.where(labels_np == cls)[0]
            n3_class = n3_array[class_indices]
            adapted_N3[cls] = n3_class.mean()

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
