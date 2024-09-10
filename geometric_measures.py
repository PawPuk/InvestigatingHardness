from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
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
        """Initialize with the data loader, curvatures, and set K for KNN."""
        self.loader = loader
        self.k = k
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
        """Compute proximity metrics for each sample in the dataset, including KNN curvature."""
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

        # Prepare KNN classifier
        flattened_samples = self.samples.view(self.samples.size(0), -1).cpu().numpy()
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(flattened_samples)

        for idx, (sample, target) in tqdm(enumerate(zip(self.samples, self.labels)),
                                          desc='Computing sample-level proximity metrics.'):
            # Flatten the sample for KNN
            sample_np = sample.view(-1).cpu().numpy()
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

            # Compute the KNN for this sample
            distances, indices = knn.kneighbors(sample_np.reshape(1, -1))
            distances = distances.flatten()
            indices = indices.flatten()

            # Exclude the sample itself (index 0) from the KNN distances
            knn_distances = distances[1:]
            knn_labels = self.labels[indices[1:]].cpu().numpy()
            knn_same_class_indices = knn_labels == target.item()
            knn_other_class_indices = knn_labels != target.item()

            # Closest distances to the same and other class samples + ratio
            if np.any(knn_same_class_indices):
                min_same_class_dist = np.min(knn_distances[knn_same_class_indices])
            else:
                min_same_class_dist = np.inf

            if np.any(knn_other_class_indices):
                min_other_class_dist = np.min(knn_distances[knn_same_class_indices])
            else:
                min_other_class_dist = np.inf

            closest_same_class_distances.append(min_same_class_dist)
            closest_other_class_distances.append(min_other_class_dist)
            closest_distance_ratios.append(min_same_class_dist / min_other_class_dist)

            # Average distances to same, other, and all samples in kNN
            avg_same_dist = np.mean(knn_distances[knn_same_class_indices]) if np.any(
                knn_same_class_indices) else np.inf
            avg_other_dist = np.mean(knn_distances[knn_other_class_indices]) if np.any(
                knn_other_class_indices) else np.inf
            avg_all_dist = np.mean(knn_distances)

            avg_same_class_distances.append(avg_same_dist)
            avg_other_class_distances.append(avg_other_dist)
            avg_all_class_distances.append(avg_all_dist)
            avg_distance_ratios.append(avg_same_dist / avg_other_dist)

            # Compute the percentage of kNN samples from same and other classes
            percentage_same_class_knn.append(np.sum(knn_same_class_indices) / self.k)
            percentage_other_class_knn.append(np.sum(knn_other_class_indices) / self.k)

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
            percentage_other_class_knn
        )


class Disjuncts:
    def __init__(self, data, data_indices):
        """Initialize with the data loader and K for custom clustering method."""
        self.data = data.reshape(data.shape[0], -1).cpu().numpy()  # Flatten the images (required for image datasets)
        self.data_indices = data_indices

    def custom_clustering(self, disjunct_metrics):
        """Custom clustering based on path-based method with KNN."""
        # Perform KNN for each sample
        knn = NearestNeighbors(n_neighbors=len(self.data), n_jobs=-1)  # Use all available CPU cores
        knn.fit(self.data)
        distances, _ = knn.kneighbors(self.data)
        # Compute the average distance to use as a threshold
        avg_distance = np.mean(distances[:, 1:])

        # Create an adjacency matrix where edges exist if distance is less than the threshold
        adjacency_matrix = distances[:, 1:] < avg_distance

        # Initialize clusters with disjoint sets
        clusters = defaultdict(list)
        visited = set()

        def dfs(node, cluster_id):
            visited.add(node)
            clusters[cluster_id].append(node)
            for neighbor, is_connected in enumerate(adjacency_matrix[node]):
                if is_connected and neighbor not in visited:
                    dfs(neighbor, cluster_id)

        # A dataset belongs to a cluster if it's within dist < avg_distance of any other sample from the cluster
        cluster_id = 0
        for i in range(len(self.data)):
            if i not in visited:
                dfs(i, cluster_id)
                cluster_id += 1

        # Compute the size of the cluster for each sample
        cluster_sizes = [len(clusters[cluster_id]) for cluster_id in clusters]
        for i in range(len(cluster_sizes)):
            for node in clusters[i]:
                if disjunct_metrics[self.data_indices[node]] is not None:
                    raise Exception
                disjunct_metrics[self.data_indices[node]] = cluster_sizes[i]

    def gmm_clustering(self, disjunct_metrics: List[None], n_components: int):
        """GMM-based clustering."""

        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(self.data)
        cluster_labels = gmm.predict(self.data)

        # Compute the size of the cluster for each sample
        cluster_sizes = np.bincount(cluster_labels)
        for i in range(len(self.data)):
            if disjunct_metrics[self.data_indices[i]] is not None:
                raise Exception
            disjunct_metrics[self.data_indices[i]] = cluster_sizes[cluster_labels[i]]

    def dbscan_clustering(self, disjunct_metrics: List[None], eps: float, min_samples: int):
        """DBSCAN-based clustering."""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(self.data)

        # Compute the size of the cluster for each sample
        cluster_sizes = np.bincount(cluster_labels)
        for i in range(len(self.data)):
            if disjunct_metrics[self.data_indices[i]] is not None:
                raise Exception
            disjunct_metrics[self.data_indices[i]] = cluster_sizes[cluster_labels[i]]

    def compute_disjunct_statistics(self, method: str, disjunct_metrics: List[None], **kwargs):
        """
        Compute disjunct statistics based on the specified method.

        :param method: The method to use ('custom', 'gmm', 'dbscan').
        :param disjunct_metrics: A list that will be populated with the disjunct metrics
        :param kwargs: Additional parameters for the clustering method.
        :return: List of disjunct statistics for each sample.
        """
        if method == 'custom':
            return self.custom_clustering(disjunct_metrics)
        elif method == 'gmm':
            n_components = kwargs.get('n_components', 10)  # Default to 10 components for GMM
            return self.gmm_clustering(disjunct_metrics, n_components)
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)  # Default epsilon for DBSCAN
            min_samples = kwargs.get('min_samples', 5)  # Default minimum samples for DBSCAN
            return self.dbscan_clustering(disjunct_metrics, eps, min_samples)
        else:
            raise ValueError("Invalid method. Choose 'custom', 'gmm', or 'dbscan'.")
