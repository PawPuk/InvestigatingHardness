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
    def __init__(self, data, k=15, pca_components=8):
        # Assuming data is a batch of images, we flatten each image into a 1D vector
        self.data = data.reshape(data.shape[0], -1)  # Flatten the images
        self.k = k
        self.pca_components = pca_components

    def quantify_local_concavity(self):
        nn = NearestNeighbors(n_neighbors=self.k + 1)
        nn.fit(self.data)
        distances, indices = nn.kneighbors(self.data)

        local_curvatures = []
        for i, point_neighbors in enumerate(indices):
            neighbor_points = self.data[point_neighbors]
            local_cov_matrix = np.cov(neighbor_points[:, 1:].T)
            eigenvalues = np.linalg.eigvals(local_cov_matrix)
            curvature = np.mean(eigenvalues)
            local_curvatures.append(curvature)

        return local_curvatures  # Return the list of local curvatures for each data point

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

    def estimate_curvatures(self, curvature_type='both'):
        nn = NearestNeighbors(n_neighbors=self.k + 1)
        nn.fit(self.data)
        distances, indices = nn.kneighbors(self.data)

        gaussian_curvatures = []
        mean_curvatures = []
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
                gaussian_curvatures.append(gaussian_curvature)
                mean_curvatures.append(mean_curvature)
            del point, neighbors, pca, coords, H, eigenvalues, gaussian_curvature, mean_curvature

        if curvature_type == 'gaussian':
            return gaussian_curvatures  # Return list of Gaussian curvatures for each point
        elif curvature_type == 'mean':
            return np.abs(mean_curvatures)  # Return list of mean curvatures for each point
        elif curvature_type == 'both':
            return gaussian_curvatures, np.abs(mean_curvatures)  # Return both curvatures for each point
        else:
            raise ValueError("Invalid curvature_type. Choose 'gaussian', 'mean', or 'both'.")

    def curvatures(self, curvature_type='PCA'):
        if curvature_type == 'PCA':
            return self.quantify_local_concavity()  # Return local concavities for each data point
        elif curvature_type == 'gaussian':
            return self.estimate_curvatures(curvature_type='gaussian')
        elif curvature_type == 'mean':
            return self.estimate_curvatures(curvature_type='mean')
        elif curvature_type == 'both':
            return self.estimate_curvatures(curvature_type='both')
        else:
            raise ValueError("Invalid curvature_type. Choose 'PCA', 'gaussian', 'mean', or 'both'.")


class Proximity:
    def __init__(self, loader: DataLoader, curvatures: List[float], k: int):
        """Initialize with the data loader, curvatures, and set K for KNN."""
        self.loader = loader
        self.curvatures = curvatures
        self.k = k
        self.centroids = self.compute_centroids()
        self.samples, self.labels = self.collect_samples()

    def compute_centroids(self):
        """Compute the centroids for each class."""
        centroids = {}
        class_counts = {}

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

        avg_same_class_curvatures = []
        avg_other_class_curvatures = []
        avg_all_class_curvatures = []

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
            centroid_ratio = min_other_class_dist / same_centroid_dist
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

            # Closest distances to the same and other class samples + ratio
            if np.any(knn_labels == target.item()):
                min_same_class_dist = np.min(knn_distances[knn_labels == target.item()])
            else:
                min_same_class_dist = np.inf

            if np.any(knn_labels != target.item()):
                min_other_class_dist = np.min(knn_distances[knn_labels != target.item()])
            else:
                min_other_class_dist = np.inf

            closest_same_class_distances.append(min_same_class_dist)
            closest_other_class_distances.append(min_other_class_dist)
            closest_distance_ratios.append(min_same_class_dist / min_other_class_dist)

            # Average distances to same, other, and all samples in kNN
            avg_same_dist = np.mean(knn_distances[knn_labels == target.item()]) if np.any(
                knn_labels == target.item()) else np.inf
            avg_other_dist = np.mean(knn_distances[knn_labels != target.item()]) if np.any(
                knn_labels != target.item()) else np.inf
            avg_all_dist = np.mean(knn_distances)

            avg_same_class_distances.append(avg_same_dist)
            avg_other_class_distances.append(avg_other_dist)
            avg_all_class_distances.append(avg_all_dist)
            avg_distance_ratios.append(avg_same_dist / avg_other_dist)

            # Compute the percentage of kNN samples from same and other classes
            same_class_count = np.sum(knn_labels == target.item())
            other_class_count = np.sum(knn_labels != target.item())

            percentage_same_class_knn.append(same_class_count / self.k)
            percentage_other_class_knn.append(other_class_count / self.k)

            # Compute the average curvature of the K-nearest neighbors
            knn_curvatures_all = [self.curvatures[i] for i in indices[1:]]
            knn_curvatures_same = [self.curvatures[i] for i in indices[1:] if knn_labels[i - 1] == target.item()]
            knn_curvatures_other = [self.curvatures[i] for i in indices[1:] if knn_labels[i - 1] != target.item()]

            avg_all_curv = np.mean(knn_curvatures_all)
            avg_same_curv = np.mean(knn_curvatures_same) if knn_curvatures_same else np.inf
            avg_other_curv = np.mean(knn_curvatures_other) if knn_curvatures_other else np.array(0.0)

            avg_all_class_curvatures.append(avg_all_curv)
            avg_same_class_curvatures.append(avg_same_curv)
            avg_other_class_curvatures.append(avg_other_curv)

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
            avg_same_class_curvatures,
            avg_other_class_curvatures,
            avg_all_class_curvatures
        )


class Disjuncts:
    def __init__(self, loader: DataLoader, k: int):
        """Initialize with the data loader and K for custom clustering method."""
        self.loader = loader
        self.k = k
        self.samples, self.labels = self.collect_samples()

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

    def custom_clustering(self):
        """Custom clustering based on path-based method with KNN."""
        flattened_samples = self.samples.view(self.samples.size(0), -1).cpu().numpy()

        # Perform KNN for each sample
        knn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1)  # Use all available CPU cores
        knn.fit(flattened_samples)
        distances, _ = knn.kneighbors(flattened_samples)
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

        cluster_id = 0
        for i in range(len(flattened_samples)):
            if i not in visited:
                dfs(i, cluster_id)
                cluster_id += 1

        # Compute the size of the cluster for each sample
        cluster_sizes = [len(clusters[cluster_id]) for cluster_id in clusters]
        disjunct_statistics = [cluster_sizes[cluster_id] for i in range(len(flattened_samples))]

        return disjunct_statistics

    def gmm_clustering(self, n_components: int):
        """GMM-based clustering."""
        flattened_samples = self.samples.view(self.samples.size(0), -1).cpu().numpy()

        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(flattened_samples)
        cluster_labels = gmm.predict(flattened_samples)

        # Compute the size of the cluster for each sample
        cluster_sizes = np.bincount(cluster_labels)
        disjunct_statistics = [cluster_sizes[label] for label in cluster_labels]

        return disjunct_statistics

    def dbscan_clustering(self, eps: float, min_samples: int):
        """DBSCAN-based clustering."""
        flattened_samples = self.samples.view(self.samples.size(0), -1).cpu().numpy()

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(flattened_samples)

        # Compute the size of the cluster for each sample, ignoring noise (-1)
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)
        cluster_sizes = {label: sum(cluster_labels == label) for label in unique_labels}
        disjunct_statistics = [cluster_sizes.get(label, 0) for label in cluster_labels]

        return disjunct_statistics

    def compute_disjunct_statistics(self, method: str = 'custom', **kwargs):
        """
        Compute disjunct statistics based on the specified method.

        :param method: The method to use ('custom', 'gmm', 'dbscan').
        :param kwargs: Additional parameters for the clustering method.
        :return: List of disjunct statistics for each sample.
        """
        if method == 'custom':
            return self.custom_clustering()
        elif method == 'gmm':
            n_components = kwargs.get('n_components', 10)  # Default to 10 components for GMM
            return self.gmm_clustering(n_components)
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)  # Default epsilon for DBSCAN
            min_samples = kwargs.get('min_samples', 5)  # Default minimum samples for DBSCAN
            return self.dbscan_clustering(eps, min_samples)
        else:
            raise ValueError("Invalid method. Choose 'custom', 'gmm', or 'dbscan'.")
