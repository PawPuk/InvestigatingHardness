from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
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
        self.samples, self.labels = self.collect_samples()
        self.centroids = self.compute_centroids()

    def compute_centroids(self):
        """Compute the centroids for each class using vectorized operations."""
        # Flatten samples
        samples_flat = self.samples.view(self.samples.size(0), -1)
        labels = self.labels

        # Get unique classes
        classes = torch.unique(labels)

        # Initialize centroids dictionary
        centroids = {}

        # Compute centroids vectorized
        for cls in classes:
            class_mask = (labels == cls)
            class_samples = samples_flat[class_mask]
            centroids[cls.item()] = class_samples.mean(dim=0)

        return centroids

    def collect_samples(self):
        """Collect all samples and their corresponding labels from the loader."""
        samples_list, labels_list = [], []
        for data, targets in self.loader:
            samples_list.append(data)
            labels_list.append(targets)
        samples = torch.cat(samples_list).to(u.DEVICE)
        labels = torch.cat(labels_list).to(u.DEVICE)
        return samples, labels

    def compute_proximity_metrics(self):
        """Compute proximity metrics for each sample in the dataset."""
        num_samples = self.samples.size(0)

        # Flatten samples
        samples_flat = self.samples.view(num_samples, -1).cpu().numpy()
        labels_np = self.labels.cpu().numpy()

        # Compute distance matrix
        distance_matrix = squareform(pdist(samples_flat, metric='euclidean'))

        # Prepare centroids
        classes = sorted(self.centroids.keys())
        centroids_list = [self.centroids[cls] for cls in classes]
        centroids_tensor = torch.stack(centroids_list).cpu()  # Shape: (n_classes, feature_dim)

        # Compute distances from all samples to all centroids
        centroids_flat = centroids_tensor.numpy()  # Shape: (n_classes, feature_dim)
        sample_to_centroid_dists = cdist(samples_flat, centroids_flat, metric='euclidean')  # (n_samples, n_classes)

        # Prepare lists for metrics
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

        batch_size = 1000
        # For each sample, compute metrics
        for start_idx in tqdm(range(0, num_samples, batch_size), desc='Computing sample-level proximity metrics.'):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = np.arange(start_idx, end_idx)
            batch_targets = labels_np[batch_indices]

            # Centroid distances for the batch
            batch_centroid_dists = sample_to_centroid_dists[batch_indices]  # Shape: (batch_size, n_classes)

            # Get same and other centroid distances
            target_class_indices = [classes.index(t) for t in batch_targets]
            same_centroid_dists_batch = batch_centroid_dists[np.arange(len(batch_indices)), target_class_indices]
            other_centroid_dists_batch = batch_centroid_dists.copy()
            other_centroid_dists_batch[np.arange(len(batch_indices)), target_class_indices] = np.inf
            min_other_centroid_dists_batch = np.min(other_centroid_dists_batch, axis=1)

            centroid_ratios_batch = same_centroid_dists_batch / min_other_centroid_dists_batch

            same_centroid_dists.extend(same_centroid_dists_batch.tolist())
            other_centroid_dists.extend(min_other_centroid_dists_batch.tolist())
            centroid_ratios.extend(centroid_ratios_batch.tolist())

            # KNN computations for the batch
            batch_distances = distance_matrix[batch_indices]  # Shape: (batch_size, num_samples)
            # Set self-distances to np.inf
            batch_distances[np.arange(len(batch_indices)), batch_indices] = np.inf

            # Get k nearest neighbors
            knn_indices = np.argsort(batch_distances, axis=1)[:, :self.k]
            knn_distances = np.take_along_axis(batch_distances, knn_indices, axis=1)
            knn_labels = labels_np[knn_indices]  # Shape: (batch_size, k)

            # Compute metrics per batch
            knn_same_class = (knn_labels == batch_targets[:, None])
            knn_other_class = ~knn_same_class

            # Closest same class distances
            min_same_class_dist_batch = np.where(knn_same_class, knn_distances, np.inf).min(axis=1)
            min_other_class_dist_batch = np.where(knn_other_class, knn_distances, np.inf).min(axis=1)
            closest_distance_ratios_batch = min_same_class_dist_batch / min_other_class_dist_batch

            closest_same_class_distances.extend(min_same_class_dist_batch.tolist())
            closest_other_class_distances.extend(min_other_class_dist_batch.tolist())
            closest_distance_ratios.extend(closest_distance_ratios_batch.tolist())

            # Average distances
            avg_same_dist_batch = np.nanmean(np.where(knn_same_class, knn_distances, np.nan), axis=1)
            avg_other_dist_batch = np.nanmean(np.where(knn_other_class, knn_distances, np.nan), axis=1)
            avg_all_dist_batch = knn_distances.mean(axis=1)
            avg_distance_ratios_batch = avg_same_dist_batch / avg_other_dist_batch

            avg_same_class_distances.extend(avg_same_dist_batch.tolist())
            avg_other_class_distances.extend(avg_other_dist_batch.tolist())
            avg_all_class_distances.extend(avg_all_dist_batch.tolist())
            avg_distance_ratios.extend(avg_distance_ratios_batch.tolist())

            # Percentage of same and other class in kNN
            percentage_same_class_knn_batch = knn_same_class.sum(axis=1) / self.k
            percentage_other_class_knn_batch = knn_other_class.sum(axis=1) / self.k

            percentage_same_class_knn.extend(percentage_same_class_knn_batch.tolist())
            percentage_other_class_knn.extend(percentage_other_class_knn_batch.tolist())

            # Adapted N3
            nearest_neighbor_labels = knn_labels[:, 0]
            n3_different_class_batch = (nearest_neighbor_labels != batch_targets).astype(int)
            n3_different_class.extend(n3_different_class_batch.tolist())

        # Compute adapted N3 per class
        unique_classes = np.unique(labels_np)
        adapted_N3 = {}

        labels_array = labels_np
        n3_array = np.array(n3_different_class)

        for cls in unique_classes:
            class_indices = np.where(labels_array == cls)[0]
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
