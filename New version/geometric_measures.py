import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

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

    def compute_hessian(self, coords):
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
    def __init__(self, loader: DataLoader):
        """Initialize with the data loader and compute class centroids."""
        self.loader = loader
        self.centroids = self.compute_centroids()

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

    def compute_proximity_ratios(self):
        """Compute the proximity ratios for each sample in the dataset and return them as a flat list."""
        proximity_ratios = []

        for data, targets in self.loader:
            data, targets = data.to(u.DEVICE), targets.to(u.DEVICE)

            for sample, target in zip(data, targets):
                same_class_centroid = self.centroids[target.item()]
                min_other_class_dist = float('inf')

                # Compute distance to the same class centroid
                same_class_dist = torch.norm(sample - same_class_centroid).item()

                # Compute the minimum distance to centroids of other classes
                for cls, centroid in self.centroids.items():
                    if cls != target.item():
                        dist = torch.norm(sample - centroid).item()
                        if dist < min_other_class_dist:
                            min_other_class_dist = dist

                # Compute the proximity ratio
                proximity_ratio = min_other_class_dist / same_class_dist

                # Append the proximity ratio to the flat list
                proximity_ratios.append(proximity_ratio)

        return proximity_ratios
