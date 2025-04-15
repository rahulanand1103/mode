import hdbscan
import torch
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from typing import Union


class HDBSCANClusterer:
    """
    Clusters embeddings using HDBSCAN and applies KMeans splitting for large clusters.
    """

    def __init__(
        self, min_cluster_size: int, max_cluster_size: int, metric: str = "euclidean"
    ):
        """
        Initializes the HDBSCAN clusterer.

        :param min_cluster_size: Minimum cluster size for HDBSCAN.
        :param max_cluster_size: Maximum cluster size before applying KMeans splitting.
        :param metric: Distance metric for clustering (default: "euclidean").
        """
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method="eom",
            allow_single_cluster=False,
        )
        self.max_cluster_size = max_cluster_size

    def cluster(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Clusters the given embeddings using HDBSCAN.

        :param embeddings: Torch tensor of shape (N, D) containing embeddings.
        :return: Array of cluster labels.
        """

        labels = self.clusterer.fit_predict(embeddings.cpu().numpy())

        # Normalize labels if there are negative values
        min_label = min(labels)
        if min_label < 0:
            labels = labels - min_label

        # Handle large clusters
        cluster_counts = Counter(labels)
        for cluster, count in cluster_counts.items():
            if count > self.max_cluster_size:
                labels = self._split_large_cluster(
                    embeddings.cpu().numpy(), labels, cluster
                )

        return labels

    def _split_large_cluster(
        self, embeddings: np.ndarray, labels: np.ndarray, cluster_id: int
    ) -> np.ndarray:
        """
        Splits a large cluster using KMeans.

        :param embeddings: Numpy array of shape (N, D) containing embeddings.
        :param labels: List of cluster labels.
        :param cluster: Cluster ID to split.
        :return: Updated list of cluster labels.
        """
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]

        num_subclusters = len(cluster_embeddings) // self.max_cluster_size + 1

        if num_subclusters > 1:
            kmeans = KMeans(n_clusters=num_subclusters, random_state=42, n_init="auto")
            new_labels = kmeans.fit_predict(cluster_embeddings)

            new_cluster_id = labels.max() + 1
            for idx, cluster_idx in enumerate(cluster_indices):
                labels[cluster_idx] = new_cluster_id + new_labels[idx]
        return labels
