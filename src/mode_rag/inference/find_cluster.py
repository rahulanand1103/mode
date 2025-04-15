import torch
import torch.nn.functional as F
from typing import Dict, List


class CentroidFinder:
    """Finds the closest centroid to a given embedding."""

    @staticmethod
    def find_best_centroids(
        centroids: Dict[int, torch.Tensor],
        query_embedding: torch.Tensor,
        top_n_model: int = 1,
    ) -> List[int]:
        """
        Identifies the top-N closest centroids using cosine similarity.

        Args:
            centroids (Dict[int, Tensor]): Dictionary of centroid keys and their tensors.
            query_embedding (Tensor): The query embedding to compare.
            top_n_model (int): Number of closest centroids to retrieve.

        Returns:
            List[int]: List of the top-N closest centroid keys.
        """
        centroid_keys = list(centroids.keys())
        centroid_tensors = torch.stack(list(centroids.values()))
        centroid_tensors = F.normalize(centroid_tensors, p=2, dim=1)
        query_embedding = F.normalize(query_embedding, p=2, dim=0)

        similarities = torch.matmul(centroid_tensors, query_embedding)
        top_k_indices = torch.topk(similarities, top_n_model).indices

        return [int(centroid_keys[i]) for i in top_k_indices]
