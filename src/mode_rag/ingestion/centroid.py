import torch
from typing import List


class GenerateCentroid:
    """Computes the centroid of each cluster."""

    @staticmethod
    def compute_centroid(tensor_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the centroid of a list of tensors by averaging them.
        :param tensor_list: List of tensors representing a cluster.
        :return: The centroid tensor.
        """
        # Simply stack all tensors and compute the mean along the first dimension
        tensor_stack = torch.stack(tensor_list)
        centroid = torch.mean(tensor_stack, dim=0)
        return centroid
