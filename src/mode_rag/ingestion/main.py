from mode_rag.ingestion.cluster import HDBSCANClusterer
from mode_rag.ingestion.centroid import GenerateCentroid
from typing import List, Union, Dict

import os
import uuid
import torch
import joblib
import gzip
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class ModeIngestion:
    """
    Handles the ingestion, clustering, and centroid computation of text embeddings.
    """

    def __init__(
        self,
        chunks: list,
        embedding: Union[torch.Tensor, List[List[float]], np.ndarray],
        min_cluster_size: int = 10,
        max_cluster_size: int = 30,
        persist_directory: str = None,
    ):
        """
        Initializes the ingestion pipeline.

        :param chunks: List of text data.
        :param embedding: Embeddings as tensor, list of lists, or numpy array.
        :param min_cluster_size: Minimum cluster size.
        :param max_cluster_size: Maximum cluster size.
        :param persist_directory: Directory to save clustering results.
        """
        self.chunks = chunks
        self.embedding = self._prepare_embeddings(embedding)
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.persist_directory = persist_directory or str(uuid.uuid4()).replace("-", "")
        os.makedirs(self.persist_directory, exist_ok=True)

    def _save_data(
        self, data: Union[Dict, object], filename: str, use_torch: bool = False
    ):
        """
        Saves data to disk in compressed format.

        :param data: Data to save.
        :param filename: Filename for storage.
        :param use_torch: Whether to use torch serialization.
        """
        file_path = os.path.join(self.persist_directory, filename)
        with gzip.open(file_path, "wb") as f:
            if use_torch and isinstance(data, dict):
                torch.save(data, f)
            else:
                joblib.dump(data, f)
        print(f"Centroid and cluster saved here: {file_path}")

    def _prepare_embeddings(
        self, embeddings: Union[torch.Tensor, List[List[float]], np.ndarray]
    ):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings

        elif isinstance(embeddings, list):
            # Check if it's a list of lists
            if isinstance(embeddings[0], list) or isinstance(
                embeddings[0], torch.Tensor
            ):
                embeddings = torch.tensor(embeddings, dtype=torch.float32)
            else:
                raise TypeError(
                    "Expected a list of lists or list of tensors inside embeddings."
                )

        elif isinstance(embeddings, np.ndarray):
            # Good, do nothing
            pass

        else:
            raise TypeError(f"Unsupported embeddings type: {type(embeddings)}")

        return embeddings

    def process_data(self, parallel: bool = True):
        """
        Processes embeddings: clustering, centroid computation, and data storage.

        :param parallel: Whether to compute centroids in parallel.
        :return: Tuple of (clustered texts dictionary, centroids dictionary).
        """
        clusterer = HDBSCANClusterer(
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
        )
        labels = clusterer.cluster(self.embedding)

        cluster_dict = {}
        cluster_dict_text = {}

        for text, label, embedding in zip(self.chunks, labels, self.embedding):
            cluster_dict.setdefault(label, []).append(embedding)
            cluster_dict_text.setdefault(label, []).append(text)

        if parallel:
            with ThreadPoolExecutor() as executor:
                centroids = dict(
                    zip(
                        cluster_dict.keys(),
                        executor.map(
                            lambda key: GenerateCentroid.compute_centroid(
                                cluster_dict[key]
                            ),
                            cluster_dict.keys(),
                        ),
                    )
                )
        else:
            centroids = {
                key: GenerateCentroid.compute_centroid(tensors)
                for key, tensors in cluster_dict.items()
            }

        self._save_data(cluster_dict_text, "cluster_dict_text.pkl.gz")
        self._save_data(centroids, "centroids.pkl.gz", use_torch=True)

        return cluster_dict_text, centroids
