import os
import gzip
import joblib
import torch
from typing import Optional, Tuple
from mode_rag.inference.search import Search
from mode_rag.inference.types import ModelPrompt


class ModeInference:
    """Handles data loading and invoking search based on clustering."""

    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.cluster_dict_text, self.centroids = self.load()

    def load(self) -> Tuple[dict, dict]:
        """Loads processed data from compressed pickle files."""
        cluster_text_path = os.path.join(
            self.persist_directory, "cluster_dict_text.pkl.gz"
        )
        centroids_path = os.path.join(self.persist_directory, "centroids.pkl.gz")

        if not os.path.exists(cluster_text_path):
            raise FileNotFoundError(
                f"Cluster text file not found at: {cluster_text_path}"
            )

        if not os.path.exists(centroids_path):
            raise FileNotFoundError(f"centroids file not found at: {centroids_path}")

        with gzip.open(cluster_text_path, "rb") as f:
            cluster_dict_text = joblib.load(f)

        with gzip.open(centroids_path, "rb") as f:
            centroids = torch.load(f, weights_only=False)

        return cluster_dict_text, centroids

    def invoke(
        self,
        query: str,
        query_embedding: torch.Tensor,
        prompt: ModelPrompt,
        model_input: dict = {},
        parallel: bool = True,
        top_n_model: int = 1,
    ) -> str:
        """Runs the search pipeline using clustering and nearest neighbors."""
        search = Search(self.cluster_dict_text, self.centroids)
        return search.process_query(
            query=query,
            query_embedding=query_embedding,
            prompt=prompt,
            model_input=model_input,
            parallel=parallel,
            top_n_model=top_n_model,
        )
