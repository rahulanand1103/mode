from langchain_huggingface import HuggingFaceEmbeddings
import torch
from torch import Tensor
from typing import List


class EmbeddingGenerator:
    """Generates embeddings for text using HuggingFace models."""

    def __init__(self, model_name: str = "avsolatorio/GIST-large-Embedding-v0"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def generate_embeddings(self, texts: List[str]) -> Tensor:
        """Generates embeddings for a list of texts."""
        embedding_list = [self.embeddings.embed_query(text) for text in texts]
        return torch.tensor(embedding_list, dtype=torch.float32)

    def generate_embedding(self, text: str) -> Tensor:
        """Generates a single embedding for a given text."""
        return torch.tensor(self.embeddings.embed_query(text), dtype=torch.float32)
