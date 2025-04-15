import torch
from typing import List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from mode_rag.inference.find_cluster import CentroidFinder
from mode_rag.inference.model import LiteLLMModel
from mode_rag.inference.types import ModelPrompt, GenerationInput


class Search:
    """Processes queries using centroid-based cluster retrieval and local search."""

    def __init__(self, cluster_dict_text: dict, centroids: dict):
        """
        Initializes the search process.

        Args:
            cluster_dict_text (dict): Dictionary containing text clusters.
            centroids (dict): Dictionary containing cluster centroid embeddings.
        """
        self.cluster_dict_text = cluster_dict_text
        self.centroids = centroids

    def find_best_centroids(
        self, query_embedding: torch.Tensor, top_n_model: int
    ) -> List[int]:
        """
        Finds the closest centroids for the given embedding.

        Args:
            embedding (Tensor): The input embedding.
            top_n_model (int): Number of closest centroids to retrieve.

        Returns:
            List[int]: List of closest centroid keys.
        """
        return CentroidFinder.find_best_centroids(
            self.centroids, query_embedding, top_n_model
        )

    def process_cluster(
        self,
        query: str,
        query_embedding: torch.Tensor,
        prompt: ModelPrompt,
        model_input: dict,
        cluster_index: int,
    ) -> str:
        """
        Processes a single cluster by retrieving relevant texts and generating a response.

        Args:
            query (str): User query.
            query_embedding (Tensor): Query embedding.
            prompt (ModelPrompt): Prompt object.
            cluster_index (int): Cluster index to process.

        Returns:
            str: Model-generated response.
        """
        retrieved_texts = self.cluster_dict_text[cluster_index]
        retrieved_texts = "\n".join(
            f"context {i+1}: {text}"
            for i, text in enumerate(self.cluster_dict_text[cluster_index])
        )

        messages = [
            {"role": "system", "content": f"{prompt.ref_sys_prompt} {retrieved_texts}"},
            {"role": "user", "content": f"{prompt.ref_usr_prompt}{query}"},
        ]

        model_parameters = GenerationInput(**model_input)
        return LiteLLMModel().generate(messages=messages, **model_parameters.dict())

    def process_query(
        self,
        query: str,
        query_embedding: torch.Tensor,
        prompt: ModelPrompt,
        model_input: dict,
        parallel: bool = True,
        top_n_model: int = 1,
    ) -> str:
        """
        Processes a query using the closest centroid clusters and retrieves relevant responses.

        Args:
            query (str): User query.
            query_embedding (Tensor): Query embedding.
            prompt (ModelPrompt): Prompt object.
            model_input (dict): LLM model parameters.
            parallel (bool): Whether to process clusters in parallel.
            top_n_model (int): Number of clusters to consider.

        Returns:
            str: Final synthesized response.
        """
        best_centroid_keys = self.find_best_centroids(query_embedding, top_n_model)
        model_responses = []

        if parallel:
            with ThreadPoolExecutor() as executor:
                results = list(
                    tqdm(
                        executor.map(
                            lambda key: self.process_cluster(
                                query, query_embedding, prompt, model_input, key
                            ),
                            best_centroid_keys,
                        ),
                        total=len(best_centroid_keys),
                    )
                )
                model_responses.extend(results)
        else:
            for cluster_index in tqdm(best_centroid_keys):
                model_responses.append(
                    self.process_cluster(
                        query, query_embedding, prompt, model_input, cluster_index
                    )
                )

        if top_n_model == 1:
            return model_responses[0]

        outputs = "\n".join(
            f"Response {i}: {resp}" for i, resp in enumerate(model_responses, start=1)
        )

        messages = [
            {"role": "system", "content": prompt.syn_sys_prompt},
            {
                "role": "user",
                "content": f"{prompt.syn_usr_prompt.format(query=query)} {outputs}",
            },
        ]
        model_parameters = GenerationInput(**model_input)
        return LiteLLMModel().generate(messages=messages, **model_parameters.dict())
