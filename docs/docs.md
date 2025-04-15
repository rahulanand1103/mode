## Overview

MODE (Mixture of Document Experts) is a novel framework designed to enhance Retrieval-Augmented Generation (RAG) by organizing documents into semantically coherent clusters and utilizing centroid-based retrieval. Unlike traditional RAG pipelines that rely on large vector databases and re-rankers, MODE offers a scalable, interpretable, and efficient alternative, particularly suited for specialized or small to medium-sized datasets. It provides two primary classes: ModeIngestion for clustering and data preparation, and ModeInference for efficient semantic search and response generation.

---

# Installation

```bash
pip install mode_rag
```


# Modules


## `ModeIngestion`

### Description

The `ModeIngestion` class clusters text data using HDBSCAN and identifies centroids (central points) within each cluster. The results are persisted for use during inference.

### Class: `ModeIngestion`

```python
mode_rag.ModeIngestion
```

### Parameters

- **chunks** (`List[str]`):  
  A list of text documents or chunks.

- **embedding** (`Union[torch.Tensor, List[List[float]], np.ndarray]`):  
  Embeddings corresponding to the text chunks. Should be of shape `(n_chunks, embedding_dimension)`.

- **min_cluster_size** (`int`, optional):  
  Minimum samples per cluster. Defaults to `10`.

- **max_cluster_size** (`int`, optional):  
  Maximum samples per cluster. Defaults to `30`.

- **persist_directory** (`str`, optional):  
  Directory where clustering results will be saved. If not provided, a unique directory is created using a UUID.

---

### Methods

#### `process_data`

```python
process_data(parallel: bool = True) -> Tuple[Dict[int, List[str]], Dict[int, np.ndarray]]
```

Processes the embeddings: clusters them, computes centroids, and saves results to disk.

##### Parameters

- **parallel** (`bool`, optional):  
  Whether to compute centroids in parallel. Defaults to `True`.

##### Returns

- **Tuple[Dict[int, List[str]], Dict[int, np.ndarray]]**:  
  - Dictionary mapping cluster labels to lists of text chunks.
  - Dictionary mapping cluster labels to centroid embeddings.

---

### Example

```python
# ========================================
# 📄 Sample Code: 
# ========================================
#
# 1. Loading PDF using PyPDFLoader
# 2. Creating chunks using `RecursiveCharacterTextSplitter`
# 3. Embedding with langchain_huggingface
#
# Main inputs to `ModeIngestion` are `chunks` and `embeddings`:

## requirements
# pip install langchain_huggingface==0.1.2
# pip install langchain_community==0.3.4
# pip install pypdf==5.1.0


import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from mode_rag import ModeIngestion, EmbeddingGenerator
import os
import json

## Pdf reader
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://arxiv.org/pdf/1706.03762")
docs = loader.load()

print("downloaded the files")

from langchain.text_splitter import RecursiveCharacterTextSplitter

print("Chunking the pdf:doc")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
chunks = []
for doc in documents:
    chunks.append(doc.page_content)

print("doing embedding")
embed_gen = EmbeddingGenerator()
embeddings = embed_gen.generate_embeddings(chunks)
print("embedding done")
main_processor = ModeIngestion(
    chunks=chunks,
    embedding=embeddings,
    persist_directory="attention",
)
main_processor.process_data(parallel=False)

```

---

## `ModeInference`

### Description

The `ModeInference` class enables querying pre-clustered text data efficiently. It matches a query embedding against pre-computed centroids, then searches within the most relevant clusters to generate a response.

### Class: `ModeInference`

```python
mode_rag.ModeInference
```

### Parameters

- **persist_directory** (`str`):  
  Path to the directory containing clustered texts and centroids saved by `ModeIngestion`.

---

### Methods

#### `invoke`

```python
invoke(
    query: str,
    query_embedding: torch.Tensor,
    prompt: ModelPrompt,
    model_input: dict = {},
    parallel: bool = True,
    top_n_model: int = 1
) -> str
```

Performs a search based on a query and its embedding, retrieving the most relevant information from the clustered data.

##### Parameters

- **query** (`str`):  
  The search query text.

- **query_embedding** (`torch.Tensor`):  
  The embedding of the search query.

- **prompt** (`ModelPrompt`):  
  A prompt object that helps format the model's final output.

- **model_input** (`dict`, optional):  
  Additional parameters for the generation model. This can either be an empty dictionary `{}` or include keys like `temperature`, `top_p`, `max_tokens`, `model`, `stream`, etc.  
  If left empty, a default will be applied internally: `{"model": "openai/gpt-4o"}`.  
  The LLM calls internally use [LiteLLM](https://docs.litellm.ai/docs/#litellm-python-sdk) for flexible model selection and easy API integration. You can refer to the LiteLLM documentation for more details on supported parameters and providers.

- **parallel** (`bool`, optional):  
  Whether to perform computations in parallel. Defaults to `True`.

- **top_n_model** (`int`, optional):  
  Number of top matching results to retrieve. Defaults to `1`.

##### Returns

- **str**:  
  A string response generated from the most relevant search results.

---

### Example

```python
# ========================================
# 📄 Sample Code:
# ========================================
#
# 1. Load clustered data (`ModeInference`).
# 2. Generate query embedding.
# 3. Retrieve context and synthesize response using `ModelPrompt`.

import os
import json
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from mode_rag import (
    EmbeddingGenerator,
    ModeInference,
    ModelPrompt,
)


main_processor = ModeInference(
    persist_directory="attention",
)

print("====start======")
# Create a PromptManager instance

query = "What are the key mathematical operations involved in computing self-attention?"

embed_gen = EmbeddingGenerator()
embedding = embed_gen.generate_embedding(query)

prompts = ModelPrompt(
    ref_sys_prompt="Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just return you don't know.",
    ref_usr_prompt="context: ",
    syn_sys_prompt="You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\nResponses from models:",
    syn_usr_prompt="responses:",
)

response = main_processor.invoke(
    query,
    embedding,
    prompts,
    model_input={"temperature": 0.3, "model": "openai/gpt-4o-mini"},
    top_n_model=2,
)
print(response)

```


---

# License

MIT License. See [LICENSE](https://yourprojectlicenseurl.com) for details.

---

# Contributing

Contributions are welcome!  
Please submit a pull request or open an issue in the GitHub repository.

---

# Author

Developed and maintained by **Rahul Anand**.

