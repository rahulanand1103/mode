# MODE: Mixture of Document Experts for RAG

## Project Overview
MODE (Mixture of Document Experts) is an advanced framework that improves Retrieval-Augmented Generation (RAG) by integrating external knowledge retrieval with a mixture of specialized expert models.

Key features of MODE include:
* Hierarchical Clustering: Organizes documents into semantically meaningful clusters.
* Expert Models: Assigns specialized models to different document clusters for targeted expertise.
* Centroid-Based Retrieval: Selects representative documents efficiently to enhance retrieval relevance.

By combining these techniques, MODE delivers more accurate document retrieval and synthesis for query-based applications, improving answer quality while reducing retrieval noise. MODE is particularly well-suited for small to medium-sized document collections or datasets.

📄 arxiv: [https://mode-rag.readthedocs.io/en/latest/](https://arxiv.org/abs/2509.00100) <br>
📄 Docs: https://mode-rag.readthedocs.io/en/latest/ <br>
🌐 Website: https://mode-rag.netlify.app/ <br>


## 📂 Project Structure
```
.
├── benchmarking              # Evaluation and benchmarking scripts
│   ├── data.py
│   ├── eval
│   │   ├── db
│   │   └── logs
│   │       ├── ours
│   │       └── traditional_rag
│   ├── evaluate.py
│   ├── metric_to_json.py
│   ├── mode.py
│   └── traditional_rag.py
├── README.md
├── requirements.txt     
├── src                         
│   ├── __init__.py
│   ├── inference                # inference (retrieval + generation) 
│   │   ├── __init__.py
│   │   ├── find_cluster.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── search.py
│   │   └── types.py
│   ├── ingestion                # Data ingestion & clustering         
│   │   ├── __init__.py
│   │   ├── centroid.py
│   │   ├── cluster.py
│   │   └── main.py
│   └── utils                    # (chunking, embeddings, data loading)
│       ├── __init__.py
│       ├── chunker.py
│       ├── data.py
│       └── embedding.py
└── test
    ├── inference_test.py
    ├── ingestion_test.py
    └── test.py
```

## Quick start
### Installation
```bash
git clone https://github.com/rahulanand1103/mode.git
cd mode
pip install -r requirements.txt
```

```bash
pip install mode_rag
```

```python
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### 1. Ingestion Code

This is a sample using `RecursiveCharacterTextSplitter` and `EmbeddingGenerator`.
You can use your **own chunking/embedding** logic.
Main inputs to `ModeIngestion` are `chunks` and `embeddings`:

```python
# ========================================
# 📄 Sample Code: 
# ========================================
#
# 1. Loading pdf using PyPDFLoader
# 2. create chunking using `RecursiveCharacterTextSplitter`.
# 3. for embedding we are using langchain_huggingface.
# This is a sample using `RecursiveCharacterTextSplitter` and `EmbeddingGenerator`.
# You can use your **own chunking/embedding** logic.
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


### 2. Inference Code


This is a sample using `ModeInference` and `EmbeddingGenerator`.
You can use your **own embedding** method.
Main inputs to `ModeInference.invoke` are `query`, `query_embedding`, and `prompts`:

```python
# ========================================
# 📄 Sample Code:
# ========================================
#
# 1. Load clustered data (`ModeInference`).
# 2. Generate query embedding (replaceable with your `embedding.py`).
# 3. Retrieve context and synthesize response with `ModelPrompt`.

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


### 🧪 Running Sample Scripts (Same as Above)

The **same ingestion and inference logic** is provided as ready-to-run test scripts inside the `test/` folder.

You can quickly test MODE without writing any code!

#### Run Ingestion Test

```bash
cd test
python ingestion_test.py
```


#### Run Inference Test

```bash
cd test
python inference_test.py
```


---

> **Note:**  
> These scripts (`test/ingestion_test.py` and `test/inference_test.py`) use the same examples shown above.


---

## Benchmarking

Run experiments on different datasets using `mode.py` and `traditional_rag.py`.



### Setup

```bash
cd benchmarking
pip install -r bench-requirements.txt
```

### Run Benchmarks

#### HotpotQA

**Mode:**

```bash
python mode.py --dataset hotpotqa --chunks 100 --num_questions 100 --top_n_model 2
```

**Traditional RAG:**

```bash
python traditional_rag.py --dataset hotpotqa --chunks 100 --num_questions 100
```


#### SQuAD

**Mode:**

```bash
python mode.py --dataset squad --chunks 100 --num_questions 100 --top_n_model 2
```

**Traditional RAG:**

```bash
python traditional_rag.py --dataset squad --chunks 100 --num_questions 100
```

**Notes**:
- `--dataset` must be either `hotpotqa` or `squad`.
- `--top_n_model` is **only used in `mode.py`**.
- Customize `--chunks` and `--num_questions` as needed.

### 📊 Benchmark Results
[View Logs](https://github.com/rahulanand1103/mode/tree/main/benchmarking/eval/logs/ours)

#### MODE
| Dataset  | No. Chunk | No. Question | Top n Model | GPT Accuracy | GPT F1 Score | BERT Precision | BERT Recall | BERT F1 Score |
|:---------|:---------:|:------------:|:-----:|:------------:|:------------:|:--------------:|:-----------:|:-------------:|
| HotpotQA | 100       | 100          | 1     | 0.80         | 0.8889       | 0.8059         | 0.8276      | 0.8154        |
| HotpotQA | 100       | 100          | 2     | 0.70         | 0.8235       | 0.7427         | 0.7612      | 0.7493        |
| HotpotQA | 200       | 100          | 1     | 0.75         | 0.8571       | 0.8048         | 0.7582      | 0.7745        |
| HotpotQA | 200       | 100          | 2     | 0.80         | 0.8889       | 0.7746         | 0.7910      | 0.7811        |
| HotpotQA | 500       | 100          | 1     | 0.7843       | 0.8791       | 0.7777         | 0.7581      | 0.7613        |
| HotpotQA | 500       | 100          | 2     | 0.8039       | 0.8913       | 0.7208         | 0.7507      | 0.7320        |
| SQuAD    | 100       | 100          | 1     | 0.78         | 0.8764       | 0.7881         | 0.7939      | 0.7852        |
| SQuAD    | 100       | 100          | 2     | 0.89         | 0.9418       | 0.7805         | 0.8241      | 0.7993        |
| SQuAD    | 200       | 100          | 1     | 0.72         | 0.8372       | 0.7449         | 0.7380      | 0.7336        |
| SQuAD    | 200       | 100          | 2     | 0.78         | 0.8764       | 0.7429         | 0.7828      | 0.7595        |
| SQuAD    | 500       | 100          | 1     | 0.71         | 0.8304       | 0.7495         | 0.7473      | 0.7408        |
| SQuAD    | 500       | 100          | 2     | 0.82         | 0.9011       | 0.7660         | 0.8047      | 0.7825        |

#### Traditional RAG
[View Logs](https://github.com/rahulanand1103/mode/tree/main/benchmarking/eval/logs/traditional_rag)


| Dataset  | No. Chunks | GPT Accuracy | GPT F1 Score | BERT Precision | BERT F1 Score |
|:---------|:----------:|:------------:|:------------:|:--------------:|:-------------:|
| HotpotQA | 100        | 0.70         | 0.82         | 0.23           | 0.29          |
| HotpotQA | 200        | 0.70         | 0.82         | 0.37           | 0.40          |
| HotpotQA | 500        | 0.72         | 0.84         | 0.25           | 0.29          |
| SQuAD    | 100        | 0.88         | 0.94         | 0.46           | 0.51          |
| SQuAD    | 200        | 0.87         | 0.93         | 0.46           | 0.51          |
| SQuAD    | 500        | 0.86         | 0.92         | 0.46           | 0.51          |


## Contributing

We welcome contributions! Here’s how you can help:

- **Report Bugs:** Submit issues on GitHub.
- **Suggest Features:**  Open an issue with your ideas.
- **Code Contributions:** Fork, make changes, and submit a pull request.
- **Documentation:** Update and enhance our docs.


## License

This project is licensed under the [MIT License](LICENSE).
