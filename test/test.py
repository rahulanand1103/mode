"""
TO RUN
import os
export OPENAI_API_KEY=""
pytest test/test.py
"""

import os
import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mode_rag import ModeIngestion, ModeInference, EmbeddingGenerator, ModelPrompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PDF_URL = "https://arxiv.org/pdf/1706.03762"
PERSIST_DIR = "attention"
TEST_QUERY = (
    "What are the key mathematical operations involved in computing self-attention?"
)


@pytest.fixture(scope="module")
def chunks():
    # Load and split the PDF
    loader = PyPDFLoader(PDF_URL)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    return [doc.page_content for doc in split_docs]


@pytest.fixture(scope="module")
def embeddings(chunks):
    embed_gen = EmbeddingGenerator()
    return embed_gen.generate_embeddings(chunks)


@pytest.fixture(scope="module")
def embedding_for_query():
    embed_gen = EmbeddingGenerator()
    return embed_gen.generate_embedding(TEST_QUERY)


@pytest.fixture(scope="module")
def prompts():
    return ModelPrompt(
        ref_sys_prompt="Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just return you don't know.",
        ref_usr_prompt="context: ",
        syn_sys_prompt="You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\nResponses from models:",
        syn_usr_prompt="responses:",
    )


# ----------- INGESTION TEST -----------


def test_ingestion(chunks, embeddings):
    ingestion = ModeIngestion(
        chunks=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    ingestion.process_data(parallel=False)
    assert os.path.exists(PERSIST_DIR)
    assert len(os.listdir(PERSIST_DIR)) > 0


# ----------- INFERENCE TEST -----------


def test_inference(embedding_for_query, prompts):
    inference = ModeInference(persist_directory=PERSIST_DIR)
    response = inference.invoke(
        TEST_QUERY,
        embedding_for_query,
        prompts,
        model_input={"temperature": 0.3, "model": "openai/gpt-4o-mini"},
        top_n_model=2,
    )
    print("Inference Response:", response)
    assert isinstance(response, str)
    assert len(response.strip()) > 0
