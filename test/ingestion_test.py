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
