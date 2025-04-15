from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import json
import os
from typing import List
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_openai import ChatOpenAI
from data import get_dataset_loader
from evaluate import Evaluator
import sys
import argparse
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Ingestion:
    def __init__(self, chunks: List[str], persist_directory: str, collection_name: str):
        self.chunks = chunks
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="avsolatorio/GIST-large-Embedding-v0"
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def load_documents(self) -> List[Document]:
        return [Document(page_content=chunk) for chunk in self.chunks]

    def ingest(self):
        documents = self.load_documents()
        self.vector_store.add_documents(documents=documents)
        print(f"Ingestion complete for collection: {self.collection_name}")


class Inference:
    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="avsolatorio/GIST-large-Embedding-v0"
        )
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def retrieve_and_answer(self, query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=""" You are a helpful assistant.
                    Use the following context to answer the question.
                    If you don't know the answer, just say "I don't know".
                    Answer concisely with a maximum of 8 words.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer: """,
        )

        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=compression_retriever,
            chain_type_kwargs={"prompt": prompt},
        )

        return chain.invoke(query)["result"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate QA Model on Different Datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["squad", "hotpotqa"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--chunks", type=int, default=100, help="Number of chunks to load"
    )
    parser.add_argument(
        "--num_questions", type=int, default=100, help="Number of test questions"
    )
    args = parser.parse_args()

    log_file_name = f"eval/logs/traditional_rag/D-{args.dataset}_C-{args.chunks}_Q-{args.num_questions}.txt"

    # Load data
    data_loader = get_dataset_loader(args.dataset, sample_size=args.chunks)
    chunks = data_loader.chunks  # Get unique chunks
    eval_dt = data_loader.evaluate_qa_pair(args.num_questions)  # Get test QA pairs

    print(f"\nDataset: {args.dataset}")
    print(f"Number of Chunks: {len(chunks)}")
    print("Number of eval questions:", len(eval_dt))

    persist_directory = f"eval/db/traditional_rag/D-{args.dataset}_C-{args.chunks}_Q-{args.num_questions}"
    collection_name = "my_rag"

    # Ingest data
    ingestion = Ingestion(chunks, persist_directory, collection_name)
    ingestion.ingest()

    # Initialize inference and evaluator
    inference = Inference(persist_directory, collection_name)
    evaluator = Evaluator()

    # Metrics lists
    exact_matches = []
    f1_scores = []
    bert_precision = []
    bert_recall = []
    bert_f1 = []
    accuracy_gpt = []

    # Logging
    os.makedirs("eval", exist_ok=True)
    log_file = open(log_file_name, "w")

    # Evaluation loop
    for i, sample in enumerate(eval_dt):
        query = sample["question"]
        ground_truth = sample["answer"]

        generated_response = inference.retrieve_and_answer(query)

        log_file.write(f"Sample Number: {i + 1}\n")
        log_file.write(f"Question: {query}\n")
        log_file.write(f"Ground Truth: {ground_truth}\n")
        log_file.write(f"Generated Response: {generated_response}\n")

        # Exact Match
        normalized_gt = evaluator.normalize_answer(ground_truth)
        normalized_response = evaluator.normalize_answer(generated_response)
        exact_match = 1.0 if normalized_gt == normalized_response else 0.0
        exact_matches.append(exact_match)
        log_file.write(f"Exact Match: {exact_match}\n")

        # F1 Score
        f1 = evaluator.calculate_f1(ground_truth, generated_response)
        f1_scores.append(f1)
        log_file.write(f"F1 Score: {f1}\n")

        # BERTScore
        p, r, f1_bert = evaluator.compute_bertscore(
            [generated_response], [ground_truth]
        )
        bert_precision.append(p)
        bert_recall.append(r)
        bert_f1.append(f1_bert)
        log_file.write(f"BERTScore (P, R, F1): {p:.4f}, {r:.4f}, {f1_bert:.4f}\n")

        # GPT-based Accuracy Check
        gpt_accuracy = evaluator.check_accuracy_gpt(
            query, ground_truth, generated_response
        )
        accuracy_gpt.append(gpt_accuracy)
        log_file.write(f"GPT Accuracy Check: {gpt_accuracy}\n")

        log_file.write("-" * 50 + "\n")

    # Compute average metrics
    average_metrics = evaluator.compute_averages(
        exact_matches, f1_scores, bert_precision, bert_recall, bert_f1, accuracy_gpt
    )

    log_file.write("\n" + "=" * 50 + " Overall Evaluation Results " + "=" * 50 + "\n")
    for metric, value in average_metrics.items():
        log_file.write(f"{metric}: {value:.4f}\n")
    log_file.write("=" * 104 + "\n")

    log_file.close()

    print("\n" + "=" * 50 + " Evaluation Results " + "=" * 50)
    for metric, value in average_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * 104)
    print(f"Evaluation logs saved to: {log_file_name}")
