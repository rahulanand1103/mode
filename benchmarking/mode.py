import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from evaluate import Evaluator
from data import HotpotQADataLoader, SquadDataLoader

from mode_rag import ModeIngestion, EmbeddingGenerator, ModeInference, ModelPrompt
from data import get_dataset_loader


class ModeIng:
    """Handles embedding generation and ingestion into vector store."""

    def __init__(self, chunks, persist_directory):
        self.embed_gen = EmbeddingGenerator()
        self.persist_directory = persist_directory
        self.chunks = chunks  # Store chunks from SquadDataLoader
        print("Generating embeddings...")
        self.embeddings = self.create_embeddings()  # Generate embeddings

    def create_embeddings(self):
        """Generates embeddings for chunks."""
        return self.embed_gen.generate_embeddings(self.chunks)

    def ingest_data(self):
        """Ingests data into the vector store."""
        main_processor = ModeIngestion(
            chunks=self.chunks,
            max_cluster_size=20,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        main_processor.process_data(parallel=False)


class EvalInference:
    def __init__(self, persist_directory, top_n_model=1):
        self.embed_gen = EmbeddingGenerator()
        self.prompts = ModelPrompt(
            ref_sys_prompt="Use the provided context to give a answer to the question. If the answer cannot be determined from the context, respond with 'None'.Answer should be 1-3 sentences long",
            ref_usr_prompt="Context: ",
            syn_sys_prompt="Synthesize the provided responses to generate a answer to the query. Ignore any responses that are None.Answer should be 1-3 sentences long",
            syn_usr_prompt="query: {query}\nResponses:",
        )
        self.inference = ModeInference(persist_directory=persist_directory)
        self.top_n_model = top_n_model

    def generate_response(self, query):
        embedding = self.embed_gen.generate_embedding(query)
        response = self.inference.invoke(
            query, embedding, self.prompts, top_n_model=self.top_n_model
        )
        return response


if __name__ == "__main__":
    import argparse

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
    parser.add_argument(
        "--top_n_model", type=int, default=1, help="Number of Experts you want to use"
    )

    args = parser.parse_args()

    log_file_name = f"eval/logs/ours/D-{args.dataset}_C-{args.chunks}_Q-{args.num_questions}_M-{args.top_n_model}.txt"

    ### Getting the data
    data_loader = get_dataset_loader(args.dataset, sample_size=args.chunks)
    chunks = data_loader.chunks  # Get unique chunks
    eval_dt = data_loader.evaluate_qa_pair(args.num_questions)  # Get test QA pairs

    print(f"\nDataset: {args.dataset}")
    print(f"Number of Chunks: {len(chunks)}")
    print("Number of eval question:", len(eval_dt))

    ingestion = ModeIng(
        chunks,
        f"eval/db/ours/D-{args.dataset}_C-{args.chunks}_Q-{args.num_questions}_M-{args.top_n_model}",
    )
    ingestion.ingest_data()

    evaluator = Evaluator()
    eval_infer = EvalInference(
        f"eval/db/ours/D-{args.dataset}_C-{args.chunks}_Q-{args.num_questions}_M-{args.top_n_model}",
        top_n_model=args.top_n_model,
    )

    exact_matches = []
    f1_scores = []
    bert_precision = []
    bert_recall = []
    bert_f1 = []
    accuracy_gpt = []

    # Open file for logging
    log_file = open(log_file_name, "w")  # Open file in write mode

    for i, sample in enumerate(eval_dt):
        query = sample["question"]
        ground_truth = sample["answer"]

        generated_response = eval_infer.generate_response(query)

        log_file.write(f"Sample Number: {i + 1}\n")
        log_file.write(f"Question: {query}\n")
        log_file.write(f"Ground Truth: {ground_truth}\n")
        log_file.write(f"Generated Response: {generated_response}\n")

        # Calculate Exact Match
        normalized_gt = evaluator.normalize_answer(ground_truth)
        normalized_response = evaluator.normalize_answer(generated_response)
        exact_match = 1.0 if normalized_gt == normalized_response else 0.0
        exact_matches.append(exact_match)
        log_file.write(f"Exact Match: {exact_match}\n")

        # Calculate F1 Score
        f1 = evaluator.calculate_f1(ground_truth, generated_response)
        f1_scores.append(f1)
        log_file.write(f"F1 Score: {f1}\n")

        # Calculate BERTScore (Placeholder values for now)
        p, r, f1_bert = evaluator.compute_bertscore(
            [generated_response], [ground_truth]
        )
        bert_precision.append(p)
        bert_recall.append(r)
        bert_f1.append(f1_bert)
        log_file.write(f"BERTScore (P, R, F1): {p:.4f}, {r:.4f}, {f1_bert:.4f}\n")

        # Check Accuracy with GPT
        gpt_accuracy_check = evaluator.check_accuracy_gpt(
            query, ground_truth, generated_response
        )
        accuracy_gpt.append(gpt_accuracy_check)
        log_file.write(f"GPT Accuracy Check: {gpt_accuracy_check}\n")
        log_file.write("-" * 50 + "\n")

    # Compute and print average metrics
    average_metrics = evaluator.compute_averages(
        exact_matches, f1_scores, bert_precision, bert_recall, bert_f1, accuracy_gpt
    )

    log_file.write("\n" + "=" * 50 + " Overall Evaluation Results " + "=" * 50 + "\n")
    for metric, value in average_metrics.items():
        log_file.write(f"{metric}: {value:.4f}\n")
    log_file.write("=" * 104 + "\n")

    # Close log file
    log_file.close()

    print("\n" + "=" * 50 + " Evaluation Results " + "=" * 50)
    for metric, value in average_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("=" * (104))
    print(f"Evaluation logs saved to: {log_file_name}")
