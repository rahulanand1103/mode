from datasets import load_dataset
import nltk
import argparse

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


class HotpotQADataLoader:
    def __init__(self, sample_size=100):
        self.sample_size = sample_size
        self.chunks, self.qa_pair = self.load_unique_hotpotqa_data()

    def load_unique_hotpotqa_data(self):
        dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
        unique_contexts = set()
        qa_pair = []

        for entry in dataset:
            contexts = [" ".join(sent) for sent in entry["context"]["sentences"]]
            question, answer = entry["question"], entry["answer"]
            qa_pair.append({"question": question, "answer": answer})

            unique_contexts.update(contexts)
            if len(unique_contexts) >= self.sample_size:
                break

        return list(unique_contexts), qa_pair

    def evaluate_qa_pair(self, n):
        return self.qa_pair[:n]


class SquadDataLoader:
    def __init__(self, sample_size=100):
        self.sample_size = sample_size
        self.chunks, self.qa_pair = self.load_unique_squad_data()

    def load_unique_squad_data(self):
        dataset = load_dataset("squad", split="train")
        unique_contexts = {}
        qa_pair = []

        for entry in dataset:
            context, question, answers = (
                entry["context"],
                entry["question"],
                entry["answers"]["text"],
            )
            answer = answers[0] if answers else "No answer available"

            if context not in unique_contexts:
                unique_contexts[context] = True
                qa_pair.append({"question": question, "answer": answer})

            if len(unique_contexts) >= self.sample_size:
                break

        return list(unique_contexts.keys()), qa_pair

    def evaluate_qa_pair(self, n):
        return self.qa_pair[:n]


def get_dataset_loader(dataset_name, sample_size=100):
    dataset_loaders = {"hotpotqa": HotpotQADataLoader, "squad": SquadDataLoader}
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    return dataset_loaders[dataset_name](sample_size=sample_size)
