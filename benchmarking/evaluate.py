from bert_score import score
from sklearn.metrics import accuracy_score, f1_score
import re
import string
from collections import Counter
import numpy as np
from litellm import completion


class Evaluator:
    def __init__(self, model_name="microsoft/deberta-xlarge-mnli"):
        self.model_name = model_name

    def normalize_answer(self, text):
        """Lower case and remove punctuation, articles, and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        def remove_periods(text):
            return re.sub(r"\.", "", text)

        return white_space_fix(
            remove_articles(remove_punc(remove_periods(lower(text))))
        )

    def calculate_f1(self, ground_truth, response):
        ground_truth = self.normalize_answer(ground_truth).split()
        response = self.normalize_answer(response).split()
        common = Counter(ground_truth) & Counter(response)
        num_same = sum(common.values())

        if len(ground_truth) == 0 or len(response) == 0:
            return int(ground_truth == response)

        precision = num_same / len(response) if len(response) > 0 else 0
        recall = num_same / len(ground_truth) if len(ground_truth) > 0 else 0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return f1

    def check_accuracy_gpt(self, query, ground_truth, response):
        messages = [
            {
                "role": "system",
                "content": """Prompt:
            You are an expert evaluator of question-answering systems. Assess whether the generated answer is contexual equivalent to the ground truth answer. Respond with "Yes" if they are equivalent and "No" if they are not.
            For long answer also please check high level contexual they are same or not,
            Example Evaluations:
            Question: How often does the event occur?
            Ground Truth Answer: twice
            Generated Answer: twice a year
            Response: No

            Question: How many BS-level degrees are offered in the College of Engineering at Notre Dame?
            Ground Truth Answer: eight
            Generated Answer: Eight B.S. degrees are offered.
            Response: Yes

            Output format:
            yes or no
            """,
            },
            {
                "role": "user",
                "content": f"Question: {query}\nGround Truth Answer: {ground_truth}\nGenerated Answer: {response}",
            },
        ]
        try:
            final_response = completion(model="openai/gpt-4o", messages=messages)
            return final_response.choices[0].message.content.lower().strip()
        except Exception as e:
            print(f"Error during GPT evaluation: {e}")
            return "error"

    def calculate_metrics_gpt(self, predictions):
        valid_predictions = [p for p in predictions if p in ["yes", "no"]]
        if not valid_predictions:
            return 0.0, 0.0
        predictions_binary = [1 if x == "yes" else 0 for x in valid_predictions]
        actual_binary = [1] * len(
            valid_predictions
        )  # Assuming all generated answers should ideally be correct
        accuracy = accuracy_score(actual_binary, predictions_binary)
        f1 = f1_score(actual_binary, predictions_binary)
        return accuracy, f1

    def compute_bertscore(self, predictions, references):
        try:
            P, R, F1 = score(
                predictions, references, model_type=self.model_name, lang="en"
            )
            return P.mean().item(), R.mean().item(), F1.mean().item()
        except Exception as e:
            print(f"Error during BERTScore calculation: {e}")
            return 0.0, 0.0, 0.0

    def compute_averages(
        self,
        exact_matches,
        f1_scores,
        bert_precision,
        bert_recall,
        bert_f1,
        accuracy_gpt,
    ):
        accuracy, f1_gpt = self.calculate_metrics_gpt(accuracy_gpt)

        return {
            "GPT Accuracy": accuracy,
            "GPT F1 Score": f1_gpt,
            "Exact Match": np.mean(exact_matches) if exact_matches else 0.0,
            "F1 Score": np.mean(f1_scores) if f1_scores else 0.0,
            "BERT Precision": np.mean(bert_precision) if bert_precision else 0.0,
            "BERT Recall": np.mean(bert_recall) if bert_recall else 0.0,
            "BERT F1 Score": np.mean(bert_f1) if bert_f1 else 0.0,
        }
