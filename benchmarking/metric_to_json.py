import os
import re
import json
import argparse

# Regex pattern to extract metrics
pattern = r"(GPT Accuracy|GPT F1 Score|Exact Match|F1 Score|BERT Precision|BERT Recall|BERT F1 Score):\s*([\d.]+)"


# Function to rename the key
def convert_key(filename):
    # Remove extension
    name = os.path.splitext(filename)[0]

    # Example: D-hotpotqa_C-100_Q-100_M- ➔ dataset-hotpotqa_chunk-100-question-100_model-2
    name = name.replace("D-", "dataset-")
    name = name.replace("_C-", "_chunk-")
    name = name.replace("_Q-", "-question-")
    name = name.replace("_M-", "_model-")

    return name


def main(method):
    # Select folder path based on method
    if method == "ours":
        folder_path = "eval/logs/ours"
    elif method == "traditional_rag":
        folder_path = "eval/logs/traditional_rag"
    else:
        raise ValueError("Invalid method. Must be 'ours' or 'traditional_rag'.")

    # Dictionary to store results
    all_metrics = {}

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                content = file.read()

            matches = re.findall(pattern, content)

            if matches:
                metrics_dict = {key: float(value) for key, value in matches}
                custom_key = convert_key(filename)
                all_metrics[custom_key] = metrics_dict

    # Save all extracted metrics to a JSON file
    output_json = f"logs/all_metrics_{method}.json"
    with open(output_json, "w") as json_file:
        json.dump(all_metrics, json_file, indent=4)

    # Print the final dictionary
    print(json.dumps(all_metrics, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract metrics from evaluation logs."
    )
    parser.add_argument(
        "--method", type=str, required=True, help="Choose 'ours' or 'traditional_rag'."
    )

    args = parser.parse_args()
    main(args.method)
