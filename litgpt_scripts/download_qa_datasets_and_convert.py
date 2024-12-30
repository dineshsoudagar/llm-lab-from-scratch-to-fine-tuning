import json
import os
import requests


def download_dataset(url, save_path):
    """Downloads a dataset from a URL."""
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded dataset to {save_path}")


def process_natural_questions(input_path):
    """Processes the Natural Questions dataset."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for entry in data:
        question = entry.get("question", "")
        context = "\n\n".join(entry.get("context", []))
        answer = entry.get("answer", {}).get("text", "")

        processed_data.append({
            "context": context,
            "question": question,
            "answer": answer
        })
    return processed_data


def process_triviaqa(input_path):
    """Processes the TriviaQA dataset."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for entry in data:
        question = entry.get("Question", "")
        context = "\n\n".join(entry.get("Context", []))
        answer = entry.get("Answer", {}).get("text", "")

        processed_data.append({
            "context": context,
            "question": question,
            "answer": answer
        })
    return processed_data


def process_quac(input_path):
    """Processes the QuAC dataset."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for entry in data["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa.get("question", "")
                answer = qa.get("answers", [{}])[0].get("text", "")

                processed_data.append({
                    "context": context,
                    "question": question,
                    "answer": answer
                })
    return processed_data


def save_to_json(data, output_path):
    """Saves processed data to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved processed data to {output_path}")


# Example usage
def main():
    # Define dataset URLs and paths
    datasets = {
        "natural_questions": {
            "url": "https://example.com/natural_questions.json",
            "input_path": "natural_questions.json",
            "process_function": process_natural_questions,
            "output_path": "processed_natural_questions.json"
        },
        "triviaqa": {
            "url": "https://example.com/triviaqa.json",
            "input_path": "triviaqa.json",
            "process_function": process_triviaqa,
            "output_path": "processed_triviaqa.json"
        },
        "quac": {
            "url": "https://example.com/quac.json",
            "input_path": "quac.json",
            "process_function": process_quac,
            "output_path": "processed_quac.json"
        }
    }

    for name, info in datasets.items():
        print(f"Processing {name} dataset...")

        # Download dataset
        download_dataset(info["url"], info["input_path"])

        # Process dataset
        processed_data = info["process_function"](info["input_path"])

        # Save processed data
        save_to_json(processed_data, info["output_path"])


if __name__ == "__main__":
    main()
