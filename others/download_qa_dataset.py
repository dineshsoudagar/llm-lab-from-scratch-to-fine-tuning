import json
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# Define datasets to load
datasets_to_combine = [
    ("squad_v2", None),
    # ("squad", None),  # Dataset name and subset (if applicable)
    # ("natural_questions", None),
    # ("trivia_qa", "unfiltered"),
    # ("hotpot_qa", None),
    # ("quac", None)
]


# Function to load and format datasets
def load_and_format(dataset_name, subset=None):
    print(f"Loading {dataset_name}...")
    if subset:
        dataset = load_dataset(dataset_name, subset)
        dataset.save_to_disk("trivia")
    else:
        dataset = load_dataset(dataset_name, trust_remote_code=True)

    dataset_train = []

    for example in tqdm(dataset["validation"]):
        if not example["answers"]["text"]:
            answer_text = "No answer in the context given"
        else:
            answer_text = (
                example["answers"]["text"][0] if isinstance(example["answers"]["text"], list) else example["answers"]
            )
        dataset_train.append({"context": example["context"], "question": example["question"], "answer": answer_text})
    return dataset_train

    # Standardize the dataset format
    def standardize_format(example):
        if "context" in example and "question" in example and "answers" in example:
            if not example["answers"]["text"]:
                print("skipped", example)
                return
            answer_text = (
                example["answers"]["text"][0] if isinstance(example["answers"]["text"], list) else example["answers"]
            )
            return {"context": example["context"], "question": example["question"], "answer": answer_text}
        elif "paragraphs" in example:  # TriviaQA
            return {"context": example["paragraphs"][0], "question": example["question"], "answer": example["answer"]}
        else:
            return {"context": None, "question": None, "answer": None}  # Skip incompatible formats

    # Map and filter dataset
    dataset = dataset.map(standardize_format, remove_columns=dataset["train"].column_names, batched=False)
    dataset = dataset.filter(
        lambda x: x["context"] is not None and x["question"] is not None and x["answer"] is not None)

    return dataset


# Function to save dataset as JSON
def save_to_json(dataset, split, dataset_name):
    # Convert to a list of dictionaries
    data = [{"context": example["context"], "question": example["question"], "answer": example["answer"]} for example in
            dataset[split]]

    # Save to JSON file
    with open(f"{dataset_name}_{split}.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {dataset_name}_{split} to JSON.")


# Combine datasets (train, test, and validation)
for dataset_name, subset in datasets_to_combine:
    formatted_dataset = load_and_format(dataset_name, subset)
    with open(f"{dataset_name}_val.json", "w") as f:
        json.dump(formatted_dataset, f, indent=4)
    break
    # Save each split (train, validation, test) separately as JSON
    for split in ['train', 'validation', 'test']:
        if split in formatted_dataset:
            save_to_json(formatted_dataset, split, dataset_name)

print("All datasets saved in JSON format.")
