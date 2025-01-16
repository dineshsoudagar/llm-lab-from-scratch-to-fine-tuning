import json

from tqdm import tqdm


# Function to extract answer text from tokenized answers (if necessary)
def extract_answer_text(answer_tokens, tokenizer):
    # If answer is a list of tokens, convert back to text using the tokenizer
    answer_text = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer_text.strip()


def process_natural_questions(input_file, output_file, tokenizer=None):
    # Read the input file
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    processed_data = []

    # Iterate over each entry in the dataset
    for entry in tqdm(data):
        context = entry.get("document_text", "")  # Assuming "document_text" contains the context
        question = entry.get("question_text", "")  # Assuming "question_text" contains the question

        # Get the answer tokens and convert them to actual text if needed
        if 'annotations' in entry and entry['annotations']:
            answer_tokens = entry['annotations'][0].get("long_answer", {}).get("text", "")
        else:
            answer_tokens = ""

        # Convert answer tokens to text if they exist
        if tokenizer and answer_tokens:
            answer = extract_answer_text(answer_tokens, tokenizer)
        else:
            answer = answer_tokens

        # Append the formatted data
        processed_data.append({
            "context": context,
            "question": question,
            "answer": answer
        })

    # Save the processed data into the output file
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)


# Example usage:
input_file = r'C:\Others\LLMs\litgpt_scripts\v1.0-simplified_simplified-nq-train.jsonl\simplified-nq-train.jsonl'  # Input file path
output_file = 'processed_natural_questions.json'  # Output file path
# If tokenizer is needed (e.g., for tokenizing and converting answers back to text), load it here
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

# Process the dataset
process_natural_questions(input_file, output_file, tokenizer)
