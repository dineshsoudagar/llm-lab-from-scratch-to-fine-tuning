from datasets import load_dataset

# Load WikiText dataset as an example
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Save the train split to a UTF-8 encoded text file
with open("wiki_test.txt", "w", encoding="utf-8") as f:
    for text in dataset['test']['text']:
        f.write(text + "\n")