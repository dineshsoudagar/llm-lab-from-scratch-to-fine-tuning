"""
Understanding the data from https://en.wikisource.org/wiki/The_Verdict
"""
import re


def print_sample():
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))
    print("sample raw text", raw_text[:99])

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item for item in preprocessed if item]
    print("Number of Tokens", len(preprocessed))
    print("sample tokens", preprocessed[:30])
    print("unique tokens", sorted(set(preprocessed)))

def convert_tokens_to_token_ids_sample():
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break

    sample_text = raw_text[:99]
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', sample_text)
    tokens = [vocab[token] for token in preprocessed]
    print("Sample text", sample_text)
    print("Sample token ids", tokens)



# print_sample()
convert_tokens_to_token_ids_sample()