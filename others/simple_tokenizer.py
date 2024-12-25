import re


class SimpleTokenizer:
    def __init__(self, text_file):
        self.text_file = text_file
        self.vocab = self.generate_vocab()
        self.str_to_int = self.vocab
        self.int_to_str = {i: s for s, i in self.vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

    def generate_vocab(self):
        with open(self.text_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        all_words = sorted(set(preprocessed))
        vocab = {token: integer for integer, token in enumerate(all_words)}
        return vocab


def test(text_file):
    tokenizer = SimpleTokenizer(text_file)
    text = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    tokens = tokenizer.decode(ids)
    print(ids)
    print(tokens)

# file = "data/the-verdict.txt"
file = "../data/Evaluation of Multi-Task Vs. Single.txt"
test(file)