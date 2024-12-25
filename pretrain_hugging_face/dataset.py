import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.examples = []
        for text in texts:
            tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=block_size)
            self.examples.append((torch.tensor(tokenized['input_ids']),
                                  torch.tensor(tokenized['attention_mask'])))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


