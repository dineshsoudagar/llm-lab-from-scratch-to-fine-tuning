import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset import TextDataset
from torch.utils.data import Dataset, DataLoader

# Load pretrained GPT-2 model and tokenizer
model_name = "gpt2"  # Or any other GPT-2 variant like 'gpt2-medium', 'gpt2-small', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Example dataset
texts = [
    "Once upon a time, there was a brave knight.",
    "The quick brown fox jumps over the lazy dog.",
    "PyTorch is great for deep learning.",
    "Transformers have revolutionized NLP."
]

block_size = 128  # Max sequence length
dataset = TextDataset(texts, tokenizer, block_size)

# DataLoader for batching
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids, attention_mask = [b.to(device) for b in batch]

        # GPT-2 expects the labels to be shifted by 1
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss  # CrossEntropyLoss
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

