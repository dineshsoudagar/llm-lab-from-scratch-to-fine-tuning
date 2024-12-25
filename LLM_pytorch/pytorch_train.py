import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_small_gpt_model import SmallGPT
from dataset import GPTDataset
# Hyperparameters
block_size = 128
embed_dim = 128
num_heads = 4
num_layers = 4
vocab_size = 50257  # GPT-like vocabulary size (change as needed)
dropout = 0.1
learning_rate = 1e-4
batch_size = 32
num_epochs = 10

# Generate dummy text data (you should replace this with actual text data)
dummy_text = torch.randint(0, vocab_size, (100000,))
dataset = GPTDataset(dummy_text, block_size, vocab_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmallGPT(vocab_size, embed_dim, num_heads, num_layers, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits = logits.view(-1, logits.size(-1))  # Flatten for CrossEntropy
        y = y.view(-1)  # Flatten labels

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {step}, Loss: {loss.item():.4f}")
