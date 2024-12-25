import torch
import tiktoken
from small_gpt_model import (
    GPTModel, generate_text_simple,
    calc_loss_batch,
    evaluate_model,
    generate_and_print_sample,
    plot_losses,

)
from dataset import GPTDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-key-value bias
}
batch_size = 2
num_workers = 2
num_epochs = 20
learning_rate = 0.0004
weight_decay = 0.1


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in tqdm(train_loader, desc=f"Current epoch {epoch} : "):
            input_batch.to(DEVICE)
            target_batch.to(DEVICE)
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

        torch.save(model.state_dict(), f"model_wiki_epoch_{epoch}.pth")

        # Optional evaluation step
        # if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        track_tokens_seen.append(tokens_seen)
        print(f"Ep {epoch + 1} (Step {global_step:06d}): "
              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def main():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M).to(DEVICE)
    model.eval()

    with open("../data/wiki/wiki_train.txt", "r", encoding="utf-8") as file:
        train_data = file.read()
    with open("../data/wiki/wiki_test.txt", "r", encoding="utf-8") as file:
        val_data = file.read()

    # Train/validation ratio
    # train_ratio = 0.90
    # split_idx = int(train_ratio * len(text_data))
    # train_data = text_data[:split_idx]
    # val_data = text_data[split_idx:]

    torch.manual_seed(123)
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create train and val dataset
    train_dataset = GPTDataset(train_data, tokenizer, GPT_CONFIG_124M["context_length"],
                               GPT_CONFIG_124M["context_length"])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    val_dataset = GPTDataset(val_data, tokenizer, GPT_CONFIG_124M["context_length"], GPT_CONFIG_124M["context_length"])
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, DEVICE,
        num_epochs=num_epochs, eval_freq=10000, eval_iter=10000,
        start_context="AI is good only if", tokenizer=tokenizer
    )

    torch.save(model.state_dict(), "model_wiki.pth")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


if __name__ == "__main__":
    main()
