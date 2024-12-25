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
from gpt_download import download_and_load_gpt2, load_weights_into_gpt

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


def main():
    # Define model configurations in a dictionary for compactness
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-xl (1558M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    torch.manual_seed(155)
    model = GPTModel(NEW_CONFIG)
    model.eval()
    settings, params = download_and_load_gpt2(model_size="1558M", models_dir="gpt2")
    load_weights_into_gpt(model, params)
    model.to(DEVICE)
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "Deep learning is a field of "

    generate_and_print_sample(
        model, tokenizer, DEVICE, start_context
    )


if __name__ == "__main__":
    main()
