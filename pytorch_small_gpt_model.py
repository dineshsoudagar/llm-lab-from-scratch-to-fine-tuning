import torch
import torch.nn as nn

class SmallGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, dropout=0.1):
        super(SmallGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape
        assert T <= self.block_size, "Sequence length exceeds block size!"

        positions = torch.arange(0, T, device=x.device).unsqueeze(0)  # (1, T)
        x = self.token_embedding(x) + self.position_embedding(positions)  # (B, T, embed_dim)

        for block in self.transformer_blocks:
            x = block(x)  # (B, T, embed_dim)

        x = self.ln(x)  # (B, T, embed_dim)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
