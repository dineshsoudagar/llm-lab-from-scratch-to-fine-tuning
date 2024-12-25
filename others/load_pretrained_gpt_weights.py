from transformers import GPT2LMHeadModel
import torch

# Load the pretrained Hugging Face GPT-2 model
pretrained_model_name = "gpt2"  # Change to "gpt2-medium", "gpt2-large" as needed
hf_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)

def load_hf_gpt2_pretrained_weights(small_gpt, hf_model):
    # Initialize SmallGPT with same configuration as GPT-2
    embed_dim = hf_model.config.n_embd

    small_gpt.to("cuda" if torch.cuda.is_available() else "cpu")

    # Map weights from Hugging Face model to SmallGPT
    state_dict = small_gpt.state_dict()
    # Map weights from Hugging Face model to SmallGPT
    state_dict = small_gpt.state_dict()

    # Token and position embeddings
    state_dict['token_embedding.weight'] = hf_model.transformer.wte.weight
    state_dict['position_embedding.weight'] = hf_model.transformer.wpe.weight

    # Transformer blocks
    for i, block in enumerate(hf_model.transformer.h):
        # Reshape the in_proj_weight (query, key, value concatenated) to match PyTorch ordering
        qkv_weight = block.attn.c_attn.weight  # Shape: (3 * embed_dim, embed_dim)
        qkv_bias = block.attn.c_attn.bias  # Shape: (3 * embed_dim)

        # Split and rearrange weights for query, key, value
        q_weight, k_weight, v_weight = torch.split(qkv_weight, embed_dim, dim=1)
        q_bias, k_bias, v_bias = torch.split(qkv_bias, embed_dim, dim=0)

        # Concatenate weights and biases in PyTorch's expected order
        state_dict[f'transformer_blocks.{i}.self_attn.in_proj_weight'] = torch.cat([q_weight, k_weight, v_weight], dim=0)
        state_dict[f'transformer_blocks.{i}.self_attn.in_proj_bias'] = torch.cat([q_bias, k_bias, v_bias], dim=0)

        # Output projection
        state_dict[f'transformer_blocks.{i}.self_attn.out_proj.weight'] = block.attn.c_proj.weight
        state_dict[f'transformer_blocks.{i}.self_attn.out_proj.bias'] = block.attn.c_proj.bias

        # Feedforward layers
        state_dict[f'transformer_blocks.{i}.linear1.weight'] = block.mlp.c_fc.weight.t()  # Transpose for PyTorch ordering
        state_dict[f'transformer_blocks.{i}.linear1.bias'] = block.mlp.c_fc.bias
        state_dict[f'transformer_blocks.{i}.linear2.weight'] = block.mlp.c_proj.weight.t()  # Transpose for PyTorch ordering
        state_dict[f'transformer_blocks.{i}.linear2.bias'] = block.mlp.c_proj.bias

        # Layer normalization
        state_dict[f'transformer_blocks.{i}.norm1.weight'] = block.ln_1.weight
        state_dict[f'transformer_blocks.{i}.norm1.bias'] = block.ln_1.bias
        state_dict[f'transformer_blocks.{i}.norm2.weight'] = block.ln_2.weight
        state_dict[f'transformer_blocks.{i}.norm2.bias'] = block.ln_2.bias

    # Final layer normalization
    state_dict['ln.weight'] = hf_model.transformer.ln_f.weight
    state_dict['ln.bias'] = hf_model.transformer.ln_f.bias

    # Output head
    state_dict['head.weight'] = hf_model.lm_head.weight

    # Load the mapped state_dict into SmallGPT
    small_gpt.load_state_dict(state_dict)

    print("Weights loaded successfully!")

    return small_gpt
