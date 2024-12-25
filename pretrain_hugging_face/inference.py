import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def generate_text(prompt, model, tokenizer, max_length, device):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Generate text
prompt = "Once upon a time"
generated_text = generate_text(model, tokenizer, prompt, max_length=50, block_size=128)
print("Generated Text:", generated_text)
