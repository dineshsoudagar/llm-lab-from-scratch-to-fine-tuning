import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Use "gpt2" for the small GPT-2 model
model_name = "gpt2"
# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# Load the fine-tuned model and tokenizer
hf_model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")

prompt = "who am I when "
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

# Generate text
attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long().to("cuda")
hf_output = hf_model.generate(
    inputs["input_ids"],
    max_new_tokens=20,
    #max_length=20,
    #num_return_sequences=1,
    #do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    attention_mask=attention_mask,  # Add attention_mask here
)
generated_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
