from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load the tokenizer and model
model_name = "gpt2"  # Use "gpt2" for the small GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token to tokenizer (if missing)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Prepare your dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Example: Loading a sample dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # Replace with your dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# DataCollator to handle padding during batch creation
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
)

# Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tuning the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
