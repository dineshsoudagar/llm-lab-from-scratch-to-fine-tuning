from pathlib import Path
import torch
import litgpt
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L

from litgpt_scripts.LightningModule import LitLLM
from utils import load_json
from litgpt.data.json_data import JSON
from litgpt import LLM

train_file_paths = ["qa_dataset/squad_train.json"]
val_file_paths = ["squad_validation.json"]

train_data = []
val_data = []

for train_file in train_file_paths:
    train_data.extend(load_json(train_file))

#for val_file in val_file_paths:
#    val_data.extend(load_json(val_file))

data = JSON(Path(train_file_paths[0]), val_split_fraction=0.1)


epochs = 3
log_interval = 100
model_name = "meta-llama/Llama-3.2-1B-Instruct"
batch_size = 8
accumulate_grad_batches = 1

lit_model = LitLLM(checkpoint_dir=model_name)
data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)
del lit_model
trainer = L.Trainer(
    devices=1,
    accelerator="cuda",
    max_epochs=1,
    accumulate_grad_batches=accumulate_grad_batches,
    precision="bf16-true",
)
trainer.fit(lit_model, data)

lit_model.llm.model.to(lit_model.llm.preprocessor.device)
lit_model.llm.generate("hello world")
