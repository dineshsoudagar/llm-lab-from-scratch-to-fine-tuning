# Fine-Tuning and Inference with litGPT (LLMs) using PyTorch Lightning  

Link to [LitGPT](https://github.com/Lightning-AI/litgpt)

This repository provides scripts and notebooks for fine-tuning and running inference on **litGPT** models (LLMs) using **PyTorch Lightning**. It also supports **LoRA (Low-Rank Adaptation)** for efficient fine-tuning of LLMs.  

## Contents  

### Core Scripts  
- [LightningModule.py](LightningModule.py)  
  Supporting module implementation for training and fine-tuning litGPT models (LLMs).  

- [finetune_litgpt_models.py](finetune_litgpt_models.py)  
  Script for fine-tuning litGPT models.  

- [load_and_test_pretrained.py](load_and_test_pretrained.py)  
  Script for loading and evaluating pre-trained litGPT models.  

- [utils.py](utils.py)  
  Utility functions for data loading.

### Notebooks  
- [finetune_llms_with_lora.ipynb](finetune_llms_with_lora.ipynb)  
  Step-by-step notebook for fine-tuning litGPT models using **LoRA** for parameter-efficient learning.  

- [infer_finetuned_llms.ipynb](infer_finetuned_llms.ipynb)  
  Notebook for running inference on fine-tuned litGPT models and evaluating their performance.  

### Configurations  
- **[configs/](configs/)**  
  This folder contains configuration files with training parameters, model settings, and dataset details for fine-tuning tasks.  

---

This folder contains efficient framework for fine-tuning and deploying **litGPT models** based on **PyTorch Lightning**.  

Feel free to explore, adapt, or extend these resources for your LLM-related projects!  
