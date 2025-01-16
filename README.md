# LLMs Repository

This repository contains a collection of scripts, notebooks, and tools for working with **Large Language Models (LLMs)**. It includes resources for pretraining models using **Hugging Face Transformers**, fine-tuning **litGPT models** with **PyTorch Lightning**, and running inference on fine-tuned models. The repository also provides efficient fine-tuning through **LoRA (Low-Rank Adaptation)**, which helps in training large models with fewer parameters.

## Contents

### [huggingface_scripts/](huggingface_scripts/)
This folder includes scripts for pretraining and running inference on **Hugging Face models** (e.g., GPT). The scripts are designed to work with **Hugging Face Transformers** for training models from scratch and running inference on pretrained models. 

### [litgpt_scripts/](litgpt_scripts/)
This folder contains scripts and notebooks for fine-tuning **litGPT models** (LLMs) using **PyTorch Lightning**. It includes LoRA support for efficient fine-tuning of large models, allowing the training of language models with fewer parameters and faster convergence.

### [pytorch_gpt_scripts/](pytorch_gpt_scripts/)
Scripts focused on implementing and training GPT models from scratch using **PyTorch**. This folder contains tools for building and training custom GPT models using the PyTorch framework.

## Key Features

- **Pretraining with Hugging Face Models**: Easily train GPT models using Hugging Face’s popular **Transformers** library.
- **Fine-Tuning with litGPT**: Fine-tune pre-trained litGPT models for specific tasks using **PyTorch Lightning**.
- **Efficient Fine-Tuning with LoRA**: Implement LoRA for parameter-efficient fine-tuning of large models.
- **Inference Support**: Scripts for running inference on pretrained models and evaluating their performance.
- **Flexible Framework**: Supports both Hugging Face and litGPT models, allowing you to choose the model architecture and training framework that best fits your needs.

## Getting Started

This repository provides a comprehensive framework for training and fine-tuning **Large Language Models** (LLMs). It includes everything you need to start experimenting with **Hugging Face Transformers**, **litGPT models**, and **LoRA** for efficient model fine-tuning.

Feel free to explore and adapt the scripts, notebooks, and utilities in this repository to suit your LLM training and inference needs!

---

**Note:**  
- Ensure that the required libraries are installed via `requirements.txt`.  
- The scripts are modular, allowing easy integration with other training frameworks or models.

