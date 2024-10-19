# AIRepublic_Finetune_LLM_model

## Fine-tuning and Inference with LLM Models Using Hugging Face

This repository contains two Jupyter notebooks demonstrating how to fine-tune a large language model (LLM) using the Hugging Face Transformers library, followed by a notebook that performs inference with the fine-tuned model. This project is part of the activities for Week 2 of the AI Republic's AI Bootcamp.


## Notebooks

### 1. Fine-tuning an LLM Model
- **Notebook Name**: `finetune_opensource_llm_model.ipynb`
- **Description**: This notebook walks you through the process of fine-tuning a pre-trained LLM on a custom dataset. It covers:
  - Loading a pre-trained model and tokenizer
  - Preparing the dataset for training
  - Fine-tuning the model
  - Uploading the model in HuggingFace
  - Evaluation Metrics

### 2. Inference with the Fine-tuned Model
- **Notebook Name**: `inferenced_finetuned_model.ipynb`
- **Description**: In this notebook, you will use the fine-tuned model to make responses. It includes:
  - Loading the fine-tuned model and tokenizer
  - Preparing input data for inference
  - Running inference and displaying results

## Model and Dataset Information

- **Fine-Tuned Model**: The fine-tuned model can be found [here](https://huggingface.co/noelabu/gemma-2b-instruct-ft-mental-health-conv_v2).
- **Base Model**: The base model used for fine-tuning is available [here](https://huggingface.co/google/gemma-2-2b-it).
- **Dataset**: The dataset used for fine-tuning is accessible [here](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations).
