# LLM Fine-tuning Toolkit

This repository contains a comprehensive toolkit for fine-tuning Large Language Models (LLMs) on custom datasets.

## Getting Started

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Restroulner/LLM-Fine-tuning-Toolkit.git
cd LLM-Fine-tuning-Toolkit
pip install -r requirements.txt
```

## Example Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("text", data_files={"train": "train.txt"})

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# Fine-tune the model
trainer.train()
```
