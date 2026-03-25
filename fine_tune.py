from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

def fine_tune_llm(model_name, train_file, output_dir="./results", num_train_epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset("text", data_files={"train": train_file})

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"]
    )

    trainer.train()

if __name__ == "__main__":
    # Example usage
    # Create a dummy train.txt for demonstration
    with open("train.txt", "w") as f:
        f.write("This is a sample text for fine-tuning.\nAnother line of text.\n")
    fine_tune_llm("gpt2", "train.txt")
