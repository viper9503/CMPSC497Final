import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments
)

def tokenize_batch(batch, tokenizer, max_input=128, max_target=512):
    inp = tokenizer(
        batch["prompt"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_input
    )
    tgt = tokenizer(
        batch["target"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_target
    )
    inp["labels"] = tgt["input_ids"]
    return inp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--out_dir",   default="checkpoints")
    parser.add_argument("--epochs",    type=int, default=3)
    parser.add_argument("--bs",        type=int, default=4)
    parser.add_argument("--lr",        type=float, default=5e-5)
    args = parser.parse_args()

    # 1) Load data
    raw = load_dataset("json", data_files={
        "train":      "data/train.jsonl",
        "validation": "data/validation.jsonl"
    })

    # 2) Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized = raw.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["prompt", "target"]
    )

    # 3) Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 4) TrainingArguments & Trainer
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=100,
        push_to_hub=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset= tokenized["validation"],
        tokenizer=tokenizer
    )

    # 5) Train!
    trainer.train()
    trainer.save_model(f"{args.out_dir}/final")

if __name__ == "__main__":
    main()
