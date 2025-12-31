#!/usr/bin/env python3
"""
Fine-tunes a LeanProgress regression head that predicts steps remaining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LeanProgress regression model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL with goal/prefix/tactic/steps_remaining fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Base Hugging Face model to fine-tune.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_text(goal: str, prefix: str, tactic: str) -> str:
    prefix = prefix or ""
    return (
        "Goal:\n" + goal + "\n\nPrefix:\n" + prefix + "\n\nCandidate tactic:\n" + tactic
    )


def tokenize_batch(tokenizer, max_length):
    def _tokenize(batch):
        texts = [
            build_text(goal, prefix, tactic)
            for goal, prefix, tactic in zip(
                batch["goal"],
                batch.get("prefix", [""] * len(batch["goal"])),
                batch["tactic"],
            )
        ]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [float(x) for x in batch["steps_remaining"]]
        return encoded

    return _tokenize


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze()
    mse = ((preds - labels) ** 2).mean().item()
    mae = (np.abs(preds - labels)).mean().item()
    return {"mse": mse, "mae": mae}


def main() -> None:
    args = parse_args()
    raw_dataset = load_dataset(
        "json",
        data_files={"train": str(args.dataset)},
    )["train"]
    dataset = raw_dataset.train_test_split(
        test_size=args.eval_ratio,
        seed=args.seed,
        stratify_by_column=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    tokenized = dataset.map(
        tokenize_batch(tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(
        f"LeanProgress model saved to {args.output_dir.resolve()}. "
        "Set LEANPROGRESS_MODEL to this path to enable step predictions."
    )


if __name__ == "__main__":
    main()
