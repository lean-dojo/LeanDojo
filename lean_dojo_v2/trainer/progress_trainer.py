from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class ProgressDataset:
    """Loads data and produces a HF Dataset for regression training.

    Supports two formats:
    1. JSONL file with goal/prefix/tactic/steps_remaining fields (from create_sample_dataset.py)
    2. Traced tactics JSON from database export
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_data(data_path)

    def _load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file (format from create_sample_dataset.py)."""
        processed: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if all(
                        key in item for key in ["goal", "tactic", "steps_remaining"]
                    ):
                        processed.append(
                            {
                                "goal": item.get("goal", "").strip(),
                                "prefix": item.get("prefix", "").strip(),
                                "tactic": item.get("tactic", "").strip(),
                                "steps_remaining": item.get("steps_remaining", 0),
                            }
                        )
        return processed

    def to_hf(self) -> Dataset:
        return Dataset.from_list(self.data)


class ProgressTrainer:
    """Trainer for LeanProgress regression model that predicts steps remaining."""

    def __init__(
        self,
        data_path: str,
        model_name: str = "bert-base-uncased",
        output_dir: str = "outputs",
        max_length: int = 512,
        batch_size: int = 8,
        epochs: float = 3.0,
        learning_rate: float = 1e-5,
        eval_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.eval_ratio = eval_ratio
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.cls_token
            )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
        )

    def _build_text(self, goal: str, prefix: str, tactic: str) -> str:
        """Build the input text from goal, prefix, and tactic."""
        prefix = prefix or ""
        return (
            "Goal:\n"
            + goal
            + "\n\nPrefix:\n"
            + prefix
            + "\n\nCandidate tactic:\n"
            + tactic
        )

    def _tokenize_batch(self, batch):
        """Tokenize a batch of examples."""
        texts = [
            self._build_text(goal, prefix, tactic)
            for goal, prefix, tactic in zip(
                batch["goal"],
                batch.get("prefix", [""] * len(batch["goal"])),
                batch["tactic"],
            )
        ]
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = [float(x) for x in batch["steps_remaining"]]
        return encoded

    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        preds = predictions.squeeze()
        mse = ((preds - labels) ** 2).mean().item()
        mae = (np.abs(preds - labels)).mean().item()
        return {"mse": mse, "mae": mae}

    def train(self):
        """Train the progress model from a JSONL file (e.g., from create_sample_dataset.py).

        Args:
        """
        train_dataset = ProgressDataset(self.data_path).to_hf()

        # Split into train and eval
        split_dataset = train_dataset.train_test_split(
            test_size=self.eval_ratio,
            seed=self.seed,
        )

        # Tokenize datasets
        tokenized_train = split_dataset["train"].map(
            self._tokenize_batch,
            batched=True,
            remove_columns=split_dataset["train"].column_names,
        )

        tokenized_eval = split_dataset["test"].map(
            self._tokenize_batch,
            batched=True,
            remove_columns=split_dataset["test"].column_names,
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="mse",
            logging_steps=50,
            report_to="none",
            seed=self.seed,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.model = trainer.model
        self.tokenizer.save_pretrained(self.output_dir)
        print(
            f"LeanProgress model saved to {Path(self.output_dir).resolve()}. "
            "Set LEANPROGRESS_MODEL to this path to enable step predictions."
        )


if __name__ == "__main__":
    trainer = ProgressTrainer(
        model_name="bert-base-uncased",
        data_path="raid/data/sample_leanprogress_dataset.jsonl",
        output_dir="outputs-progress",
        max_length=512,
        batch_size=8,
        epochs=3.0,
        learning_rate=1e-5,
    )
    trainer.train()
