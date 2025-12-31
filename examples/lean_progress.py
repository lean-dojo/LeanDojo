"""
Usage: python examples/lean_progress.py

Example script for training a LeanProgress regression model that predicts
the number of steps remaining in a proof.

To generate a sample dataset first, run:
    python -m lean_dojo_v2.lean_progress.create_sample_dataset --output raid/data/sample_leanprogress_dataset.jsonl
"""

from pathlib import Path

from lean_dojo_v2.trainer.progress_trainer import ProgressTrainer

sample_dataset_path = Path("raid/data/sample_leanprogress_dataset.jsonl")

trainer = ProgressTrainer(
    model_name="bert-base-uncased",
    data_path=str(sample_dataset_path),
    output_dir="outputs-progress",
)

trainer.train()
