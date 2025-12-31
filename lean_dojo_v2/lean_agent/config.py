"""Configuration classes for LeanAgent."""

import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies import DDPStrategy

from lean_dojo_v2.utils.constants import BATCH_SIZE, CHECKPOINT_DIR, RAID_DIR


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Training mode configuration
    run_progressive_training: bool = True
    single_repo: bool = True
    num_repos: int = 1

    max_epochs: int = 1

    # Model configuration
    model_name: str = "kaiyuy/leandojo-lean4-retriever-byt5-small"
    lr: float = 1e-3
    warmup_steps: int = 1000
    max_seq_len: int = 512
    num_retrieved: int = 100

    # Training parameters
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = 64
    accumulate_grad_batches: int = 4
    num_gpus: int = 1
    num_workers: int = 4
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 1.0

    # Data configuration
    num_negatives: int = 3
    num_in_file_negatives: int = 1
    data_max_seq_len: int = 1024
    tokenizer_model_name: str = "google/byt5-small"

    # EWC parameters
    lambda_value: float = 0.1

    # Timeout configuration
    timeout_seconds: int = 7 * 24 * 60 * 60 * 52  # 1 year

    # Callback configuration
    early_stopping_patience: int = 5
    early_stopping_monitor: str = "Recall@10_val"
    early_stopping_mode: str = "max"
    checkpoint_save_top_k: int = -1
    checkpoint_every_n_epochs: int = 1
    checkpoint_monitor: str = "Recall@10_val"
    checkpoint_mode: str = "max"

    # Logging configuration
    log_every_n_steps: int = 1
    num_sanity_val_steps: int = 0

    # Random seed
    seed: int = 3407

    @classmethod
    def from_yaml(cls, file_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, file_path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "run_progressive_training": self.run_progressive_training,
            "single_repo": self.single_repo,
            "num_repos": self.num_repos,
            "max_epochs": self.max_epochs,
            "model_name": self.model_name,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "max_seq_len": self.max_seq_len,
            "num_retrieved": self.num_retrieved,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "num_gpus": self.num_gpus,
            "num_workers": self.num_workers,
            "precision": self.precision,
            "gradient_clip_val": self.gradient_clip_val,
            "num_negatives": self.num_negatives,
            "num_in_file_negatives": self.num_in_file_negatives,
            "data_max_seq_len": self.data_max_seq_len,
            "tokenizer_model_name": self.tokenizer_model_name,
            "lambda_value": self.lambda_value,
            "timeout_seconds": self.timeout_seconds,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_monitor": self.early_stopping_monitor,
            "early_stopping_mode": self.early_stopping_mode,
            "checkpoint_save_top_k": self.checkpoint_save_top_k,
            "checkpoint_every_n_epochs": self.checkpoint_every_n_epochs,
            "checkpoint_monitor": self.checkpoint_monitor,
            "checkpoint_mode": self.checkpoint_mode,
            "log_every_n_steps": self.log_every_n_steps,
            "num_sanity_val_steps": self.num_sanity_val_steps,
            "seed": self.seed,
        }
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            "model_name": self.model_name,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "max_seq_len": self.max_seq_len,
            "num_retrieved": self.num_retrieved,
        }

    def get_data_module_config(
        self, data_path: str, corpus_path: str
    ) -> Dict[str, Any]:
        """Get data module configuration dictionary."""
        return {
            "data_path": data_path,
            "corpus_path": corpus_path,
            "num_negatives": self.num_negatives,
            "num_in_file_negatives": self.num_in_file_negatives,
            "model_name": self.tokenizer_model_name,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "max_seq_len": self.data_max_seq_len,
            "num_workers": self.num_workers,
        }

    def setup_environment_variables(self) -> None:
        """Set up environment variables for training."""
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_TIMEOUT"] = str(self.timeout_seconds * 1000)

    def create_callbacks(self, dir_name: str) -> list:
        """Create PyTorch Lightning callbacks."""
        filename_suffix = f"_lambda_{self.lambda_value}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(RAID_DIR, CHECKPOINT_DIR),
            filename=f"{dir_name}{filename_suffix}_{{epoch}}-{{Recall@10_val:.2f}}",
            verbose=True,
            save_top_k=self.checkpoint_save_top_k,
            every_n_epochs=self.checkpoint_every_n_epochs,
            monitor=self.checkpoint_monitor,
            mode=self.checkpoint_mode,
        )

        early_stop_callback = EarlyStopping(
            monitor=self.early_stopping_monitor,
            patience=self.early_stopping_patience,
            mode=self.early_stopping_mode,
            verbose=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        return [lr_monitor, checkpoint_callback, early_stop_callback]

    def create_trainer(self, dir_name: str, custom_log_dir: str) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        # Set up environment variables
        self.setup_environment_variables()

        # Create callbacks
        callbacks = self.create_callbacks(dir_name)

        # Initialize DDP strategy
        ddp_strategy = DDPStrategy(timeout=timedelta(seconds=self.timeout_seconds))

        return pl.Trainer(
            accelerator="gpu",
            gradient_clip_val=self.gradient_clip_val,
            precision=self.precision,
            strategy=ddp_strategy,
            devices=self.num_gpus,
            accumulate_grad_batches=self.accumulate_grad_batches,
            callbacks=callbacks,
            max_epochs=self.max_epochs,
            log_every_n_steps=self.log_every_n_steps,
            num_sanity_val_steps=self.num_sanity_val_steps,
            default_root_dir=custom_log_dir,
        )

    def get_custom_log_dir(self, dir_name: str) -> str:
        """Get custom log directory path."""
        custom_log_dir = os.path.join(
            RAID_DIR,
            "lightning_logs",
            f"{dir_name}_{False}_lambda_{self.lambda_value}",
        )
        os.makedirs(custom_log_dir, exist_ok=True)
        return custom_log_dir

    def get_lambda_value(self) -> float:
        """Get lambda value based on training mode."""
        if self.run_progressive_training:
            return self.lambda_value
        else:
            return 0.0


@dataclass
class ProverConfig:
    """Configuration for the proving pipeline."""

    use_vllm: bool = False
    num_workers: int = 4
    timeout: int = 600
    max_expansions: Optional[int] = None
    num_sampled_tactics: int = 64
    debug: bool = False
    batch_size: int = 12

    @classmethod
    def from_yaml(cls, file_path: str) -> "ProverConfig":
        """Load configuration from YAML file."""
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, file_path: str) -> "ProverConfig":
        """Load configuration from JSON file."""
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "use_vllm": self.use_vllm,
            "num_workers": self.num_workers,
            "timeout": self.timeout,
            "max_expansions": self.max_expansions,
            "num_sampled_tactics": self.num_sampled_tactics,
            "debug": self.debug,
            "batch_size": self.batch_size,
        }
        with open(file_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def get_prover_config(self) -> Dict[str, Any]:
        """Get prover configuration dictionary."""
        return {
            "use_vllm": self.use_vllm,
            "tactic": None,  # `None` since we are not using a fixed tactic generator
            "module": None,  # `None` since we are not using a fixed tactic generator
            "num_workers": self.num_workers,
            "timeout": self.timeout,
            "max_expansions": self.max_expansions,
            "num_sampled_tactics": self.num_sampled_tactics,
            "raid_dir": RAID_DIR,
            "checkpoint_dir": CHECKPOINT_DIR,
            "debug": self.debug,
        }


def save_configs_to_file(
    training_config: TrainingConfig, prover_config: ProverConfig, config_file: str
) -> None:
    """Save both configurations to a YAML file."""
    config_dict = {
        "training": {
            "run_progressive_training": training_config.run_progressive_training,
            "single_repo": training_config.single_repo,
            "num_repos": training_config.num_repos,
            "max_epochs": training_config.max_epochs,
            "model_name": training_config.model_name,
            "lr": training_config.lr,
            "warmup_steps": training_config.warmup_steps,
            "max_seq_len": training_config.max_seq_len,
            "num_retrieved": training_config.num_retrieved,
            "batch_size": training_config.batch_size,
            "eval_batch_size": training_config.eval_batch_size,
            "accumulate_grad_batches": training_config.accumulate_grad_batches,
            "num_gpus": training_config.num_gpus,
            "num_workers": training_config.num_workers,
            "precision": training_config.precision,
            "gradient_clip_val": training_config.gradient_clip_val,
            "num_negatives": training_config.num_negatives,
            "num_in_file_negatives": training_config.num_in_file_negatives,
            "data_max_seq_len": training_config.data_max_seq_len,
            "tokenizer_model_name": training_config.tokenizer_model_name,
            "lambda_value": training_config.lambda_value,
            "timeout_seconds": training_config.timeout_seconds,
            "early_stopping_patience": training_config.early_stopping_patience,
            "early_stopping_monitor": training_config.early_stopping_monitor,
            "early_stopping_mode": training_config.early_stopping_mode,
            "checkpoint_save_top_k": training_config.checkpoint_save_top_k,
            "checkpoint_every_n_epochs": training_config.checkpoint_every_n_epochs,
            "checkpoint_monitor": training_config.checkpoint_monitor,
            "checkpoint_mode": training_config.checkpoint_mode,
            "log_every_n_steps": training_config.log_every_n_steps,
            "num_sanity_val_steps": training_config.num_sanity_val_steps,
            "seed": training_config.seed,
        },
        "prover": {
            "use_vllm": prover_config.use_vllm,
            "num_workers": prover_config.num_workers,
            "timeout": prover_config.timeout,
            "max_expansions": prover_config.max_expansions,
            "num_sampled_tactics": prover_config.num_sampled_tactics,
            "debug": prover_config.debug,
            "batch_size": prover_config.batch_size,
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
