from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import trl
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
from lean_dojo_v2.utils import remove_marks
from lean_dojo_v2.utils.constants import DATA_DIR, RAID_DIR


class GRPODataset:
    """Loads your traced tactics json and produces a HF Dataset with a 'messages' column."""

    def __init__(self, data_path: str):
        self.data_path = data_path

        with open(data_path) as f:
            self.json_data = json.load(f)
        self.data = self._process_data(self.json_data)

    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for item in data:
            for tactic in item.get("traced_tactics", []):
                if tactic.get("tactic") == "sorry":
                    continue

                processed.append(
                    {
                        "problem": remove_marks(tactic["state_before"]).strip(),
                        "prompt": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a Lean 4 tactic generator. Given a goal state, "
                                    "output exactly ONE Lean tactic that advances or solves the goal.\n"
                                    "Rules:\n"
                                    "- Output only the tactic text; no prose or code fences.\n"
                                    "- Single line only; no `by` blocks.\n"
                                    "- Never use `sorry` or `admit`.\n"
                                ),
                            },
                            {
                                "role": "user",
                                "content": remove_marks(tactic["state_before"]).strip(),
                            },
                        ],
                    }
                )
        return processed

    def to_hf(self) -> Dataset:
        return Dataset.from_list(self.data)


class GRPOTrainer:
    def __init__(
        self,
        model_name: str,
        reward_func,
        output_dir: str = "outputs",
        epochs_per_repo: int = 1,
        batch_size: int = 1,
        lr: float = 2e-5,
        lora_config: Optional[LoraConfig] = None,
    ):
        self.model_name = model_name
        self.reward_func = reward_func
        self.output_dir = output_dir
        self.lora_config = lora_config
        self.use_lora = lora_config is not None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        if self.use_lora:
            self.model = self._apply_lora()

        self.config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=epochs_per_repo,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
        )

    def _apply_lora(self):
        """Apply LoRA configuration to the model."""
        if self.lora_config is None:
            raise ValueError("LoRA config is required when use_lora is True")

        model = get_peft_model(self.model, self.lora_config)

        model.print_trainable_parameters()

        return model

    def train(
        self,
        repos: List[LeanGitRepo],
        database: DynamicDatabase,
        data_path: Path,
    ):
        repos_to_process = []

        for repo in repos:
            repos_to_process.append(repo)

            database.export_merged_data(repos_to_process, data_path)

            train_dataset = GRPODataset(
                os.path.join(data_path, "random", "train.json")
            ).to_hf()

            trainer = trl.GRPOTrainer(
                model=self.model,
                args=self.config,
                train_dataset=train_dataset,
                reward_funcs=[self.reward_func],
            )
            trainer.train()

            self.model = trainer.model

            if self.use_lora:
                self.model.save_pretrained(self.output_dir)
            else:
                trainer.save_model(self.output_dir)

        self.tokenizer.save_pretrained(self.output_dir)


if __name__ == "__main__":

    def reward_func(completions, **kwargs):
        """Reward function for GRPO training."""
        return torch.tensor([1.0] * len(completions))

    repo = LeanGitRepo(
        url="https://github.com/durant42040/lean4-example",
        commit="005de00d03f1aaa32cb2923d5e3cbaf0b954a192",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
    )

    sft = GRPOTrainer(
        model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
        output_dir="outputs-deepseek-lora",
        reward_func=reward_func,
        epochs_per_repo=1,
        batch_size=8,
        lr=2e-5,
        # lora_config=lora_config,
    )

    sft.train(repo)
