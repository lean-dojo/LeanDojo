"""
Usage: python examples/grpo.py

Example script for training a model with group relative policy optimization (GRPO) on a GitHub repository.
"""

import torch

from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.grpo_trainer import GRPOTrainer


def reward_func(completions, **kwargs):
    return torch.tensor([1.0] * len(completions))


url = "https://github.com/durant42040/lean4-example"
commit = "b14fef0ceca29a65bc3122bf730406b33c7effe5"

trainer = GRPOTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    reward_func=reward_func,
    epochs_per_repo=1,
    batch_size=8,
    lr=2e-5,
)

agent = HFAgent(trainer=trainer)
agent.setup_github_repository(url=url, commit=commit)
agent.train()
agent.prove()
