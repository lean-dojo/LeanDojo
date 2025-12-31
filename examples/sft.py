"""
Usage: python examples/sft.py

Example script for training a model with supervised fine-tuning (SFT) on a GitHub repository.
"""

from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

url = "https://github.com/durant42040/lean4-example"
commit = "b14fef0ceca29a65bc3122bf730406b33c7effe5"

trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    epochs_per_repo=1,
    batch_size=2,
    lr=2e-5,
)

agent = HFAgent(trainer=trainer)

agent.setup_github_repository(url=url, commit=commit)

agent.train()
agent.prove()
