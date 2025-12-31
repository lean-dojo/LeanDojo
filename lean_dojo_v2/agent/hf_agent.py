from typing import Optional

from peft import LoraConfig

from lean_dojo_v2.lean_agent.config import ProverConfig, TrainingConfig
from lean_dojo_v2.prover import HFProver
from lean_dojo_v2.trainer import SFTTrainer
from lean_dojo_v2.utils.constants import *

from .base_agent import BaseAgent


class HFAgent(BaseAgent):
    def __init__(
        self,
        trainer: SFTTrainer,
        database_path: str = "dynamic_database.json",
        training_config: Optional[TrainingConfig] = None,
        prover_config: Optional[ProverConfig] = None,
    ):
        super().__init__(database_path)
        self.config = training_config or TrainingConfig()
        self.prover_config = prover_config or ProverConfig()
        self.trainer = trainer
        self.output_dir = self.trainer.output_dir
        self.use_lora = self.trainer.lora_config is not None

    def _get_build_deps(self) -> bool:
        """HFAgent doesn't build dependencies by default."""
        return False

    def _setup_prover(self):
        """Set up the HFProver for HFAgent."""
        self.prover = HFProver(
            ckpt_path=self.output_dir,
            use_lora=self.use_lora,
        )


def main():
    """
    Main function to run HFAgent.
    """
    url = "https://github.com/durant42040/lean4-example"
    commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"
    model_name = "deepseek-ai/DeepSeek-Prover-V2-7B"
    output_dir = "outputs-deepseek"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
    )

    trainer = SFTTrainer(
        model_name=model_name,
        output_dir=output_dir,
        epochs_per_repo=1,
        batch_size=2,
        lr=2e-5,
        lora_config=lora_config,
    )

    agent = HFAgent(trainer=trainer)
    agent.setup_github_repository(url=url, commit=commit)
    agent.train()
    agent.prove()


if __name__ == "__main__":
    main()
