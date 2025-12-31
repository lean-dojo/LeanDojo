import os
from typing import Optional

from lean_dojo_v2.lean_agent.config import ProverConfig, TrainingConfig
from lean_dojo_v2.prover import RetrievalProver
from lean_dojo_v2.trainer import RetrievalTrainer
from lean_dojo_v2.utils.constants import RAID_DIR
from lean_dojo_v2.utils.filesystem import find_latest_checkpoint

from .base_agent import BaseAgent


class LeanAgent(BaseAgent):
    def __init__(
        self,
        database_path: str = "dynamic_database.json",
        training_config: Optional[TrainingConfig] = None,
        prover_config: Optional[ProverConfig] = None,
    ):
        super().__init__(database_path)
        self.config = training_config or TrainingConfig()
        self.prover_config = prover_config or ProverConfig()
        self.trainer = RetrievalTrainer(config=self.config)

    def _get_build_deps(self) -> bool:
        """LeanAgent builds dependencies by default."""
        return True

    def _setup_prover(self):
        """Set up the RetrievalProver for LeanAgent."""
        ret_ckpt_path = find_latest_checkpoint()
        self.prover = RetrievalProver(
            ret_ckpt_path=ret_ckpt_path,
            gen_ckpt_path=os.path.join(RAID_DIR, "model_lightning.ckpt"),
            indexed_corpus_path=os.path.join(self.data_path, "corpus.jsonl"),
        )


def main():
    """
    Main function to run LeanAgent.
    """
    url = "https://github.com/durant42040/lean4-example"
    commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

    agent = LeanAgent()
    agent.setup_github_repository(url=url, commit=commit)
    agent.train()
    agent.prove()


if __name__ == "__main__":
    main()
