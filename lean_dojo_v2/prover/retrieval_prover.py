"""RAG agent for Lean theorem proving."""

import random
from typing import Optional

import numpy as np
import torch
from pantograph.expr import GoalState, Tactic

from lean_dojo_v2.database.models.theorems import Theorem
from lean_dojo_v2.lean_agent.generator.model import RetrievalAugmentedGenerator
from lean_dojo_v2.prover.base_prover import BaseProver


class RetrievalProver(BaseProver):
    """Retrieval-Augmented Generation agent for Lean theorem proving.

    This agent uses a retrieval-augmented generator to generate tactics
    by first retrieving relevant premises and then generating tactics
    based on the retrieved context.
    """

    def __init__(self, ret_ckpt_path, gen_ckpt_path, indexed_corpus_path):
        super().__init__()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        config = {
            "model_name": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
            "lr": 1e-3,
            "warmup_steps": 1000,
            "num_beams": 5,
            "eval_num_retrieved": 10,
            "eval_num_workers": 1,
            "eval_num_gpus": 1,
            "eval_num_theorems": 100,
            "max_inp_seq_len": 512,
            "max_oup_seq_len": 128,
            "ret_ckpt_path": ret_ckpt_path,
        }

        self.tactic_generator = RetrievalAugmentedGenerator.load(
            gen_ckpt_path,
            device=device,
            freeze=True,
            config=config,
        )

        if self.tactic_generator.retriever is not None:
            self.tactic_generator.retriever.load_corpus(indexed_corpus_path)
            self.tactic_generator.retriever.reindex_corpus(batch_size=32)

        self.tactic_generator.eval()

    def next_tactic(
        self,
        state: GoalState,
        goal_id: int,
    ) -> Optional[Tactic]:
        """Generate the next tactic using RAG model."""
        if not hasattr(self, "theorem") or self.theorem is None:
            return None

        suggestions = self.tactic_generator.generate(
            state=str(state),
            file_path=str(self.theorem.file_path),
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.theorem.start,
            num_samples=10,
        )

        tactics, log_probs = zip(*suggestions)
        probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
        selected_tactic = random.choices(tactics, weights=probs, k=1)[0]

        return selected_tactic

    def generate_whole_proof(self, theorem: Theorem) -> str:
        raise NotImplementedError(
            "RetrievalProver does not support whole proof generation"
        )
