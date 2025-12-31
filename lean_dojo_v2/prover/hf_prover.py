"""RAG agent for Lean theorem proving."""

import random
from typing import Optional

import torch
from pantograph.expr import GoalState, Tactic
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from lean_dojo_v2.database.models.theorems import Theorem
from lean_dojo_v2.prover.base_prover import BaseProver


class HFProver(BaseProver):
    """Retrieval-Augmented Generation agent for Lean theorem proving.

    This agent uses a retrieval-augmented generator to generate tactics
    by first retrieving relevant premises and then generating tactics
    based on the retrieved context.
    """

    def __init__(self, ckpt_path: str, use_lora: bool = False, device: str = "auto"):
        super().__init__()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        if use_lora:
            self.model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path).to(
                self.device
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(ckpt_path).to(self.device)

        self.model.eval()

    def next_tactic(
        self,
        state: GoalState,
        goal_id: int,
    ) -> Optional[Tactic]:
        """Generate the next tactic using the loaded HuggingFace model."""
        if not hasattr(self, "theorem") or self.theorem is None:
            return None

        prompt = (
            "### System:\n"
            "You are a Lean 4 tactic generator. Given a goal state, "
            "output exactly ONE Lean tactic that advances or solves the goal.\n"
            "Rules:\n"
            "- Output only the tactic text; no prose, quotes, or code fences.\n"
            "- Single line only; no `by` blocks.\n"
            "- Never use `sorry` or `admit`.\n"
            "### User:\n"
            "{goal_str}\n\n"
            "### Assistant:\n"
        ).format(goal_str=str(state))

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=64,
                num_return_sequences=5,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        tactics = []
        for text in generated_texts:
            tactic_part = text[len(prompt) :].strip()
            tactic_part = tactic_part.split("\n")[0].split("<;>")[0].strip()
            if tactic_part and tactic_part != "sorry":
                tactics.append(tactic_part)

        selected_tactic = random.choice(tactics)

        return selected_tactic

    def generate_whole_proof(self, theorem: Theorem) -> str:
        self.theorem = theorem

        prompt = (
            "### System:\n"
            "Given a theorem statement, "
            "output the complete proof of the theorem in Lean 4 code.\n"
            "Only output the proof, no explanation, no comments, no theorem, nothing else."
            "### User:\n"
            "{theorem_str}\n\n"
            "### Assistant:\n"
        ).format(theorem_str=str(self.theorem))

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        proof = generated_texts[0][len(prompt) :].strip().replace("<;> ", "")

        return proof
