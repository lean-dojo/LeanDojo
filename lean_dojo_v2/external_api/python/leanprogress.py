from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _resolve_device(device: Optional[str]) -> torch.device:
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class PromptTemplate:
    template: str = (
        "Goal:\n{goal}\n\nPrefix:\n{prefix}\n\nCandidate tactic:\n{tactic}\n"
    )

    def render(self, goal: str, tactic: str, prefix: Optional[str]) -> str:
        return self.template.format(
            goal=goal, tactic=tactic, prefix=prefix or ""
        ).strip()


class LeanProgressScorer:
    """Predicts remaining steps using a regression head."""

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        template: Optional[str] = None,
    ) -> None:
        if model_name.startswith("~"):
            model_name = os.path.expanduser(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        resolved_device = _resolve_device(device)
        logger.info(f"Loading LeanProgress model {model_name} on {resolved_device}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
        ).to(resolved_device)
        self.template = PromptTemplate(template or PromptTemplate.template)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def predict(self, goal: str, tactic: str, prefix: Optional[str] = None) -> float:
        text = self.template.render(goal=goal, tactic=tactic, prefix=prefix)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze().item()
        return float(max(0.0, logits))
