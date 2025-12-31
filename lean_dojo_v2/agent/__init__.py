"""
Agent package for Lean theorem proving.

This package provides different types of theorem proving agents:
- BaseAgent: Abstract base class defining the common interface
- HFAgent: HuggingFace-based agent using SFT training
- LeanAgent: Lean-specific agent using retrieval training
"""

from .base_agent import BaseAgent
from .external_agent import ExternalAgent
from .hf_agent import HFAgent
from .lean_agent import LeanAgent

__all__ = [
    "BaseAgent",
    "HFAgent",
    "LeanAgent",
    "ExternalAgent",
]
