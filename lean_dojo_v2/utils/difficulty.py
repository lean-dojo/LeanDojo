"""
Utility functions for calculating and categorizing theorem difficulties.
"""

import math
from typing import Dict, List, Union

from lean_dojo_v2.database.models import Theorem
from lean_dojo_v2.database.models.repository import Repository


def calculate_theorem_difficulty(theorem: Theorem) -> Union[float, None]:
    """Calculates the difficulty of a theorem."""
    proof_steps = theorem.traced_tactics
    if any("sorry" in step.tactic for step in proof_steps):
        return float("inf")  # Hard (no proof)
    if len(proof_steps) == 0:
        return None  # To be distributed later
    return math.exp(len(proof_steps))


def categorize_difficulty(
    difficulty: Union[float, None], percentiles: List[float]
) -> str:
    """Categorizes the difficulty of a theorem."""
    if difficulty is None:
        return "To_Distribute"
    if difficulty == float("inf"):
        return "Hard (No proof)"
    elif difficulty <= percentiles[0]:
        return "Easy"
    elif difficulty <= percentiles[1]:
        return "Medium"
    else:
        return "Hard"


def print_difficulty_summary(
    categorized_theorems: Dict[Repository, Dict], percentiles: List[float]
) -> None:
    """
    Print detailed summary of theorem difficulties by repository and overall statistics.

    Args:
        categorized_theorems: Dictionary mapping repositories to their categorized theorems
        percentiles: List of percentile thresholds [33rd, 67th]
    """
    categories = ["Easy", "Medium", "Hard", "Hard (No proof)"]
    print("Summary of theorem difficulties by URL:")

    for repo in categorized_theorems:
        print(f"\nURL: {repo.url}")
        for category in categories:
            theorems = categorized_theorems[repo][category]
            if theorems:
                sorted_theorems = sorted(
                    theorems,
                    key=lambda x: (x[4] if x[4] is not None else -float("inf")),
                    reverse=True,
                )[:3]
                for name, path, start, end, diff in sorted_theorems:
                    diff_str = f"{diff:.2f}" if diff is not None else "N/A"
                    print(f"    - {name} (File: {path}, Difficulty: {diff_str})")

    print("\nOverall Statistics:")
    total_theorems = sum(
        len(theorems)
        for categories in categorized_theorems.values()
        for theorems in categories.values()
    )
    for category in categories:
        count = sum(
            len(categories[category]) for categories in categorized_theorems.values()
        )
        percentage = (count / total_theorems) * 100
        print(f"{category}: {count} theorems ({percentage:.2f}%)")

    print(
        f"\nPercentile thresholds: Easy <= {percentiles[0]:.2f}, Medium <= {percentiles[1]:.2f}, Hard > {percentiles[1]:.2f}"
    )
