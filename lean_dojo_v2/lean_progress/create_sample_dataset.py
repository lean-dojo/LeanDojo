#!/usr/bin/env python3
"""
Creates a toy LeanProgress dataset showing how to store remaining-step labels.

Each line:
{
    "goal": "... Lean goal ...",
    "prefix": "... optional proof prefix ...",
    "tactic": "... candidate tactic ...",
    "steps_remaining": <non-negative integer>
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

SAMPLE_DATA = [
    {
        "goal": "n : ℕ ⊢ gcd n n = n",
        "prefix": dedent(
            """
            have : n.gcd n ∣ n := by
              exact gcd_dvd_right n n
            """
        ).strip(),
        "tactic": "simpa [Nat.gcd_comm] using Nat.gcd_self n",
        "steps_remaining": 1,
    },
    {
        "goal": "a b : ℝ, ha : a = b ⊢ b = a",
        "prefix": "",
        "tactic": "exact ha.symm",
        "steps_remaining": 0,
    },
    {
        "goal": "G : Group, g h : G ⊢ g * h = h * g",
        "prefix": "",
        "tactic": "sorry",
        "steps_remaining": 12,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sample LeanProgress dataset in JSONL format."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("raid/data/sample_leanprogress_dataset.jsonl"),
        help="Where to save the sample JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in SAMPLE_DATA:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(SAMPLE_DATA)} LeanProgress examples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
