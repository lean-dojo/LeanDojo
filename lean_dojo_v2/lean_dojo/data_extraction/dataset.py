"""
Benchmark dataset generation for Lean 4 theorem proving.

This module provides functionality to generate training datasets from Lean 4 repositories
by tracing theorem proofs and splitting them into train/validation/test sets.
"""

import json
import os
import random
from collections import defaultdict
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import networkx as nx
from loguru import logger

from lean_dojo_v2.utils.constants import LEAN4_PACKAGES_DIR, __version__

# Import utility functions from the lean_dojo module
from .lean import LeanGitRepo, get_lean4_version_from_config, is_supported_version
from .trace import trace
from .traced_data import TracedRepo, TracedTheorem

# Set random seed for reproducibility
random.seed(3407)  # https://arxiv.org/abs/2109.08203

# Type aliases for better readability
SPLIT_NAME = str  # train/val/test
SPLIT = Dict[SPLIT_NAME, List[TracedTheorem]]
SPLIT_STRATEGY = str


# =============================================================================
# DATA SPLITTING FUNCTIONS
# =============================================================================


def _split_sequentially(
    traced_theorems: List[TracedTheorem], num_val: int, num_test: int
) -> SPLIT:
    """Split theorems sequentially into train/val/test sets."""
    num_theorems = len(traced_theorems)
    num_train = num_theorems - num_val - num_test

    return {
        "train": traced_theorems[:num_train],
        "val": traced_theorems[num_train : num_train + num_val],
        "test": traced_theorems[num_train + num_val :],
    }


def split_randomly(
    traced_theorems: List[TracedTheorem], num_val: int, num_test: int
) -> SPLIT:
    """Split theorems randomly into train/val/test sets."""
    logger.info("Splitting theorems randomly")
    traced_theorems = copy(traced_theorems)
    random.shuffle(traced_theorems)
    return _split_sequentially(traced_theorems, num_val, num_test)


def split_by_premise(
    traced_theorems: List[TracedTheorem], num_val: int, num_test: int
) -> SPLIT:
    """
    Split theorems so that val/test proofs rely on novel premises not in train.

    This ensures that validation and test sets contain theorems that use premises
    not seen during training, making the evaluation more realistic.
    """
    logger.info("Splitting theorems by novel premises")

    num_val_test = num_val + num_test
    theorems_val_test = set()

    # Map each premise to theorems that use it
    theorems_by_premises = defaultdict(list)
    for theorem in traced_theorems:
        for premise in theorem.get_premise_full_names():
            theorems_by_premises[premise].append(theorem)

    # Sort premises by usage (ascending) to prioritize rare premises
    premises_sorted = sorted(theorems_by_premises.items(), key=lambda x: len(x[1]))

    # Assign theorems to val/test based on novel premises
    for _, theorems in premises_sorted:
        if len(theorems_val_test) < num_val_test:
            theorems_val_test.update(theorems)

    # Remaining theorems go to training
    theorems_train = [t for t in traced_theorems if t not in theorems_val_test]
    theorems_val_test = list(theorems_val_test)
    random.shuffle(theorems_val_test)

    return {
        "train": theorems_train,
        "val": theorems_val_test[:num_val],
        "test": theorems_val_test[num_val:],
    }


def split_data(
    traced_repo: TracedRepo, num_val_pct: float = 0.2, num_test_pct: float = 0.2
) -> Dict[SPLIT_STRATEGY, SPLIT]:
    """
        Split traced theorems into training, validation, and test sets.
    c
        Args:
            traced_repo: Repository containing theorems to split
            num_val_pct: Percentage of theorems for validation (default: 0.2)
            num_test_pct: Percentage of theorems for testing (default: 0.2)

        Returns:
            Dictionary mapping split strategies to their corresponding splits
    """
    # Exclude Lean 4 repository theorems
    traced_theorems = [
        thm for thm in traced_repo.get_traced_theorems() if not thm.repo.is_lean4
    ]

    num_theorems = len(traced_theorems)
    num_val = int(num_theorems * num_val_pct)
    num_test = int(num_theorems * num_test_pct)

    logger.info(
        f"Total theorems: {num_theorems}, " f"Validation: {num_val}, Test: {num_test}"
    )

    return {
        "random": split_randomly(traced_theorems, num_val, num_test),
        "novel_premises": split_by_premise(traced_theorems, num_val, num_test),
    }


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def _get_file_path(traced_repo: TracedRepo, thm: TracedTheorem) -> str:
    """Get the file path for a theorem, handling dependencies."""
    if thm.repo == traced_repo.repo:
        # Theorem belongs to the traced repo itself
        return str(thm.theorem.file_path)
    else:
        # Theorem belongs to a dependency
        for name, dep in traced_repo.dependencies.items():
            if dep == thm.repo:
                return f"{LEAN4_PACKAGES_DIR}/{name}/{thm.theorem.file_path}"
        raise ValueError(f"Unable to find dependency {thm.repo}")


def export_proofs(
    splits: Dict[SPLIT_STRATEGY, SPLIT], dst_path: Path, traced_repo: TracedRepo
) -> int:
    """
    Export theorem proofs to JSON files organized by split strategy.

    Args:
        splits: Dictionary mapping strategies to theorem splits
        dst_path: Destination directory for exported proofs
        traced_repo: Repository containing the theorems

    Returns:
        Total number of theorems exported
    """
    # Collect all unique theorems across all strategies
    # (all strategies contain the same theorems, just split differently)
    unique_theorems = set()
    
    for strategy, split in splits.items():
        strategy_dir = dst_path / strategy
        strategy_dir.mkdir(parents=True)

        for split_name, theorems in split.items():
            data = []
            num_tactics = 0

            for theorem in theorems:
                # Track unique theorems by their identity
                unique_theorems.add(id(theorem))
                
                # Filter out tactics with "no goals" or containing "·"
                tactics = [
                    {
                        "tactic": tactic.tactic,
                        "annotated_tactic": tactic.get_annotated_tactic(),
                        "state_before": tactic.state_before,
                        "state_after": tactic.state_after,
                    }
                    for tactic in theorem.get_traced_tactics()
                    if (tactic.state_before != "no goals" and "·" not in tactic.tactic)
                ]
                num_tactics += len(tactics)

                # Get theorem statement if available
                theorem_statement = None
                if (
                    theorem.has_tactic_proof()
                    and theorem.get_tactic_proof() is not None
                ):
                    theorem_statement = theorem.get_theorem_statement()

                data.append(
                    {
                        "url": traced_repo.repo.url,
                        "commit": traced_repo.repo.commit,
                        "file_path": _get_file_path(traced_repo, theorem),
                        "full_name": theorem.theorem.full_name,
                        "theorem_statement": theorem_statement,
                        "start": list(theorem.start),
                        "end": list(theorem.end),
                        "traced_tactics": tactics,
                    }
                )

            # Save to JSON file
            output_path = strategy_dir / f"{split_name}.json"
            json.dump(data, output_path.open("wt"))

            logger.info(
                f"Saved {len(theorems)} theorems with {num_tactics} tactics "
                f"to {output_path}"
            )

    total_theorems = len(unique_theorems)
    logger.info(f"Total theorems exported: {total_theorems}")
    return total_theorems


def export_premises(traced_repo: TracedRepo, dst_path: Path) -> tuple[int, int]:
    """
    Export premise definitions and traced file information.

    Args:
        traced_repo: Repository containing traced files
        dst_path: Destination directory for exported data

    Returns:
        Tuple of (number of premises, number of traced files)
    """
    # Export premises in topological order
    corpus_path = dst_path / "corpus.jsonl"
    num_premises = 0

    with corpus_path.open("wt") as output:
        graph = traced_repo.traced_files_graph

        if not graph:
            logger.warning("No traced files found in repository")
            pass
        else:
            for node in reversed(list(nx.topological_sort(graph))):
                traced_file = graph.nodes[node]["traced_file"]
                imports = [str(imp) for imp in graph.successors(node)]
                premises = traced_file.get_premise_definitions()
                num_premises += len(premises)

                output.write(
                    json.dumps(
                        {
                            "path": str(traced_file.path),
                            "imports": imports,
                            "premises": premises,
                        }
                    )
                    + "\n"
                )

    logger.info(
        f"Exported {num_premises} premises from "
        f"{len(traced_repo.traced_files)} files to {corpus_path}"
    )

    # Export traced file paths
    traced_files_path = dst_path / "traced_files.jsonl"
    with traced_files_path.open("wt") as output:
        for traced_file in traced_repo.traced_files:
            file_path = traced_file.lean_file.path
            output.write(json.dumps({"traced_file_path": str(file_path)}) + "\n")

    return num_premises, len(traced_repo.traced_files)


def export_metadata(traced_repo: TracedRepo, dst_path: Path, **kwargs) -> None:
    """
    Export repository metadata to JSON file.

    Args:
        traced_repo: Repository containing metadata
        dst_path: Destination directory
        **kwargs: Additional metadata to include
    """
    metadata = dict(kwargs)
    metadata.update(
        {
            "creation_time": str(datetime.now()),
            "from_repo": {
                "url": traced_repo.repo.url,
                "commit": traced_repo.repo.commit,
            },
            "leandojo_version": __version__,
        }
    )

    metadata_path = dst_path / "metadata.json"
    json.dump(metadata, metadata_path.open("wt"))


def export_data(
    traced_repo: TracedRepo,
    splits: Dict[SPLIT_STRATEGY, SPLIT],
    dst_path: Union[str, Path],
    **kwargs,
) -> tuple[int, int, int]:
    """
    Export complete dataset from traced repository.

    Args:
        traced_repo: Repository containing data to export
        splits: Dictionary mapping strategies to theorem splits
        dst_path: Destination directory for exported data
        **kwargs: Additional metadata

    Returns:
        Tuple of (num_premises, num_files_traced, total_theorems)
    """
    if isinstance(dst_path, str):
        dst_path = Path(dst_path)

    # Clear existing directory
    if dst_path.exists():
        import shutil

        shutil.rmtree(dst_path)

    # Export all components
    total_theorems = export_proofs(splits, dst_path, traced_repo)
    num_premises, num_files_traced = export_premises(traced_repo, dst_path)
    export_metadata(traced_repo, dst_path, **kwargs)

    return num_premises, num_files_traced, total_theorems


# =============================================================================
# LEAN ENVIRONMENT SETUP
# =============================================================================


def setup_lean_environment(repo: LeanGitRepo) -> None:
    """
    Configure Lean environment to match repository requirements.

    Args:
        repo: Repository to configure environment for
    """
    # we need to change the toolchain version that LeanAgent uses
    # to match the repo we are currently tracing
    config = repo.get_config("lean-toolchain")
    v = get_lean4_version_from_config(config["content"])
    v = v[1:]  # ignore "v" at beginning

    lean_dir_1 = f"/.elan/toolchains/leanprover--lean4---{v}"
    lean_dir_2 = f"~/.elan/toolchains/leanprover--lean4---{v}"

    if os.path.exists(lean_dir_1):
        lean_dir = lean_dir_1
    else:
        lean_dir = lean_dir_2

    os.environ["LEAN4_PATH"] = lean_dir
    os.environ["PATH"] = f"{lean_dir}/bin:{os.environ.get('PATH', '')}"


# =============================================================================
# MAIN DATASET GENERATION
# =============================================================================


def generate_benchmark(
    repo: LeanGitRepo, dst_dir: str, build_deps: bool
) -> tuple[TracedRepo, int]:
    """
    Generate benchmark dataset from Lean 4 repository.

    This function:
    1. Clones and checks out the specified repository
    2. Configures Lean environment to match repository requirements
    3. Traces the repository to extract theorem proofs
    4. Splits data into train/validation/test sets
    5. Exports all data to the specified directory

    Args:
        url: Git repository URL
        commit: Commit hash to check out
        dst_dir: Destination directory for dataset

    Returns:
        Tuple of (traced_repo, num_premises, num_files_traced, total_theorems)
        Returns (None, 0, 0, 10) if tracing fails
    """
    logger.info(f"Repository: {repo}")

    # Configure Lean environment
    setup_lean_environment(repo)

    # Trace repository
    traced_repo = trace(repo, build_deps=build_deps)
    logger.info("Repository tracing completed successfully")

    # Generate and export dataset
    if os.path.exists(dst_dir):
        import shutil

        shutil.rmtree(dst_dir)

    splits = split_data(traced_repo)
    logger.info("Data splitting completed")

    num_premises, num_files_traced, total_theorems = export_data(
        traced_repo, splits, dst_dir
    )

    logger.info(
        f"Dataset generation complete: "
        f"{total_theorems} theorems, "
        f"{num_premises} premises, "
        f"{num_files_traced} files"
    )

    return traced_repo, total_theorems
