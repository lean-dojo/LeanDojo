from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

import lean_dojo_v2.lean_dojo
from lean_dojo_v2.lean_dojo import LeanGitRepo, generate_benchmark
from lean_dojo_v2.utils.constants import (
    BATCH_SIZE,
    DATA_DIR,
    KNOWN_REPOSITORIES,
    RAID_DIR,
)
from lean_dojo_v2.utils.difficulty import (
    calculate_theorem_difficulty,
    categorize_difficulty,
    print_difficulty_summary,
)
from lean_dojo_v2.utils.filesystem import remove_dir
from lean_dojo_v2.utils.git import (
    find_and_save_compatible_commits,
    get_version_for_commit,
    search_github_repositories,
)
from lean_dojo_v2.utils.lean import get_lean4_version_from_config

from .models import Repository, Theorem


class DynamicDatabase:
    """
    A class that manages a collection of repositories containing Lean theorem proofs.
    The DynamicDatabase class provides functionality for:
    1. Managing repositories (adding, retrieving, updating, deleting)
    2. Generating merged datasets from multiple repositories
    3. Splitting theorem data for training/validation/testing
    4. Exporting proofs, corpus data, and metadata
    """

    def __init__(self, json_path: str = "dynamic_database.json"):
        """initialization hook for dataclass."""
        # Ensure the directory for json_path exists
        json_dir = os.path.dirname(json_path)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)

        self.json_path = json_path
        self.repositories = []

        logger.info(f"Initializing new database at {self.json_path}")
        self.to_json(self.json_path)

    SPLIT = Dict[str, List[Theorem]]

    def export_merged_data(
        self,
        repositories: List[Repository],
        output_path: Path,
    ) -> None:
        """
        Export a dataset from a repository in the database.

        :param repo: repository to include in the dataset.
        """
        random.seed(3407)

        output_path.mkdir(parents=True, exist_ok=True)

        all_theorems = {}
        all_traced_files = set()

        for repository in repositories:
            for theorem in repository.get_all_theorems:
                key = (
                    theorem.file_path,
                    theorem.full_name,
                    list(theorem.start)[0],
                    list(theorem.start)[1],
                    list(theorem.end)[0],
                    list(theorem.end)[1],
                )
                date_processed = repository.metadata["date_processed"]
                if isinstance(date_processed, str):
                    date_processed = datetime.fromisoformat(date_processed)
                if key not in all_theorems or date_processed > all_theorems[key][1]:
                    all_theorems[key] = (theorem, date_processed)

            all_traced_files.update(repository.files_traced)

        theorems = [t for t, _ in all_theorems.values()]
        splits = self._split_data(theorems)

        remove_dir(output_path)

        self._export_proofs(splits, output_path)
        logger.info(f"Exported proofs to {output_path}")

        self._merge_corpus(repositories, output_path)
        logger.info(f"Merged and exported corpus to {output_path}")

        self._export_traced_files(all_traced_files, output_path)
        logger.info(f"Exported traced files to {output_path}")

        self._export_metadata(repositories, output_path)
        logger.info(f"Exported metadata to {output_path}")

    def _merge_corpus(self, repos: List[Repository], output_path: Path) -> None:
        merged_corpus = {}
        for repo in repos:
            for premise_file in repo.premise_files:
                file_data = {
                    "path": str(premise_file.path),
                    "imports": premise_file.imports,
                    "premises": [
                        {
                            "full_name": premise.full_name,
                            "code": premise.code,
                            "start": list(premise.start),
                            "end": list(premise.end),
                            "kind": premise.kind,
                        }
                        for premise in premise_file.premises
                    ],
                }
                path = file_data["path"]
                if path not in merged_corpus:
                    merged_corpus[path] = json.dumps(file_data)

        with open(output_path / "corpus.jsonl", "w") as f:
            for line in merged_corpus.values():
                f.write(line + "\n")

    def _split_data(
        self,
        theorems: List[Theorem],
        num_val_pct: float = 0.2,
        num_test_pct: float = 0.2,
    ) -> Dict[str, SPLIT]:
        num_theorems = len(theorems)
        num_val = int(num_theorems * num_val_pct)
        num_test = int(num_theorems * num_test_pct)

        return {
            "random": self._split_randomly(theorems, num_val, num_test),
            "novel_premises": self._split_by_premise(theorems, num_val, num_test),
        }

    def _split_randomly(
        self, theorems: List[Theorem], num_val: int, num_test: int
    ) -> SPLIT:
        random.shuffle(theorems)
        num_train = len(theorems) - num_val - num_test
        return {
            "train": theorems[:num_train],
            "val": theorems[num_train : num_train + num_val],
            "test": theorems[num_train + num_val :],
        }

    def _split_by_premise(
        self, theorems: List[Theorem], num_val: int, num_test: int
    ) -> SPLIT:
        num_val_test = num_val + num_test
        theorems_val_test = []

        theorems_by_premises = defaultdict(list)
        for t in theorems:
            if t.traced_tactics:
                for tactic in t.traced_tactics:
                    for annotation in tactic.annotated_tactic[1]:
                        theorems_by_premises[annotation.full_name].append(t)

        theorems_by_premises = sorted(
            theorems_by_premises.items(), key=lambda x: len(x[1])
        )

        for _, thms in theorems_by_premises:
            if len(theorems_val_test) < num_val_test:
                theorems_val_test.extend(
                    [t for t in thms if t not in theorems_val_test]
                )
            else:
                break

        theorems_train = [t for t in theorems if t not in theorems_val_test]
        random.shuffle(theorems_val_test)

        return {
            "train": theorems_train,
            "val": theorems_val_test[:num_val],
            "test": theorems_val_test[num_val:],
        }

    def _export_proofs(self, splits: Dict[str, SPLIT], output_path: Path) -> None:
        for strategy, split in splits.items():
            strategy_dir = output_path / strategy
            strategy_dir.mkdir(parents=True, exist_ok=True)

            for name, theorems in split.items():
                data = []
                for thm in theorems:
                    tactics = [
                        {
                            "tactic": t.tactic,
                            "annotated_tactic": [
                                t.annotated_tactic[0],
                                [
                                    {
                                        "full_name": a.full_name,
                                        "def_path": str(a.def_path),
                                        "def_pos": list(a.def_pos),
                                        "def_end_pos": list(a.def_end_pos),
                                    }
                                    for a in t.annotated_tactic[1]
                                ],
                            ],
                            "state_before": t.state_before,
                            "state_after": t.state_after,
                        }
                        for t in thm.traced_tactics
                        if t.state_before != "no goals" and "·" not in t.tactic
                    ]
                    data.append(
                        {
                            "url": thm.url,
                            "commit": thm.commit,
                            "file_path": str(thm.file_path),
                            "full_name": thm.full_name,
                            "theorem_statement": thm.theorem_statement,
                            "start": list(thm.start),
                            "end": list(thm.end),
                            "traced_tactics": tactics,
                        }
                    )

                output_file = strategy_dir / f"{name}.json"
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

    def _export_traced_files(
        self, all_traced_files: Set[Path], output_path: Path
    ) -> None:
        with open(output_path / "traced_files.jsonl", "w") as f:
            for file in all_traced_files:
                f.write(json.dumps({"traced_file_path": str(file)}) + "\n")

    def _export_metadata(self, repos: List[Repository], output_path: Path) -> None:
        metadata = {
            "repositories": [
                {
                    "url": repo.url,
                    "name": repo.name,
                    "commit": repo.commit,
                    "lean_version": repo.lean_version,
                    "lean_dojo_version": repo.lean_dojo_version,
                    "metadata": repo.metadata,
                }
                for repo in repos
            ],
            "total_theorems": sum(repo.total_theorems for repo in repos),
            "num_proven_theorems": sum(repo.num_proven_theorems for repo in repos),
            "num_sorry_theorems": sum(repo.num_sorry_theorems for repo in repos),
            "num_premise_files": sum(repo.num_premise_files for repo in repos),
            "num_premises": sum(repo.num_premises for repo in repos),
            "num_files_traced": sum(repo.num_files_traced for repo in repos),
        }

        for repo_data in metadata["repositories"]:
            if isinstance(repo_data["metadata"]["date_processed"], datetime):
                repo_data["metadata"]["date_processed"] = repo_data["metadata"][
                    "date_processed"
                ].isoformat()

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def add_repository(self, repo: Repository) -> None:
        if repo not in self.repositories:
            self.repositories.append(repo)
        else:
            logger.info(
                f"Repository '{repo.url}' with commit '{repo.commit}' already exists in the database."
            )

    def get_repository(self, url: str, commit: str) -> Optional[Repository]:
        for repo in self.repositories:
            if repo.url == url and repo.commit == commit:
                return repo
        return None

    def update_repository(self, updated_repo: Repository) -> None:
        for i, repo in enumerate(self.repositories):
            if repo == updated_repo:
                self.repositories[i] = updated_repo
                logger.info(
                    f"Updated repository: {updated_repo.url} (commit: {updated_repo.commit})"
                )
                return
        logger.error(
            f"Repository '{updated_repo.url}' with commit '{updated_repo.commit}' not found for update."
        )
        raise ValueError(
            f"Repository '{updated_repo.url}' with commit '{updated_repo.commit}' not found."
        )

    def print_database_contents(self):
        logger.info("Current database contents:")
        for repo in self.repositories:
            logger.info(f"  - {repo.url} (commit: {repo.commit})")

    def delete_repository(self, url: str, commit: str) -> None:
        for i, repo in enumerate(self.repositories):
            if repo.url == url and repo.commit == commit:
                del self.repositories[i]
                return
        raise ValueError(f"Repository '{url}' with commit '{commit}' not found.")

    def to_dict(self) -> Dict:
        return {"repositories": [repo.to_dict() for repo in self.repositories]}

    def _load_from_json(self, file_path: str) -> None:
        """Load data from JSON file without creating new instances."""
        with open(file_path, "r") as f:
            data = json.load(f)

        if "repositories" not in data:
            raise ValueError("Invalid DynamicDatabase data format")

        # Clear existing repositories and load new ones
        self.repositories.clear()
        for repo_data in data["repositories"]:
            repo = Repository.from_dict(repo_data)
            self.add_repository(repo)

    @classmethod
    def from_dict(cls, data: Dict) -> DynamicDatabase:
        if "repositories" not in data:
            raise ValueError("Invalid DynamicDatabase data format")
        db = cls()
        for repo_data in data["repositories"]:
            repo = Repository.from_dict(repo_data)
            db.add_repository(repo)
        return db

    def to_json(self, file_path: Optional[str] = None) -> None:
        """Serialize the database to a JSON file."""
        if file_path is None:
            file_path = self.json_path
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, file_path: str) -> DynamicDatabase:
        """Create a new instance from JSON file."""
        db = cls()
        db._load_from_json(file_path)
        return db

    def update_json(self, file_path: str) -> None:
        """Update an existing JSON file with the current database state."""
        try:
            existing_db = self.from_json(file_path)
        except FileNotFoundError:
            existing_db = DynamicDatabase()

        for repo in self.repositories:
            existing_db.update_repository(repo)

        existing_db.to_json(file_path)

    def sort_repositories_by_difficulty(
        self,
    ) -> Tuple[List[Repository], Dict, List[float]]:
        """
        Sorts repositories by the difficulty of their theorems and optionally saves results.

        Returns:
            Tuple containing:
            - List of repositories sorted by difficulty (most easy theorems first)
            - Dictionary mapping repositories to categorized theorems
            - List of percentile thresholds [33rd, 67th]
        """
        difficulties_by_repo = defaultdict(list)
        all_difficulties = []

        for repo in self.repositories:
            for theorem in repo.get_all_theorems:
                difficulty = calculate_theorem_difficulty(theorem)
                theorem.difficulty_rating = difficulty
                difficulties_by_repo[repo].append(
                    (
                        theorem.full_name,
                        str(theorem.file_path),
                        tuple(theorem.start),
                        tuple(theorem.end),
                        difficulty,
                    )
                )
                if difficulty is not None:
                    all_difficulties.append(difficulty)

            self.update_repository(repo)

        percentiles = np.percentile(all_difficulties, [33, 67])

        categorized_theorems = defaultdict(lambda: defaultdict(list))

        for repo, theorems in difficulties_by_repo.items():
            for theorem_name, file_path, start, end, difficulty in theorems:
                category = categorize_difficulty(difficulty, percentiles)
                categorized_theorems[repo][category].append(
                    (theorem_name, file_path, start, end, difficulty)
                )

        for repo in categorized_theorems:
            to_distribute = categorized_theorems[repo]["To_Distribute"]
            chunk_size = len(to_distribute) // 3
            for i, category in enumerate(["Easy", "Medium", "Hard"]):
                start = i * chunk_size
                end = start + chunk_size if i < 2 else None
                categorized_theorems[repo][category].extend(to_distribute[start:end])
            del categorized_theorems[repo]["To_Distribute"]

        # Sort repositories based on the number of easy theorems
        sorted_repos = sorted(
            categorized_theorems.keys(),
            key=lambda r: len(categorized_theorems[r]["Easy"]),
            reverse=True,
        )

        # Save results if path is provided
        if self.json_path:
            self.to_json(self.json_path)

            # Print summary of theorem difficulties
            print_difficulty_summary(categorized_theorems, percentiles)

        return sorted_repos

    def trace_repository(
        self, url: str, commit: str, build_deps: bool
    ) -> Optional[Repository]:
        """
        Trace a LeanGitRepo and create a Repository object with processed data.

        Args:
            lean_git_repo: The LeanGitRepo object to trace

        Returns:
            Repository object with processed theorem data if successful, None if failed
        """
        url = url.replace(".git", "")
        repo = LeanGitRepo(url, commit)

        v = get_version_for_commit(repo.url, repo.commit)

        dst_dir = os.path.join(RAID_DIR, DATA_DIR, str(repo))

        logger.info(f"Generating benchmark at {dst_dir}")

        traced_repo, total_theorems = generate_benchmark(repo, dst_dir, build_deps)

        if not traced_repo:
            logger.info(f"Failed to trace {repo.url}")
            return None

        if (
            total_theorems < 3 * BATCH_SIZE
        ):  # Should be enough theorems for train/val/test
            logger.info(f"Not enough theorems found in {repo.url}")
            return None

        config = repo.get_config("lean-toolchain")
        v = get_lean4_version_from_config(config["content"])
        theorems_folder = os.path.join(dst_dir, "random")
        premise_files_corpus = os.path.join(dst_dir, "corpus.jsonl")
        files_traced = os.path.join(dst_dir, "traced_files.jsonl")
        data = {
            "url": repo.url,
            "name": "/".join(repo.url.split("/")[-2:]),
            "commit": repo.commit,
            "lean_version": v,
            "lean_dojo_version": lean_dojo_v2.lean_dojo.__version__,
            "metadata": {
                "date_processed": datetime.now(),
            },
            "theorems_folder": theorems_folder,
            "premise_files_corpus": premise_files_corpus,
            "files_traced": files_traced,
            "pr_url": None,
        }

        repository = Repository.from_dict(data)
        self.to_json(self.json_path)

        return repository

    def discover_repositories(
        self,
        num_repos: int,
        curriculum_learning: bool,
        build_deps: bool = True,
    ) -> List[LeanGitRepo]:
        """
        Initialize the database and discover repositories.

        Args:
            num_repos: Number of repositories to discover
            curriculum_learning: Whether to enable curriculum learning

        Returns:
            List of LeanGitRepo objects
        """
        logger.info("Starting the main process")
        logger.info(f"Found {num_repos} repositories")

        lean_git_repos = search_github_repositories(
            "Lean", num_repos, KNOWN_REPOSITORIES
        )

        for repo in lean_git_repos:
            logger.info(f"Processing {repo.url}")
            traced_repo = self.trace_repository(
                repo.url, repo.commit, build_deps=build_deps
            )
            if traced_repo:
                self.add_repository(traced_repo)
                logger.info(f"Successfully added repo {traced_repo.url}")
            else:
                logger.info(f"Failed to add repo {repo.url}")

        # If curriculum learning is enabled, initialize repositories and sort them by difficulty
        if curriculum_learning:
            logger.info("Starting curriculum learning")
            lean_git_repos = self.sort_repositories_by_difficulty()
        else:
            logger.info("Starting without curriculum learning")

        logger.info("Finding compatible repositories...")
        repo_info_file = f"{RAID_DIR}/{DATA_DIR}/repo_info_compatible.json"
        updated_repos = find_and_save_compatible_commits(repo_info_file, lean_git_repos)

        lean_git_repos = [
            LeanGitRepo(repo["url"], repo["commit"]) for repo in updated_repos
        ]

        return lean_git_repos
