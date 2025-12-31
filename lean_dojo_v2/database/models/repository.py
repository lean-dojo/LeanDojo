from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .premises import PremiseFile
from .theorems import Theorem


@dataclass
class Repository:
    """
    Repository class represents a repository with various attributes and methods
    to manage theorems and premise files.

    Attributes:
        url (str): URL of the repository.
        name (str): Name of the repository.
        commit (str): Commit hash of the repository.
        lean_version (str): Version of Lean used in the repository.
        lean_dojo_version (str): Version of Lean Dojo used in the repository.
        metadata (Dict[str, str]): Metadata associated with the repository.
        proven_theorems (List[Theorem]): List of proven theorems.
        sorry_theorems_proved (List[Theorem]): List of sorry theorems that have been proved.
        sorry_theorems_unproved (List[Theorem]): List of sorry theorems that are unproved.
        premise_files (List[PremiseFile]): List of premise files.
        files_traced (List[Path]): List of traced files.
        pr_url (Optional[str]): URL of the pull request.
    """

    url: str
    name: str
    commit: str
    lean_version: str
    lean_dojo_version: str
    metadata: Dict[str, str]
    proven_theorems: List[Theorem] = field(default_factory=list)
    sorry_theorems_proved: List[Theorem] = field(default_factory=list)
    sorry_theorems_unproved: List[Theorem] = field(default_factory=list)
    premise_files: List[PremiseFile] = field(default_factory=list)
    files_traced: List[Path] = field(default_factory=list)
    pr_url: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, Repository):
            return NotImplemented
        return (
            self.url == other.url
            and self.name == other.name
            and self.commit == other.commit
            and self.lean_version == other.lean_version
            and self.lean_dojo_version == other.lean_dojo_version
        )

    def __hash__(self):
        return hash(
            (
                self.url,
                self.name,
                self.commit,
                self.lean_version,
                self.lean_dojo_version,
            )
        )

    def __str__(self) -> str:
        """Directory-friendly string representation.

        Example: "repo-name_commit-hash"
        """
        return f"{self.name}_{self.commit}"

    @property
    def total_theorems(self) -> int:
        return self.num_proven_theorems + self.num_sorry_theorems

    @property
    def num_proven_theorems(self) -> int:
        return len(self.proven_theorems)

    @property
    def num_sorry_theorems_proved(self) -> int:
        return len(self.sorry_theorems_proved)

    @property
    def num_sorry_theorems_unproved(self) -> int:
        return len(self.sorry_theorems_unproved)

    @property
    def num_sorry_theorems(self) -> int:
        return self.num_sorry_theorems_proved + self.num_sorry_theorems_unproved

    @property
    def num_premise_files(self) -> int:
        return len(self.premise_files)

    @property
    def num_premises(self) -> int:
        return sum(len(pf.premises) for pf in self.premise_files)

    @property
    def num_files_traced(self) -> int:
        return len(self.files_traced)

    @property
    def get_all_theorems(self) -> List[Theorem]:
        return (
            self.proven_theorems
            + self.sorry_theorems_proved
            + self.sorry_theorems_unproved
        )

    def get_theorem(self, full_name: str, file_path: str) -> Optional[Theorem]:
        for thm_list in [
            self.proven_theorems,
            self.sorry_theorems_proved,
            self.sorry_theorems_unproved,
        ]:
            for thm in thm_list:
                if thm.full_name == full_name and (
                    str(thm.file_path) == file_path
                    or (file_path == "" and str(thm.file_path) == ".")
                ):
                    return thm
        return None

    def update_theorem(self, theorem: Theorem) -> None:
        for thm_list in [
            self.proven_theorems,
            self.sorry_theorems_proved,
            self.sorry_theorems_unproved,
        ]:
            for i, thm in enumerate(thm_list):
                if thm.is_same_theorem(theorem):
                    thm_list[i] = theorem
                    return
        raise ValueError(f"Theorem '{theorem.full_name}' not found.")

    def get_premise_file(self, path: str) -> Optional[PremiseFile]:
        return next((pf for pf in self.premise_files if str(pf.path) == path), None)

    def get_file_traced(self, path: str) -> Optional[Path]:
        return next((f for f in self.files_traced if str(f) == path), None)

    @classmethod
    def from_dict(cls, data: Dict) -> Repository:
        if not all(
            key in data
            for key in [
                "url",
                "name",
                "commit",
                "lean_version",
                "lean_dojo_version",
                "metadata",
            ]
        ):
            raise ValueError("Invalid Repository data format")
        if "date_processed" not in data["metadata"]:
            raise ValueError("Metadata must contain the 'date_processed' key")

        metadata = data["metadata"].copy()
        if isinstance(metadata["date_processed"], str):
            metadata["date_processed"] = datetime.datetime.fromisoformat(
                metadata["date_processed"]
            )

        repo = cls(
            url=data["url"],
            name=data["name"],
            commit=data["commit"],
            lean_version=data["lean_version"],
            lean_dojo_version=data["lean_dojo_version"],
            metadata=metadata,
            files_traced=[],
            pr_url=data.get("pr_url"),
        )

        if all(
            key in data
            for key in ["theorems_folder", "premise_files_corpus", "files_traced"]
        ):
            if not all(
                os.path.exists(data[key])
                for key in ["theorems_folder", "premise_files_corpus", "files_traced"]
            ):
                raise ValueError(
                    "Paths to data cannot be empty when creating repo from dataset"
                )

            theorems_folder = Path(data["theorems_folder"])
            for file in theorems_folder.glob("*.json"):
                with open(file, "r") as f:
                    theorem_data = json.load(f)
                for t_data in tqdm(theorem_data):
                    theorem = Theorem.from_dict(t_data, repo.url, repo.commit)
                    if any(
                        "sorry" in step.tactic
                        for step in (theorem.traced_tactics or [])
                    ):
                        repo.sorry_theorems_unproved.append(theorem)
                    else:
                        repo.proven_theorems.append(theorem)

            with open(data["premise_files_corpus"], "r") as f:
                for line in f:
                    premise_file_data = json.loads(line)
                    premise_file = PremiseFile.from_dict(premise_file_data)
                    repo.premise_files.append(premise_file)

            with open(data["files_traced"], "r") as f:
                for line in f:
                    traced_file_data = json.loads(line)
                    repo.files_traced.append(Path(traced_file_data["traced_file_path"]))
        else:
            # Process theorems and premises from the existing data structure
            repo.proven_theorems = [
                Theorem.from_dict(t, repo.url, repo.commit)
                for t in data.get("proven_theorems", [])
            ]
            repo.sorry_theorems_proved = [
                Theorem.from_dict(t, repo.url, repo.commit)
                for t in data.get("sorry_theorems_proved", [])
            ]
            repo.sorry_theorems_unproved = [
                Theorem.from_dict(t, repo.url, repo.commit)
                for t in data.get("sorry_theorems_unproved", [])
            ]
            repo.premise_files = [
                PremiseFile.from_dict(pf) for pf in data.get("premise_files", [])
            ]
            repo.files_traced = [Path(file) for file in data.get("files_traced", [])]

        return repo

    def to_dict(self) -> Dict:
        metadata_copy = self.metadata.copy()
        if isinstance(metadata_copy["date_processed"], datetime.datetime):
            metadata_copy["date_processed"] = metadata_copy[
                "date_processed"
            ].isoformat()
        return {
            "url": self.url,
            "name": self.name,
            "commit": self.commit,
            "lean_version": self.lean_version,
            "lean_dojo_version": self.lean_dojo_version,
            "metadata": metadata_copy,
            "total_theorems": self.total_theorems,
            "num_proven_theorems": self.num_proven_theorems,
            "num_sorry_theorems": self.num_sorry_theorems,
            "num_sorry_theorems_proved": self.num_sorry_theorems_proved,
            "num_sorry_theorems_unproved": self.num_sorry_theorems_unproved,
            "num_premise_files": self.num_premise_files,
            "num_premises": self.num_premises,
            "num_files_traced": self.num_files_traced,
            "proven_theorems": [t.to_dict() for t in self.proven_theorems],
            "sorry_theorems_proved": [t.to_dict() for t in self.sorry_theorems_proved],
            "sorry_theorems_unproved": [
                t.to_dict() for t in self.sorry_theorems_unproved
            ],
            "premise_files": [pf.to_dict() for pf in self.premise_files],
            "files_traced": [str(file) for file in self.files_traced],
            "pr_url": self.pr_url,
        }

    def change_sorry_to_proven(self, theorem: Theorem, log_file: str) -> None:
        if theorem in self.sorry_theorems_unproved:
            self.sorry_theorems_unproved.remove(theorem)
            self.sorry_theorems_proved.append(theorem)

            message = f"Theorem proved: {theorem.full_name} in {theorem.file_path} for repo {self.name} (commit: {self.commit})"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"{timestamp} - {message}\n"

            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            with open(log_file, "a") as f:
                f.write(log_entry)
        else:
            raise ValueError(
                "The theorem is not in the list of unproved sorry theorems."
            )
