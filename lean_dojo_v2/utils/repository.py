"""
Utility functions for managing repository data, operations, and skipping functionality.
"""

import json
import os
from typing import List, Tuple

from lean_dojo_v2.database.models import Repository
from lean_dojo_v2.utils.constants import DATA_DIR, RAID_DIR


def save_sorted_repos(sorted_repos: List[Repository], file_path: str) -> None:
    """
    Saves the sorted repositories to a file.

    Args:
        sorted_repos: List of Repository objects to save
        file_path: Path to the file where repositories will be saved
    """
    sorted_repo_data = [
        {"url": repo.url, "commit": repo.commit, "name": repo.name}
        for repo in sorted_repos
    ]
    with open(file_path, "w") as f:
        json.dump(sorted_repo_data, f, indent=2)


def load_sorted_repos(file_path: str) -> List[Tuple[str, str, str]]:
    """
    Loads the sorted repositories from a file.

    Args:
        file_path: Path to the file containing repository data

    Returns:
        List of tuples containing (url, commit, name) for each repository
    """
    with open(file_path, "r") as f:
        sorted_repo_data = json.load(f)
    return [(repo["url"], repo["commit"], repo["name"]) for repo in sorted_repo_data]
