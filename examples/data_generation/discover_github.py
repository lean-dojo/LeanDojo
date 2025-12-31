"""
Example script for discovering popular Lean 4 repositories on Github. 
The repositories are saved at <RAID_DIR>/<DATA_DIR>/repo_info_compatible.json

Usage: python examples/data_generation/discover_github.py
"""

from lean_dojo_v2.database import DynamicDatabase

database = DynamicDatabase()

database.discover_repositories(
    num_repos=5,
    curriculum_learning=True,
    build_deps=False,
)
