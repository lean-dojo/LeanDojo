"""
Example script for generating dataset from a local Lean 4 repository. 
The data is saved at <RAID_DIR>/<DATA_DIR>/<repo.name>_<repo.commit>.
e.g. LeanDojo-v2/raid/data/lean4-example_005de00d03f1aaa32cb2923d5e3cbaf0b954a192

Usage: python examples/data_generation/trace_local.py
"""

from lean_dojo_v2.database import DynamicDatabase

path = "path/to/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

database = DynamicDatabase()

database.trace_repository(
    url=path,
    commit=commit,
    build_deps=False,
)
