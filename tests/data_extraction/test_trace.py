from pathlib import Path
from lean_dojo import *


def test_trace(traced_repo):
    traced_repo.check_sanity()


def test_get_traced_repo_path(mathlib4_repo):
    path = get_traced_repo_path(mathlib4_repo)
    assert isinstance(path, Path) and path.exists()
