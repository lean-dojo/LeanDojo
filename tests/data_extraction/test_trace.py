from pathlib import Path
from lean_dojo import *
from lean_dojo.data_extraction.cache import cache


def test_example_trace(lean4_example_repo):
    trace_repo = trace(lean4_example_repo)
    repo = trace_repo.repo
    path = cache.get(repo.url, repo.commit)
    assert path is not None


def test_trace(traced_repo):
    traced_repo.check_sanity()


def test_get_traced_repo_path(mathlib4_repo):
    path = get_traced_repo_path(mathlib4_repo)
    assert isinstance(path, Path) and path.exists()
