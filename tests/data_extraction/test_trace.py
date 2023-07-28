import pytest
from pathlib import Path
from lean_dojo import *


def test_trace(traced_repo):
    traced_repo.check_sanity()


def test_get_traced_repo_path(mathlib_repo):
    path = get_traced_repo_path(mathlib_repo)
    assert isinstance(path, Path) and path.exists()


@pytest.mark.skip()
def test_trace_local_repo(local_traced_repo):
    local_traced_repo.check_sanity()


@pytest.mark.skip()
def test_trace_local_file():
    pass
