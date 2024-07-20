import pytest
from lean_dojo import *
from git import Repo
import os, shutil
from pathlib import Path

# Avoid using remote cache
os.environ['DISABLE_REMOTE_CACHE'] = 'true'

@pytest.fixture(scope="session")
def local_trace_dir(local_test_path):
    return os.path.join(local_test_path, 'lean4-example-traced')

def test_remote_trace(remote_example_url, local_trace_dir):
    remote_repo = LeanGitRepo(url=remote_example_url, commit="main")
    assert remote_repo.repo_type == 'remote'
    if os.path.exists(local_trace_dir):
        shutil.rmtree(local_trace_dir)
    traced_repo = trace(remote_repo, local_trace_dir)
    traced_repo.check_sanity()
    assert traced_repo.repo.repo_type == 'local'
    
def test_local_trace(clean_clone_and_checkout, lean4_example_url, local_test_path, local_trace_dir):
    local_repo_path = os.path.join(local_test_path, 'lean4-example')
    clean_clone_and_checkout(lean4_example_url, local_repo_path)
    local_repo = LeanGitRepo(url=local_repo_path, commit="main")
    assert local_repo.repo_type == 'local'
    if os.path.exists(local_trace_dir):
        shutil.rmtree(local_trace_dir)
    traced_repo = trace(local_repo, local_trace_dir)
    traced_repo.check_sanity()
    assert traced_repo.repo.repo_type == 'local'

def test_github_trace(lean4_example_url, local_trace_dir):
    remote_repo = LeanGitRepo(url=lean4_example_url, commit="main")
    assert remote_repo.repo_type == 'github'
    if os.path.exists(local_trace_dir):
        shutil.rmtree(local_trace_dir)
    traced_repo = trace(remote_repo, local_trace_dir)
    traced_repo.check_sanity()
    assert traced_repo.repo.repo_type == 'local'

def test_trace(traced_repo):
    traced_repo.check_sanity()

def test_get_traced_repo_path(mathlib4_repo):
    path = get_traced_repo_path(mathlib4_repo)
    assert isinstance(path, Path) and path.exists()
