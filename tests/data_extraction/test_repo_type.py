import pytest
from lean_dojo import LeanGitRepo
import os

def test_local_with_branch(clean_clone_and_checkout, lean4_example_url, local_test_path):
    # Initialize GitHub repo
    github_repo = LeanGitRepo(url=lean4_example_url, commit="main")

    # Initialize local repo
    local_repo_path = os.path.join(local_test_path, 'lean4-example')
    clean_clone_and_checkout(lean4_example_url, local_repo_path, 'main')
    local_repo = LeanGitRepo(url=local_repo_path, commit="main")
    from_path_repo = LeanGitRepo.from_path(local_repo_path)

    # Check if commit hashes match
    assert github_repo.commit == local_repo.commit == from_path_repo.commit
    assert github_repo.lean_version == local_repo.lean_version == from_path_repo.lean_version
    
    # check the repo type
    assert github_repo.repo_type == 'github'
    assert local_repo.repo_type == 'local'
    assert from_path_repo.repo_type == 'local'

def test_local_with_commit(clean_clone_and_checkout, lean4_example_url, local_test_path):
    # GitHub commit hash from conftest.py
    COMMIT_HASH = "3f8c5eb303a225cdef609498b8d87262e5ef344b"

    # Initialize GitHub repo
    github_repo = LeanGitRepo(url=lean4_example_url, commit=COMMIT_HASH)

    # Initialize local repo
    local_repo_path = os.path.join(local_test_path, 'lean4-example')
    clean_clone_and_checkout(lean4_example_url, local_repo_path, 'main') # use main branch
    local_repo = LeanGitRepo(url=local_repo_path, commit=COMMIT_HASH) # checkout to commit hash

    # Check if commit hashes match
    assert github_repo.commit == local_repo.commit
    assert github_repo.lean_version == local_repo.lean_version
    
    # check the repo type
    assert github_repo.repo_type == 'github'
    assert local_repo.repo_type == 'local'

def test_remote_url(lean4_example_url, remote_example_url):
    # GitHub commit hash from conftest.py
    COMMIT_HASH = "3f8c5eb303a225cdef609498b8d87262e5ef344b"

    # Initialize GitHub repo
    github_repo = LeanGitRepo(url=lean4_example_url, commit=COMMIT_HASH)
    # Initialize Gitee repo
    _ = LeanGitRepo(url=remote_example_url, commit="main") # get commit by branch
    gitee_repo = LeanGitRepo(url=remote_example_url, commit=COMMIT_HASH)
    
    # Check if commit hashes match
    assert github_repo.commit == gitee_repo.commit
    assert github_repo.lean_version == gitee_repo.lean_version
    
    # check the repo type
    assert gitee_repo.repo_type == 'remote'
