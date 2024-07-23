import pytest
from lean_dojo import LeanGitRepo
import os
from lean_dojo.data_extraction.lean import url_to_repo, _to_commit_hash
from github import Repository
from git import Repo
from lean_dojo.utils import repo_type_of_url, get_repo_info
from pathlib import Path

def test_repo_types(clean_clone_and_checkout, local_test_path, lean4_example_url, remote_example_url, example_commit_hash):
    # init: local repo
    local_repo_path = os.path.join(local_test_path, 'lean4-example')
    clean_clone_and_checkout(lean4_example_url, local_repo_path, 'main')
    url, _ = get_repo_info(Path(local_repo_path))
    assert url == local_repo_path

    # test repo type of url
    assert repo_type_of_url(lean4_example_url) == 'github'
    assert repo_type_of_url(remote_example_url) == 'remote'
    assert repo_type_of_url(local_repo_path) == 'local'

    # test url to repo
    ## test github url
    github_repo = url_to_repo(lean4_example_url, num_retries=2)
    assert isinstance(github_repo, Repository.Repository)
    assert github_repo.full_name == 'yangky11/lean4-example'

    ## test remote url
    remote_repo = url_to_repo(remote_example_url, repo_type='remote', num_retries=2)
    assert isinstance(remote_repo, Repo)
    assert remote_repo.remotes[0].url == remote_example_url

    ## test local path
    local_repo = url_to_repo(local_repo_path, repo_type='local')
    assert isinstance(local_repo, Repo)
    assert local_repo.remotes[0].url == lean4_example_url
    assert (
        local_repo.working_dir != local_repo_path
    ), "working_dir should not be the same as local_repo_path to avoid changing the original repo"

    # test commit hash
    ## test github url
    commit_hash = _to_commit_hash(github_repo, 'main')
    assert len(commit_hash) == 40
    commit_hash = _to_commit_hash(github_repo, example_commit_hash)
    assert commit_hash == example_commit_hash

    ## test remote url
    commit_hash = _to_commit_hash(remote_repo, 'main')
    assert len(commit_hash) == 40
    commit_hash = _to_commit_hash(remote_repo, example_commit_hash)
    assert commit_hash == example_commit_hash

    ## test local path
    commit_hash = _to_commit_hash(local_repo, 'main')
    assert len(commit_hash) == 40
    commit_hash = _to_commit_hash(local_repo, example_commit_hash)
    assert commit_hash == example_commit_hash


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

    # check repo name
    assert github_repo.name == 'lean4-example' == local_repo.name == from_path_repo.name

    # test get_config
    assert github_repo.get_config('lean-toolchain') == {'content': 'leanprover/lean4:v4.9.0-rc3\n'}
    assert local_repo.get_config('lean-toolchain') == {'content': 'leanprover/lean4:v4.9.0-rc3\n'}
    assert from_path_repo.get_config('lean-toolchain') == {'content': 'leanprover/lean4:v4.9.0-rc3\n'}

def test_local_with_commit(clean_clone_and_checkout, lean4_example_url, local_test_path, example_commit_hash):
    # Initialize GitHub repo
    github_repo = LeanGitRepo(url=lean4_example_url, commit=example_commit_hash)

    # Initialize local repo
    local_repo_path = os.path.join(local_test_path, 'lean4-example')
    clean_clone_and_checkout(lean4_example_url, local_repo_path, 'main') # use main branch
    local_repo = LeanGitRepo(url=local_repo_path, commit=example_commit_hash) # checkout to commit hash

    # Check if commit hashes match
    assert github_repo.commit == local_repo.commit
    assert github_repo.lean_version == local_repo.lean_version
    
    # check the repo type
    assert github_repo.repo_type == 'github'
    assert local_repo.repo_type == 'local'

    # check repo name
    assert github_repo.name == 'lean4-example' == local_repo.name

    # test get_config
    assert github_repo.get_config('lean-toolchain') == {'content': 'leanprover/lean4:v4.9.0-rc3\n'}
    assert local_repo.get_config('lean-toolchain') == {'content': 'leanprover/lean4:v4.9.0-rc3\n'}

def test_remote_url(lean4_example_url, remote_example_url, example_commit_hash):
    # Initialize GitHub repo
    github_repo = LeanGitRepo(url=lean4_example_url, commit=example_commit_hash)
    # Initialize Gitee repo
    _ = LeanGitRepo(url=remote_example_url, commit="main") # get commit by branch
    gitee_repo = LeanGitRepo(url=remote_example_url, commit=example_commit_hash)
    
    # Check if commit hashes match
    assert github_repo.commit == gitee_repo.commit
    assert github_repo.lean_version == gitee_repo.lean_version
    
    # check the repo type
    assert gitee_repo.repo_type == 'remote'

    # check repo name
    assert gitee_repo.name == 'lean4-example' == github_repo.name

    # test get_config
    assert gitee_repo.get_config('lean-toolchain') == {'content': 'leanprover/lean4:v4.9.0-rc3\n'}
    assert github_repo.get_config('lean-toolchain') == {'content': 'leanprover/lean4:v4.9.0-rc3\n'}
