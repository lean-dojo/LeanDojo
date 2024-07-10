from pathlib import Path
from lean_dojo import *
from git import Repo
import os, shutil

# repository details
GITHUB_REPO_URL = "https://github.com/yangky11/lean4-example"
GITHUB_COMMIT_HASH = "3f8c5eb303a225cdef609498b8d87262e5ef344b"
GITEE_REPO_URL = "https://gitee.com/rexzong/lean4-example"

LOCAL_REPO_PATH = f"{os.path.dirname(__file__)}/testdata/lean4-example"
LOCAL_TRACE_DIR = f"{os.path.dirname(__file__)}/testdata/lean4-example-traced"


def clone_repo_and_remove_remote(repo_url, local_path, label='main'):
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    repo = Repo.clone_from(repo_url, local_path)
    repo.git.checkout(label)
    remote = repo.remote(name='origin')
    remote.remove(repo, 'origin')

# Clone the GitHub repository to the local path
clone_repo_and_remove_remote(GITHUB_REPO_URL, LOCAL_REPO_PATH)

def test_remote_trace():
    remote_repo = LeanGitRepo(url=GITEE_REPO_URL, commit="main")
    assert remote_repo.repo_type == 'remote'
    if os.path.exists(LOCAL_TRACE_DIR):
        shutil.rmtree(LOCAL_TRACE_DIR)
    traced_repo = trace(remote_repo, LOCAL_TRACE_DIR)
    traced_repo.check_sanity()
    assert traced_repo.repo.repo_type == 'remote'
    
def test_local_trace():
    local_repo = LeanGitRepo(url=LOCAL_REPO_PATH, commit="main")
    assert local_repo.repo_type == 'local'
    if os.path.exists(LOCAL_TRACE_DIR):
        shutil.rmtree(LOCAL_TRACE_DIR)
    traced_repo = trace(local_repo, LOCAL_TRACE_DIR)
    traced_repo.check_sanity()
    assert traced_repo.repo.repo_type == 'local'

def test_github_trace():
    remote_repo = LeanGitRepo(url=GITHUB_REPO_URL, commit="main")
    assert remote_repo.repo_type == 'github'
    if os.path.exists(LOCAL_TRACE_DIR):
        shutil.rmtree(LOCAL_TRACE_DIR)
    traced_repo = trace(remote_repo, LOCAL_TRACE_DIR)
    traced_repo.check_sanity()
    assert traced_repo.repo.repo_type == 'github'

def test_trace(traced_repo):
    traced_repo.check_sanity()

def test_get_traced_repo_path(mathlib4_repo):
    path = get_traced_repo_path(mathlib4_repo)
    assert isinstance(path, Path) and path.exists()
