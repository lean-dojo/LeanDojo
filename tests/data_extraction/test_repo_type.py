from lean_dojo import LeanGitRepo
from git import Repo
import os, shutil

# GitHub repository details
GITHUB_REPO_URL = "https://github.com/yangky11/lean4-example"
GITHUB_COMMIT_HASH = "3f8c5eb303a225cdef609498b8d87262e5ef344b"
GITEE_REPO_URL = "https://gitee.com/rexzong/lean4-example"

# Local repository path (make sure this path exists and is the clone of the above GitHub repo)

LOCAL_REPO_PATH = f"{os.path.dirname(__file__)}/testdata/lean4-example"

def clone_repo_if_not_exists(repo_url, local_path, label='main'):
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    repo = Repo.clone_from(repo_url, local_path)
    repo.git.checkout(label)

# Clone the GitHub repository to the local path
clone_repo_if_not_exists(GITHUB_REPO_URL, LOCAL_REPO_PATH)

def test_local_with_branch():
    
    # Initialize GitHub repo
    github_repo = LeanGitRepo(url=GITHUB_REPO_URL, commit="main")

    # Initialize local repo
    local_repo = LeanGitRepo(url=LOCAL_REPO_PATH, commit="main")

    # Check if commit hashes match
    assert github_repo.commit == local_repo.commit
    assert github_repo.lean_version == local_repo.lean_version
    
    # check the repo type
    assert github_repo.repo_type == 'github'
    assert local_repo.repo_type == 'local'

def test_local_with_commit():
    # Clone the GitHub repository to the local path
    clone_repo_if_not_exists(GITHUB_REPO_URL, LOCAL_REPO_PATH, GITHUB_COMMIT_HASH)
    
    # Initialize GitHub repo
    github_repo = LeanGitRepo(url=GITHUB_REPO_URL, commit=GITHUB_COMMIT_HASH)

    # Initialize local repo
    local_repo = LeanGitRepo(url=LOCAL_REPO_PATH, commit=GITHUB_COMMIT_HASH)

    # Check if commit hashes match
    assert github_repo.commit == local_repo.commit
    assert github_repo.lean_version == local_repo.lean_version
    
    # check the repo type
    assert github_repo.repo_type == 'github'
    assert local_repo.repo_type == 'local'

def test_remote_url():
    # Initialize GitHub repo
    github_repo = LeanGitRepo(url=GITHUB_REPO_URL, commit=GITHUB_COMMIT_HASH)
    # Initialize Gitee repo
    LeanGitRepo(url=GITEE_REPO_URL, commit="main") # get commit by branch
    gitee_repo = LeanGitRepo(url=GITEE_REPO_URL, commit=GITHUB_COMMIT_HASH)
    # Check if commit hashes match
    assert github_repo.commit == gitee_repo.commit
    assert github_repo.lean_version == gitee_repo.lean_version
    # check the repo type
    assert gitee_repo.repo_type == 'remote'
