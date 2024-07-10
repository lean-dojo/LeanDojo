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

def test_remote_interact():
    repo = LeanGitRepo(url=GITEE_REPO_URL, commit="main")
    assert repo.repo_type == 'remote'
    theorem = Theorem(repo, "Lean4Example.lean", "hello_world")
    # initial state
    dojo, state_0 = Dojo(theorem).__enter__()
    assert state_0.pp == 'a b c : Nat\n⊢ a + b + c = a + c + b'
    # state after running a tactic
    state_1 = dojo.run_tac(state_0, "rw [add_assoc]")
    assert state_1.pp == 'a b c : Nat\n⊢ a + (b + c) = a + c + b'
    # state after running another a sorry tactic
    assert dojo.run_tac(state_1, "sorry") == ProofGivenUp()
    # finish proof
    final_state = dojo.run_tac(state_1, "rw [add_comm b, ←add_assoc]")
    assert isinstance(final_state, ProofFinished)

def test_local_interact():
    repo = LeanGitRepo(url=LOCAL_REPO_PATH, commit="main")
    assert repo.repo_type == 'local'
    theorem = Theorem(repo, "Lean4Example.lean", "hello_world")
    # initial state
    dojo, state_0 = Dojo(theorem).__enter__()
    assert state_0.pp == 'a b c : Nat\n⊢ a + b + c = a + c + b'
    # state after running a tactic
    state_1 = dojo.run_tac(state_0, "rw [add_assoc]")
    assert state_1.pp == 'a b c : Nat\n⊢ a + (b + c) = a + c + b'
    # state after running another a sorry tactic
    assert dojo.run_tac(state_1, "sorry") == ProofGivenUp()
    # finish proof
    final_state = dojo.run_tac(state_1, "rw [add_comm b, ←add_assoc]")
    assert isinstance(final_state, ProofFinished)
