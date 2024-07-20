import pytest
from lean_dojo import *
from git import Repo
import os

# Avoid using remote cache
os.environ['DISABLE_REMOTE_CACHE'] = 'true'

def test_remote_interact(remote_example_url):
    repo = LeanGitRepo(url=remote_example_url, commit="main")
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

def test_local_interact(clean_clone_and_checkout, lean4_example_url, local_test_path):
    # Clone the GitHub repository to the local path
    local_repo_path = os.path.join(local_test_path, 'lean4-example')
    clean_clone_and_checkout(lean4_example_url, local_repo_path)
    repo = LeanGitRepo(url=local_repo_path, commit="main")
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
