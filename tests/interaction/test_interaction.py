import os
from git import Repo
from lean_dojo.utils import working_directory
from lean_dojo.data_extraction.lean import RepoType
from lean_dojo import LeanGitRepo, Dojo, ProofFinished, ProofGivenUp, Theorem


# Avoid using remote cache
os.environ["DISABLE_REMOTE_CACHE"] = "1"


def test_github_interact(lean4_example_url):
    repo = LeanGitRepo(url=lean4_example_url, commit="main")
    assert repo.repo_type == RepoType.GITHUB
    theorem = Theorem(repo, "Lean4Example.lean", "hello_world")
    # initial state
    dojo, state_0 = Dojo(theorem).__enter__()
    assert state_0.pp == "a b c : Nat\n⊢ a + b + c = a + c + b"
    # state after running a tactic
    state_1 = dojo.run_tac(state_0, "rw [add_assoc]")
    assert state_1.pp == "a b c : Nat\n⊢ a + (b + c) = a + c + b"
    # state after running another a sorry tactic
    assert dojo.run_tac(state_1, "sorry") == ProofGivenUp()
    # finish proof
    final_state = dojo.run_tac(state_1, "rw [add_comm b, ←add_assoc]")
    assert isinstance(final_state, ProofFinished)


def test_local_interact(lean4_example_url):
    # Clone the GitHub repository to the local path
    with working_directory() as tmp_dir:
        # git repo placed in `tmp_dir / repo_name`
        Repo.clone_from(lean4_example_url, "lean4-example")

        local_dir = str((tmp_dir / "lean4-example"))
        repo = LeanGitRepo(local_dir, commit="main")
        assert repo.repo_type == RepoType.LOCAL
        theorem = Theorem(repo, "Lean4Example.lean", "hello_world")
        # initial state
        dojo, state_0 = Dojo(theorem).__enter__()
        assert state_0.pp == "a b c : Nat\n⊢ a + b + c = a + c + b"
        # state after running a tactic
        state_1 = dojo.run_tac(state_0, "rw [add_assoc]")
        assert state_1.pp == "a b c : Nat\n⊢ a + (b + c) = a + c + b"
        # state after running another a sorry tactic
        assert dojo.run_tac(state_1, "sorry") == ProofGivenUp()
        # finish proof
        final_state = dojo.run_tac(state_1, "rw [add_comm b, ←add_assoc]")
        assert isinstance(final_state, ProofFinished)
