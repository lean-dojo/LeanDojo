from lean_dojo import LeanGitRepo, Dojo, ProofFinished, ProofGivenUp, Theorem


def test_remote_interact(lean4_example_url):
    repo = LeanGitRepo(url=lean4_example_url, commit="main")
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
