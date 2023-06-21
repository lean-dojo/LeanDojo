from lean_dojo import *


def test_sorry_1(mathlib_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib_repo,
        "src/ring_theory/subring/basic.lean",
        "subring.mem_supr_of_directed",
    )
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(init_state, "sorry")
        assert isinstance(res, ProofGivenUp)
        assert not dojo.is_proved


def test_sorry_2(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/CategoryTheory/Arrow.lean",
        "CategoryTheory.Arrow.id_left",
    )
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(init_state, "sorry")
        assert isinstance(res, ProofGivenUp)
        assert not dojo.is_proved
