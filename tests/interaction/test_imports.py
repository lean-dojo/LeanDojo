import pytest
from lean_dojo import *


def test_unimported_tactic(lean4_example_repo) -> None:
    thm = Theorem(lean4_example_repo, "Lean4Example.lean", "hello_world")
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(init_state, "aesop")
        assert isinstance(res, LeanError) and "unknown tactic" in res.error


def test_additional_imports_failure_1(lean4_example_repo) -> None:
    """This error is because aesop is not imported by any file in this repo.
    Therefore, it hasn't been built yet, and we can't import it.
    """
    thm = Theorem(lean4_example_repo, "Lean4Example.lean", "foo")
    with pytest.raises(DojoInitError):
        with Dojo(thm, additional_imports=["Aesop"]):
            pass


def test_additional_imports(mathlib4_repo) -> None:
    """This test doesn't have the problem above because aesop is used by other files in mathlib4."""
    thm = Theorem(mathlib4_repo, "Mathlib/Data/Subtype.lean", "Subtype.restrict_apply")
    with Dojo(thm, additional_imports=["Aesop"]) as (dojo, init_state):
        res = dojo.run_tac(init_state, "aesop")
        assert isinstance(res, ProofFinished) and dojo.is_successful
