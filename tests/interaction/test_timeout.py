import pytest

from lean_dojo import *


def test_timeout_1(lean4_example_repo: LeanGitRepo) -> None:
    thm = Theorem(
        lean4_example_repo,
        "Lean4Example.lean",
        "hello_world",
    )
    with Dojo(thm, timeout=30) as (dojo, init_state):
        with pytest.raises(DojoTacticTimeoutError):
            dojo.run_tac(init_state, "sleep 99999999999999")
