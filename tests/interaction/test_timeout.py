import pytest

from lean_dojo import *


def test_timeout_1(minif2f_repo: LeanGitRepo) -> None:
    thm = Theorem(minif2f_repo, "lean/src/test.lean", "amc12_2000_p6")
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(
            init_state,
            "{ revert p q h₀ h₁ h₂, intros p q hpq, rintros ⟨hp, hq⟩, rintro ⟨h, h⟩, intro h, have h₁ := nat.prime.ne_zero hpq.1, have h₂ : q ≠ 0, { rintro rfl, simp * at * }, apply h₁, revert hpq, intro h, simp * at *, apply h₁, have h₃ : q = 10 * q, apply eq.symm, all_goals { dec_trivial! } }",
        )
        assert isinstance(res, TimeoutError)


def test_timeout_2(lean4_example_repo: LeanGitRepo) -> None:
    thm = Theorem(
        lean4_example_repo,
        "Lean4Example.lean",
        "hello_world",
    )
    with Dojo(thm, hard_timeout=10) as (dojo, init_state):
        with pytest.raises(DojoHardTimeoutError):
            dojo.run_tac(init_state, "sleep 99999999999999")
