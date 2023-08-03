import sys
import pytest
from lean_dojo import *
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def test_private_name_failure_1(mathlib_repo: LeanGitRepo) -> None:
    thm = Theorem(mathlib_repo, "src/data/part.lean", "part.mem_to_option")
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(
            init_state,
            "{ unfold to_option, by_cases h : o.dom; simp [h], { exact ⟨λ h, ⟨_, h⟩, λ ⟨_, h⟩, h⟩ }, { exact mt Exists.fst h } }",
        )
        assert isinstance(res, LeanError)
        assert not dojo.is_successful


def test_private_name_failure_2(mathlib_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib_repo, "src/data/analysis/filter.lean", "filter.realizer.ne_bot_iff"
    )
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(
            init_state,
            "{ classical, rw [not_iff_comm, ← le_bot_iff, F.le_iff realizer.bot, not_forall], simp only [set.not_nonempty_iff_eq_empty], exact ⟨λ ⟨x, e⟩ _, ⟨x, le_of_eq e⟩, λ h, let ⟨x, h⟩ := h () in ⟨x, le_bot_iff.1 h⟩⟩ }",
        )
        assert isinstance(res, LeanError)
        assert not dojo.is_successful


def test_parse_tactic_failure_1(mathlib_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib_repo,
        "src/set_theory/ordinal/arithmetic.lean",
        "ordinal.le_sup_shrink_equiv",
    )
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(
            init_state,
            "{ convert le_sup.{u u} _ ((@equiv_shrink s hs) ⟨a, ha⟩), rw symm_apply_apply }",
        )
        assert isinstance(res, LeanError)
        assert not dojo.is_successful


def test_proof_check_failure_1(mathlib_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib_repo,
        "src/order/initial_seg.lean",
        "initial_seg.le_lt_apply",
    )
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(
            init_state,
            "{ delta initial_seg.le_lt, cases h : f.lt_or_eq with f' f', { simp only [principal_seg.trans_apply, f.lt_or_eq_apply_left] }, { simp only [principal_seg.equiv_lt_apply, f.lt_or_eq_apply_right] } }",
        )
        assert isinstance(res, LeanError)
        assert not dojo.is_successful


def test_prelude_failure_1(lean_repo: LeanGitRepo) -> None:
    thm = Theorem(lean_repo, "library/init/data/int/order.lean", "int.lt_iff_le_not_le")
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(
            init_state,
            "{ simp [int.lt_iff_le_and_ne], split; intro h, { cases h with hab hn, split, { assumption }, { intro hba, simp [int.le_antisymm hab hba] at *, contradiction } } { cases h with hab hn, split, { assumption }, { intro h, simp [*] at * } } }",
        )
        assert isinstance(res, LeanError)
        assert not dojo.is_successful


def test_prelude_failure_2(lean_repo: LeanGitRepo) -> None:
    thm = Theorem(lean_repo, "library/init/data/nat/bitwise.lean", "nat.bodd_add")
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(
            init_state,
            "{ induction n with n IH, { simp, cases bodd m; refl }, { simp [IH], cases bodd m; cases bodd n; refl } }",
        )
        assert isinstance(res, LeanError)
        assert not dojo.is_successful


def test_prelude_failure_3(lean_repo: LeanGitRepo) -> None:
    thm = Theorem(lean_repo, "library/init/data/nat/bitwise.lean", "nat.bodd_mul")
    with Dojo(thm) as (dojo, init_state):
        res = dojo.run_tac(
            init_state,
            "{ induction n with n IH, { simp, cases bodd m; refl }, { simp [mul_succ, IH], cases bodd m; cases bodd n; refl } }",
        )
        assert isinstance(res, LeanError)
        assert not dojo.is_successful


def test_deep_recursion_1(minif2f_repo: LeanGitRepo) -> None:
    thm = Theorem(minif2f_repo, "lean/src/test.lean", "mathd_algebra_296")
    with Dojo(thm) as (dojo, init_state):
        with pytest.raises(DojoCrashError):
            dojo.run_tac(init_state, "{ rw abs_of_nonpos, norm_num, norm_num }")


def test_deep_recursion_2(minif2f_repo: LeanGitRepo) -> None:
    thm = Theorem(minif2f_repo, "lean/src/test.lean", "mathd_algebra_314")
    with Dojo(thm) as (dojo, init_state):
        with pytest.raises(DojoCrashError):
            dojo.run_tac(init_state, "{ rw h₀, norm_num }")
