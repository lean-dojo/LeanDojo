import sys
import pytest
from lean_dojo import *
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def test_prelude_failure_1(lean_repo: LeanGitRepo) -> None:
    thm = Theorem(lean_repo, "library/init/data/int/basic.lean", "int.to_nat_sub")
    with pytest.raises(DojoInitError):
        with Dojo(thm):
            pass


def test_nonexistent_theorem_1(mathlib_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib_repo,
        "src/ring_theory/non_existent.lean",
        "hello.world",
    )
    with pytest.raises(DojoInitError):
        with Dojo(thm):
            pass


def test_nonexistent_theorem_2(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/LinearAlgebra/Basic.lean",
        "hello.world",
    )
    with pytest.raises(DojoInitError):
        with Dojo(thm):
            pass


def test_not_theorem_1(mathlib_repo: LeanGitRepo) -> None:
    thm = Theorem(mathlib_repo, "library/data/dlist.lean", "dlist.cons")
    with pytest.raises(DojoInitError):
        with Dojo(thm):
            pass


def test_not_theorem_2(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(mathlib4_repo, "Mathlib/Data/Bool/Basic.lean", "Bool.ofNat")
    with pytest.raises(DojoInitError):
        with Dojo(thm):
            pass
