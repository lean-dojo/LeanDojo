import sys
import pytest
from lean_dojo import *
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


def test_nonexistent_theorem_1(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/LinearAlgebra/Basic.lean",
        "hello.world",
    )
    with pytest.raises(DojoInitError):
        with Dojo(thm):
            pass


def test_not_theorem_1(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(mathlib4_repo, "Mathlib/Data/Bool/Basic.lean", "Bool.ofNat")
    with pytest.raises(DojoInitError):
        with Dojo(thm):
            pass
