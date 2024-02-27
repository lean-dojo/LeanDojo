import pytest

from lean_dojo import *


def test_example_1(mathlib4_repo: LeanGitRepo) -> None:
    entry = (mathlib4_repo, "Mathlib/LinearAlgebra/Basic.lean", 90)
    with Dojo(entry) as (dojo, s0):
        s1 = dojo.run_cmd(s0, "#eval 1")
        assert isinstance(s1, CommandState) and s1.message == "1"

        s2 = dojo.run_cmd(s1, "#eval x")
        assert isinstance(s2, LeanError)
        with pytest.raises(RuntimeError):
            dojo.run_cmd(s2, "def x := 1")
        s3 = dojo.run_cmd(s0, "def x := 1")
        s4 = dojo.run_cmd(s3, "#eval x")
        assert isinstance(s4, CommandState) and s4.message == "1"

        s5 = dojo.run_cmd(s0, "#print addMonoidHomLequivNat")
        assert isinstance(s5, CommandState) and s5.message != ""
        s6 = dojo.run_cmd(s0, "#print addMonoidEndRingEquivInt")
        assert isinstance(s6, LeanError)
