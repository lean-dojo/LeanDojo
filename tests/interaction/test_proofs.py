from lean_dojo import *


def test_example_mk0_eq_one_iff_correct_1(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/RingTheory/ClassGroup.lean",
        "ClassGroup.mk0_eq_one_iff",
    )
    assert check_proof(
        thm, "ClassGroup.mk_eq_one_iff.trans (coeSubmodule_isPrincipal R _)"
    )


def test_example_mk0_eq_one_iff_correct_2(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/RingTheory/ClassGroup.lean",
        "ClassGroup.mk0_eq_one_iff",
    )
    assert check_proof(
        thm,
        "by\n    exact (ClassGroup.mk_eq_one_iff.trans (coeSubmodule_isPrincipal R _))",
    )


def test_example_mk0_eq_one_iff_incorrect_1(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/RingTheory/ClassGroup.lean",
        "ClassGroup.mk0_eq_one_iff",
    )
    assert not check_proof(
        thm, "ClassGroup.mk_eq_one_iff.trans coeSubmodule_isPrincipal R"
    )


def test_example_mk0_eq_one_iff_incorrect_2(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/RingTheory/ClassGroup.lean",
        "ClassGroup.mk0_eq_one_iff",
    )
    assert not check_proof(thm, "sorry")
