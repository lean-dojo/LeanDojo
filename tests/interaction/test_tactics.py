from lean_dojo import *


def test_example_hello_world(lean4_example_repo: LeanGitRepo) -> None:
    thm = Theorem(
        lean4_example_repo,
        "Lean4Example.lean",
        "hello_world",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "rw [add_assoc, add_comm b, ←add_assoc]")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_pow_two_pow_sub_pow_two_pow(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/NumberTheory/Multiplicity.lean",
        "pow_two_pow_sub_pow_two_pow",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "induction' n with d hd")
        s2 = dojo.run_tac(
            s1,
            "· simp only [pow_zero, pow_one, range_zero, prod_empty, one_mul, Nat.zero_eq]",
        )
        s3 = dojo.run_tac(
            s2,
            "· suffices x ^ 2 ^ d.succ - y ^ 2 ^ d.succ = (x ^ 2 ^ d + y ^ 2 ^ d) * (x ^ 2 ^ d - y ^ 2 ^ d) by rw [this, hd, Finset.prod_range_succ, ← mul_assoc, mul_comm (x ^ 2 ^ d + y ^ 2 ^ d)]",
        )
        assert not isinstance(s3, ProofFinished)
        assert not dojo.is_successful


def test_example_mem_nil_iff(std4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        std4_repo,
        "Std/Data/List/Lemmas.lean",
        "List.mem_nil_iff",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "simp")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_not_intro(aesop_repo: LeanGitRepo) -> None:
    thm = Theorem(
        aesop_repo,
        "Aesop/BuiltinRules.lean",
        "Aesop.BuiltinRules.not_intro",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact h")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_nsmul_zero(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Algebra/GroupPower/Basic.lean",
        "nsmul_zero",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact nsmul_zero n")
        assert isinstance(s1, LeanError)
        assert not dojo.is_successful


def test_example_div_im(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Data/Complex/Basic.lean",
        "Complex.div_im",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(
            s0, "simp [div_eq_mul_inv, mul_assoc, sub_eq_add_neg, add_comm]"
        )
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_mulIndicator_inv(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Algebra/Function/Indicator.lean",
        "Set.mulIndicator_inv'",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact (mulIndicatorHom G s).map_inv f")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_Iio_def(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Data/Set/Intervals/Basic.lean",
        "Set.Iio_def",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "rfl")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_sum_biUnion_boxes(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Analysis/BoxIntegral/Partition/Basic.lean",
        "BoxIntegral.Prepartition.sum_biUnion_boxes",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(
            s0,
            "refine' Finset.sum_biUnion fun J₁ h₁ J₂ h₂ hne => Finset.disjoint_left.2 fun J' h₁' h₂' => _",
        )
        s2 = dojo.run_tac(
            s1,
            "exact hne (π.eq_of_le_of_le h₁ h₂ ((πi J₁).le_of_mem h₁') ((πi J₂).le_of_mem h₂'))",
        )
        assert isinstance(s2, ProofFinished)
        assert dojo.is_successful


def test_example_coe_monoidHom_mk(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Algebra/Ring/Hom/Defs.lean",
        "RingHom.coe_monoidHom_mk",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "rfl")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_length_le(std4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        std4_repo,
        "Std/Data/List/Lemmas.lean",
        "List.IsSuffix.length_le",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact h.sublist.length_le")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_compl_F_eq_I(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Order/PrimeIdeal.lean",
        "Order.Ideal.PrimePair.compl_F_eq_I",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact IF.isCompl_I_F.eq_compl.symm")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_gcd_singleton(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Algebra/GCDMonoid/Multiset.lean",
        "Multiset.gcd_singleton",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact (fold_singleton _ _ _).trans <| gcd_zero_right _")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_principalSeg_coe(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/SetTheory/Ordinal/Basic.lean",
        "Ordinal.typein.principalSeg_coe",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact rfl")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_comp_hasFDerivAt_iff(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Analysis/Calculus/FDeriv/Equiv.lean",
        "ContinuousLinearEquiv.comp_hasFDerivAt_iff'",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(
            s0, "simp_rw [← hasFDerivWithinAt_univ, iso.comp_hasFDerivWithinAt_iff']"
        )
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_map_inv(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Algebra/Group/Hom/Defs.lean",
        "MonoidHom.map_inv",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(
            s0, "exact eq_inv_of_mul_eq_one_left <| map_mul_eq_one f <| inv_mul_self _"
        )
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_equivFunOnFinite_single(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Data/Finsupp/Defs.lean",
        "Finsupp.equivFunOnFinite_single",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "ext")
        s2 = dojo.run_tac(s1, "simp [Finsupp.single_eq_pi_single, equivFunOnFinite]")
        assert isinstance(s2, ProofFinished)
        assert dojo.is_successful


def test_example_card_erase_le(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Data/Finset/Card.lean",
        "Finset.card_erase_le",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact Multiset.card_erase_le")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_sometimes_eq(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Logic/Function/Basic.lean",
        "Function.sometimes_eq",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact dif_pos ⟨a⟩")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_measurable_prod(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/MeasureTheory/Group/Arithmetic.lean",
        "List.measurable_prod'",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "induction' l with f l ihl")
        s2 = dojo.run_tac(s1, "exact measurable_one")
        s3 = dojo.run_tac(s2, "rw [List.forall_mem_cons] at hl")
        s4 = dojo.run_tac(s3, "rw [List.prod_cons]")
        s5 = dojo.run_tac(s4, "exact hl.1.mul (ihl hl.2)")
        assert isinstance(s5, ProofFinished)
        assert dojo.is_successful


def test_example_mem_eqLocus(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/LinearAlgebra/Basic.lean",
        "LinearMap.mem_eqLocus",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(s0, "exact Iff.rfl")
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_neg_lt_sub_right_of_lt_add(std4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        std4_repo,
        "Std/Data/Int/Order.lean",
        "Int.neg_lt_sub_right_of_lt_add",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(
            s0, "exact Int.lt_sub_left_of_add_lt (Int.sub_right_lt_of_lt_add h)"
        )
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_eq(mathlib4_repo: LeanGitRepo) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Algebra/CharP/Basic.lean",
        "CharP.eq",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(
            s0,
            "exact Nat.dvd_antisymm ((CharP.cast_eq_zero_iff R p q).1 (CharP.cast_eq_zero _ _)) ((CharP.cast_eq_zero_iff R q p).1 (CharP.cast_eq_zero _ _))",
        )
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful


def test_example_nonempty_iInter_of_nonempty_biInter(
    mathlib4_repo: LeanGitRepo,
) -> None:
    thm = Theorem(
        mathlib4_repo,
        "Mathlib/Topology/MetricSpace/Bounded.lean",
        "Metric.nonempty_iInter_of_nonempty_biInter",
    )
    with Dojo(thm) as (dojo, s0):
        s1 = dojo.run_tac(
            s0,
            "exact (hs 0).isComplete.nonempty_iInter_of_nonempty_biInter hs h's h h'",
        )
        assert isinstance(s1, ProofFinished)
        assert dojo.is_successful
