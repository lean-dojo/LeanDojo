from lean_dojo import *


def test_parse_goal_1() -> None:
    goals = parse_goals(
        "case inl\n\n⊢ orderOf (r 1) = 0\n\ncase inr\nn : ℕ\nhn : NeZero n\n⊢ orderOf (r 1) = n"
    )
    assert len(goals) == 2
    assert goals == parse_goals(
        "case inl\n⊢ orderOf (r 1) = 0\n\ncase inr\nn : ℕ\nhn : NeZero n\n⊢ orderOf (r 1) = n"
    )
