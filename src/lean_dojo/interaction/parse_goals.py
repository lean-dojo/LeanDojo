"""Utilities for parsing Lean's pretty-printed proof goals.
"""
import re
from typing import List
from dataclasses import dataclass


_DECL_REGEX = re.compile(
    r"(?<=\n)(?P<idents>.+?)\s+\:(?P<lean_type>.+?)\n(?=\S)", re.DOTALL
)
"""Regex for a line of declarations in the local context.

It can be a single declaration such as ``x : Nat`` or multiple declarations such as ``x y : Nat``.
"""


_CASE_REGEX = re.compile(r"case\s\S+\n")


_SPACE_REGEX = re.compile(r"\s+")


@dataclass(frozen=True)
class Declaration:
    """A declaration in the local context."""

    ident: str
    lean_type: str

    def __post_init__(self) -> None:
        assert _SPACE_REGEX.search(self.ident) is None


def _parse_local_context(ctx_pp: str) -> List[Declaration]:
    """Parse the local context of a goal."""
    m = _CASE_REGEX.match(ctx_pp)
    if m is not None:
        ctx_pp = ctx_pp[m.end() :]

    decls = []
    for m in _DECL_REGEX.finditer("\n" + ctx_pp + "⊢"):
        lean_type = m["lean_type"].strip()
        if lean_type.endswith(","):
            lean_type = lean_type[:-1].strip()
        for ident in m["idents"].strip().split():
            decls.append(Declaration(ident.strip(), lean_type))
    return decls


@dataclass(frozen=True)
class Goal:
    """A goal in Lean."""

    assumptions: List[Declaration]
    conclusion: str

    @classmethod
    def from_pp(cls, pp: str) -> "Goal":
        """Parse a pretty-printed goal."""
        assert pp.count("⊢") == 1
        ctx, concl = pp.split("⊢")
        assumptions = _parse_local_context(ctx)
        return cls(assumptions, concl.strip())


def parse_goals(pp: str) -> List[Goal]:
    """Parse a list of pretty-printed goals."""
    return [Goal.from_pp(g) for g in pp.split("\n\n") if "⊢" in g]
