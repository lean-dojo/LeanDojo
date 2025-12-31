from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from lean_dojo_v2.lean_dojo.data_extraction.lean import Pos

from .annotations import AnnotatedTactic, parse_pos


@dataclass
class Theorem:
    """
    A class to represent a theorem with its associated metadata.

    Attributes:
        full_name (str): The full name of the theorem.
        file_path (Path): The file path where the theorem is located.
        start (Pos): The starting position of the theorem in the file.
        end (Pos): The ending position of the theorem in the file.
        url (str): The URL associated with the theorem.
        commit (str): The commit hash associated with the theorem.
        theorem_statement (str, optional): The statement of the theorem.
        traced_tactics (Optional[List[AnnotatedTactic]], optional): A list of traced tactics.
        difficulty_rating (Optional[float], optional): The difficulty rating of the theorem.
    """

    full_name: str
    file_path: Path
    start: Pos
    end: Pos
    url: str
    commit: str
    theorem_statement: str = None
    traced_tactics: Optional[List[AnnotatedTactic]] = field(default_factory=list)
    difficulty_rating: Optional[float] = None

    def __eq__(self, other):
        if not isinstance(other, Theorem):
            return NotImplemented
        return self.is_same_theorem(other)

    def is_same_theorem(self, other: Theorem) -> bool:
        return (
            self.full_name == other.full_name
            and self.file_path == other.file_path
            and self.start == other.start
            and self.end == other.end
        )

    @classmethod
    def from_dict(cls, data: Dict, url: str, commit: str) -> Theorem:
        if not all(key in data for key in ["full_name", "file_path", "start", "end"]):
            raise ValueError("Invalid Theorem data format")
        return cls(
            full_name=data["full_name"],
            theorem_statement=data.get("theorem_statement"),
            file_path=Path(data["file_path"]),
            start=parse_pos(data["start"]),
            end=parse_pos(data["end"]),
            url=url,
            commit=commit,
            traced_tactics=[
                AnnotatedTactic.from_dict(t) for t in data.get("traced_tactics", [])
            ],
            difficulty_rating=data.get("difficulty_rating"),
        )

    def __str__(self) -> str:
        return self.theorem_statement

    def to_dict(self) -> Dict:
        return {
            "full_name": self.full_name,
            "theorem_statement": self.theorem_statement,
            "file_path": str(self.file_path),
            "start": repr(self.start),
            "end": repr(self.end),
            "url": self.url,
            "commit": self.commit,
            "traced_tactics": [t.to_dict() for t in (self.traced_tactics or [])],
            "difficulty_rating": self.difficulty_rating,
        }

    def get_theorem_identifier(
        self,
    ) -> tuple[str, str, tuple[int, int], tuple[int, int]]:
        """Returns a unique identifier for a theorem."""
        return (
            self.full_name,
            str(self.file_path),
            tuple(self.start),
            tuple(self.end),
        )
