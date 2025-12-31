from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from lean_dojo_v2.lean_dojo.data_extraction.lean import Pos

from .annotations import parse_pos


@dataclass
class Premise:
    """
    A class representing a Premise with various attributes.

    Attributes:
        full_name (str): The full name of the premise.
        code (str): The code associated with the premise.
        start (Pos): The starting position of the premise.
        end (Pos): The ending position of the premise.
        kind (str): The kind or type of the premise.
    """

    full_name: str
    code: str
    start: Pos
    end: Pos
    kind: str

    @classmethod
    def from_dict(cls, data: Dict) -> Premise:
        if not all(
            key in data for key in ["full_name", "code", "start", "end", "kind"]
        ):
            raise ValueError("Invalid Premise data format")
        return cls(
            full_name=data["full_name"],
            code=data["code"],
            start=parse_pos(data["start"]),
            end=parse_pos(data["end"]),
            kind=data["kind"],
        )

    def to_dict(self) -> Dict:
        return {
            "full_name": self.full_name,
            "code": self.code,
            "start": repr(self.start),
            "end": repr(self.end),
            "kind": self.kind,
        }


@dataclass
class PremiseFile:
    """
    Represents a file containing premises and their associated imports.

    Attributes:
        path (Path): The file path.
        imports (List[str]): A list of import statements.
        premises (List[Premise]): A list of premises.
    """

    path: Path
    imports: List[str]
    premises: List[Premise]

    @classmethod
    def from_dict(cls, data: Dict) -> PremiseFile:
        if not all(key in data for key in ["path", "imports", "premises"]):
            raise ValueError("Invalid PremiseFile data format")
        return cls(
            path=Path(data["path"]),
            imports=data["imports"],
            premises=[Premise.from_dict(p) for p in data["premises"]],
        )

    def to_dict(self) -> Dict:
        return {
            "path": str(self.path),
            "imports": self.imports,
            "premises": [p.to_dict() for p in self.premises],
        }
