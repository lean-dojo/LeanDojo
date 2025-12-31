from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from lean_dojo_v2.lean_dojo.data_extraction.lean import Pos


def parse_pos(pos_str):
    """
    Parses a position string or list into a Pos object.

    Args:
        pos_str (str or list): The position data, either as a string in the format 'Pos(x, y)'
                               or as a list [x, y].

    Returns:
        Pos: A Pos object initialized with the parsed coordinates.

    Raises:
        ValueError: If the input format is neither a string nor a list.
    """
    if isinstance(pos_str, str):
        # pos_str came from a JSON file
        pos_parts = (
            pos_str.replace("Pos", "").replace("(", "").replace(")", "").split(",")
        )
        return Pos(int(pos_parts[0]), int(pos_parts[1]))
    elif isinstance(pos_str, list):
        # pos_str came from a dictionary initialization
        return Pos(*pos_str)
    else:
        raise ValueError(f"Unexpected format for Pos: {pos_str}")


@dataclass
class Annotation:
    """
    Annotation class represents a code annotation with its full name, definition path,
    and position details.

    Attributes:
        full_name (str): The full name of the annotation.
        def_path (str): The file path where the annotation is defined.
        def_pos (Pos): The starting position of the annotation definition.
        def_end_pos (Pos): The ending position of the annotation definition.
    """

    full_name: str
    def_path: str
    def_pos: Pos
    def_end_pos: Pos

    @classmethod
    def from_dict(cls, data: Dict) -> Annotation:
        if not all(
            key in data for key in ["full_name", "def_path", "def_pos", "def_end_pos"]
        ):
            raise ValueError("Invalid Annotation data format")
        return cls(
            full_name=data["full_name"],
            def_path=data["def_path"],
            def_pos=parse_pos(data["def_pos"]),
            def_end_pos=parse_pos(data["def_end_pos"]),
        )

    def to_dict(self) -> Dict:
        return {
            "full_name": self.full_name,
            "def_path": self.def_path,
            "def_pos": repr(self.def_pos),
            "def_end_pos": repr(self.def_end_pos),
        }


@dataclass
class AnnotatedTactic:
    """
    AnnotatedTactic is a data class that represents a tactic with its annotations
    and states before and after its application.

    Attributes:
        tactic (str): The tactic applied.
        annotated_tactic (Tuple[str, List[Annotation]]): A tuple containing the tactic and a list of annotations.
        state_before (str): The state before the tactic is applied.
        state_after (str): The state after the tactic is applied.
    """

    tactic: str
    annotated_tactic: Tuple[str, List[Annotation]]
    state_before: str
    state_after: str

    @classmethod
    def from_dict(cls, data: Dict) -> AnnotatedTactic:
        if not all(
            key in data
            for key in ["tactic", "annotated_tactic", "state_before", "state_after"]
        ):
            raise ValueError("Invalid AnnotatedTactic data format")
        return cls(
            tactic=data["tactic"],
            annotated_tactic=(
                data["annotated_tactic"][0],
                [Annotation.from_dict(a) for a in data["annotated_tactic"][1]],
            ),
            state_before=data["state_before"],
            state_after=data["state_after"],
        )

    def to_dict(self) -> Dict:
        return {
            "tactic": self.tactic,
            "annotated_tactic": [
                self.annotated_tactic[0],
                [a.to_dict() for a in self.annotated_tactic[1]],
            ],
            "state_before": self.state_before,
            "state_after": self.state_after,
        }
