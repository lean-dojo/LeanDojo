from typing import Any, Dict, Tuple

from ...lean import Pos


def parse_pos(d: Dict[str, Any]) -> Tuple[Pos, Pos]:
    # Lean uses 1-index for lines and 0-index for columns.
    # We always use 1-index.
    start_line, start_column = d["start"]
    start = Pos(start_line, 1 + start_column)
    end_line, end_column = d["end"]
    end = Pos(end_line, 1 + end_column)
    return start, end
