import os
from .data_extraction.trace import (
    trace,
    get_traced_repo_path,
    is_available_in_cache,
)

from .data_extraction.traced_data import (
    TracedRepo,
    TracedFile,
    TracedTheorem,
    TracedTactic,
)
from .interaction.dojo import (
    CommandState,
    TacticState,
    LeanError,
    TimeoutError,
    TacticResult,
    DojoCrashError,
    DojoHardTimeoutError,
    DojoInitError,
    Dojo,
    ProofFinished,
    ProofGivenUp,
)
from .interaction.parse_goals import Declaration, Goal, parse_goals
from .data_extraction.lean import LeanGitRepo, LeanFile, Theorem, Pos
from .constants import __version__

if os.geteuid() == 0:
    raise RuntimeError("LeanDojo should not be run as root.")
