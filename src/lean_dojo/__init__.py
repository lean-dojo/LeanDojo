import os
from loguru import logger
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
    DojoTacticTimeoutError,
    DojoInitError,
    Dojo,
    ProofFinished,
    ProofGivenUp,
)
from .interaction.parse_goals import Declaration, Goal, parse_goals
from .data_extraction.lean import get_latest_commit, LeanGitRepo, LeanFile, Theorem, Pos
from .constants import __version__

if os.geteuid() == 0:
    logger.warning(
        "Running LeanDojo as the root user may cause unexpected issues. Proceed with caution."
    )
