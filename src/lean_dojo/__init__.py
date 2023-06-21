import os
from .data_extraction.trace import trace, get_traced_repo_path, is_available_in_cache

from .data_extraction.traced_data import (
    TracedRepo,
    TracedFile,
    TracedTheorem,
    TracedTactic,
)
from .utils import set_lean_dojo_logger
from .interaction.dojo import (
    TacticState,
    TacticError,
    TimeoutError,
    TacticResult,
    DojoCrashError,
    DojoHardTimeoutError,
    DojoInitError,
    Dojo,
    ProofFinished,
    ProofGivenUp,
)
from .data_extraction.lean import LeanGitRepo, LeanFile, Theorem, Pos
from .constants import __version__

if "VERBOSE" in os.environ or "DEBUG" in os.environ:
    set_lean_dojo_logger(verbose=True)
else:
    set_lean_dojo_logger(verbose=False)
