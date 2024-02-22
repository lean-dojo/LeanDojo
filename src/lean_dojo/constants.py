"""Constants controlling LeanDojo's behaviors. 
Many of them are configurable via :ref:`environment-variables`.
"""

import os
import re
import sys
import subprocess
import multiprocessing
from pathlib import Path
from typing import Tuple
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

__version__ = "1.6.0"

logger.remove()
if "VERBOSE" in os.environ or "DEBUG" in os.environ:
    logger.add(sys.stderr, level="DEBUG")
else:
    logger.add(sys.stderr, level="INFO")

CACHE_DIR = (
    Path(os.environ["CACHE_DIR"])
    if "CACHE_DIR" in os.environ
    else Path.home() / ".cache/lean_dojo"
)
CACHE_DIR = CACHE_DIR.resolve() if not CACHE_DIR.is_absolute() else CACHE_DIR
"""Cache directory for storing traced repos (see :ref:`caching`).
"""

REMOTE_CACHE_URL = "https://lean-dojo.s3.amazonaws.com"
"""URL of the remote cache (see :ref:`caching`)."""

DISABLE_REMOTE_CACHE = "DISABLE_REMOTE_CACHE" in os.environ
"""Whether to disable remote caching (see :ref:`caching`) and build all repos locally.
"""

TMP_DIR = Path(os.environ["TMP_DIR"]) if "TMP_DIR" in os.environ else None
TMP_DIR = TMP_DIR.resolve() if TMP_DIR and not TMP_DIR.is_absolute() else TMP_DIR
"""Temporary directory used by LeanDojo for storing intermediate files
"""

MAX_NUM_PROCS = 32

NUM_PROCS = int(os.getenv("NUM_PROCS", min(multiprocessing.cpu_count(), MAX_NUM_PROCS)))
"""Number of threads to use
"""

NUM_WORKERS = NUM_PROCS - 1

LEAN3_URL = "https://github.com/leanprover-community/lean"
"""The URL of the Lean 3 repo."""

LEAN3_PACKAGES_DIR = Path("_target/deps")
"""The directory where Lean 3 dependencies are stored."""

LEAN4_URL = "https://github.com/leanprover/lean4"
"""The URL of the Lean 4 repo."""

LEAN4_PACKAGES_DIR_OLD = Path("lake-packages")
"""The directory where Lean 4 dependencies are stored (before v4.3.0-rc2)."""

LEAN4_PACKAGES_DIR = Path(".lake/packages")
"""The directory where Lean 4 dependencies are stored (since v4.3.0-rc2)."""

LOAD_USED_PACKAGES_ONLY = "LOAD_USED_PACKAGES_ONLY" in os.environ
"""Only load depdendency files that are actually used by the target repo."""

LEAN4_BUILD_DIR = Path(".lake/build")

LEAN_BUILD_DIR_OLD = Path("build")

TACTIC_TIMEOUT = int(os.getenv("TACTIC_TIMEOUT", 5000))
"""Maximum time (in milliseconds) before interrupting a tactic when interacting with Lean (only for Lean 3).
"""

TACTIC_CPU_LIMIT = int(os.getenv("TACTIC_CPU_LIMIT", 1))
"""Number of CPUs for executing tactics when interacting with Lean (only useful when running within Docker).
"""

TACTIC_MEMORY_LIMIT = os.getenv("TACTIC_MEMORY_LIMIT", "32g")
"""Maximum memory when interacting with Lean (only useful when running within Docker).
"""

CONTAINER = os.getenv("CONTAINER", "native")
"""Container to use for running LeanDojo. Default to ``native`` but also support ``docker``. Using ``docker`` is recommended for Lean 3.
"""

DOCKER_AVAILABLE = os.system("docker version 1>/dev/null 2>/dev/null") == 0

DOCKER_TAG = "yangky11/lean-dojo"

if CONTAINER == "docker":
    assert (
        DOCKER_AVAILABLE
    ), "Failed to access Docker. Please make sure Docker is running and you have access. Alternatively, you can try to run without Docker by setting the `CONTAINER` environment variable to `native` (see https://leandojo.readthedocs.io/en/latest/user-guide.html#advanced-running-within-docker)."
    os.system(f"docker pull {DOCKER_TAG} 1>/dev/null 2>/dev/null")

MIN_LEAN3_VERSION = "v3.42.1"
"""The minimum version of Lean 3 that LeanDojo supports.
"""


def check_git_version(min_version: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Check the version of Git installed on the system."""
    res = subprocess.run("git --version", shell=True, capture_output=True, check=True)
    output = res.stdout.decode()
    error = res.stderr.decode()
    assert error == "", error
    m = re.match(r"git version (?P<version>[0-9.]+)", output)
    version = tuple(int(_) for _ in m["version"].split("."))

    version_str = ".".join(str(_) for _ in version)
    min_version_str = ".".join(str(_) for _ in min_version)
    assert (
        version >= min_version
    ), f"Git version {version_str} is too old. Please upgrade to at least {min_version_str}."


check_git_version((2, 25, 0))
