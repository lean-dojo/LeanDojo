"""
Constants and configuration for LeanAgent.
"""

import multiprocessing
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

# Batch size for training
BATCH_SIZE = 1

# Environment variables
RAID_DIR = os.path.join(os.getcwd(), "raid")
if "GITHUB_ACCESS_TOKEN" not in os.environ:
    raise ValueError("GITHUB_ACCESS_TOKEN environment variable must be set")
GITHUB_ACCESS_TOKEN = os.environ["GITHUB_ACCESS_TOKEN"]

os.environ["RAY_TMPDIR"] = f"/tmp/ray"

# Directory paths
REPO_DIR = "repos_new"
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
FISHER_DIR = "fisher"  # Optional

# File paths
EVAL_RESULTS_FILE_PATH = f"{RAID_DIR}/eval_results.json"
DB_FILE_NAME = "dynamic_database.json"
PROOF_LOG_FILE_NAME = "proof_log.txt"
ENCOUNTERED_THEOREMS_FILE = f"{RAID_DIR}/encountered_theorems.json"

# List of known repositories to process or skip
# Feel free to remove any repos from this list if you would like to test on them
KNOWN_REPOSITORIES: List[str] = [
    "leanprover-community/mathlib4",  # ReProver is trained on this
    "leanprover-community/batteries",  # functional programming instead of math
    "leanprover-community/aesop",
    "leanprover/lean4",
    "leanprover-community/mathlib",  # Mathlib3 version
    "leanprover-community/mathlib3",
    "leanprover/std4",  # moved to batteries
    "leanprover-community/duper",  # functional programming instead of math
    "leanprover/lake",
    "openai/lean-gym",
    "leanprover-community/lean4-metaprogramming-book",
    "kmill/lean4-raytracer",  # no theorems
    "argumentcomputer/yatima",  # trace problems
    "ImperialCollegeLondon/formalising-mathematics-2024",  # trace problems
    "leanprover-community/ProofWidgets4",  # trace problems
    "leanprover/verso",  # trace problems
    "leanprover-community/NNG4",  # trace problems
    "ufmg-smite/lean-smt",  # fails to trace due to windows-style line endings
    "teorth/symmetric_project",  # no compatible commit
    "cmu-l3/llmlean",  # irrelevant + only 4 theorems
    "PatrickMassot/GlimpseOfLean",  # strange trace problems with _parse_deps
    "avigad/lamr",  # trace problems
    "leanprover-community/quote4",  # no theorems
    "leanprover-community/iris-lean",  # trace problems
    "aripiprazole/rinha",  # incompatible commit
    "leanprover/lean4-cli",  # no theorems
    "leanprover/LeanInk",  # no theorems
    "leanprover-community/lean-auto",
    "leanprover-community/repl",  # no theorems
    "leanprover/doc-gen4",  # no theorems
    "leanprover/SampCert",  # trace problems
    "nomeata/loogle",
    "risc0/risc0-lean4",
    "PatrickMassot/verbose-lean4",  # no theorems
    "tydeu/lean4-alloy",  # no theorems
    "leanprover/leansat",  # deprecated
    "BoltonBailey/formal-snarks-project",  # two theorems
    "dwrensha/lean4-maze",  # two theorems
    "leanprover-community/mathport",  # irrelevant
    "argumentcomputer/LSpec",  # one theorem
    "reaslab/jixia",  # no theorems
    "riccardobrasca/flt3",  # no theorems
    "dwrensha/animate-lean-proofs",  # irrelevant
    "lean-ja/lean-by-example",  # irrelevant
    "NethermindEth/Clear",  # no theorems
    "fgdorais/lean4-parser",  # irrelevant
    "semorrison/lean-training-data",  # irrelevant
    "verse-lab/lean-ssr",  # irrelevant
    "GaloisInc/lean-llvm",  # irrelevant
    "argumentcomputer/Wasm.lean",  # irrelevant
    "NethermindEth/EVMYulLean",  # irrelevant
    "rwbarton/advent-of-lean-4",  # irrelevant
    "leanprover-community/tutorials4",  # irrelevant
    "haruhisa-enomoto/mathlib4-all-tactics",  # irrelevant
    "leanprover/LNSym",
    "leanprover-community/flt-regular",
    "opencompl/lean-mlir-old",
    "rami3l/plfl",
    "HEPLean/HepLean",
    "forked-from-1kasper/ground_zero",
    "verified-optimization/CvxLean",
    "leanprover-community/sphere-eversion",
    "optsuite/optlib",
    "YaelDillies/LeanCamCombi",
    "JamesGallicchio/LeanColls",
    "T-Brick/c0deine",
    "jjdishere/EG",
    "alexkeizer/QpfTypes",
    "fpvandoorn/LeanCourse23",
    "marcusrossel/lean-egg",
    "reilabs/proven-zk",
    "algebraic-dev/soda",
    "leanprover-community/llm",
    "dignissimus/Untangle",
    "argumentcomputer/Megaparsec.lean",
    "emilyriehl/infinity-cosmos",
    "BartoszPiotrowski/lean-premise-selection",
    "djvelleman/HTPILeanPackage",
    "girving/ray",
    "Anderssorby/SDL.lean",
    "pandaman64/lean-regex",
    "brown-cs22/CS22-Lean-2023",
    "hhu-adam/GameSkeleton",
    "FR-vdash-bot/Algorithm",
    "PeterKementzey/graph-library-for-lean4",
    "arthurpaulino/LeanMySQL",
    "arthurpaulino/NumLean",
    "FormalSAT/trestle",
    "nomeata/lean-wf-induct",
    "leanprover/lean4checker",
    "IPDSnelting/tba-2022",
    "digama0/mm-lean4",
    "KislyjKisel/Raylib.lean",
    "algebraic-dev/melp",
    "hhu-adam/Robo",  # same as other tutorials but has lots of sorries
    "hargoniX/socket.lean",
    "kovach/etch",
    "damek/gd-lean",
    "0art0/lean-slides",
    "forked-from-1kasper/lean4-categories",
    "katydid/proofs",
    "alexjbest/leaff",
    "sinhp/Poly",
    "lftcm2023/lftcm2023",  # same as other tutorials but has lots of sorries
    "lean-ja/lean99",
    "leanprover/SHerLOC",
    "Seasawher/mdgen",
    "opencompl/egg-tactic-code",
    "david-christiansen/ssft24",
    "T-Brick/lean2wasm",
    "hargoniX/cpdt-lean",
    "jsm28/AperiodicMonotilesLean",
    "draperlaboratory/ELFSage",
    "rookie-joe/automatic-lean4-compilation",
    "madvorak/fecssk",
    "david-christiansen/bob24",
    "awodey/joyal",
    "BrownCS1951x/fpv2023",  # same as other tutorials but has lots of sorries
    "paulch42/lean-spec",
    "siddhartha-gadgil/MetaExamples",
    "dannypsnl/violet",
    "arthurpaulino/LeanREPL",
    "Kha/do-supplement",
    "joehendrix/lean-sat-checker",
    "ammkrn/timelib",
    "kmill/LeanTeX",
    "leanprover/lean4export",
    "leanprover-community/mathlib3port",
    "brown-cs22/CS22-Lean-2024",  # same as other tutorials but has lots of sorries
    "T-Brick/lean-wasm",
    "crabbo-rave/Soup",
    "argumentcomputer/RustFFI.lean",
    "suhr/tmath",
    "leanprover/leanbv",
    "arthurpaulino/FxyLang",
    "SchrodingerZhu/LeanGccBackend",
    "lecopivo/lean4-karray",
    "ImperialCollegeLondon/M1F-explained",
    "proost-assistant/ProostLean",
    "DavePearce/LeanEVM",
    "algebraic-dev/ash",
    "FormalizedFormalLogic/Arithmetization",
    "cmu-l3/ntp-toolkit",
    "dwrensha/tryAtEachStep",
    "yangky11/lean4-example",
    "T-Brick/DateTime",
    "model-checking/rust-lean-models",
    "MichaelStollBayreuth/EulerProducts",
    "hargoniX/Flame",
    "argumentcomputer/Http.lean",
    "madvorak/vcsp",
    "teorth/newton",
    "apnelson1/Matroid",
    "smorel394/TS1",
    "ianjauslin-rutgers/pythagoras4",
    "mortarsanjaya/IMOSLLean4",
    "dupuisf/BibtexQuery",
    "nomeata/lean-calcify",
    "argumentcomputer/FFaCiL.lean",
    "javra/iit",
    "arthurpaulino/viper",
    "lindy-labs/aegis",
    "PatrickMassot/NNG4",
    "argumentcomputer/YatimaStdLib.lean",
    "fgdorais/lean4-unicode-basic",
    "mhuisi/Uniq",
    "Kha/macro-supplement",
    "chenjulang/rubikcubegroup",
    "arthurpaulino/LeanMusic",
    "argumentcomputer/Ipld.lean",
    "Odomontois/advent2022-lean",
    "kbuzzard/IISc-experiments",  # same as other tutorials but has lots of sorries
    "ykonstant1/InfinitePrimes",
    "alexkassil/natural_number_game_lean4",
    "seewoo5/lean-poly-abc",
    "rah4927/lean-dojo-mew",
    "siddhartha-gadgil/proofs-and-programs-2023",
    "PatrickMassot/lean4-game-server",
    "knowsys/Formale-Systeme-in-LEAN",  # same as other tutorials but has lots of sorries
    "katydid/symbolic-automatic-derivatives",
    "girving/interval",
    "ImperialCollegeLondon/group-theory-experiments",
    "knowsys/CertifyingDatalog",
    "bergmannjg/leanCurl",
    "vasnesterov/HadwigerNelson",
    "FWuermse/lean-postgres",
    "leanprover-community/import-graph",
    "Human-Oriented-ATP/lean-tactics",  # more about tactics than premises
    "paulcadman/lean4-leetcode",
    "argumentcomputer/Lurk.lean",
    "AlexDuchnowski/rubiks-cube",
    "SchrodingerZhu/lean-gccjit",
    "JamesGallicchio/http",
    "jtristan/UnicodeSkipListTableExample",
    "adomani/MA4N1_2023",  # same as other tutorials but has lots of sorries
    "remimimimimi/leansec",
    "hhu-adam/lean-i18n",
    "RemyDegenne/testing-lower-bounds",
    "mariainesdff/LocalClassFieldTheory",
    "AviCraimer/relational-calculus-library-lean4",
    "JLimperg/regensburg-itp-school-2023",
    "jaalonso/Calculemus2",
    "mseri/BET",
    "xubaiw/Reservoir.lean",
    "hargoniX/nest-core",
    "siddhartha-gadgil/Polylean",
    "MichaelStollBayreuth/Weights",
    "sanchace/FRACTRAN",
    "argumentcomputer/Poseidon.lean",
    "madvorak/chomsky",
    "T-Brick/ControlFlow",
    "pa-ba/guarded-lean",
]

# Mark symbols for annotations
MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"


def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")


load_dotenv()

__version__ = "4.20.0"

# Cache and temporary directories
CACHE_DIR = (
    Path(os.environ["CACHE_DIR"])
    if "CACHE_DIR" in os.environ
    else Path.home() / ".cache/lean_dojo"
).absolute()
"""Cache directory for storing traced repos (see :ref:`caching`).
"""

REMOTE_CACHE_URL = "https://dl.fbaipublicfiles.com/lean-dojo"
"""URL of the remote cache (see :ref:`caching`)."""

DISABLE_REMOTE_CACHE = "DISABLE_REMOTE_CACHE" in os.environ
"""Whether to disable remote caching (see :ref:`caching`) and build all repos locally.
"""

TMP_DIR = Path(os.environ["TMP_DIR"]).absolute() if "TMP_DIR" in os.environ else None
"""Temporary directory used by LeanDojo for storing intermediate files
"""
if TMP_DIR is not None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)


# Process and worker configuration
MAX_NUM_PROCS = 32

NUM_PROCS = int(os.getenv("NUM_PROCS", min(multiprocessing.cpu_count(), MAX_NUM_PROCS)))
"""Number of processes to use
"""

NUM_WORKERS = NUM_PROCS - 1

# Lean 4 configuration
LEAN4_URL = "https://github.com/leanprover/lean4"
"""The URL of the Lean 4 repo."""

LEAN4_PACKAGES_DIR = Path(".lake/packages")
"""The directory where Lean 4 dependencies are stored (since v4.3.0-rc2)."""

LOAD_USED_PACKAGES_ONLY = "LOAD_USED_PACKAGES_ONLY" in os.environ
"""Only load depdendency files that are actually used by the target repo."""

LEAN4_BUILD_DIR = Path(".lake/build")

# Tactic execution limits
TACTIC_CPU_LIMIT = int(os.getenv("TACTIC_CPU_LIMIT", 1))
"""Number of CPUs for executing tactics when interacting with Lean.
"""

TACTIC_MEMORY_LIMIT = os.getenv("TACTIC_MEMORY_LIMIT", "32g")
"""Maximum memory when interacting with Lean.
"""

assert re.fullmatch(r"\d+g", TACTIC_MEMORY_LIMIT)


def check_git_version(min_version: Tuple[int, int, int]) -> None:
    """Check the version of Git installed on the system."""
    res = subprocess.run("git --version", shell=True, capture_output=True, check=True)
    output = res.stdout.decode().strip()
    error = res.stderr.decode()
    assert error == "", error
    m = re.search(r"git version (\d+\.\d+\.\d+)", output)
    assert m, f"Could not parse Git version from: {output}"
    # Convert version number string to tuple of integers
    version = tuple(int(_) for _ in m.group(1).split("."))
    version_str = ".".join(str(_) for _ in version)
    min_version_str = ".".join(str(_) for _ in min_version)
    assert (
        version >= min_version
    ), f"Git version {version_str} is too old. Please upgrade to at least {min_version_str}."


check_git_version((2, 25, 0))
