# TODO: Add a hard timeout for Lean 4.
import re
import os
import sys
import json
import time
import signal
import shutil
from pathlib import Path
from loguru import logger
from tempfile import mkdtemp
from shutil import ignore_patterns
from subprocess import TimeoutExpired
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Dict, Any, Optional

from ..utils import execute, to_json_path
from ..container import get_container, Mount
from ..data_extraction.traced_data import TracedFile
from ..data_extraction.lean import Theorem, LeanGitRepo
from ..data_extraction.trace import get_traced_repo_path
from ..constants import (
    TMP_DIR,
    LEAN3_DEPS_DIR,
    LEAN4_DEPS_DIR,
    TACTIC_TIMEOUT,
    TACTIC_CPU_LIMIT,
    TACTIC_MEMORY_LIMIT,
)


_REPL_PROMPT = "REPL>"
_DECL_REGEX = re.compile(
    r"(?<=\n)(?P<idents>.+?)\s+\:(?P<lean_type>.+?)\n(?=\S)", re.DOTALL
)


@dataclass(frozen=True)
class Declaration:
    ident: str
    lean_type: str


def _parse_local_context(goal_pp: str) -> List[Declaration]:
    decls = []
    for m in _DECL_REGEX.finditer("\n" + goal_pp):
        lean_type = m["lean_type"].strip()
        if lean_type.endswith(","):
            lean_type = lean_type[:-1].strip()
        for ident in m["idents"].strip().split():
            decls.append(Declaration(ident.strip(), lean_type))
    return decls


@dataclass(frozen=True)
class Goal:
    assumptions: List[Declaration]
    conclusion: str

    @classmethod
    def from_pp(cls, pp) -> "Goal":
        _, concl = pp.split("⊢")
        assumptions = _parse_local_context(pp)
        return cls(assumptions, concl.strip())


@dataclass(frozen=True)
class TacticState:
    pp: str
    id: int = field(compare=False)
    message: Optional[str] = field(default=None, compare=False)
    goals: List[Goal] = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        goals = [Goal.from_pp(_) for _ in self.pp.split("\n\n")]
        assert len(goals) == self.pp.count("⊢")
        object.__setattr__(self, "goals", goals)

    @property
    def num_goals(self) -> int:
        return len(self.goals)


@dataclass(frozen=True)
class ProofFinished:
    tactic_state_id: int
    message: Optional[str] = field(default=None, compare=False)


@dataclass(frozen=True)
class ProofGivenUp:
    pass


@dataclass(frozen=True)
class TacticError:
    error: str


@dataclass(frozen=True)
class TimeoutError:
    error: str


TacticResult = Union[
    TacticState,
    ProofFinished,
    TacticError,
    TimeoutError,
    ProofGivenUp,
]


class DojoCrashError(Exception):
    @property
    def is_out_of_memory(self) -> bool:
        return str(self) == "OOM"


class DojoHardTimeoutError(Exception):
    pass


class DojoInitError(Exception):
    @property
    def is_not_a_theorem(self) -> bool:
        return "not_a_theorem" in str(self)


def _get_all_dependencies(
    root_dir: Path, lean_path: Path, repo: LeanGitRepo
) -> List[Path]:
    all_deps = []
    stack = [lean_path]

    while stack != []:
        json_path = to_json_path(root_dir, stack.pop(), repo.uses_lean4)
        tf = TracedFile.from_traced_file(root_dir, json_path, repo)
        deps = tf.get_direct_dependencies()

        for d in deps:
            if d not in all_deps:
                all_deps.append(d)
                stack.append(d)

    return all_deps


@dataclass
class Dojo:
    """Gym-like environment for programmatic interaction with a given theorem in Lean 3."""

    theorem: Theorem
    hard_timeout: Optional[float] = None
    origin_dir: Path = field(init=False, repr=False)
    tmp_dir: Path = field(init=False)
    start_time: float = field(init=False, repr=False)
    is_proved: bool = False
    is_crashed: bool = False
    has_timedout: bool = False

    def __post_init__(self):
        if (
            self.theorem.repo.name == "lean4"
            and self.theorem.file_path.parts[0] == "src"
        ):
            file_path = Path("src/lean") / self.theorem.file_path.relative_to("src")
            object.__setattr__(self.theorem, "file_path", file_path)

        if self.uses_lean4 and self.hard_timeout is None:
            logger.warning(
                "Using Lean 4 without a hard timeout may lead to problems if a tactic hangs indefinitely."
            )

    @property
    def uses_lean4(self) -> bool:
        return self.theorem.repo.uses_lean4

    def _handle_hard_timeout(self, signum, frame) -> None:
        logger.debug(f"Hard timeout when proving {self.theorem}")
        self.has_timedout = True
        raise DojoHardTimeoutError()

    def __enter__(self) -> Tuple["Dojo", TacticState]:
        """Initialize Dojo.

        Raises:
            DojoInitError: _description_

        Returns:
            _type_: _description_
        """
        logger.debug(f"Initializing Dojo for {self.theorem}")

        # Work in a temporary directory.
        self.origin_dir = Path.cwd()
        self.tmp_dir = Path(mkdtemp(dir=TMP_DIR))

        try:
            self.old_sigint = signal.signal(signal.SIGINT, self._exit_gracefully)
            self.old_sigterm = signal.signal(signal.SIGTERM, self._exit_gracefully)
            os.chdir(self.tmp_dir)

            repo = self.theorem.repo
            if repo.is_lean4:
                raise NotImplementedError("InteractingLean 4 is not supported yet.")
            traced_repo_path = get_traced_repo_path(repo)

            # Copy and `cd` into the repo.
            shutil.copytree(
                traced_repo_path,
                repo.name,
                ignore=ignore_patterns("*.dep_paths", "*.ast.json", "*.trace.xml"),
            )
            execute(f"chmod -R a+w {repo.name}")
            os.chdir(repo.name)

            # Replace the human-written proof with a `repl` tactic.
            json_path = to_json_path(
                traced_repo_path, self.theorem.file_path, repo.uses_lean4
            )

            try:
                traced_file = TracedFile.from_traced_file(
                    traced_repo_path, json_path, repo
                )
            except FileNotFoundError:
                raise DojoInitError(f"Cannot find the file {json_path}")

            self._modify_proof(traced_file)

            if repo.uses_lean3:
                if repo.is_lean:
                    path = Path("library/system/io.lean")
                else:
                    path = LEAN3_DEPS_DIR / "lean/library/system/io.lean"
            else:
                if repo.is_lean:
                    path = Path("src/lean/Lean/Elab/Tactic.lean")
                else:
                    path = LEAN4_DEPS_DIR / "lean4/src/lean/Lean/Elab/Tactic.lean"
            deps = [path] + _get_all_dependencies(traced_repo_path, path, repo)

            # Run the modified proof in a container.
            self.container = get_container()
            logger.debug(f"Launching the proof using {type(self.container)}")
            mts = [Mount(Path.cwd(), f"/workspace/{repo.name}")]
            if repo.uses_lean3:
                cmd = f"lean {self.theorem.file_path}"
            elif repo.is_lean4:
                cmd = f"./build/release/stage1/bin/lean {self.theorem.file_path}"
            else:
                self.container.run(
                    f"lake build Lean4Repl",
                    mts,
                    as_current_user=True,
                    capture_output=True,
                    work_dir=f"/workspace/{repo.name}",
                )
                cmd = f"lake env lean {self.theorem.file_path}"

            self.proc = self.container.run_interactive(
                cmd,
                mts,
                cpu_limit=TACTIC_CPU_LIMIT,
                memory_limit=TACTIC_MEMORY_LIMIT,
                work_dir=f"/workspace/{repo.name}",
                as_current_user=True,
            )

            # Get the initial tactic state.
            try:
                res = json.loads(self._read_next_line()[0])
            except Exception as ex:
                if traced_file.path in deps or traced_file.has_prelude:
                    raise DojoInitError(
                        "Currently LeanDojo does not support interacting with proofs in prelude files or files imported by system/io.lean."
                    )
                elif isinstance(ex, EOFError):
                    raise DojoInitError("EOF")
                else:
                    raise ex

            assert res["error"] is None and res["tactic_state"] != "no goals"
            # logger.debug(f"Response: {res}")
            tactic_state = self._post_process(res["tactic_state"])
            init_state = TacticState(
                tactic_state,
                res["tsid"],
            )

            self.start_time = time.monotonic()
            if self.hard_timeout is not None:
                signal.signal(signal.SIGALRM, self._handle_hard_timeout)
                signal.alarm(self.hard_timeout)

            return self, init_state

        except Exception as ex:
            os.chdir(self.origin_dir)
            shutil.rmtree(self.tmp_dir)
            raise ex

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        """Exit Dojo.

        Args:
            exc_type (None): _description_
            exc_val (None): _description_
            exc_tb (None): _description_
        """
        # Cancel the hard timeout.
        if self.hard_timeout is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

        if not self.is_proved:
            logger.debug(f"Failed to prove {self.theorem}")

        if not self.is_crashed and not self.has_timedout:
            if self.uses_lean4:
                req = "exit"
            else:
                req = json.dumps(["exit_repl", []])
            try:
                self._submit_request(req)
            except Exception:
                pass

        self._cleanup()

    def _exit_gracefully(self, signum, frame):
        logger.debug("Exiting gracefully.")
        self._cleanup()
        sys.exit(-1)

    def _cleanup(self):
        logger.debug("Cleaning up.")
        try:
            self._cleanup_container()
            self._cleanup_proc()
        finally:
            self._cleanup_tmp_dir()
            signal.signal(signal.SIGINT, self.old_sigint)
            signal.signal(signal.SIGTERM, self.old_sigterm)

    def _cleanup_container(self) -> None:
        """Clean up the container."""
        logger.debug("Cleaning up the container.")
        self.container.cleanup()

    def _cleanup_proc(self) -> None:
        """Clean up the subprocess."""
        self.proc.terminate()
        try:
            self.proc.wait(timeout=1)
        except TimeoutExpired:
            self.proc.kill()

    def _cleanup_tmp_dir(self) -> None:
        """Clean up the temporary directory."""
        logger.debug("Cleaning up the temporary directory.")
        os.chdir(self.origin_dir)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _post_process(self, tactic_state: str) -> str:
        """Post-process the pretty-printed tactic state.

        Args:
            tactic_state (str): _description_

        Returns:
            str: _description_
        """
        m = re.match(r"\d+ goals\n", tactic_state)
        if m is not None:
            return tactic_state[m.end() :]
        else:
            return tactic_state

    def _modify_proof(self, traced_file: TracedFile) -> None:
        # Modify the proof and set up the `repl` tactic.
        logger.debug("Modifying the proof")

        traced_theorem = traced_file.get_traced_theorem(self.theorem)
        if traced_theorem is None:
            raise DojoInitError(
                f"Failed to locate the theorem with `{self.theorem.full_name}` as its fully qualified name"
            )
        proof_start, proof_end = traced_theorem.locate_proof()

        # logger.debug("Modifiying the proof and set up the `repl` tactic")
        lean_file = traced_file.lean_file

        if lean_file.uses_lean4:
            code_import = "import Lean4Repl\n\n"
            code_proof = "\nby\n  lean_dojo_repl\n  sorry\n"
            code_before_theorem = lean_file[: traced_theorem.start]
            code_thereom = lean_file[traced_theorem.start : proof_start]
            modified_code = (
                code_import
                + code_before_theorem
                + "set_option maxHeartbeats 0 in\n"
                + code_thereom
                + code_proof
                + lean_file[proof_end:]
            )
            repl_file = "Lean4Repl.lean"
            repl_dst = Path(repl_file)

        else:
            code_before_proof = lean_file[:proof_start].strip()
            if not code_before_proof.endswith(":="):
                code_before_proof += " :="
            code_import = "import .lean3_repl\n\n"
            code_proof = "\nbegin\n  lean_dojo.repl,\n  sorry\nend\n"
            modified_code = (
                code_import + code_before_proof + code_proof + lean_file[proof_end:]
            )
            repl_file = "lean3_repl.lean"
            repl_dst = self.theorem.file_path.parent / repl_file

        # Copy "repl.lean" to the same directory of the target file.
        repl_src = Path(__file__).with_name(repl_file)
        repl_code = (
            repl_src.open().read().replace("$TACTIC_TIMEOUT", str(TACTIC_TIMEOUT))
        )

        if repl_dst.exists():
            raise DojoInitError(f"{repl_dst} exists")

        with repl_dst.open("wt") as oup:
            oup.write(repl_code)

        if lean_file.uses_lean4:
            with open("lakefile.lean", "a") as oup:
                oup.write("\nlean_lib Lean4Repl {\n\n}\n")

        with self.theorem.file_path.open("wt") as oup:
            oup.write(modified_code)

    def run_tac(self, state: TacticState, tactic: str) -> TacticResult:
        if not isinstance(state, TacticState):
            raise RuntimeError(
                f"Attempting to run a tactic on an invalid state {state}."
            )
        assert isinstance(tactic, str), f"Invalid tactic {tactic}"

        tsid = state.id
        if self.uses_lean4:
            req = json.dumps({"tsid": tsid, "tac": tactic})
        else:
            req = json.dumps(["run_tac", [tsid, tactic]])
        res = self._submit_request(req)

        if res["error"] is not None:
            if "proof contains `sorry`" in res["error"]:
                return ProofGivenUp()
            elif "try_for_time tactic failed, timeout" in res["error"]:
                return TimeoutError(res["error"].strip())
            else:
                return TacticError(res["error"].strip())
        elif res["tactic_state"] == "no goals":
            self.is_proved = True
            return ProofFinished(res["tsid"], res["message"])
        else:
            tactic_state = self._post_process(res["tactic_state"])
            return TacticState(
                tactic_state,
                res["tsid"],
                res["message"],
            )

    def query_env(self, state: TacticState):
        if self.uses_lean4:
            raise NotImplementedError
        req = json.dumps(["query_env", [state.id]])
        res = self._submit_request(req)
        return res["environment"]

    def query_decl(self, state: TacticState, name: str):
        if self.uses_lean4:
            raise NotImplementedError
        req = json.dumps(["query_decl", [state.id, name]])
        res = self._submit_request(req)
        return res["declaration"]

    def _submit_request(self, req: str) -> Dict[str, Any]:
        """Submit a request to Lean and get the response.

        Args:
            req (str): _description_

        Raises:
            DojoCrashError: _description_

        Returns:
            Dict[str, Any]: _description_
        """
        logger.debug(f"Request: {req}")
        self._check_alive()
        self.proc.stdin.write(req + "\n")
        try:
            res, msg = self._read_next_line()
        except EOFError:
            raise DojoCrashError("EOF")
        # logger.debug(f"Response: {res}")
        try:
            res = json.loads(res)
        except json.decoder.JSONDecodeError:
            raise DojoCrashError(f"Invalid JSON: {res}")

        assert "message" not in res
        res["message"] = msg
        return res

    def _check_alive(self) -> None:
        exit_code = self.proc.poll()
        if exit_code is None:
            return
        elif exit_code == 137:
            raise DojoCrashError("OOM")
        else:
            raise DojoCrashError(f"Unknown exit code: {exit_code}")

    def _read_next_line(self) -> Tuple[str, str]:
        """Read the next line from `self.proc`.

        Raises:
            EOFError: _description_
            DojoCrashError: _description_
            DojoInitError: _description_

        Returns:
            str: _description_
        """
        msg = []
        while True:
            line = self.proc.stdout.readline().strip()
            logger.debug(line)
            if line == "":
                raise EOFError
            if line.startswith(_REPL_PROMPT):
                self._check_alive()
                return line[len(_REPL_PROMPT) :].strip(), "\n".join(msg)
            elif "error: " in line:
                if "error: deep recursion was detected" in line:
                    self.is_crashed = True
                    raise DojoCrashError(line)
                elif "error: [fatal] not_a_theorem" in line:
                    self.is_crashed = True
                    raise DojoInitError(line)
                else:
                    pass
            else:
                msg.append(line)
