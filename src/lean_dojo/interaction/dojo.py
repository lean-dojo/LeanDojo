import re
import os
import json
import time
import psutil
import pexpect
import tempfile
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from subprocess import CalledProcessError
from typing import Union, Tuple, List, Dict, Any, Optional, TextIO

from .parse_goals import parse_goals, Goal
from ..data_extraction.trace import get_traced_repo_path
from ..utils import to_json_path, working_directory, execute
from ..data_extraction.lean import Theorem, LeanGitRepo, Pos
from ..constants import TACTIC_CPU_LIMIT, TACTIC_MEMORY_LIMIT
from ..data_extraction.traced_data import TracedFile, get_code_without_comments


@dataclass(frozen=True)
class CommandState:
    id: int = field(compare=False)
    message: Optional[str] = field(default=None, compare=False)


@dataclass(frozen=True)
class TacticState:
    pp: str
    id: int = field(compare=False)
    message: Optional[str] = field(default=None, compare=False)
    goals: List[Goal] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        goals = parse_goals(self.pp)
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
class LeanError:
    error: str


TacticResult = Union[
    TacticState,
    ProofFinished,
    LeanError,
    ProofGivenUp,
]

CommandResult = Union[CommandState, LeanError]

State = Union[CommandState, TacticState]


class DojoCrashError(Exception):
    @property
    def is_out_of_memory(self) -> bool:
        return str(self) == "OOM"


class DojoTacticTimeoutError(Exception):
    pass


class DojoInitError(Exception):
    pass


def kill_descendants(pid: int) -> None:
    try:
        _kill_descendants(psutil.Process(pid))
    except psutil.NoSuchProcess:
        pass


def _kill_descendants(proc: psutil.Process) -> None:
    for child in proc.children():
        _kill_descendants(child)
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass


_SORRY_WARNING_REGEX = re.compile(
    r"(?P<line>\d+)\:\d+\:\s+warning\:\s+declaration uses \'sorry\'"
)


def check_proof(thm: Theorem, proof: str) -> bool:
    """Check if a proof is correct.

    Args:
        thm (Theorem): The theorem statement.
        proof (str): The proof to check.
    """
    # Replace the original human-written proof.
    traced_repo_path = get_traced_repo_path(thm.repo)
    repl_path = traced_repo_path / "Lean4Repl.lean"
    assert (
        repl_path.exists()
    ), "Unable to find Lean4Repl.lean in the traced repo. The traced repo was likely produced by an outdated version of LeanDojo. See https://github.com/lean-dojo/LeanDojo/releases/tag/v2.0.0."
    try:
        json_path = to_json_path(traced_repo_path, thm.file_path, thm.repo)
        traced_file = TracedFile.from_traced_file(traced_repo_path, json_path, thm.repo)
    except FileNotFoundError:
        raise DojoInitError(
            f"Cannot find the *.ast.json file for {thm} in {traced_repo_path}."
        )
    traced_theorem = traced_file.get_traced_theorem(thm)
    if traced_theorem is None:
        raise DojoInitError(
            f"Failed to locate the theorem with `{thm.full_name}` as its fully qualified name."
        )

    # Modify the code and write it to a temporary file.
    modified_file = tempfile.NamedTemporaryFile(  # type: ignore
        "wt",
        prefix=thm.file_path.stem,
        suffix=thm.file_path.suffix,
        dir=traced_file.abs_path.parent,
        delete=True,
    ).__enter__()
    logger.debug(f"Modifying `{thm.file_path}` into `{modified_file.name}`")
    proof_start, proof_end = traced_theorem.locate_proof()
    lean_file = traced_file.lean_file

    code_before_theorem = get_code_without_comments(
        lean_file, lean_file.start_pos, traced_theorem.start, traced_file.comments
    )
    code_thereom = get_code_without_comments(
        lean_file, traced_theorem.start, proof_start, traced_file.comments
    ).strip()
    if code_thereom.endswith(" where"):
        raise DojoInitError("Cannot interact with theorems with the `where` keyword.")
    if not code_thereom.endswith(":="):
        code_thereom += " := "
    modified_code = (
        "import Lean4Repl\n"
        + code_before_theorem
        + "\n\nset_option maxHeartbeats 0 in\n"
    )
    start_line = modified_code.count("\n") + 1
    modified_code += code_thereom + f"{proof}\n"
    end_line = modified_code.count("\n") + 1
    modified_code += lean_file[proof_end:]

    modified_file.write(modified_code)
    modified_file.flush()

    if os.path.exists("lakefile.olean"):
        os.remove("lakefile.olean")
    if os.path.exists(".lake/lakefile.olean"):
        os.remove(".lake/lakefile.olean")

    # Run the modified file.
    with working_directory(traced_repo_path):
        memory_limit = 1024 * int(TACTIC_MEMORY_LIMIT[:-1])
        modified_path = Path(modified_file.name).relative_to(traced_repo_path)
        cmd = f"lake env lean --threads={TACTIC_CPU_LIMIT} --memory={memory_limit} {modified_path}"
        try:
            oup, _ = execute(cmd, capture_output=True)
            for m in _SORRY_WARNING_REGEX.finditer(oup):
                line = int(m.group("line"))
                if start_line <= line <= end_line:
                    return False
        except CalledProcessError:
            return False
        return True


class Dojo:
    """Gym-like environment for programmatic interaction with Lean through tactics or commands."""

    entry: Union[Theorem, Tuple[LeanGitRepo, Path, int]]
    additional_imports: List[str]
    repo: LeanGitRepo
    file_path: Path
    modified_file: TextIO
    is_successful: Optional[bool] = None
    is_crashed: bool = False
    has_timedout: bool = False

    def __init__(
        self,
        entry: Union[Theorem, Tuple[LeanGitRepo, Path, int]],
        timeout: int = 600,
        additional_imports: List[str] = [],
        build_deps: bool = True,
    ):
        """Initialize Dojo.

        Args:
            entry (Union[Theorem, Tuple[LeanGitRepo, Path, int]]): When a Theorem is given,
                the :class:`Dojo` object enables interaction with the theorem through tactics.
                When a tuple of (repo, file_path, line_nb) is given (only supported in Lean 4),
                the :class:`Dojo` object enables interaction with Lean through commands (similar to a REPL).
            timeout (int): The maximum number of seconds for a single interaction (e.g., tactic).
        """
        self.entry = entry
        self.timeout = timeout
        self.additional_imports = additional_imports
        self.build_deps = build_deps

        if self.uses_tactics:
            assert isinstance(entry, Theorem)
            self.repo, self.file_path = entry.repo, entry.file_path
            self.is_successful = False
        else:
            assert self.uses_commands
            assert isinstance(entry, tuple)
            self.repo, self.file_path, _ = entry
            self.file_path = Path(self.file_path)

    @property
    def uses_tactics(self) -> bool:
        return isinstance(self.entry, Theorem)

    @property
    def uses_commands(self) -> bool:
        return isinstance(self.entry, tuple)

    def __enter__(self) -> Tuple["Dojo", State]:
        """Initialize Dojo."""
        logger.debug(f"Initializing Dojo for {self.entry}")

        # Replace the human-written proof with a `repl` tactic.
        traced_repo_path = get_traced_repo_path(self.repo, self.build_deps)
        repl_path = traced_repo_path / "Lean4Repl.lean"
        assert (
            repl_path.exists()
        ), "Unable to find Lean4Repl.lean in the traced repo. The traced repo was likely produced by an outdated version of LeanDojo. See https://github.com/lean-dojo/LeanDojo/releases/tag/v2.0.0."

        try:
            traced_file = self._locate_traced_file(traced_repo_path)
        except FileNotFoundError:
            raise DojoInitError(
                f"Cannot find the *.ast.json file for {self.entry} in {traced_repo_path}."
            )

        self._modify_file(traced_file)

        # Run the modified file.
        with working_directory(traced_repo_path):
            memory_limit = 1024 * int(TACTIC_MEMORY_LIMIT[:-1])
            modified_path = Path(self.modified_file.name).relative_to(traced_repo_path)
            cmd = f"lake env lean --threads={TACTIC_CPU_LIMIT} --memory={memory_limit} {modified_path}"
            self.proc = pexpect.spawn(
                cmd, timeout=self.timeout, maxread=1, encoding="utf-8", echo=False
            )

        # Get the initial tactic state.
        try:
            res = json.loads(self._read_next_line()[0])
        except Exception as ex:
            if traced_file.has_prelude:
                raise DojoInitError(
                    "Currently LeanDojo does not support interacting with proofs in prelude files."
                )
            elif isinstance(ex, EOFError):
                raise DojoInitError("Unexpected EOF")
            elif isinstance(ex, DojoTacticTimeoutError):
                raise DojoInitError("Timeout during initialization")
            else:
                raise ex

        assert res["error"] is None

        if self.uses_tactics:
            assert res["tacticState"] != "no goals"
            init_state: State = TacticState(
                self._post_process(res["tacticState"]),
                res["sid"],
            )
        else:
            assert self.uses_commands
            init_state = CommandState(int(res["sid"]))

        self.start_time = time.monotonic()
        return self, init_state

    def _locate_traced_file(self, traced_repo_path: Path) -> TracedFile:
        json_path = to_json_path(traced_repo_path, self.file_path, self.repo)
        return TracedFile.from_traced_file(traced_repo_path, json_path, self.repo)

    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        """Exit Dojo.

        Args:
            exc_type (None): _description_
            exc_val (None): _description_
            exc_tb (None): _description_
        """
        logger.debug("Cleaning up.")
        kill_descendants(self.proc.pid)
        self.modified_file.__exit__(exc_type, exc_val, exc_tb)

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

    def _get_imports(self) -> str:
        imports = ["Lean4Repl"] + self.additional_imports
        return "\n".join(f"import {_}" for _ in imports) + "\n\n"

    def _modify_file(self, traced_file: TracedFile) -> None:
        self.modified_file = tempfile.NamedTemporaryFile(  # type: ignore
            "wt",
            prefix=self.file_path.stem,
            suffix=self.file_path.suffix,
            dir=traced_file.abs_path.parent,
            delete=True,
        ).__enter__()
        logger.debug(f"Modifying `{self.file_path}` into `{self.modified_file.name}`")

        # Modify the code and write it to a temporary file.
        if self.uses_tactics:
            # Interaction through tactics.
            modified_code = self._get_modified_proof(traced_file)
        else:
            # Interaction through commands (via CommandElabM).
            lean_file = traced_file.lean_file
            pos = Pos(line_nb=self.entry[2], column_nb=1)
            code_before = get_code_without_comments(
                lean_file, lean_file.start_pos, pos, traced_file.comments
            )
            modified_code = (
                self._get_imports()
                + code_before
                + "set_option maxHeartbeats 0 in\n#lean_dojo_repl\n\n"
                + lean_file[pos:]
            )
        self.modified_file.write(modified_code)
        self.modified_file.flush()

        if os.path.exists("lakefile.olean"):
            os.remove("lakefile.olean")
        if os.path.exists(".lake/lakefile.olean"):
            os.remove(".lake/lakefile.olean")

    def _get_modified_proof(self, traced_file: TracedFile) -> str:
        # Modify the proof and set up the `repl` tactic.
        assert isinstance(self.entry, Theorem)
        traced_theorem = traced_file.get_traced_theorem(self.entry)
        if traced_theorem is None:
            raise DojoInitError(
                f"Failed to locate the theorem with `{self.entry.full_name}` as its fully qualified name."
            )
        proof_start, proof_end = traced_theorem.locate_proof()
        lean_file = traced_file.lean_file

        code_import = self._get_imports()
        code_proof = "by\n  lean_dojo_repl\n  sorry\n"
        code_before_theorem = get_code_without_comments(
            lean_file, lean_file.start_pos, traced_theorem.start, traced_file.comments
        )
        code_thereom = get_code_without_comments(
            lean_file, traced_theorem.start, proof_start, traced_file.comments
        ).strip()
        if code_thereom.endswith(" where"):
            raise DojoInitError(
                "Cannot interact with theorems with the `where` keyword."
            )
        if not code_thereom.endswith(":="):
            code_thereom += " := "
        modified_code = (
            code_import
            + code_before_theorem
            + "\n\nset_option maxHeartbeats 0 in\n"
            + code_thereom
            + code_proof
            + lean_file[proof_end:]
        )

        return str(modified_code)

    def run_tac(self, state: TacticState, tactic: str) -> TacticResult:
        if not isinstance(state, TacticState):
            raise RuntimeError(
                f"Attempting to run a tactic on an invalid state {state}."
            )
        assert isinstance(tactic, str), f"Invalid tactic {tactic}"

        tsid = state.id
        req = json.dumps({"sid": tsid, "cmd": tactic}, ensure_ascii=False)
        res = self._submit_request(req)

        if res["error"] is not None:
            if "proof contains `sorry`" in res["error"]:
                return ProofGivenUp()
            else:
                return LeanError(res["error"].strip())
        elif res["tacticState"] == "no goals":
            self.is_successful = True
            return ProofFinished(res["sid"], res["message"])
        else:
            tactic_state = self._post_process(res["tacticState"])
            return TacticState(
                tactic_state,
                res["sid"],
                res["message"],
            )

    def run_cmd(self, state: CommandState, command: str) -> CommandResult:
        if not isinstance(state, CommandState):
            raise RuntimeError(
                f"Attempting to run a command on an invalid state {state}."
            )
        assert isinstance(command, str), f"Invalid command {command}"

        csid = state.id
        req = json.dumps({"sid": csid, "cmd": command}, ensure_ascii=False)
        res = self._submit_request(req)

        if res["error"] is not None:
            return LeanError(res["error"].strip())
        else:
            return CommandState(res["sid"], res["message"])

    def _submit_request(self, req: str) -> Dict[str, Any]:
        """Submit a request to Lean and get the response.

        Args:
            req (str): _description_

        Raises:
            DojoCrashError: _description_

        Returns:
            Dict[str, Any]: _description_
        """
        self._check_alive()
        logger.debug(req)
        self.proc.sendline(req)
        try:
            res, msg = self._read_next_line()
        except EOFError:
            raise DojoCrashError("Unexpected EOF")
        try:
            result: Dict[str, Any] = json.loads(res)
        except json.decoder.JSONDecodeError:
            raise DojoCrashError(f"Invalid JSON: {res}")

        result["message"] = msg
        logger.debug(result)
        return result

    def _check_alive(self) -> None:
        if self.proc.isalive():
            return
        exit_code = self.proc.exitstatus
        assert exit_code is not None
        if exit_code == 137:
            raise DojoCrashError("OOM")
        else:
            raise DojoCrashError(f"Unexpected exit code: {exit_code}")

    def _read_next_line(self) -> Tuple[str, str]:
        """Read the next line from `self.proc`.

        Raises:
            EOFError: _description_
            DojoCrashError: _description_
            DojoInitError: _description_

        Returns:
            str: _description_
        """
        _REPL_PROMPT = "REPL>"
        msg: List[str] = []
        while True:
            try:
                index = self.proc.expect(["\n", f"{_REPL_PROMPT}.*?\n"])
                if index == 0:
                    if self.proc.before == "":
                        raise EOFError
                    else:
                        msg.append(self.proc.before.strip())
                        continue
                self._check_alive()
                res = self.proc.match.string[len(_REPL_PROMPT) :].strip()
                return res, "\n".join(msg) + self.proc.before
            except pexpect.EOF:
                raise EOFError
            except pexpect.TIMEOUT:
                logger.debug(f"Tactic timed out")
                self.has_timedout = True
                raise DojoTacticTimeoutError()
