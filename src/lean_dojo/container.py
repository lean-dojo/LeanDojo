"""Containers provide runtime environment for running LeanDojo. 
Currently, LeanDojo supports two types of containers: ``docker`` and ``native``. 
The former is the default and recommended option, while the latter is experimental.
"""
import os
import shlex
import signal
import shutil
import tempfile
import subprocess
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Tuple, Optional

from .constants import CONTAINER, DOCKER_TAG
from .utils import execute, report_critical_failure, working_directory


@dataclass(frozen=True)
class Mount:
    """A mount is a pair of source and destination paths."""

    src: Path
    dst: Path

    def __post_init__(self):
        object.__setattr__(self, "src", Path(self.src))
        object.__setattr__(self, "dst", Path(self.dst))

    def __iter__(self):
        yield self.src
        yield self.dst


def create_mounts(mts: Dict[Union[str, Path], Union[str, Path]]) -> List[Mount]:
    """Create a list of mounts from a dictionary."""
    return [Mount(Path(k), Path(v)) for k, v in mts.items()]


class Container(ABC):
    """Abstract base class for containers."""

    @abstractmethod
    def run(
        self,
        command: str,
        mounts: List[Mount],
        envs: Dict[str, str],
        as_current_user: bool,
        capture_output: bool,
        cpu_limit: Optional[int],
        memory_limit: Optional[str],
        work_dir: Optional[str],
    ) -> None:
        """Run a command in the container.

        Args:
            command (str): _description_
            mounts (List[Mount]): _description_
            envs (Dict[str, str]): _description_
            as_current_user (bool): _description_
            capture_output (bool): _description_
            cpu_limit (Optional[int]): _description_
            memory_limit (Optional[str]): _description_
            work_dir (Optional[str]): _description_
        """
        raise NotImplementedError

    @abstractmethod
    def run_interactive(
        self,
        command: str,
        mounts: List[Mount],
        envs: Dict[str, str],
        as_current_user: bool,
        cpu_limit: Optional[int],
        memory_limit: Optional[str],
        work_dir: Optional[str],
    ) -> subprocess.Popen:
        """Run a command in the container interactively.

        Args:
            command (str): _description_
            mounts (List[Mount]): _description_
            envs (Dict[str, str]): _description_
            as_current_user (bool): _description_
            cpu_limit (Optional[int]): _description_
            memory_limit (Optional[str]): _description_
            work_dir (Optional[str]): _description_

        Returns:
            subprocess.Popen: _description_
        """
        raise NotImplementedError


def _copy_file_or_dir(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copy(src, dst)
    else:
        assert src.is_dir() and not src.is_relative_to(dst)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


class NativeContainer(Container):
    """A container that runs commands natively."""

    def _mount_files(self, mounts: List[Mount]) -> None:
        cwd = Path.cwd()

        for src, dst in mounts:
            if dst.is_absolute():
                dst = cwd / dst.relative_to(dst.root)
            if src == cwd:
                for path in src.glob("*"):
                    p = dst / path.relative_to(src)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    _copy_file_or_dir(path, p)
                continue
            assert not cwd.is_relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            _copy_file_or_dir(src, dst)

    def _unmount_files(self, mounts: List[Mount]) -> None:
        cwd = Path.cwd()

        for src, dst in mounts:
            if dst.is_absolute():
                dst = cwd / dst.relative_to(dst.root)

            if dst.exists():
                if src.is_file():
                    shutil.move(dst, src)
                elif dst.is_relative_to(src):
                    for path in dst.glob("*"):
                        p = src / path.relative_to(dst)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        _copy_file_or_dir(path, p)
                    shutil.rmtree(dst)
                else:
                    with report_critical_failure(
                        f"Failed to override the directory {src}"
                    ):
                        shutil.rmtree(src)
                        shutil.move(dst, src)

            for path in dst.parents:
                if (
                    path.exists()
                    and path.is_relative_to(cwd)
                    and len(list(path.glob("**/*"))) == 0
                ):
                    path.rmdir()

    def _build_native_command(self, command: str, envs: Dict[str, str]) -> str:
        if len(envs) == 0:
            return command
        else:
            return " ".join(f"{k}={v}" for k, v in envs.items()) + " " + command

    def run(
        self,
        command: str,
        mounts: List[Mount] = [],
        envs: Dict[str, str] = {},
        as_current_user: bool = True,
        capture_output: bool = False,
        cpu_limit: Optional[int] = None,
        memory_limit: Optional[str] = None,
        work_dir: Union[Path, str, None] = None,
    ) -> None:
        assert as_current_user, "NativeContainer can only run as the current user."
        assert memory_limit is None, "NativeContainer does not support memory limit."
        assert cpu_limit is None, "NativeContainer does not support CPU limit."

        self._mount_files(mounts)

        cmd = self._build_native_command(command, envs)
        logger.debug(cmd)

        if work_dir is None:
            work_dir = Path.cwd()
        else:
            work_dir = Path(work_dir)
            if work_dir.is_absolute():
                work_dir = Path.cwd() / work_dir.relative_to(work_dir.root)

        with working_directory(work_dir):
            execute(cmd, capture_output=capture_output)

        self._unmount_files(mounts)

    def run_interactive(
        self,
        command: str,
        mounts: List[Mount] = [],
        envs: Dict[str, str] = {},
        as_current_user: bool = True,
        cpu_limit: Optional[int] = None,
        memory_limit: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> subprocess.Popen:
        assert as_current_user, "NativeContainer can only run as the current user."
        if cpu_limit is not None:
            logger.warning(
                f"Disregarding `cpu_limit = {cpu_limit} since NativeContainer does not support CPU limit.`"
            )
        if memory_limit is not None:
            logger.warning(
                f"Disregarding `memory_limit = {memory_limit}` since NativeContainer does not support memory limit."
            )

        self._mount_files(mounts)
        self.mounts = mounts

        cmd = self._build_native_command(command, envs)
        logger.debug(cmd)

        if work_dir is None:
            work_dir = Path.cwd()
        else:
            work_dir = Path(work_dir)
            if work_dir.is_absolute():
                work_dir = Path.cwd() / work_dir.relative_to(work_dir.root)

        with working_directory(work_dir):
            proc = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding="utf-8",
                bufsize=1,
            )

        return proc

    def cleanup(self) -> None:
        self._unmount_files(self.mounts)


class DockerContainer(Container):
    """A container that runs commands in a Docker container."""

    def __init__(self, image: str) -> None:
        self.image = image
        self.cid_file = None

    def _build_docker_command(
        self,
        command: str,
        mounts: List[Mount],
        envs: Dict[str, str],
        as_current_user: bool,
        cpu_limit: Optional[int] = None,
        memory_limit: Optional[str] = None,
        work_dir: Optional[str] = None,
        interactive: bool = False,
    ) -> Tuple[str, Path]:
        cid_file = Path(next(tempfile._get_candidate_names()) + ".cid")
        cmd = f"docker run --cidfile {cid_file} --rm"
        if as_current_user:
            cmd += f" -u {os.getuid()}"
        for src, dst in mounts:
            cmd += f' --mount type=bind,src="{src}",target="{dst}"'
        for k, v in envs.items():
            cmd += f" --env {k}={v}"
        if cpu_limit:
            cmd += f" --cpus {cpu_limit}"
        if memory_limit:
            cmd += f" --memory {memory_limit}"
        if work_dir:
            cmd += f" --workdir {work_dir}"
        if interactive:
            cmd += " -i"
        cmd += f" {self.image} {command}"
        return cmd, cid_file

    def run(
        self,
        command: str,
        mounts: List[Mount] = [],
        envs: Dict[str, str] = {},
        as_current_user: bool = True,
        capture_output: bool = False,
        cpu_limit: Optional[int] = None,
        memory_limit: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> None:
        cmd, cid_file = self._build_docker_command(
            command,
            mounts,
            envs,
            as_current_user,
            cpu_limit,
            memory_limit,
            work_dir,
            interactive=False,
        )
        logger.debug(cmd)

        def _exit_gracefully(signum, frame):
            cid = open(cid_file).read().strip()
            execute(f"docker stop -t 1 {cid}", capture_output=True)
            raise RuntimeError(f"Failed to execute {cmd}")

        old_sigint = signal.signal(signal.SIGINT, _exit_gracefully)
        old_sigterm = signal.signal(signal.SIGTERM, _exit_gracefully)

        execute(cmd, capture_output=capture_output)

        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if cid_file.exists():
            cid_file.unlink()

    def run_interactive(
        self,
        command: str,
        mounts: List[Mount] = [],
        envs: Dict[str, str] = {},
        as_current_user: bool = False,
        cpu_limit: Optional[int] = None,
        memory_limit: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> subprocess.Popen:
        cmd, self.cid_file = self._build_docker_command(
            command,
            mounts,
            envs,
            as_current_user,
            cpu_limit,
            memory_limit,
            work_dir,
            interactive=True,
        )
        logger.debug(cmd)
        proc = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            bufsize=1,
        )
        return proc

    def cleanup(self) -> None:
        # Cannot use `self.proc.terminate()` to stop Docker since it may be running as root.
        if self.cid_file is None or not self.cid_file.exists():
            return
        cid = self.cid_file.open().read().strip()
        os.system(f"docker stop -t 1 {cid} 1>/dev/null 2>/dev/null")


def get_container() -> Container:
    if CONTAINER == "docker":
        return DockerContainer(DOCKER_TAG)
    else:
        assert (
            CONTAINER == "native"
        ), "Currently only `docker` and `native` are supported."
        return NativeContainer()
