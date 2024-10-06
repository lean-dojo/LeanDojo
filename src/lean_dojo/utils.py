"""Utility functions used internally by LeanDojo.
"""

import re
import os
import ray
import time
import urllib
import typing
import hashlib
import tempfile
import subprocess
from pathlib import Path
from loguru import logger
from functools import cache
from contextlib import contextmanager
from ray.util.actor_pool import ActorPool
from typing import Tuple, Union, List, Generator, Optional

from .constants import NUM_WORKERS, TMP_DIR, LEAN4_PACKAGES_DIR, LEAN4_BUILD_DIR


@contextmanager
def working_directory(
    path: Optional[Union[str, Path]] = None
) -> Generator[Path, None, None]:
    """Context manager setting the current working directory (CWD) to ``path`` (or a temporary directory if ``path`` is None).

    The original CWD is restored after the context manager exits.

    Args:
        path (Optional[Union[str, Path]], optional): The desired CWD. Defaults to None.

    Yields:
        Generator[Path, None, None]: A ``Path`` object representing the CWD.
    """
    origin = Path.cwd()
    if path is None:
        tmp_dir = tempfile.TemporaryDirectory(dir=TMP_DIR)
        path = tmp_dir.__enter__()
        is_temporary = True
    else:
        is_temporary = False

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    os.chdir(path)

    try:
        yield path
    finally:
        os.chdir(origin)
        if is_temporary:
            tmp_dir.__exit__(None, None, None)


@contextmanager
def ray_actor_pool(
    actor_cls: type, *args, **kwargs
) -> Generator[ActorPool, None, None]:
    """Create a pool of Ray Actors of class ``actor_cls``.

    Args:
        actor_cls (type): A Ray Actor class (annotated by ``@ray.remote``).
        *args: Position arguments passed to ``actor_cls``.
        **kwargs: Keyword arguments passed to ``actor_cls``.

    Yields:
        Generator[ActorPool, None, None]: A :class:`ray.util.actor_pool.ActorPool` object.
    """
    assert not ray.is_initialized()
    ray.init()
    pool = ActorPool([actor_cls.remote(*args, **kwargs) for _ in range(NUM_WORKERS)])  # type: ignore
    try:
        yield pool
    finally:
        ray.shutdown()


@contextmanager
def report_critical_failure(msg: str) -> Generator[None, None, None]:
    """Context manager logging ``msg`` in case of any exception.

    Args:
        msg (str): The message to log in case of exceptions.

    Raises:
        ex: Any exception that may be raised within the context manager.
    """
    try:
        yield
    except Exception as ex:
        logger.error(msg)
        raise ex


def execute(
    cmd: Union[str, List[str]], capture_output: bool = False
) -> Optional[Tuple[str, str]]:
    """Execute the shell command ``cmd`` and optionally return its output.

    Args:
        cmd (Union[str, List[str]]): The shell command to execute.
        capture_output (bool, optional): Whether to capture and return the output. Defaults to False.

    Returns:
        Optional[Tuple[str, str]]: The command's output, including stdout and stderr (None if ``capture_output == False``).
    """
    logger.debug(cmd)
    try:
        res = subprocess.run(cmd, shell=True, capture_output=capture_output, check=True)
    except subprocess.CalledProcessError as ex:
        if capture_output:
            logger.info(ex.stdout.decode())
            logger.error(ex.stderr.decode())
        raise ex
    if not capture_output:
        return None
    output = res.stdout.decode()
    error = res.stderr.decode()
    return output, error


def compute_md5(path: Path) -> str:
    """Return the MD5 hash of the file ``path``."""
    # The file could be large
    # See: https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file
    hasher = hashlib.md5()
    with path.open("rb") as inp:
        while True:
            block = inp.read(64 * (1 << 20))
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


_CAMEL_CASE_REGEX = re.compile(r"(_|-)+")


def camel_case(s: str) -> str:
    """Convert the string ``s`` to camel case."""
    return _CAMEL_CASE_REGEX.sub(" ", s).title().replace(" ", "")


def is_optional_type(tp: type) -> bool:
    """Test if ``tp`` is Optional[X]."""
    if typing.get_origin(tp) != Union:
        return False
    args = typing.get_args(tp)
    return len(args) == 2 and args[1] == type(None)


def remove_optional_type(tp: type) -> type:
    """Given Optional[X], return X."""
    assert typing.get_origin(tp) == Union
    args = typing.get_args(tp)
    if len(args) == 2 and args[1] == type(None):
        return args[0]
    else:
        raise ValueError(f"{tp} is not Optional")


@cache
def read_url(url: str, num_retries: int = 2) -> str:
    """Read the contents of the URL ``url``. Retry if failed"""
    backoff = 1
    while True:
        try:
            request = urllib.request.Request(url)  # type: ignore
            gh_token = os.getenv("GITHUB_ACCESS_TOKEN")
            if gh_token is not None:
                request.add_header("Authorization", f"token {gh_token}")
            with urllib.request.urlopen(request) as f:  # type: ignore
                return f.read().decode()
        except Exception as ex:
            if num_retries <= 0:
                raise ex
            num_retries -= 1
            logger.debug(f"Request to {url} failed. Retrying...")
            time.sleep(backoff)
            backoff *= 2


@cache
def url_exists(url: str) -> bool:
    """Return True if the URL ``url`` exists, using the GITHUB_ACCESS_TOKEN for authentication if provided."""
    try:
        request = urllib.request.Request(url)  # type: ignore
        gh_token = os.getenv("GITHUB_ACCESS_TOKEN")
        if gh_token is not None:
            request.add_header("Authorization", f"token {gh_token}")
        with urllib.request.urlopen(request) as _:  # type: ignore
            return True
    except urllib.error.HTTPError:  # type: ignore
        return False


def parse_int_list(s: str) -> List[int]:
    assert s.startswith("[") and s.endswith("]")
    return [int(_) for _ in s[1:-1].split(",") if _ != ""]


def parse_str_list(s: str) -> List[str]:
    assert s.startswith("[") and s.endswith("]")
    return [_.strip()[1:-1] for _ in s[1:-1].split(",") if _ != ""]


@cache
def is_git_repo(path: Path) -> bool:
    """Check if ``path`` is a Git repo."""
    with working_directory(path):
        return (
            os.system("git rev-parse --is-inside-work-tree 1>/dev/null 2>/dev/null")
            == 0
        )


def _from_lean_path(root_dir: Path, path: Path, repo, ext: str) -> Path:
    assert path.suffix == ".lean"
    if path.is_absolute():
        path = path.relative_to(root_dir)

    assert root_dir.name != "lean4"
    if path.is_relative_to(LEAN4_PACKAGES_DIR / "lean4/src/lean/lake"):
        # E.g., "lake-packages/lean4/src/lean/lake/Lake/CLI/Error.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR / "lean4/src/lean/lake")
        return LEAN4_PACKAGES_DIR / "lean4/lib/lean" / p.with_suffix(ext)
    elif path.is_relative_to(LEAN4_PACKAGES_DIR / "lean4/src"):
        # E.g., "lake-packages/lean4/src/lean/Init.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR / "lean4/src").with_suffix(ext)
        return LEAN4_PACKAGES_DIR / "lean4/lib" / p
    elif path.is_relative_to(LEAN4_PACKAGES_DIR):
        # E.g., "lake-packages/std/Std.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR).with_suffix(ext)
        repo_name = p.parts[0]
        return (
            LEAN4_PACKAGES_DIR
            / repo_name
            / LEAN4_BUILD_DIR
            / "ir"
            / p.relative_to(repo_name)
        )
    else:
        # E.g., "Mathlib/LinearAlgebra/Basics.lean"
        return LEAN4_BUILD_DIR / "ir" / path.with_suffix(ext)


def to_xml_path(root_dir: Path, path: Path, repo) -> Path:
    return _from_lean_path(root_dir, path, repo, ext=".trace.xml")


def to_dep_path(root_dir: Path, path: Path, repo) -> Path:
    return _from_lean_path(root_dir, path, repo, ext=".dep_paths")


def to_json_path(root_dir: Path, path: Path, repo) -> Path:
    return _from_lean_path(root_dir, path, repo, ext=".ast.json")


def to_lean_path(root_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        path = path.relative_to(root_dir)

    if path.suffix in (".xml", ".json"):
        path = path.with_suffix("").with_suffix(".lean")
    else:
        assert path.suffix == ".dep_paths"
        path = path.with_suffix(".lean")

    assert root_dir.name != "lean4"
    if path == LEAN4_PACKAGES_DIR / "lean4/lib/lean/Lake.lean":
        return LEAN4_PACKAGES_DIR / "lean4/src/lean/lake/Lake.lean"
    elif path == LEAN4_PACKAGES_DIR / "lean4/lib/lean/LakeMain.lean":
        return LEAN4_PACKAGES_DIR / "lean4/src/lean/lake/LakeMain.lean"
    elif path.is_relative_to(LEAN4_PACKAGES_DIR / "lean4/lib/lean/Lake"):
        # E.g., "lake-packages/lean4/lib/lean/Lake/Util/List.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR / "lean4/lib/lean/Lake")
        return LEAN4_PACKAGES_DIR / "lean4/src/lean/lake/Lake" / p
    elif path.is_relative_to(LEAN4_PACKAGES_DIR / "lean4/lib"):
        # E.g., "lake-packages/lean4/lib/lean/Init.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR / "lean4/lib")
        return LEAN4_PACKAGES_DIR / "lean4/src" / p
    elif path.is_relative_to(LEAN4_PACKAGES_DIR):
        # E.g., "lake-packages/std/build/ir/Std.lean"
        p = path.relative_to(LEAN4_PACKAGES_DIR)
        repo_name = p.parts[0]
        return (
            LEAN4_PACKAGES_DIR
            / repo_name
            / p.relative_to(Path(repo_name) / LEAN4_BUILD_DIR / "ir")
        )
    else:
        # E.g., ".lake/build/ir/Mathlib/LinearAlgebra/Basics.lean" or "build/ir/Mathlib/LinearAlgebra/Basics.lean"
        assert path.is_relative_to(LEAN4_BUILD_DIR / "ir"), path
        return path.relative_to(LEAN4_BUILD_DIR / "ir")
