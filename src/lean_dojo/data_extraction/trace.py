"""This module provides the main interfaces for tracing Lean repos, i.e., extracting data from them.
To estimate the time for tracing a repo, a good rule of thumb is 1.5x the time for compiling the repo using :code:`leanpkg build`.
A repo has to be traced only once, and the traced repo will be stored in a cache for fast access in the future.
"""
import shutil
from pathlib import Path
from loguru import logger
from typing import Union, Optional
from subprocess import CalledProcessError

from ..utils import (
    working_directory,
    parse_lean3_version,
    execute,
)
from .cache import cache
from ..constants import (
    NUM_PROCS,
    LEAN3_URL,
    MIN_LEAN3_VERSION,
)
from .lean import LeanGitRepo
from .traced_data import TracedRepo
from ..container import create_mounts, get_container, NativeContainer


MODIFIED_LEAN3_PATCH_PATH = Path(__file__).with_name(
    "0001-Modify-Lean-for-proof-recording.patch"
)
LEAN3_BUILD_SCRIPT_PATH = Path(__file__).with_name("build_lean3_repo.py")
LEAN4_BUILD_SCRIPT_PATH = Path(__file__).with_name("build_lean4_repo.py")
LEAN4_DATA_EXTRACTOR_PATH = Path(__file__).with_name("ExtractData.lean")


def _modify_lean3(version: str) -> None:
    """Modify Lean 3 by applying the modification patch."""
    logger.debug(f"Modifying Lean {version}")
    execute(f"git clone {LEAN3_URL}", capture_output=True)

    with working_directory("lean"):
        execute(f"git checkout {version}", capture_output=True)
        if version.startswith("v") and parse_lean3_version(
            version
        ) < parse_lean3_version(MIN_LEAN3_VERSION):
            logger.warning(
                f"Lean {version} is too outdated. We support Lean >= {MIN_LEAN3_VERSION}. Proceed with caution."
            )
        logger.debug(f"Applying the modification patch")
        try:
            execute(f"git apply {MODIFIED_LEAN3_PATCH_PATH}", capture_output=True)
        except CalledProcessError as ex:
            logger.error(
                f"`git apply` failed, probably because Lean {version} is too outdated"
            )
            raise ex


def _trace(repo: LeanGitRepo) -> None:
    assert (
        repo.exists()
    ), f"The {repo} does not exist. Please check the URL `{repo.commit_url}`."
    if repo.uses_lean3:
        _trace_lean3(repo)
    else:
        assert repo.uses_lean4
        _trace_lean4(repo)


def _trace_lean3(repo: LeanGitRepo) -> None:
    # Trace `repo` in the current working directory.
    if repo.is_lean:
        _modify_lean3(repo.lean_version)
    else:
        repo.clone_and_checkout()
        with working_directory(Path(repo.name) / "_target/deps"):
            _modify_lean3(repo.lean_version)

    logger.debug(f"Tracing {repo}")
    container = get_container()
    if isinstance(container, NativeContainer):
        logger.warning(
            "Docker is strongly recommended when using LeanDojo with Lean 3. See https://leandojo.readthedocs.io/en/latest/user-guide.html#advanced-running-within-docker."
        )
    mts = {
        Path.cwd() / repo.name: f"/workspace/{repo.name}",
        LEAN3_BUILD_SCRIPT_PATH: f"/workspace/{LEAN3_BUILD_SCRIPT_PATH.name}",
    }
    try:
        container.run(
            f"python3 build_lean3_repo.py {repo.name}",
            create_mounts(mts),
            {"NUM_PROCS": NUM_PROCS},
            as_current_user=True,
            work_dir="/workspace",
        )
    except CalledProcessError as ex:
        if isinstance(container, NativeContainer):
            logger.error(
                "Failed to build the modified Lean 3 without Docker. See https://leandojo.readthedocs.io/en/latest/user-guide.html#advanced-running-within-docker."
            )
        raise ex


def _trace_lean4(repo: LeanGitRepo) -> None:
    # Trace `repo` in the current working directory.
    assert not repo.is_lean4, "Cannot trace the Lean 4 repo itself."
    repo.clone_and_checkout()

    logger.debug(f"Tracing {repo}")
    container = get_container()
    mts = {
        Path.cwd() / repo.name: f"/workspace/{repo.name}",
        LEAN4_BUILD_SCRIPT_PATH: f"/workspace/{LEAN4_BUILD_SCRIPT_PATH.name}",
        LEAN4_DATA_EXTRACTOR_PATH: f"/workspace/{repo.name}/{LEAN4_DATA_EXTRACTOR_PATH.name}",
    }
    try:
        container.run(
            f"python3 build_lean4_repo.py {repo.name}",
            create_mounts(mts),
            {"NUM_PROCS": NUM_PROCS},
            as_current_user=True,
            work_dir="/workspace",
        )
    except CalledProcessError as ex:
        if repo.is_lean4 and isinstance(container, NativeContainer):
            logger.error(
                "Failed to build Lean 4 without Docker. See https://leandojo.readthedocs.io/en/latest/user-guide.html#advanced-running-within-docker."
            )
        raise ex


def is_available_in_cache(repo: LeanGitRepo) -> bool:
    """Check if ``repo`` has a traced repo available in the cache (including the remote cache)."""
    return cache.get(repo.url, repo.commit) is not None


def get_traced_repo_path(repo: LeanGitRepo) -> Path:
    """Return the path of a traced repo in the cache.

    The function will trace a repo if it is not available in the cache. See :ref:`caching` for details.

    Args:
        repo (LeanGitRepo): The Lean repo to trace.

    Returns:
        Path: The path of the traced repo in the cache, e.g. :file:`/home/kaiyu/.cache/lean_dojo/leanprover-community-mathlib-2196ab363eb097c008d4497125e0dde23fb36db2`
    """
    path = cache.get(repo.url, repo.commit)
    if path is None:
        logger.info(f"Tracing {repo}")
        with working_directory() as tmp_dir:
            logger.debug(f"Working in the temporary directory {tmp_dir}")
            _trace(repo)
            traced_repo = TracedRepo.from_traced_files(tmp_dir / repo.name)
            traced_repo.save_to_disk()
            path = cache.store(tmp_dir)
    else:
        logger.debug("The traced repo is available in the cache.")
    return path


def trace(repo: LeanGitRepo, dst_dir: Optional[Union[str, Path]] = None) -> TracedRepo:
    """Trace a repo (and its dependencies), saving the results to ``dst_dir``.

    The function only traces the repo when it's not available in the cache. Otherwise,
    it directly copies the traced repo from the cache to ``dst_dir``. See :ref:`caching` for details.

    Args:
        repo (LeanGitRepo): The Lean repo to trace.
        dst_dir (Union[str, Path]): The directory for saving the traced repo. If None, the traced repo is only saved in the cahe.

    Returns:
        TracedRepo: A :class:`TracedRepo` object corresponding to the files at ``dst_dir``.
    """
    if dst_dir is not None:
        dst_dir = Path(dst_dir)
        assert (
            not dst_dir.exists()
        ), f"The destination directory {dst_dir} already exists."

    cached_path = get_traced_repo_path(repo)
    logger.info(f"Loading the traced repo from {cached_path}")
    traced_repo = TracedRepo.load_from_disk(cached_path)
    traced_repo.check_sanity()

    if dst_dir is not None:
        dst_dir.mkdir(parents=True)
        shutil.copytree(cached_path, dst_dir / cached_path.name)

    return traced_repo
