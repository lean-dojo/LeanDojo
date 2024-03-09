"""This module provides the main interfaces for tracing Lean repos, i.e., extracting data from them.
To estimate the time for tracing a repo, a good rule of thumb is 1.5x the time for compiling the repo using :code:`leanpkg build`.
A repo has to be traced only once, and the traced repo will be stored in a cache for fast access in the future.
"""

import shutil
from pathlib import Path
from loguru import logger
from typing import Union, Optional
from subprocess import CalledProcessError

from .cache import cache
from .lean import LeanGitRepo
from ..constants import NUM_PROCS
from .traced_data import TracedRepo
from ..utils import working_directory
from ..container import create_mounts, get_container, NativeContainer


LEAN4_BUILD_SCRIPT_PATH = Path(__file__).with_name("build_lean4_repo.py")
LEAN4_DATA_EXTRACTOR_PATH = Path(__file__).with_name("ExtractData.lean")


def _trace(repo: LeanGitRepo, build_deps: bool) -> None:
    assert (
        repo.exists()
    ), f"The {repo} does not exist. Please check the URL `{repo.commit_url}`."

    # Trace `repo` in the current working directory.
    assert not repo.is_lean4, "Cannot trace Lean 4 itself."
    repo.clone_and_checkout()

    logger.debug(f"Tracing {repo}")
    container = get_container()
    mts = {
        Path.cwd() / repo.name: f"/workspace/{repo.name}",
        LEAN4_BUILD_SCRIPT_PATH: f"/workspace/{LEAN4_BUILD_SCRIPT_PATH.name}",
        LEAN4_DATA_EXTRACTOR_PATH: f"/workspace/{repo.name}/{LEAN4_DATA_EXTRACTOR_PATH.name}",
    }
    cmd = f"python build_lean4_repo.py {repo.name}"
    if not build_deps:
        cmd += " --no-deps"

    try:
        container.run(
            cmd,
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


def get_traced_repo_path(repo: LeanGitRepo, build_deps: bool = True) -> Path:
    """Return the path of a traced repo in the cache.

    The function will trace a repo if it is not available in the cache. See :ref:`caching` for details.

    Args:
        repo (LeanGitRepo): The Lean repo to trace.
        build_deps (bool): Whether to build the dependencies of ``repo``. Defaults to True.

    Returns:
        Path: The path of the traced repo in the cache, e.g. :file:`/home/kaiyu/.cache/lean_dojo/leanprover-community-mathlib-2196ab363eb097c008d4497125e0dde23fb36db2`
    """
    path = cache.get(repo.url, repo.commit)
    if path is None:
        logger.info(f"Tracing {repo}")
        with working_directory() as tmp_dir:
            logger.debug(f"Working in the temporary directory {tmp_dir}")
            _trace(repo, build_deps)
            traced_repo = TracedRepo.from_traced_files(tmp_dir / repo.name, build_deps)
            traced_repo.save_to_disk()
            path = cache.store(tmp_dir / repo.name)
    else:
        logger.debug("The traced repo is available in the cache.")
    return path


def trace(
    repo: LeanGitRepo,
    dst_dir: Optional[Union[str, Path]] = None,
    build_deps: bool = True,
) -> TracedRepo:
    """Trace a repo (and its dependencies), saving the results to ``dst_dir``.

    The function only traces the repo when it's not available in the cache. Otherwise,
    it directly copies the traced repo from the cache to ``dst_dir``. See :ref:`caching` for details.

    Args:
        repo (LeanGitRepo): The Lean repo to trace.
        dst_dir (Union[str, Path]): The directory for saving the traced repo. If None, the traced repo is only saved in the cahe.
        build_deps (bool): Whether to build the dependencies of ``repo``. Defaults to True.

    Returns:
        TracedRepo: A :class:`TracedRepo` object corresponding to the files at ``dst_dir``.
    """
    if dst_dir is not None:
        dst_dir = Path(dst_dir)
        assert (
            not dst_dir.exists()
        ), f"The destination directory {dst_dir} already exists."

    cached_path = get_traced_repo_path(repo, build_deps)
    logger.info(f"Loading the traced repo from {cached_path}")
    traced_repo = TracedRepo.load_from_disk(cached_path, build_deps)
    traced_repo.check_sanity()

    if dst_dir is not None:
        dst_dir.mkdir(parents=True)
        shutil.copytree(cached_path, dst_dir / cached_path.name)

    return traced_repo
