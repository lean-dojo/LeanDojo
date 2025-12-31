"""This module provides the main interfaces for tracing Lean repos, i.e., extracting data from them.
To estimate the time for tracing a repo, a good rule of thumb is 1.5x the time for compiling the repo using :code:`leanpkg build`.
A repo has to be traced only once, and the traced repo will be stored in a cache for fast access in the future.
"""

import itertools
import os
import re
import shutil
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path
from time import monotonic, sleep
from typing import Generator, List, Optional, Union

from loguru import logger
from tqdm import tqdm

from lean_dojo_v2.utils.common import execute
from lean_dojo_v2.utils.constants import NUM_PROCS
from lean_dojo_v2.utils.filesystem import working_directory

from .cache import cache
from .lean import LeanGitRepo
from .traced_data import TracedRepo

LEAN4_DATA_EXTRACTOR_PATH = Path(__file__).with_name("ExtractData.lean")

_PROGRESSBAR_UPDATE_INTERNAL = 5


def _modify_dependency_files(packages_path: Path) -> None:
    """Modify dependency files to replace 'import all' with 'public import all'."""
    logger.debug(
        "Modifying dependency files to replace 'import all' with 'public import all'"
    )
    for lean_file in packages_path.rglob("*.lean"):
        with open(lean_file, "r", encoding="utf-8") as f:
            content = f.read()

        modified_content = re.sub(
            r"^(\s*)import all", r"\1public import all", content, flags=re.MULTILINE
        )

        if modified_content != content:
            with open(lean_file, "w", encoding="utf-8") as f:
                f.write(modified_content)


def _monitor(paths: List[Path], num_total: int) -> None:
    with tqdm(total=num_total) as pbar:
        while True:
            time_start = monotonic()
            try:
                num_done = len(
                    list(
                        itertools.chain.from_iterable(
                            p.glob(f"**/*.ast.json") for p in paths
                        )
                    )
                )
            except Exception:
                continue
            time_elapsed = monotonic() - time_start
            if time_elapsed < _PROGRESSBAR_UPDATE_INTERNAL:
                sleep(_PROGRESSBAR_UPDATE_INTERNAL - time_elapsed)
            pbar.update(num_done - pbar.n)
            if num_done >= num_total:
                break
    print("")


@contextmanager
def launch_progressbar(paths: List[Path]) -> Generator[None, None, None]:
    """Launch an async progressbar to monitor the progress of tracing the repo."""
    paths = [Path(p) for p in paths]
    olean_files = list(
        itertools.chain.from_iterable(p.glob("**/*.olean") for p in paths)
    )
    num_total = len(olean_files)
    p = Process(target=_monitor, args=(paths, num_total), daemon=True)
    p.start()
    yield
    p.kill()


def get_lean_version() -> str:
    """Get the version of Lean."""
    output = execute("lean --version", capture_output=True)[0].strip()
    m = re.match(r"Lean \(version (?P<version>\S+?),", output)
    return m["version"]  # type: ignore


def is_new_version(v: str) -> bool:
    """Check if ``v`` is at least `4.3.0-rc2`."""
    major, minor, patch = [int(_) for _ in v.split("-")[0].split(".")]
    if major < 4 or (major == 4 and minor < 3):
        return False
    if (
        major > 4
        or (major == 4 and minor > 3)
        or (major == 4 and minor == 3 and patch > 0)
    ):
        return True
    assert major == 4 and minor == 3 and patch == 0
    if "4.3.0-rc" in v:
        rc = int(v.split("-")[1][2:])
        return rc >= 2
    else:
        return True


def check_files(packages_path: Path, no_deps: bool) -> None:
    """Check if all :file:`*.lean` files have been processed to produce :file:`*.ast.json` and :file:`*.dep_paths` files."""
    cwd = Path.cwd()
    packages_path = cwd / packages_path
    jsons = {
        p.with_suffix("").with_suffix("")
        for p in cwd.glob("**/build/ir/**/*.ast.json")
        if not no_deps or not p.is_relative_to(packages_path)
    }
    deps = {
        p.with_suffix("")
        for p in cwd.glob("**/build/ir/**/*.dep_paths")
        if not no_deps or not p.is_relative_to(packages_path)
    }
    oleans = {
        Path(str(p.with_suffix("")).replace("/build/lib/lean/", "/build/ir/"))
        for p in cwd.glob("**/build/lib/lean/**/*.olean")
        if not no_deps or not p.is_relative_to(packages_path)
    }
    assert len(jsons) <= len(oleans) and len(deps) <= len(oleans)
    missing_jsons = {p.with_suffix(".ast.json") for p in oleans - jsons}
    missing_deps = {p.with_suffix(".dep_paths") for p in oleans - deps}
    if len(missing_jsons) > 0 or len(missing_deps) > 0:
        for p in missing_jsons.union(missing_deps):
            logger.warning(f"Missing {p}")


def _trace(repo: LeanGitRepo, build_deps: bool) -> None:
    assert (
        repo.exists()
    ), f"The {repo} does not exist. Please check the URL `{repo.commit_url}`."

    # Trace `repo` in the current working directory.
    assert not repo.is_lean4, "Cannot trace Lean 4 itself."
    if not os.path.exists(repo.name):
        repo.clone_and_checkout()
    logger.debug(f"Tracing {repo}")

    with working_directory(repo.name):
        # Build the repo using lake.
        execute("lake build")

        # Copy the Lean 4 stdlib into the path of packages.
        lean_prefix = execute(f"lean --print-prefix", capture_output=True)[0].strip()
        if is_new_version(get_lean_version()):
            packages_path = Path(".lake/packages")
            build_path = Path(".lake/build")
        else:
            packages_path = Path("lake-packages")
            build_path = Path("build")

        shutil.copytree(
            Path(lean_prefix), str(packages_path / "lean4"), dirs_exist_ok=True
        )

        # Modify dependency files to replace 'import all' with 'public import all'
        if build_deps:
            _modify_dependency_files(packages_path)

        # Run ExtractData.lean to extract ASTs, tactic states, and premise information.
        shutil.copyfile(LEAN4_DATA_EXTRACTOR_PATH, LEAN4_DATA_EXTRACTOR_PATH.name)

        dirs_to_monitor = [build_path]
        if build_deps:
            dirs_to_monitor.append(packages_path)
        with launch_progressbar(dirs_to_monitor):
            cmd = f"lake env lean --threads {NUM_PROCS} --run ExtractData.lean"
            if not build_deps:
                cmd += " noDeps"
            execute(cmd)

        check_files(packages_path, not build_deps)
        os.remove(LEAN4_DATA_EXTRACTOR_PATH.name)


def is_available_in_cache(repo: LeanGitRepo) -> bool:
    """Check if ``repo`` has a traced repo available in the cache (including the remote cache)."""
    rel_cache_dir = repo.get_cache_dirname() / repo.name
    return cache.get(rel_cache_dir) is not None


def get_traced_repo_path(repo: LeanGitRepo, build_deps: bool = True) -> Path:
    """Return the path of a traced repo in the cache.

    The function will trace a repo if it is not available in the cache. See :ref:`caching` for details.

    Args:
        repo (LeanGitRepo): The Lean repo to trace.
        build_deps (bool): Whether to build the dependencies of ``repo``. Defaults to True.

    Returns:
        Path: The path of the traced repo in the cache, e.g. :file:`/home/kaiyu/.cache/lean_dojo/leanprover-community-mathlib-2196ab363eb097c008d4497125e0dde23fb36db2`
    """
    rel_cache_dir = repo.get_cache_dirname() / (
        repo.name + ("_d" if build_deps else "")
    )

    path = cache.get(rel_cache_dir)

    if path is None and not build_deps:
        path = cache.get(rel_cache_dir.parent / (repo.name + "_d"))

    if path is None:
        logger.info(f"Tracing {repo}")
        with working_directory() as tmp_dir:
            logger.debug(f"Working in the temporary directory {tmp_dir}")
            _trace(repo, build_deps)
            src_dir = tmp_dir / repo.name
            traced_repo = TracedRepo.from_traced_files(src_dir, build_deps)
            traced_repo.save_to_disk()
            path = cache.store(src_dir, rel_cache_dir)
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
        shutil.copytree(cached_path, dst_dir / cached_path.name, dirs_exist_ok=True)

    return traced_repo
