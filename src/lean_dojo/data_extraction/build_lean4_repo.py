"""Build Lean 4 projects in Docker.

Only this file runs in Docker. So it must be self-contained.
"""
import os
import re
import shutil
import argparse
import itertools
import subprocess
from tqdm import tqdm
from loguru import logger
from time import sleep, monotonic
from pathlib import Path, PurePath
from multiprocessing import Process
from contextlib import contextmanager
from typing import Union, List, Optional, Generator


def run_cmd(cmd: Union[str, List[str]], capture_output: bool = False) -> Optional[str]:
    """Run a shell command.

    Args:
        cmd (Union[str, List[str]]): A command or a list of commands.
    """
    if isinstance(cmd, list):
        cmd = " && ".join(cmd)
    res = subprocess.run(cmd, shell=True, capture_output=capture_output, check=True)
    if capture_output:
        return res.stdout.decode()
    else:
        return None


def record_paths(dir: Path, root: Path, lean_bin: Path) -> None:
    """Run ``lean --deps`` for all Lean files in ``dir`` to record its dependencies.

    Args:
        dir (Path): The directory containing Lean files.
    """
    dir = Path(dir)

    for p in dir.glob("**/*.lean"):
        with p.with_suffix(".dep_paths").open("wt") as oup:
            for line in run_cmd(
                f"{lean_bin} --deps {p}", capture_output=True
            ).splitlines():
                olean_path = PurePath(line.strip())
                assert olean_path.suffix == ".olean"
                lean_path = olean_path.relative_to(root).with_suffix(".lean")
                oup.write(str(lean_path) + "\n")


def remove_files(dir: Path, suffix: str) -> None:
    """Remove all files in ``dir`` that end with ``suffix``."""
    for p in Path(dir).glob(f"**/*{suffix}"):
        p.unlink()


_PROGRESSBAR_UPDATE_INTERNAL = 5


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
def launch_progressbar(paths: List[Union[str, Path]]) -> Generator[None, None, None]:
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
    output = run_cmd("lean --version", capture_output=True).strip()
    m = re.match(r"Lean \(version (?P<version>\S+?),", output)
    return m["version"]


def check_files(packages_path: str, no_deps: bool) -> None:
    """Check if all *.lean files have been processed to produce *.ast.json and *.dep_paths files."""
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
        Path(str(p.with_suffix("")).replace("/build/lib/", "/build/ir/"))
        for p in cwd.glob("**/build/lib/**/*.olean")
        if not no_deps or not p.is_relative_to(packages_path)
    }
    assert len(jsons) <= len(oleans) and len(deps) <= len(oleans)
    missing_jsons = {p.with_suffix(".ast.json") for p in oleans - jsons}
    missing_deps = {p.with_suffix(".dep_paths") for p in oleans - deps}
    if len(missing_jsons) > 0 or len(missing_deps) > 0:
        for p in missing_jsons.union(missing_deps):
            logger.warning(f"Missing {p}")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_name")
    parser.add_argument("--no-deps", action="store_true")
    args = parser.parse_args()

    num_procs = int(os.environ["NUM_PROCS"])
    repo_name = args.repo_name
    os.chdir(repo_name)

    # Build the repo using lake.
    logger.info(f"Building {repo_name}")
    if args.no_deps:
        # The additional *.olean files wouldn't matter.
        try:
            run_cmd("lake exe cache get")
        except subprocess.CalledProcessError:
            pass
    run_cmd("lake build")

    # Copy the Lean 4 stdlib into the path of packages.
    lean_prefix = run_cmd(f"lean --print-prefix", capture_output=True).strip()
    if is_new_version(get_lean_version()):
        packages_path = ".lake/packages"
        build_path = ".lake/build"
    else:
        packages_path = "lake-packages"
        build_path = "build"
    shutil.copytree(lean_prefix, f"{packages_path}/lean4")

    # Run ExtractData.lean to extract ASTs, tactic states, and premise information.
    dirs_to_monitor = [build_path]
    if not args.no_deps:
        dirs_to_monitor.append(packages_path)
    logger.info(f"Tracing {repo_name}")
    with launch_progressbar(dirs_to_monitor):
        cmd = f"lake env lean --threads {num_procs} --run ExtractData.lean"
        if args.no_deps:
            cmd += " nodeps"
        run_cmd(cmd, capture_output=True)

    check_files(packages_path, args.no_deps)


if __name__ == "__main__":
    main()
