"""Build Lean 3 projects in Docker.

Only this file runs in Docker. So it must be self-contained.
"""
import os
import sys
import shutil
import itertools
import subprocess
from glob import glob
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


def main() -> None:
    num_procs = int(os.environ["NUM_PROCS"])
    repo_name = sys.argv[1]
    os.chdir(repo_name)

    logger.info(f"Building {repo_name}")
    if repo_name == "lean4":
        # Build Lean 4 from source.
        run_cmd(
            [
                f"mkdir -p build/release",
                f"cd build/release",
                "cmake ../..",
                f"make -j{num_procs}",
                "cd ../..",
                "rm build/release/stage1/src/lean",  # Remove symbolic link.
                "rm build/release/stage0/src/lean",  # Remove symbolic link.
                "mkdir lib && cp -r build/release/stage1/lib/lean lib/lean",
                "mv src src_tmp && mkdir src && mv src_tmp src/lean",
            ]
        )

        logger.info(f"Tracing {repo_name}")
        with launch_progressbar(["lib"]):
            run_cmd(
                f"./build/release/stage1/bin/lean --threads {num_procs} --run ExtractData.lean",
                capture_output=True,
            )

    else:
        # Build the repo using lake.
        run_cmd(f"lake build")

        # Copy the Lean 4 stdlib into lake-packages.
        lean_prefix = run_cmd(f"lean --print-prefix", capture_output=True).strip()
        shutil.copytree(lean_prefix, "lake-packages/lean4")

        # Run ExtractData.lean to extract ASTs and tactic states.
        logger.info(f"Tracing {repo_name}")
        with launch_progressbar(["build", "lake-packages"]):
            run_cmd(
                f"lake env lean --threads {num_procs} --run ExtractData.lean",
                capture_output=True,
            )

        num_json = len(glob("build/ir/**/*.ast.json", recursive=True))
        num_dep = len(glob("build/ir/**/*.dep_paths", recursive=True))
        num_c = len(glob("build/ir/**/*.c", recursive=True))
        assert num_json == num_dep == num_c, f"{num_json} {num_dep} {num_c}"


if __name__ == "__main__":
    main()
