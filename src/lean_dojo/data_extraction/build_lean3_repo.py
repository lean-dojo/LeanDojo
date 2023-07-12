"""Build Lean 3 projects in Docker.

Only this file runs in Docker. So it must be self-contained.
"""
import os
import sys
import toml
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
    lean_files = list(itertools.chain.from_iterable(p.glob("**/*.lean") for p in paths))
    num_total = len(lean_files)
    p = Process(target=_monitor, args=(paths, num_total), daemon=True)
    p.start()
    yield
    p.kill()


def main() -> None:
    num_procs = int(os.environ["NUM_PROCS"])
    repo_name = sys.argv[1]
    traced_repo_root = Path.cwd() / repo_name
    if repo_name == "lean":
        modified_lean_root = traced_repo_root
    else:
        modified_lean_root = traced_repo_root / "_target/deps/lean"
    modifed_lean_lib = modified_lean_root / "library"
    modifed_lean_bin = modified_lean_root / "bin/lean"
    modified_leanpkg_bin = modified_lean_root / "bin/leanpkg"

    # Build modified Lean without installing it.
    logger.info("Building modifed Lean")
    run_cmd(
        [
            f"mkdir -p {modified_lean_root}/build/release",
            f"cd {modified_lean_root}/build/release",
            "cmake ../../src",
            f"make -j{num_procs}",
            "cd ../../../..",
        ]
    )

    # Record the paths of modified Lean.
    os.chdir(repo_name)

    if repo_name == "lean":
        # Use modified Lean to trace it self.
        src_dir = "library"
    else:
        # Build the repo and all its dependencies.
        modifed_lean_bin = modified_lean_root / "bin/lean"
        record_paths(modifed_lean_lib, traced_repo_root, modifed_lean_bin)
        config = toml.load(open("leanpkg.toml"))
        src_dir = config["package"]["path"]
        run_cmd(f"{modified_leanpkg_bin} configure")

        path_file = Path("leanpkg.path")
        lines = [
            line.replace("builtin_path", "path _target/deps/lean/library")
            for line in path_file.open()
        ]
        with open(path_file, "w") as f:
            f.writelines(lines)

        for dep in Path("_target/deps").glob("*"):
            if dep.name == "lean":
                continue
            dep_config = toml.load((dep / "leanpkg.toml").open())
            dep_src_dir = dep / dep_config["package"]["path"]
            record_paths(dep_src_dir, traced_repo_root, modifed_lean_bin)

    logger.info(f"Tracing {repo_name}")
    remove_files(modifed_lean_lib, ".olean")
    modified_lean = f"{modifed_lean_bin} --ast --tsast --tspp --recursive --make --threads={num_procs}"
    io_path = modifed_lean_lib / "system/io.lean"
    run_cmd(f"{modified_lean} {io_path}", capture_output=True)
    with launch_progressbar(["_target/deps", src_dir]):
        try:
            run_cmd(f"{modified_lean} {src_dir}", capture_output=True)
        except subprocess.CalledProcessError as ex:
            logger.error(ex)
            logger.error("Please check if the repo can be built with `leanpkg build`.")
    record_paths(src_dir, traced_repo_root, modifed_lean_bin)
    os.chdir("..")


if __name__ == "__main__":
    main()
