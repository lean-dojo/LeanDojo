"""
Utility functions for filesystem operations.
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from loguru import logger

from lean_dojo_v2.utils.constants import CHECKPOINT_DIR, RAID_DIR, TMP_DIR


def remove_dir(dir_path: Union[str, Path]) -> None:
    """
    Safely removes a directory path if it exists, with retries.

    This function attempts to delete a directory and retries multiple times if permission errors occur,
    which can happen if files are temporarily locked by another process or if the directory
    contains read-only files.

    Args:
        dir_path (Union[str, Path]): Path to the directory to be removed.

    Raises:
        PermissionError: If the directory cannot be removed after multiple retries
                         due to permission issues.

    Example:
        ```
        remove_dir('/path/to/directory')
        remove_dir(Path('/path/to/directory'))
        ```
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if dir_path.exists():
        logger.warning(f"{dir_path} already exists. Removing it now.")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                shutil.rmtree(dir_path)
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait a bit before retrying
                else:
                    logger.error(
                        f"Failed to remove {dir_path} after {max_retries} attempts: {e}"
                    )
                    raise


def find_latest_checkpoint() -> str:
    """
    Finds the most recent checkpoint file by modification time.

    Returns:
        str: Path to the latest checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint files are found.

    Example:
        ```
        checkpoint_path = find_latest_checkpoint()
        print(f"Using checkpoint: {checkpoint_path}")
        ```
    """
    checkpoint_dir = os.path.join(RAID_DIR, CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".ckpt")
    ]

    if not all_checkpoints:
        download_checkpoints()
        all_checkpoints = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".ckpt")
        ]

    if not all_checkpoints:
        raise FileNotFoundError("No checkpoints found.")

    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    return latest_checkpoint


@contextmanager
def working_directory(
    path: Optional[Union[str, Path]] = None,
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


def is_git_repo(path: Path) -> bool:
    """Check if ``path`` is a Git repo."""
    with working_directory(path):
        return (
            os.system("git rev-parse --is-inside-work-tree 1>/dev/null 2>/dev/null")
            == 0
        )


def download_checkpoints() -> None:
    """
    Download required checkpoints for LeanAgent based on README instructions.

    Downloads the following checkpoints:
    1. ReProver's Tactic Generator
    2. ReProver's Starting Retriever
    3. Latest LeanAgent checkpoint from the paper
    """
    logger.info("Downloading required checkpoints for LeanAgent...")

    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.join(RAID_DIR, CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if gdown is available
    try:
        subprocess.run(["gdown", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(
            "gdown is not installed. Please install it with: pip install gdown"
        )
        logger.info(
            "You can also manually download the checkpoints from the URLs in the README"
        )
        return

    # Define checkpoint URLs from README
    checkpoints = {
        "reprover_tactic_generator": {
            "url": "https://drive.google.com/uc?id=11DXxixg6S4-hUA-u-78geOxEB7J7rCoX",
            "dest": os.path.join(RAID_DIR, "model_lightning.ckpt"),
        },
        "reprover_starting_retriever": {
            "url": "https://drive.google.com/uc?id=1aRd1jQPu_TX15Ib5htzn3wZqHYowl3ax",
            "dest": os.path.join(
                checkpoint_dir, "mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
            ),
        },
        "leanagent_latest": {
            "url": "https://drive.google.com/uc?id=1plkC7Y5n0OVCJ0Ad6pH8_mwbALKYY_FY",
            "dest": os.path.join(checkpoint_dir, "leanagent.ckpt"),
        },
    }

    for name, info in checkpoints.items():
        url = info["url"]
        dest = info["dest"]

        # Skip if file already exists
        if os.path.exists(dest):
            logger.info(f"Checkpoint {name} already exists at {dest}")
            continue

        logger.info(f"Downloading {name} from {url}")
        try:
            # Download using gdown
            subprocess.run(["gdown", "--output", dest, url], check=True)
            logger.info(f"Successfully downloaded {name} to {dest}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {name}: {e}")
            logger.info(f"You can manually download from: {url}")

    logger.info(
        "Checkpoint download completed. Check the checkpoint directory for downloaded files."
    )
