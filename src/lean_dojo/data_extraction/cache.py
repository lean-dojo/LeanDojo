"""Cache manager of traced repos.
"""

import os
import shutil
import tarfile
from pathlib import Path
from loguru import logger
from filelock import FileLock
from dataclasses import dataclass, field
from typing import Optional, Tuple, Generator

from ..utils import (
    execute,
    url_exists,
    get_repo_info,
    report_critical_failure,
)
from ..constants import (
    CACHE_DIR,
    DISABLE_REMOTE_CACHE,
    REMOTE_CACHE_URL,
)


def _split_git_url(url: str) -> Tuple[str, str]:
    """Split a Git URL into user name and repo name."""
    if url.endswith("/"):
        url = url[:-1]
        assert not url.endswith("/"), f"Unexpected URL: {url}"
    fields = url.split("/")
    user_name = fields[-2]
    repo_name = fields[-1]
    return user_name, repo_name


def _format_dirname(url: str, commit: str) -> str:
    user_name, repo_name = _split_git_url(url)
    return f"{user_name}-{repo_name}-{commit}"


_CACHE_CORRPUTION_MSG = "The cache may have been corrputed!"


@dataclass(frozen=True, eq=False)
class Cache:
    """Cache manager."""

    cache_dir: Path
    lock: FileLock = field(init=False, repr=False)

    def __iter__(self) -> Generator[Path, None, None]:
        """Iterate over all traced repos in the cache."""
        yield from self.cache_dir.glob("*")

    def __post_init__(self):
        if not os.path.exists(self.cache_dir):
            self.cache_dir.mkdir(parents=True)
        lock_path = self.cache_dir.with_suffix(".lock")
        object.__setattr__(self, "lock", FileLock(lock_path))

    def get(self, url: str, commit: str) -> Optional[Path]:
        """Get the path of a traced repo with URL ``url`` and commit hash ``commit``. Return None if no such repo can be found."""
        _, repo_name = _split_git_url(url)
        dirname = _format_dirname(url, commit)
        dirpath = self.cache_dir / dirname

        with self.lock:
            if dirpath.exists():
                assert (dirpath / repo_name).exists()
                return dirpath / repo_name

            elif not DISABLE_REMOTE_CACHE:
                url = os.path.join(REMOTE_CACHE_URL, f"{dirname}.tar.gz")
                if not url_exists(url):
                    return None
                logger.info(
                    f"Downloading the traced repo from the remote cache. Set the environment variable `DISABLE_REMOTE_CACHE` if you want to trace the repo locally."
                )
                execute(f"wget {url} -O {dirpath}.tar.gz")

                with report_critical_failure(_CACHE_CORRPUTION_MSG):
                    with tarfile.open(f"{dirpath}.tar.gz") as tar:
                        tar.extractall(self.cache_dir)
                    os.remove(f"{dirpath}.tar.gz")
                    assert (dirpath / repo_name).exists()

                return dirpath / repo_name

            else:
                return None

    def store(self, src: Path) -> Path:
        """Store a traced repo at path ``src``. Return its path in the cache."""
        url, commit = get_repo_info(src)
        dirpath = self.cache_dir / _format_dirname(url, commit)
        _, repo_name = _split_git_url(url)
        if not dirpath.exists():
            with self.lock:
                with report_critical_failure(_CACHE_CORRPUTION_MSG):
                    shutil.copytree(src, dirpath / repo_name)
        return dirpath / repo_name


cache = Cache(CACHE_DIR)
"""A global :class:`Cache` object managing LeanDojo's caching of traced repos (see :ref:`caching`).
"""
