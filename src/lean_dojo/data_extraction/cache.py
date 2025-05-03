"""Cache manager of traced repos."""

import os
import shutil
import tarfile
from pathlib import Path
from loguru import logger
from filelock import FileLock
from typing import Optional, Generator
from dataclasses import dataclass, field

from ..utils import (
    execute,
    url_exists,
    report_critical_failure,
)
from ..constants import (
    CACHE_DIR,
    DISABLE_REMOTE_CACHE,
    REMOTE_CACHE_URL,
)


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

    def get(self, rel_cache_dir: Path) -> Optional[Path]:
        """Get the cache repo at ``CACHE_DIR / rel_cache_dir`` from the cache.

        Args:
            rel_cache_dir (Path): The relative path of the stored repo in the cache.
        """
        dirname = rel_cache_dir.parent
        dirpath = self.cache_dir / dirname
        cache_path = self.cache_dir / rel_cache_dir

        with self.lock:
            if dirpath.exists():
                assert cache_path.exists()
                return cache_path

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
                    assert (cache_path).exists()

                return cache_path

            else:
                return None

    def store(self, src: Path, rel_cache_dir: Path) -> Path:
        """Store a repo at path ``src``. Return its path in the cache.

        Args:
            src (Path): Path to the repo.
            rel_cache_dir (Path): The relative path of the stored repo in the cache.
        """
        dirpath = self.cache_dir / rel_cache_dir.parent
        cache_path = self.cache_dir / rel_cache_dir
        if not dirpath.exists():
            with self.lock:
                with report_critical_failure(_CACHE_CORRPUTION_MSG):
                    shutil.copytree(src, cache_path)
        return cache_path


cache = Cache(CACHE_DIR)
"""A global :class:`Cache` object managing LeanDojo's caching of traced repos (see :ref:`caching`).
"""
