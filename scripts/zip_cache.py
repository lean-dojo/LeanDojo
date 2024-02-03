"""Script to zip all traced repos in the cache."""

import ray
import tarfile
from pathlib import Path
from loguru import logger
from lean_dojo.data_extraction.cache import cache


@ray.remote
def compress(dir: Path) -> None:
    logger.info(dir)

    dst = dir.with_suffix(".tar.gz")
    if dst.exists():
        logger.info(f"Removing {dst}")
        dst.unlink()

    with tarfile.open(dst, "w:gz") as tar:
        logger.info(f"Zipping {dir}")
        tar.add(dir, arcname=dir.name)


def main() -> None:
    ray.get([compress.remote(dir) for dir in cache])
    logger.info("Done!")


if __name__ == "__main__":
    main()
