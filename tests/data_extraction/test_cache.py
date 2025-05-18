# test for cache manager
from git import Repo
from pathlib import Path
from lean_dojo.utils import working_directory
from lean_dojo.data_extraction.cache import cache


def test_local_repo_cache(lean4_example_url, example_commit_hash):
    # Note: The `git.Repo` requires the local repo to be cloned in a directory
    # all cached repos are stored in CACHE_DIR/repos
    prefix = "repos"
    repo_name = "lean4-example"
    with working_directory() as tmp_dir:
        repo = Repo.clone_from(lean4_example_url, repo_name)
        repo.git.checkout(example_commit_hash)
        local_dir = tmp_dir / repo_name
        rel_cache_dir = (
            prefix / Path(f"gitpython-{repo_name}-{example_commit_hash}") / repo_name
        )
        cache.store(local_dir, rel_cache_dir)
    # get the cache
    repo_cache_dir = cache.get(rel_cache_dir)
    assert repo_cache_dir is not None
