# test for cache manager
from git import Repo
from lean_dojo.utils import working_directory
from pathlib import Path
from lean_dojo.data_extraction.lean import _format_dirname
from lean_dojo.data_extraction.cache import cache


def test_get_cache(lean4_example_url, remote_example_url, example_commit_hash):
    # Note: The `git.Repo` requires the local repo to be cloned in a directory
    # all cached repos are stored in CACHE_DIR/repos
    prefix = "repos"

    # test local repo cache
    with working_directory() as tmp_dir:
        # assume that the local repo placed in `/.../testrepo/lean4-example`
        repo = Repo.clone_from(lean4_example_url, "testrepo/lean4-example")
        repo.git.checkout(example_commit_hash)
        local_dir = tmp_dir / "testrepo/lean4-example"
        # use local_dir as the key to store the cache
        rel_cache_dir = (
            prefix
            / Path(_format_dirname(str(local_dir), example_commit_hash))
            / local_dir.name
        )
        cache.store(local_dir, rel_cache_dir)
    # get the cache
    local_url, local_commit = str(local_dir), example_commit_hash
    repo_cache = cache.get(local_url, local_commit, prefix)
    assert (
        _format_dirname(local_url, local_commit)
        == f"{local_dir.parent.name}-{local_dir.name}-{local_commit}"
    )
    assert repo_cache is not None

    # test remote repo cache
    with working_directory() as tmp_dir:
        repo = Repo.clone_from(remote_example_url, "lean4-example")
        repo.git.checkout(example_commit_hash)
        tmp_remote_dir = tmp_dir / "lean4-example"
        # use remote url as the key to store the cache
        rel_cache_dir = (
            prefix
            / Path(_format_dirname(str(remote_example_url), example_commit_hash))
            / tmp_remote_dir.name
        )
        cache.store(tmp_remote_dir, rel_cache_dir)
    # get the cache
    remote_url, remote_commit = remote_example_url, example_commit_hash
    repo_cache = cache.get(remote_url, remote_commit, prefix)
    assert repo_cache is not None
    assert (
        _format_dirname(remote_url, remote_commit)
        == f"rexzong-lean4-example-{example_commit_hash}"
    )
