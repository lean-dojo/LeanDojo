# test for the class `LeanGitRepo`
from lean_dojo import LeanGitRepo
from lean_dojo.data_extraction.lean import _to_commit_hash
from lean_dojo.constants import LEAN4_URL


def test_lean_git_repo(lean4_example_url, example_commit_hash):
    repo = LeanGitRepo(lean4_example_url, example_commit_hash)
    assert repo.url == lean4_example_url
    assert repo.commit == example_commit_hash
    assert repo.exists()
    assert repo.name == "lean4-example"
    assert repo.commit_url == f"{lean4_example_url}/tree/{example_commit_hash}"
    # test cache directory
    assert (
        repo.format_dirname
        == "yangky11-lean4-example-3f8c5eb303a225cdef609498b8d87262e5ef344b"
    )
    # test commit hash
    ## test commit hash
    assert _to_commit_hash(repo, example_commit_hash) == example_commit_hash
    ## test branch, assume the branch is not changed
    assert _to_commit_hash(repo, "paper") == "8bf74cf67d1acf652a0c74baaa9dc3b9b9e4098c"
    ## test tag
    lean4_repo = LeanGitRepo(LEAN4_URL, "master")
    assert (
        _to_commit_hash(lean4_repo, "v4.9.1")
        == "1b78cb4836cf626007bd38872956a6fab8910993"
    )
