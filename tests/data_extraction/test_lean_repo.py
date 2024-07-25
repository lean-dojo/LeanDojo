# test for the class `LeanGitRepo`
from lean_dojo import LeanGitRepo
from lean_dojo.constants import LEAN4_URL
from git import Repo
from github.Repository import Repository
from lean_dojo.utils import working_directory, repo_type_of_url
from lean_dojo.data_extraction.lean import (
    _to_commit_hash,
    url_to_repo,
    get_latest_commit,
    is_commit_hash,
    GITHUB,
)


def test_url_to_repo(lean4_example_url, remote_example_url):
    repo_name = "lean4-example"

    # 1. github
    ## test get_latest_commit
    gh_cm_hash = get_latest_commit(lean4_example_url)
    assert is_commit_hash(gh_cm_hash)

    ## test url_to_repo & repo_type_of_url
    github_repo = url_to_repo(lean4_example_url)
    assert repo_type_of_url(lean4_example_url) == "github"
    assert isinstance(github_repo, Repository)
    assert github_repo.name == repo_name

    # 2. local
    with working_directory() as tmp_dir:

        ## clone from github
        Repo.clone_from(lean4_example_url, repo_name)

        ## test get_latest_commit
        local_url = str((tmp_dir / repo_name).absolute())
        assert get_latest_commit(local_url) == gh_cm_hash

        ## test url_to_repo & repo_type_of_url
        local_repo = url_to_repo(local_url, repo_type="local")
        assert repo_type_of_url(local_url) == "local"
        assert isinstance(local_repo, Repo)
        assert (
            local_repo.working_dir != local_url
        ), "The working directory should not be the same as the original repo"

    # 3. remote
    with working_directory():
        remote_repo = url_to_repo(remote_example_url)
        assert repo_type_of_url(remote_example_url) == "remote"
        assert isinstance(remote_repo, Repo)
        re_cm_hash = get_latest_commit(remote_example_url)
        tmp_repo_path = str(remote_repo.working_dir)
        assert re_cm_hash == get_latest_commit(tmp_repo_path)


def test_to_commit_hash(lean4_example_url, remote_example_url, example_commit_hash):
    # 1. github
    repo = GITHUB.get_repo("yangky11/lean4-example")
    ## commit hash
    assert _to_commit_hash(repo, example_commit_hash) == example_commit_hash
    ## branch, assume this branch is not changing
    assert _to_commit_hash(repo, "paper") == "8bf74cf67d1acf652a0c74baaa9dc3b9b9e4098c"
    gh_main_hash = _to_commit_hash(repo, "main")
    ## git tag
    assert (
        _to_commit_hash(GITHUB.get_repo("leanprover/lean4"), "v4.9.1")
        == "1b78cb4836cf626007bd38872956a6fab8910993"
    )
    # 2. local
    with working_directory():
        repo = Repo.clone_from(lean4_example_url, "lean4-example")
        repo.git.checkout(example_commit_hash)
        repo.create_tag("v0.1.0")  # create a tag
        repo.git.checkout("main")
        assert _to_commit_hash(repo, example_commit_hash) == example_commit_hash
        assert _to_commit_hash(repo, "main") == gh_main_hash
        assert (
            _to_commit_hash(repo, "origin/paper")
            == "8bf74cf67d1acf652a0c74baaa9dc3b9b9e4098c"
        )
        assert _to_commit_hash(repo, "v0.1.0") == example_commit_hash

    # 3. remote
    with working_directory():
        repo = url_to_repo(remote_example_url)
        assert _to_commit_hash(repo, example_commit_hash) == example_commit_hash
        assert (
            _to_commit_hash(repo, "origin/paper")
            == "8bf74cf67d1acf652a0c74baaa9dc3b9b9e4098c"
        )
        # no tags in the remote repo


def test_git_lean_repo(lean4_example_url, example_commit_hash):
    repo = LeanGitRepo(lean4_example_url, example_commit_hash)
    assert repo.url == lean4_example_url
    assert repo.commit == example_commit_hash
    assert repo.exists()
    assert repo.name == "lean4-example"
    assert repo.commit_url == f"{lean4_example_url}/tree/{example_commit_hash}"
    # test cache directory
    assert repo.format_dirname == f"yangky11-lean4-example-{example_commit_hash}"
