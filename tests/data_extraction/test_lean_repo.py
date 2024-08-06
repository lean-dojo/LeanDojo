# Tests for the class `LeanGitRepo`
from git import Repo
from lean_dojo import LeanGitRepo
from github.Repository import Repository
from lean_dojo.data_extraction.lean import (
    _to_commit_hash,
    get_repo_type,
    url_to_repo,
    get_latest_commit,
    is_commit_hash,
    GITHUB,
    RepoType,
)
from lean_dojo.utils import working_directory


def test_github_type(lean4_example_url, example_commit_hash):
    repo_name = "lean4-example"

    ## get_latest_commit
    gh_cm_hash = get_latest_commit(lean4_example_url)
    assert is_commit_hash(gh_cm_hash)

    ## url_to_repo & get_repo_type
    github_repo = url_to_repo(lean4_example_url)
    assert get_repo_type(lean4_example_url) == RepoType.GITHUB
    assert get_repo_type("git@github.com:yangky11/lean4-example.git") == RepoType.GITHUB
    assert get_repo_type("git@github.com:yangky11/lean4-example") == RepoType.GITHUB
    assert isinstance(github_repo, Repository)
    assert github_repo.name == repo_name

    ## commit hash
    assert _to_commit_hash(github_repo, example_commit_hash) == example_commit_hash
    ### test branch, assume this branch is not changing
    assert (
        _to_commit_hash(github_repo, "paper")
        == "8bf74cf67d1acf652a0c74baaa9dc3b9b9e4098c"
    )
    ### test git tag
    assert (
        _to_commit_hash(GITHUB.get_repo("leanprover/lean4"), "v4.9.1")
        == "1b78cb4836cf626007bd38872956a6fab8910993"
    )

    ## LeanGitRepo
    LeanGitRepo(lean4_example_url, "main")  # init with branch
    repo = LeanGitRepo(lean4_example_url, example_commit_hash)
    assert repo.url == lean4_example_url
    assert repo.repo_type == RepoType.GITHUB
    assert repo.commit == example_commit_hash
    assert repo.exists()
    assert repo.name == repo_name
    assert repo.lean_version == "v4.7.0"
    assert repo.commit_url == f"{lean4_example_url}/tree/{example_commit_hash}"
    # cache name
    assert isinstance(repo.repo, Repository)
    assert (
        str(repo.get_cache_dirname()) == f"yangky11-{repo_name}-{example_commit_hash}"
    )


def test_remote_type(remote_example_url, example_commit_hash):
    repo_name = "lean4-example"

    remote_repo = url_to_repo(remote_example_url)
    assert get_repo_type(remote_example_url) == RepoType.REMOTE
    assert isinstance(remote_repo, Repo)
    re_cm_hash = get_latest_commit(remote_example_url)
    assert re_cm_hash == get_latest_commit(str(remote_repo.working_dir))
    assert _to_commit_hash(remote_repo, example_commit_hash) == example_commit_hash
    assert (
        _to_commit_hash(remote_repo, "origin/paper")
        == "8bf74cf67d1acf652a0c74baaa9dc3b9b9e4098c"
    )

    ## LeanGitRepo
    LeanGitRepo(remote_example_url, "main")
    repo = LeanGitRepo(remote_example_url, example_commit_hash)
    assert repo.url == remote_example_url
    assert repo.repo_type == RepoType.REMOTE
    assert repo.commit == example_commit_hash
    assert repo.exists()
    assert repo.name == repo_name
    assert repo.lean_version == "v4.7.0"
    assert repo.commit_url == f"{remote_example_url}/tree/{example_commit_hash}"
    # cache name
    assert isinstance(repo.repo, Repo)
    assert (
        str(repo.get_cache_dirname()) == f"gitpython-{repo_name}-{example_commit_hash}"
    )


def test_local_type(lean4_example_url, example_commit_hash):
    repo_name = "lean4-example"
    gh_cm_hash = get_latest_commit(lean4_example_url)

    with working_directory() as tmp_dir:
        # git repo placed in `tmp_dir / repo_name`
        Repo.clone_from(lean4_example_url, repo_name)

        ## get_latest_commit
        local_url = str((tmp_dir / repo_name).absolute())
        assert get_latest_commit(local_url) == gh_cm_hash

        ## url_to_repo & get_repo_type
        local_repo = url_to_repo(local_url, repo_type=RepoType.LOCAL)
        assert get_repo_type(local_url) == RepoType.LOCAL
        assert isinstance(local_repo, Repo)
        assert (
            local_repo.working_dir != local_url
        ), "The working directory should not be the same as the original repo"

        ## commit hash
        repo = Repo(local_url)
        repo.git.checkout(example_commit_hash)
        repo.create_tag("v0.1.0")  # create a tag for the example commit hash
        repo.git.checkout("main")  # switch back to main branch
        assert _to_commit_hash(repo, example_commit_hash) == example_commit_hash
        assert (
            _to_commit_hash(repo, "origin/paper")
            == "8bf74cf67d1acf652a0c74baaa9dc3b9b9e4098c"
        )
        assert _to_commit_hash(repo, "v0.1.0") == example_commit_hash

        ## LeanGitRepo
        LeanGitRepo(local_url, "main")
        repo = LeanGitRepo(local_url, example_commit_hash)
        repo2 = LeanGitRepo.from_path(local_url)  # test from_path
        assert repo.url == local_url == repo2.url
        assert repo.repo_type == RepoType.LOCAL == repo2.repo_type
        assert repo.commit == example_commit_hash and repo2.commit == gh_cm_hash
        assert repo.exists() and repo2.exists()
        assert repo.name == repo_name == repo2.name
        assert repo.lean_version == "v4.7.0"
        # cache name
        assert isinstance(repo.repo, Repo) and isinstance(repo2.repo, Repo)
        assert (
            str(repo.get_cache_dirname())
            == f"gitpython-{repo_name}-{example_commit_hash}"
        )
        assert str(repo2.get_cache_dirname()) == f"gitpython-{repo_name}-{gh_cm_hash}"
