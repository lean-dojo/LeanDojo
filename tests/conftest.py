import pytest
import os, shutil
from git import Repo
from lean_dojo import *


BATTERIES_URL = "https://github.com/leanprover-community/batteries"
AESOP_URL = "https://github.com/leanprover-community/aesop"
MATHLIB4_URL = "https://github.com/leanprover-community/mathlib4"
LEAN4_EXAMPLE_URL = "https://github.com/yangky11/lean4-example"
URLS = [
    BATTERIES_URL,
    AESOP_URL,
    MATHLIB4_URL,
    LEAN4_EXAMPLE_URL,
]

EXAMPLE_COMMIT_HASH = "3f8c5eb303a225cdef609498b8d87262e5ef344b"
REMOTE_EXAMPLE_URL = "https://gitee.com/rexzong/lean4-example"
LOCAL_TEST_PATH = f"{os.path.dirname(__file__)}/testdata"

@pytest.fixture(scope="session")
def clean_clone_and_checkout():
    def _clean_clone_and_checkout(repo_url, local_path, label='main'):
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        repo = Repo.clone_from(repo_url, local_path)
        repo.git.checkout(label)
        return repo
    return _clean_clone_and_checkout

@pytest.fixture(scope="session")
def lean4_example_url():
    return LEAN4_EXAMPLE_URL

@pytest.fixture(scope="session")
def example_commit_hash():
    return EXAMPLE_COMMIT_HASH

@pytest.fixture(scope="session")
def remote_example_url():
    return REMOTE_EXAMPLE_URL

@pytest.fixture(scope="session")
def local_test_path():
    return LOCAL_TEST_PATH

@pytest.fixture(scope="session")
def monkeysession():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="session")
def lean4_example_repo():
    commit = get_latest_commit(LEAN4_EXAMPLE_URL)
    return LeanGitRepo(LEAN4_EXAMPLE_URL, commit)


@pytest.fixture(scope="session")
def batteries_repo():
    commit = get_latest_commit(BATTERIES_URL)
    return LeanGitRepo(BATTERIES_URL, commit)


@pytest.fixture(scope="session")
def mathlib4_repo():
    commit = "29dcec074de168ac2bf835a77ef68bbe069194c5"
    return LeanGitRepo(MATHLIB4_URL, commit)


@pytest.fixture(scope="session")
def latest_mathlib4_repo():
    commit = get_latest_commit(MATHLIB4_URL)
    return LeanGitRepo(MATHLIB4_URL, commit)


@pytest.fixture(scope="session")
def aesop_repo():
    commit = get_latest_commit(AESOP_URL)
    return LeanGitRepo(AESOP_URL, commit)


@pytest.fixture(scope="session", params=URLS)
def traced_repo(request):
    url = request.param
    commit = get_latest_commit(url)
    repo = LeanGitRepo(url, commit)
    traced_repo = trace(repo)
    yield traced_repo
