import pytest

from lean_dojo import *


MINIF2F_URL = "git@github.com:yangky11/miniF2F-lean4.git"
BATTERIES_URL = "https://github.com/leanprover-community/batteries"
AESOP_URL = "https://github.com/leanprover-community/aesop"
MATHLIB4_URL = "https://github.com/leanprover-community/mathlib4"
LEAN4_EXAMPLE_URL = "https://github.com/yangky11/lean4-example"
EXAMPLE_COMMIT_HASH = "3f8c5eb303a225cdef609498b8d87262e5ef344b"
REMOTE_EXAMPLE_URL = "https://gitee.com/rexzong/lean4-example"
URLS = [
    MINIF2F_URL,
    BATTERIES_URL,
    AESOP_URL,
    MATHLIB4_URL,
    LEAN4_EXAMPLE_URL,
]


@pytest.fixture(scope="session")
def remote_example_url():
    return REMOTE_EXAMPLE_URL


@pytest.fixture(scope="session")
def example_commit_hash():
    return EXAMPLE_COMMIT_HASH


@pytest.fixture(scope="session")
def lean4_example_url():
    return LEAN4_EXAMPLE_URL


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
