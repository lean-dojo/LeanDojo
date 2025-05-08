import pytest

from lean_dojo import *


MINIF2F_URL = "git@github.com:yangky11/miniF2F-lean4.git"
BATTERIES_URL = "https://github.com/leanprover-community/batteries"
AESOP_URL = "https://github.com/leanprover-community/aesop"
MATHLIB4_URL = "https://github.com/leanprover-community/mathlib4"
LEAN4_EXAMPLE_URL = "https://github.com/yangky11/lean4-example"
EXAMPLE_COMMIT_HASH = "e2602e8d4b1d9cf9240f1a20160a47cfc35165b8"
URLS = [
    MINIF2F_URL,
    BATTERIES_URL,
    AESOP_URL,
    MATHLIB4_URL,
    LEAN4_EXAMPLE_URL,
]


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
    return LeanGitRepo(BATTERIES_URL, "stable")


@pytest.fixture(scope="session")
def mathlib4_repo():
    return LeanGitRepo(MATHLIB4_URL, "stable")


@pytest.fixture(scope="session")
def aesop_repo():
    return LeanGitRepo(AESOP_URL, "stable")


@pytest.fixture(scope="session", params=URLS)
def traced_repo(request):
    url = request.param
    repo = LeanGitRepo(url, "stable")
    traced_repo = trace(repo)
    yield traced_repo
