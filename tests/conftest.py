import pytest

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
    commit = "fe4454af900584467d21f4fd4fe951d29d9332a7"
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
