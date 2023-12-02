import pytest

from lean_dojo import *
from lean_dojo.utils import get_latest_commit


LEAN3_URL = "https://github.com/leanprover-community/lean"
MATHLIB_URL = "https://github.com/leanprover-community/mathlib"
MINIF2F_URL = "https://github.com/facebookresearch/miniF2F"
PROOFNET_URL = "https://github.com/zhangir-azerbayev/ProofNet"
LEAN_EXAMPLE_URL = "https://github.com/yangky11/lean-example"
STD4_URL = "https://github.com/leanprover/std4"
AESOP_URL = "https://github.com/JLimperg/aesop"
MATHLIB4_URL = "https://github.com/leanprover-community/mathlib4"
LEAN4_EXAMPLE_URL = "https://github.com/yangky11/lean4-example/"
URLS = [
    LEAN3_URL,
    MINIF2F_URL,
    MATHLIB_URL,
    PROOFNET_URL,
    LEAN_EXAMPLE_URL,
    STD4_URL,
    AESOP_URL,
    MATHLIB4_URL,
    LEAN4_EXAMPLE_URL,
]


@pytest.fixture(scope="session")
def monkeysession():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="session")
def lean_repo():
    commit = get_latest_commit(LEAN3_URL)
    return LeanGitRepo(LEAN3_URL, commit)


@pytest.fixture(scope="session")
def lean4_example_repo():
    commit = get_latest_commit(LEAN4_EXAMPLE_URL)
    return LeanGitRepo(LEAN4_EXAMPLE_URL, commit)


@pytest.fixture(scope="session")
def std4_repo():
    commit = get_latest_commit(STD4_URL)
    return LeanGitRepo(STD4_URL, commit)


@pytest.fixture(scope="session")
def mathlib_repo():
    commit = "19c869efa56bbb8b500f2724c0b77261edbfa28c"
    return LeanGitRepo(MATHLIB_URL, commit)


@pytest.fixture(scope="session")
def mathlib4_repo():
    commit = "3ce43c18f614b76e161f911b75a3e1ef641620ff"
    return LeanGitRepo(MATHLIB4_URL, commit)


@pytest.fixture(scope="session")
def latest_mathlib4_repo():
    commit = get_latest_commit(MATHLIB4_URL)
    return LeanGitRepo(MATHLIB4_URL, commit)


@pytest.fixture(scope="session")
def aesop_repo():
    commit = get_latest_commit(AESOP_URL)
    return LeanGitRepo(AESOP_URL, commit)


@pytest.fixture(scope="session")
def minif2f_repo():
    commit = get_latest_commit(MINIF2F_URL)
    return LeanGitRepo(MINIF2F_URL, commit)


@pytest.fixture(scope="session", params=URLS)
def traced_repo(request):
    url = request.param
    commit = get_latest_commit(url)
    repo = LeanGitRepo(url, commit)
    traced_repo = trace(repo)
    yield traced_repo
