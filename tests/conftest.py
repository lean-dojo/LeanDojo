import pytest
from loguru import logger

from lean_dojo import *
from lean_dojo.utils import get_latest_commit


LEAN3_URL = "https://github.com/leanprover-community/lean"
MATHLIB_URL = "https://github.com/leanprover-community/mathlib"
MINIF2F_URL = "https://github.com/facebookresearch/miniF2F"
LEAN_LIQUID_URL = "https://github.com/leanprover-community/lean-liquid"
PROOFNET_URL = "https://github.com/zhangir-azerbayev/ProofNet"
LEAN_EXAMPLE_URL = "https://github.com/yangky11/lean-example"
LEAN4_URL = "https://github.com/leanprover/lean4"
STD4_URL = "https://github.com/leanprover/std4"
AESOP_URL = "https://github.com/JLimperg/aesop"
MATHLIB4_URL = "https://github.com/leanprover-community/mathlib4"
LEAN4_EXAMPLE_URL = "https://github.com/yangky11/lean4-example"
URLS = [
    LEAN3_URL,
    MINIF2F_URL,
    MATHLIB_URL,
    LEAN_LIQUID_URL,
    PROOFNET_URL,
    LEAN_EXAMPLE_URL,
    LEAN4_URL,
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
def lean4_repo():
    commit = get_latest_commit(LEAN4_URL)
    return LeanGitRepo(LEAN4_URL, commit)


@pytest.fixture(scope="session")
def lean4_example_repo():
    commit = get_latest_commit(LEAN4_EXAMPLE_URL)
    return LeanGitRepo(LEAN4_EXAMPLE_URL, commit)


@pytest.fixture(scope="session")
def std4_repo():
    commit = "ccbe74d4406be21b91c04d62b4c93dec9adfc546"
    return LeanGitRepo(STD4_URL, commit)


@pytest.fixture(scope="session")
def mathlib_repo():
    commit = "8c1b484d6a214e059531e22f1be9898ed6c1fd47"
    return LeanGitRepo(MATHLIB_URL, commit)


@pytest.fixture(scope="session")
def mathlib4_repo():
    commit = "5a919533f110b7d76410134a237ee374f24eaaad"
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
