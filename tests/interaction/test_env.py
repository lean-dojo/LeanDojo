import sys
import pytest
from tqdm import tqdm
from loguru import logger
from lean_dojo import *


logger.remove()
logger.add(sys.stderr, level="INFO")


miniF2F = LeanGitRepo(
    "https://github.com/openai/miniF2F",
    "6acdd4b9b9743e2036884a15b37d2f320b858508",
)


@pytest.mark.skip()
def test_env():
    theorem = Theorem(
        miniF2F,
        "lean/src/valid.lean",
        "mathd_algebra_101",
    )
    logger.info(f"Proving {theorem}")

    with Dojo(theorem) as (dojo, init_state):
        env = dojo.query_env(init_state)
        for name in tqdm(env):
            assert dojo.query_type(init_state, name) is not None
