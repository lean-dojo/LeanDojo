import ray
import random
import json
import shlex
import tempfile
from pathlib import Path
import toml
import argparse
import subprocess
from tqdm import tqdm
from loguru import logger
from typing import Tuple

from lean_dojo import *
from lean_dojo.constants import TACTIC_MEMORY_LIMIT
from lean_dojo.utils import ray_actor_pool, working_directory


def read_next_response(proc):
    while True:
        line = proc.stdout.readline().strip()
        # logger.info(line)
        if line == "":
            raise EOFError
        try:
            return json.loads(line)
        except json.decoder.JSONDecodeError:
            continue


def _validate_ground_truth(thm) -> Tuple[bool, bool]:
    logger.info(thm)
    theorem = thm["theorem"]
    proof = thm["proof"]

    # Validate using LeanDojo.
    lean_dojo_result = None
    init_ctx = None

    try:
        with Dojo(theorem) as (dojo, init_state):
            assert init_state.num_goals == 1
            init_ctx = [decl.ident for decl in init_state.goals[0].assumptions]

            res = dojo.run_tac(init_state, proof)
            if isinstance(res, ProofFinished):
                lean_dojo_result = True
            else:
                logger.error(f"LeanDojo error: {res}")
                lean_dojo_result = False
    except Exception as ex:
        logger.error(f"LeanDojo error: {ex}")
        lean_dojo_result = False

    # Validate using lean-gym.
    namespaces = " ".join(thm["namespaces"])
    lean_gym_result = True

    with working_directory("lean-gym"):
        cid_file = Path(next(tempfile._get_candidate_names()) + ".cid")
        proc = subprocess.Popen(
            shlex.split(
                f'docker run --cidfile {cid_file} --memory {TACTIC_MEMORY_LIMIT} -i --rm --mount type=bind,src="{Path.cwd()}",target="/workspace/lean-gym" --workdir "/workspace/lean-gym" yangky11/lean-dojo lean --threads=1 --run src/repl.lean'
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            bufsize=1,
        )

        if init_ctx is not None:
            intro_tac = f"tactic.intron {len(init_ctx)}"
        else:
            intro_tac = f"intros"
        lean_gym_commands = [
            f'["init_search", ["{theorem.full_name}", "{namespaces}"]]\n',
            f'["run_tac",["0","0","{intro_tac}"]]\n',
            f'["run_tac",["0","1","{proof}"]]\n',
        ]

        for cmd in lean_gym_commands:
            # logger.info(cmd)
            proc.stdin.write(cmd)
            try:
                res = read_next_response(proc)
            except EOFError:
                logger.error("lean-gym error: EOFError")
                lean_gym_result = False
                break
            if res["error"] is not None:
                logger.error(f"lean-gym error: {res}")
                lean_gym_result = False
                break

        if lean_gym_result and res["tactic_state"] != "no goals":
            logger.error(f"lean-gym error: {res}")
            lean_gym_result = False

    if cid_file.exists():
        cid = cid_file.open().read().strip()
        cid_file.unlink()
        os.system(f"docker stop -t 1 {cid}")

    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()

    return lean_dojo_result, lean_gym_result


@ray.remote
class RayHelper:
    def validate_ground_truth(self, thm) -> bool:
        return _validate_ground_truth(thm)


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    logger.info(args)

    os.environ["TACTIC_TIMEOUT"] = "600000"

    config = toml.load("lean-gym/leanpkg.toml")
    url = config["dependencies"]["mathlib"]["git"]
    commit = config["dependencies"]["mathlib"]["rev"]
    repo = LeanGitRepo(url, commit)
    traced_repo = trace(repo)

    theorems = {}

    logger.info("Loading the theorems")
    for t in tqdm(traced_repo.get_traced_theorems()):
        if t.is_private:  # Discard private theorems.
            continue

        proof = t.get_single_tactic_proof()
        if proof is None:  # Discard theorems without tactic-style proofs.
            continue

        inside, opened = t.get_namespaces()
        namespaces = list(set(inside + opened))
        assert t.theorem.full_name not in theorems
        theorems[t.theorem.full_name] = {
            "theorem": t.theorem,
            "proof": proof,
            "namespaces": namespaces,
        }

    theorems = list(theorems.values())
    random.shuffle(theorems)
    num_theorems = len(theorems)
    logger.info(f"Evaluating {num_theorems} theorems")

    # for thm in theorems:
    #    _validate_ground_truth(thm)
    # return

    with ray_actor_pool(RayHelper) as pool:
        results = list(
            tqdm(
                pool.map_unordered(
                    lambda a, thm: a.validate_ground_truth.remote(thm), theorems
                ),
                total=len(theorems),
            )
        )
        num_lean_dojo_correct = 0
        num_lean_gym_correct = 0
        for x, y in results:
            num_lean_dojo_correct += x
            num_lean_gym_correct += y

        logger.info(f"LeanDojo: {num_lean_dojo_correct}/{num_theorems}")
        logger.info(f"lean-gym: {num_lean_gym_correct}/{num_theorems}")


if __name__ == "__main__":
    main()
