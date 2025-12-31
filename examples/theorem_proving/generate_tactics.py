"""
Usage: python examples/theorem_proving/generate_tactics.py

Example script for proving a theorem with a locally trained Hugging Face model using proof search.
"""

from pantograph.server import Server

from lean_dojo_v2.prover import HFProver

server = Server()
prover = HFProver(ckpt_path="outputs-deepseek")

result, used_tactics = prover.search(
    server=server, goal="∀ {p q : Prop}, p ∧ q → q ∧ p", verbose=False
)

print(result)
if result.success:
    for tactic in used_tactics:
        print(tactic)
