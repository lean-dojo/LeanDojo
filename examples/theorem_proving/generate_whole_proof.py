"""
Usage: python examples/theorem_proving/prove_theorem.py

Example script for proving a theorem using the Hugging Face API.
"""

from lean_dojo_v2.prover import ExternalProver

theorem = "theorem my_and_comm : ∀ {p q : Prop}, And p q → And q p := by"
prover = ExternalProver()
proof = prover.generate_whole_proof(theorem)

print(proof)
