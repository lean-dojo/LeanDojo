"""
Usage: python examples/lean_agent.py
"""

from lean_dojo_v2.agent.lean_agent import LeanAgent

url1 = "https://github.com/durant42040/lean4-example"
commit1 = "b14fef0ceca29a65bc3122bf730406b33c7effe5"

url2 = "https://github.com/leanprover/lean4"
commit2 = "995161396a25153797460473262555e1792f4823"

agent = LeanAgent()
agent.setup_github_repository(url=url1, commit=commit1)
agent.setup_github_repository(url=url2, commit=commit2)

agent.train()
agent.prove()
