"""
Usage: python examples/external_api.py

Example script for using the Hugging Face API to prove sorry theorems in a GitHub repository.
"""

from lean_dojo_v2.agent import ExternalAgent

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

agent = ExternalAgent(model_name="deepseek-ai/DeepSeek-Prover-V2-671B:novita")
agent.setup_github_repository(url=url, commit=commit)
agent.prove(whole_proof=True)
