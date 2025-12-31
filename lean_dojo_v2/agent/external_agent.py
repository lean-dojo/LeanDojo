from lean_dojo_v2.prover.external_prover import ExternalProver

from .base_agent import BaseAgent


class ExternalAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-Prover-V2-671B:novita",
    ):
        super().__init__()
        self.model_name = model_name

    def _get_build_deps(self) -> bool:
        """ExternalAgent doesn't build dependencies by default."""
        return False

    def _setup_prover(self):
        """Set up the RetrievalProver for ExternalAgent."""
        self.prover = ExternalProver(model_name=self.model_name)


def main():
    """
    Main function to run LeanAgent.
    """
    url = "https://github.com/durant42040/lean4-example"
    commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

    agent = ExternalAgent(model_name="deepseek-ai/DeepSeek-Prover-V2-671B:novita")
    agent.setup_github_repository(url=url, commit=commit)
    agent.prove()


if __name__ == "__main__":
    main()
