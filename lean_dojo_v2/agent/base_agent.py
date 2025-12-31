import os
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
from pantograph import Server

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.database.models import Repository
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
from lean_dojo_v2.lean_dojo.data_extraction.trace import get_traced_repo_path
from lean_dojo_v2.utils.constants import DATA_DIR, RAID_DIR


class BaseAgent(ABC):
    """Abstract base class for theorem proving agents.

    This class defines the common interface and functionality for different types
    of theorem proving agents, such as HFAgent and LeanAgent.
    """

    def __init__(self, database_path: str = "dynamic_database.json"):
        self.database = DynamicDatabase(json_path=database_path)
        self.data_path = Path(os.path.join(RAID_DIR, DATA_DIR, "merged"))
        self.repos = []

    @abstractmethod
    def _get_build_deps(self) -> bool:
        """Get whether to build dependencies. Must be implemented by subclasses."""
        pass

    def trace_repository(self, url: str, commit: str, build_deps: bool) -> Repository:
        """Trace a repository and return a Repository object."""
        return self.database.trace_repository(url, commit, build_deps)

    def add_repository(self, repo: Repository):
        """Add a repository to the database."""
        self.database.add_repository(repo)
        self.repos.append(LeanGitRepo(repo.url, repo.commit))

    def train(self):
        """Train the model on the repository.

        Raises:
            ValueError: If no repository is loaded
        """
        sorted_repos = self.database.sort_repositories_by_difficulty()

        if len(sorted_repos) == 0:
            raise ValueError(
                "No repository loaded. Call setup_github_repository() or setup_local_repository() first."
            )

        self.trainer.train(
            repos=sorted_repos, database=self.database, data_path=self.data_path
        )

    def evaluate(self):
        """Evaluate the trained model."""
        self.trainer.evaluate()

    def setup_github_repository(self, url: str, commit: str):
        """Set up a GitHub repository for processing."""
        traced_repo = self.trace_repository(
            url, commit, build_deps=self._get_build_deps()
        )
        if traced_repo:
            self.add_repository(traced_repo)
        else:
            raise ValueError(f"Failed to setup github repository {url}")

    def setup_local_repository(self, path: str):
        """Set up a local repository for processing."""
        repo = LeanGitRepo.from_path(path)
        traced_repo = self.trace_repository(
            repo.url, repo.commit, build_deps=self._get_build_deps()
        )
        if traced_repo:
            self.add_repository(traced_repo)
        else:
            raise ValueError(f"Failed to setup local repository {path}")

    def initialize_prover(self):
        """Initialize the theorem prover.

        Returns:
            List of sorry theorems to prove
        """
        sorry_theorems = []
        for repo in self.repos:
            repository = self.database.get_repository(repo.url, repo.commit)
            for theorem in repository.sorry_theorems_unproved:
                sorry_theorems.append((theorem, repo))

        self._setup_prover()

        return sorry_theorems

    @abstractmethod
    def _setup_prover(self):
        """Set up the prover agent. Must be implemented by subclasses."""
        pass

    def prove(self, whole_proof: bool = False):
        """Prove sorry theorems."""
        sorry_theorems = self.initialize_prover()

        if not sorry_theorems:
            print("No sorry theorems found to prove.")
            return

        print(f"Found {len(sorry_theorems)} sorry theorems to prove")
        for theorem, repo in sorry_theorems:
            self.prove_theorem(theorem, repo, whole_proof)

    def prove_theorem(self, theorem, repo, whole_proof: bool = False):
        """Processes a single theorem."""
        if whole_proof:
            proof = self.prover.generate_whole_proof(theorem)
            print(proof)
            return

        traced_repo_path = get_traced_repo_path(repo, build_deps=self._get_build_deps())

        server = Server(
            imports=["Init", str(theorem.file_path).replace(".lean", "")],
            project_path=traced_repo_path,
        )

        print(f"Proving {theorem.full_name}")
        result, used_tactics = self.prover.search(
            server=server, theorem=theorem, verbose=False
        )
        print(result)
        if result.success:
            for tactic in used_tactics:
                print(tactic)
        else:
            logger.info(f"No proof found for {theorem.full_name}")

        return result.success
