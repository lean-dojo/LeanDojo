"""RAG Trainer for Lean theorem proving models."""

import os
from pathlib import Path
from typing import List, Optional

import torch
from loguru import logger
from pytorch_lightning import seed_everything
from tqdm import tqdm

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.database.models.repository import Repository
from lean_dojo_v2.lean_agent.config import TrainingConfig
from lean_dojo_v2.lean_agent.retrieval.datamodule import RetrievalDataModule
from lean_dojo_v2.lean_agent.retrieval.model import PremiseRetriever
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
from lean_dojo_v2.utils.constants import DATA_DIR, RAID_DIR
from lean_dojo_v2.utils.filesystem import find_latest_checkpoint


class RetrievalTrainer:
    """Trainer class for RAG-based theorem proving models.

    This class handles the training of retrieval-augmented generation models
    for Lean theorem proving, including model loading, data preparation,
    and training execution.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the RAG trainer.

        Args:
            config: Training configuration. If None, uses default TrainingConfig.
        """
        self.config = config or TrainingConfig()

    def train(
        self,
        repos: List[Repository],
        database: DynamicDatabase,
        data_path: Path = Path(os.path.join(RAID_DIR, DATA_DIR, "merged")),
        model_checkpoint_path: Optional[str] = None,
    ) -> None:
        """Train the model on the specified repository.

        Args:
            repo_url: URL of the repository being trained on
            repo_commit: Commit hash of the repository
            model_checkpoint_path: Path to the model checkpoint to load from.
                                 If None, finds the latest checkpoint.
        """
        repos_to_process = []

        for repo in repos:
            repos_to_process.append(repo)

            database.export_merged_data(repos_to_process, data_path)

            if model_checkpoint_path is None:
                model_checkpoint_path = find_latest_checkpoint()
            logger.info(f"Found latest checkpoint: {model_checkpoint_path}")

            seed_everything(self.config.seed)

            if not torch.cuda.is_available():
                logger.warning("Indexing the corpus using CPU can be very slow.")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda")

            model_config = self.config.get_model_config()
            model = PremiseRetriever.load(
                model_checkpoint_path, device, freeze=False, config=model_config
            )
            model.train()
            logger.info(f"Loaded premise retriever at {model_checkpoint_path}")

            # Create custom log directory
            custom_log_dir = self.config.get_custom_log_dir(str(repo))

            # Create trainer using configuration
            trainer = self.config.create_trainer(str(repo), custom_log_dir)

            model.set_lambda(self.config.get_lambda_value())

            corpus_path = os.path.join(data_path, "corpus.jsonl")
            random_path = os.path.join(data_path, "random")

            data_module_config = self.config.get_data_module_config(
                random_path, corpus_path
            )
            data_module = RetrievalDataModule(**data_module_config)
            data_module.setup(stage="fit")

            logger.info(f"Training dataset size: {len(data_module.ds_train)}")
            logger.info(f"Validation dataset size: {len(data_module.ds_val)}")
            logger.info(f"Testing dataset size: {len(data_module.ds_pred)}")

            trainer.fit(
                model,
                datamodule=data_module,
            )

    def evaluate(self, dataset_path: Optional[str] = None) -> None:
        """Evaluate the trained model.

        Args:
            dataset_path: Path to the dataset directory. If None, uses default RAID_DIR/DATA_DIR.
        """
        best_model_path = find_latest_checkpoint()
        logger.info("Testing...")

        if dataset_path is None:
            dataset_path = os.path.join(RAID_DIR, DATA_DIR)

        testing_paths = [
            os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
        ]

        for data_path in testing_paths:
            if not os.path.isdir(data_path):
                continue

            # Import here to avoid circular imports
            from lean_dojo_v2.lean_agent.retrieval.main import run_cli

            run_cli(best_model_path, data_path)

            num_gpus = self.config.num_gpus
            preds_map = {}
            for gpu_id in range(num_gpus):
                pickle_path = f"test_pickle_{gpu_id}.pkl"
                if os.path.exists(pickle_path):
                    import pickle

                    with open(pickle_path, "rb") as f:
                        preds = pickle.load(f)
                        preds_map.update(preds)

            logger.info("Loaded the predictions pickle files")
            test_data_path = os.path.join(data_path, "random", "test.json")

            if os.path.exists(test_data_path):
                import json

                with open(test_data_path, "r") as f:
                    data = json.load(f)

                R1, R10, MRR = self._evaluate_retrieval_metrics(data, preds_map)
                logger.info(f"R@1: {R1:.2f}, R@10: {R10:.2f}, MRR: {MRR:.2f}")

    def _evaluate_retrieval_metrics(self, data, preds_map):
        """Evaluate retrieval metrics on the test data.

        Args:
            data: Test data
            preds_map: Predictions map

        Returns:
            Tuple of (R@1, R@10, MRR) metrics
        """
        R1 = []
        R10 = []
        MRR = []

        for thm in tqdm(data):
            for i, _ in enumerate(thm["traced_tactics"]):
                pred = None
                key = (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
                if key in preds_map:
                    pred = preds_map[key]
                else:
                    continue
                all_pos_premises = set(pred["all_pos_premises"])
                if len(all_pos_premises) == 0:
                    continue

                retrieved_premises = pred["retrieved_premises"]
                TP1 = retrieved_premises[0] in all_pos_premises
                R1.append(float(TP1) / len(all_pos_premises))
                TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
                R10.append(float(TP10) / len(all_pos_premises))

                for j, p in enumerate(retrieved_premises):
                    if p in all_pos_premises:
                        MRR.append(1.0 / (j + 1))
                        break
                else:
                    MRR.append(0.0)

        R1 = 100 * sum(R1) / len(R1) if R1 else 0.0
        R10 = 100 * sum(R10) / len(R10) if R10 else 0.0
        MRR = sum(MRR) / len(MRR) if MRR else 0.0
        return R1, R10, MRR
