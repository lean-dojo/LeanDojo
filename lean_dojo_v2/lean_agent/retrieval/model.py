"""Ligihtning module for the premise retriever."""

import math
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch.distributed import barrier
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from lean_dojo_v2.lean_agent.common import Context, Corpus, Premise, get_optimizers
from lean_dojo_v2.lean_dojo import Pos
from lean_dojo_v2.utils.common import (
    cpu_checkpointing_enabled,
    load_checkpoint,
    zip_strict,
)

torch.set_float32_matmul_precision("medium")


class PremiseRetriever(pl.LightningModule):
    """
    A PyTorch Lightning module implementing a premise retriever for theorem proving.
    This class implements the premise retrieval component in a theorem proving system,
    using a T5 encoder model to generate embeddings for premises and contexts.
    It includes functionality for:
    - Training with contrastive learning
    - Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
    - Corpus indexing and retrieval
    - Evaluation using metrics like Recall@K and MRR
    The model encodes both contexts (theorem proving states) and premises into a
    shared embedding space, and retrieves relevant premises based on cosine similarity.
    Attributes:
        tokenizer: Tokenizer for encoding inputs
        encoder: T5 encoder model for generating embeddings
        corpus: Collection of premises for retrieval
        corpus_embeddings: Cached embeddings of all premises in the corpus
        embeddings_staled: Flag indicating if corpus embeddings need recomputation
        train_loss: List tracking training losses
        previous_params: Dictionary of model parameters before training (for EWC)
        fisher_info: Fisher information matrix for EWC
        lamda: Weight for EWC loss term
    Args:
        model_name (str): Name/path of the pretrained T5 model to use
        lr (float): Learning rate for optimization
        warmup_steps (int): Number of warmup steps for the learning rate scheduler
        max_seq_len (int): Maximum sequence length for tokenization
        num_retrieved (int, optional): Number of premises to retrieve. Defaults to 100.
    """

    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        max_seq_len: int,
        num_retrieved: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.embeddings_staled = True
        self.train_loss = []
        self.previous_params = {}
        self.fisher_info = {}
        self.lamda = 0  # No EWC by default

    def set_fisher_info(self, fisher_info):
        if fisher_info is not None:
            self.fisher_info = fisher_info
            logger.info("Fisher Information has been updated in the model.")
        else:
            logger.warning("No Fisher Information provided to update.")

    def set_lambda(self, lambda_value):
        self.lamda = lambda_value

    def set_previous_params(self):
        self.previous_params = {
            name: param.clone().detach() for name, param in self.named_parameters()
        }

    def ewc_loss(self):
        """
        Calculate the Elastic Weight Consolidation (EWC) loss.
        EWC loss is used to prevent catastrophic forgetting in neural networks by
        penalizing changes to important parameters. The penalty is based on the
        Fisher Information matrix and the difference between current and previous
        parameter values.
        Returns:
            float: The calculated EWC loss. If Fisher information is not available
                   or lambda is zero, returns 0.0.
        """
        if not self.fisher_info or self.lamda == 0:
            return 0.0

        ewc_loss = 0
        for name, param in self.named_parameters():
            if name in self.fisher_info and name in self.previous_params:
                # EWC Penalty is the sum of the squares of differences times the Fisher Information
                fisher = self.fisher_info[name].to(param.device)
                prev_param = self.previous_params[name].to(param.device)
                ewc_loss += (fisher * (param - prev_param) ** 2).sum()
            else:
                logger.warning(f"Parameter {name} not found in previous params.")
        total_loss = self.lamda * ewc_loss
        logger.info(f"Total EWC loss: {total_loss.item()}, lambda: {self.lamda}")
        return total_loss

    @classmethod
    def load(
        cls, ckpt_path: str, device, freeze: bool, config: dict
    ) -> "PremiseRetriever":
        return load_checkpoint(cls, ckpt_path, device, freeze, config)

    @classmethod
    def load_hf(
        cls, ckpt_path: str, max_seq_len: int, device: int, dtype=None
    ) -> "PremiseRetriever":
        model = PremiseRetriever(ckpt_path, 0.0, 0, max_seq_len, 100).to(device).eval()
        if dtype is not None:
            return model.to(dtype)
        elif (
            model.dtype == torch.float32
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
        ):
            return model.to(torch.bfloat16)
        else:
            return model

    def load_corpus(self, path_or_corpus: Union[str, Corpus]) -> None:
        """Associate the retriever with a corpus."""
        if isinstance(path_or_corpus, Corpus):
            self.corpus = path_or_corpus
            self.corpus_embeddings = None
            self.embeddings_staled = True
            return

        path = path_or_corpus
        if path.endswith(".jsonl"):  # A raw corpus without embeddings.
            self.corpus = Corpus(path)
            self.corpus_embeddings = None
            self.embeddings_staled = True
        else:  # A corpus with pre-computed embeddings.
            indexed_corpus = pickle.load(open(path, "rb"))
            self.corpus = indexed_corpus.corpus
            self.corpus_embeddings = indexed_corpus.embeddings
            self.embeddings_staled = False
            logger.info(
                f"Embeddings staled load corpus pickle: {self.embeddings_staled}"
            )

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.encoder.config.hidden_size

    def _encode(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        if cpu_checkpointing_enabled(self):
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.encoder, input_ids, attention_mask, use_reentrant=False
            )[0]
        else:
            hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        # Masked average.
        lens = attention_mask.sum(dim=1)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        return F.normalize(features, dim=1)

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_mask: torch.LongTensor,
        pos_premise_ids: torch.LongTensor,
        pos_premise_mask: torch.LongTensor,
        neg_premises_ids: torch.LongTensor,
        neg_premises_mask: torch.LongTensor,
        label: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        # Encode the query and positive/negative documents.
        context_emb = self._encode(context_ids, context_mask)
        pos_premise_emb = self._encode(pos_premise_ids, pos_premise_mask)
        neg_premise_embs = [
            self._encode(ids, mask)
            for ids, mask in zip_strict(neg_premises_ids, neg_premises_mask)
        ]
        all_premise_embs = torch.cat([pos_premise_emb, *neg_premise_embs], dim=0)

        # Cosine similarities for unit-norm vectors are just inner products.
        similarity = torch.mm(context_emb, all_premise_embs.t())
        assert -1 <= similarity.min() <= similarity.max() <= 1
        loss = F.mse_loss(similarity, label)
        return loss

    ############
    # Training #
    ############

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)

        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True

        self.set_previous_params()

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        loss = self(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )
        loss += self.ewc_loss()
        self.train_loss.append(loss.item())
        self.log(
            "loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )
        return loss

    def on_train_batch_end(self, outputs, batch, _) -> None:
        """Mark the embeddings as staled after a training batch."""
        self.embeddings_staled = True

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    ##############
    # Validation #
    ##############

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """
        Re-index the retrieval corpus using the up-to-date encoder.

        This method updates the embeddings of the retrieval corpus if they are marked as stale.
        It processes the corpus in batches, tokenizes the premises, and encodes them to update
        the corpus embeddings.

        Args:
            batch_size (int): The size of the batches to process the corpus.

        Returns:
            None
        """
        if not self.embeddings_staled:
            return
        logger.info("Re-indexing the retrieval corpus")

        self.corpus_embeddings = torch.zeros(
            len(self.corpus.all_premises),
            self.embedding_size,
            dtype=self.encoder.dtype,
            device=self.device,
        )

        for i in tqdm(range(0, len(self.corpus), batch_size)):
            batch_premises = self.corpus.all_premises[i : i + batch_size]
            tokenized_premises = self.tokenizer(
                [p.serialize() for p in batch_premises],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.corpus_embeddings[i : i + batch_size] = self._encode(
                tokenized_premises.input_ids, tokenized_premises.attention_mask
            )
        self.embeddings_staled = False

    def on_validation_start(self) -> None:
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """
        Perform a validation step by retrieving premises and calculating metrics such as Recall@K and MRR.

        Args:
            batch (Dict[str, Any]): A dictionary containing the batch data, including context IDs, context mask, and all positive premises.
            batch_idx (int): The index of the current batch.

        Returns:
            None

        The method performs the following steps:
        1. Encodes the context using the provided context IDs and mask.
        2. Retrieves the nearest premises from the corpus based on the encoded context.
        3. Evaluates the retrieval by calculating Recall@K and Mean Reciprocal Rank (MRR) metrics.
        4. Logs the metrics and the first example's ground truth and retrieved premises to TensorBoard.
        """
        # Retrieval.
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, _ = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        num_with_premises = 0
        tb = self.logger.experiment

        for i, (all_pos_premises, premises) in enumerate(
            zip_strict(batch["all_pos_premises"], retrieved_premises)
        ):
            # Only log the first example in the batch.
            if i == 0:
                msg_gt = "\n\n".join([p.serialize() for p in all_pos_premises])
                msg_retrieved = "\n\n".join(
                    [f"{j}. {p.serialize()}" for j, p in enumerate(premises)]
                )
                TP = len(set(premises).intersection(all_pos_premises))
                if len(all_pos_premises) == 0:
                    r = math.nan
                else:
                    r = float(TP) / len(all_pos_premises)
                msg = f"Recall@{self.num_retrieved}: {r}\n\nGround truth:\n\n```\n{msg_gt}\n```\n\nRetrieved:\n\n```\n{msg_retrieved}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            all_pos_premises = set(all_pos_premises)
            if len(all_pos_premises) == 0:
                logger.warning(f"No premises found for {batch['full_name']}")
                continue
            else:
                num_with_premises += 1
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premises))
                if premises[j] in all_pos_premises and not first_match_found:
                    MRR.append(1.0 / (j + 1))
                    first_match_found = True
            if not first_match_found:
                MRR.append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            logger.info(f"Recall@{j+1}_val: {recall[j]}")
            self.log(
                f"Recall@{j+1}_val",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=num_with_premises,
            )

        logger.info(f"MRR: {np.mean(MRR)}")
        self.log(
            "MRR",
            np.mean(MRR),
            on_epoch=True,
            sync_dist=True,
            batch_size=num_with_premises,
        )
        logger.info("End of validation_step")

    ##############
    # Prediction #
    ##############

    def on_predict_start(self) -> None:
        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True
        logger.info(f"Embeddings staled on predict start: {self.embeddings_staled}")
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)
        self.predict_step_outputs = []

    def predict_step(self, batch: Dict[str, Any], _):
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )

        for (
            url,
            commit,
            file_path,
            full_name,
            start,
            tactic_idx,
            ctx,
            pos_premises,
            premises,
            s,
        ) in zip_strict(
            batch["url"],
            batch["commit"],
            batch["file_path"],
            batch["full_name"],
            batch["start"],
            batch["tactic_idx"],
            batch["context"],
            batch["all_pos_premises"],
            retrieved_premises,
            scores,
        ):
            self.predict_step_outputs.append(
                {
                    "url": url,
                    "commit": commit,
                    "file_path": file_path,
                    "full_name": full_name,
                    "start": start,
                    "tactic_idx": tactic_idx,
                    "context": ctx,
                    "all_pos_premises": pos_premises,
                    "retrieved_premises": premises,
                    "scores": s,
                }
            )

    def on_predict_epoch_end(self) -> None:
        if self.trainer.log_dir is not None:
            logger.info("About to construct predictions map")
            gpu_id = self.trainer.local_rank

            preds_map = {
                (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
                for p in self.predict_step_outputs
            }

            path = f"test_pickle_{gpu_id}.pkl"
            with open(path, "wb") as oup:
                pickle.dump(preds_map, oup)
            logger.info(f"Retrieval predictions saved to {path}")

        self.predict_step_outputs.clear()

        if self.trainer.is_global_zero:
            logger.info("All GPUs have completed their predictions and saved the data.")

    def retrieve(
        self,
        state: List[str],
        file_name: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        k: int,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        self.reindex_corpus(batch_size=32)

        ctx = [
            Context(*_)
            for _ in zip_strict(file_name, theorem_full_name, theorem_pos, state)
        ]
        ctx_tokens = self.tokenizer(
            [_.serialize() for _ in ctx],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        context_emb = self._encode(
            ctx_tokens.input_ids.to(self.device),
            ctx_tokens.attention_mask.to(self.device),
        )

        if self.corpus_embeddings.device != context_emb.device:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.device)
        if self.corpus_embeddings.dtype != context_emb.dtype:
            self.corpus_embeddings = self.corpus_embeddings.to(context_emb.dtype)

        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            ctx,
            context_emb,
            k,
        )
        return retrieved_premises, scores
