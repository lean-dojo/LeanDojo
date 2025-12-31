"""Lightning module for the tactic generator."""

import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import openai
import pytorch_lightning as pl
import torch
from loguru import logger
from torchmetrics import Metric
from transformers import AutoTokenizer, T5ForConditionalGeneration

from lean_dojo_v2.lean_agent.common import (
    IndexedCorpus,
    format_augmented_state,
    get_optimizers,
)
from lean_dojo_v2.lean_agent.retrieval.model import PremiseRetriever
from lean_dojo_v2.lean_dojo import Pos
from lean_dojo_v2.utils.common import load_checkpoint, zip_strict
from lean_dojo_v2.utils.constants import remove_marks
from lean_dojo_v2.utils.filesystem import remove_dir

torch.set_float32_matmul_precision("medium")


class TopkAccuracy(Metric):
    """
    A metric class for calculating top-k accuracy for text predictions.

    This metric evaluates whether the ground truth string is present within the top k predicted strings.
    The strings are processed by removing marks before comparison.

    Attributes:
        is_differentiable (Optional[bool]): Indicates if the metric is differentiable. Default is False.
        higher_is_better (Optional[bool]): Indicates if higher values are better. Default is True.
        full_state_update (bool): Whether to update the state completely. Default is True.
        k (int): The number of top predictions to consider.
        correct (torch.Tensor): Running count of correct predictions.
        total (torch.Tensor): Running count of total predictions.

    Methods:
        update(batch_preds, batch_gt): Updates the state with batch statistics.
        compute(): Computes the accuracy based on collected state.
    """

    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch_preds: List[List[str]], batch_gt: List[str]):
        assert len(batch_preds) == len(batch_gt)
        for preds, gt in zip(batch_preds, batch_gt):
            # This still doesn't account for short names vs. full names.
            gt = remove_marks(gt)
            preds = [remove_marks(p) for p in preds]
            self.correct += gt in preds[: self.k]
        self.total += len(batch_gt)

    def compute(self) -> float:
        return self.correct.float() / self.total


class TacticGenerator(ABC):
    """A tactic generator takes a state and generates multiple tactic candidates."""

    @abstractmethod
    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError


class RetrievalAugmentedGenerator(TacticGenerator, pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        eval_num_retrieved: int,
        eval_num_workers: int,
        eval_num_gpus: int,
        eval_num_theorems: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        length_penalty: float = 0.0,
        ret_ckpt_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the RetrievalAugmentedGenerator.

        The generator can optionally use a retriever to augment the generation process.

        Parameters
        ----------
        model_name : str
            Name of the pre-trained model to use for generation
        lr : float
            Learning rate for the optimizer
        warmup_steps : int
            Number of warmup steps for learning rate scheduler
        num_beams : int
            Number of beams to use for beam search during generation
        eval_num_retrieved : int
            Number of premises to retrieve during evaluation
        eval_num_workers : int
            Number of worker processes for evaluation
        eval_num_gpus : int
            Number of GPUs to use for evaluation
        eval_num_theorems : int
            Number of theorems to evaluate on
        max_inp_seq_len : int
            Maximum input sequence length
        max_oup_seq_len : int
            Maximum output sequence length
        length_penalty : float, optional
            Length penalty for beam search, by default 0.0
        ret_ckpt_path : Optional[str], optional
            Path to the retriever checkpoint, by default None.
            If None, the generator will not use retrieval augmentation.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.eval_num_retrieved = eval_num_retrieved
        self.eval_num_workers = eval_num_workers
        self.eval_num_gpus = eval_num_gpus
        self.eval_num_theorems = eval_num_theorems
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len

        config = {
            "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
            "lr": 1e-3,
            "warmup_steps": 1000,
            "max_seq_len": 512,
            "num_retrieved": 100,
        }

        if ret_ckpt_path is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            logger.info(f"Loading the retriever from {ret_ckpt_path}")
            self.retriever = PremiseRetriever.load(
                ret_ckpt_path, self.device, freeze=True, config=config
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name)

        self.topk_accuracies = dict()
        for k in range(1, num_beams + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"top{k}_acc_val", acc)

    @classmethod
    def load(
        cls, ckpt_path: str, device, freeze: bool, config: dict
    ) -> "RetrievalAugmentedGenerator":
        return load_checkpoint(cls, ckpt_path, device, freeze, config)

    def forward(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        tactic_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=tactic_ids,
        ).loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["tactic_ids"],
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        self._log_io_texts("train", batch["state_ids"], batch["tactic_ids"])
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    def _log_io_texts(
        self,
        split: str,
        state_ids: torch.LongTensor,
        tactic_ids: torch.LongTensor,
    ) -> None:
        tb = self.logger.experiment
        inp = self.tokenizer.decode(state_ids[0], skip_special_tokens=True)
        oup_ids = torch.where(
            tactic_ids[0] == -100, self.tokenizer.pad_token_id, tactic_ids[0]
        )
        oup = self.tokenizer.decode(oup_ids, skip_special_tokens=True)
        tb.add_text(f"{split}_state", f"```\n{inp}\n```", self.global_step)
        tb.add_text(f"{split}_tactic", f"`{oup}`", self.global_step)

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

        if self.retriever is not None:
            self.retriever.load_corpus(self.trainer.datamodule.corpus)

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        """
        Performs a validation step on a batch of data.

        The method computes the loss on the validation data, logs the loss, and generates
        tactic candidates using Beam Search. It also logs example inputs/outputs and
        calculates top-k accuracy metrics for the generated tactics.

        Args:
            batch: A dictionary containing batch data with the following keys:
                - state_ids: Tensor of input state token IDs
                - state_mask: Attention mask for state input
                - tactic_ids: Tensor of target tactic token IDs
                - tactic: List of reference tactic strings
            _: Batch index (unused)

        Returns:
            None

        Side effects:
            - Logs validation loss
            - Logs example inputs/outputs as text
            - Generates tactic predictions using beam search
            - Computes and logs top-k accuracy metrics
        """
        state_ids = batch["state_ids"]
        state_mask = batch["state_mask"]
        tactic_ids = batch["tactic_ids"]

        loss = self(state_ids, state_mask, tactic_ids)
        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._log_io_texts("val", state_ids, tactic_ids)

        # Generate topk tactic candidates via Beam Search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_oup_seq_len,
            num_beams=self.num_beams,
            do_sample=False,
            num_return_sequences=self.num_beams,
            early_stopping=False,
        )
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        batch_size = state_ids.size(0)
        assert len(output_text) == batch_size * self.num_beams
        tactics_pred = [
            output_text[i * self.num_beams : (i + 1) * self.num_beams]
            for i in range(batch_size)
        ]

        tb = self.logger.experiment
        msg = "\n".join(tactics_pred[0])
        tb.add_text(f"preds_val", f"```\n{msg}\n```", self.global_step)

        # Log the topk accuracies.
        for k in range(1, self.num_beams + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(tactics_pred, batch["tactic"])
            self.log(
                f"top{k}_acc_val",
                topk_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def on_validation_epoch_end(self) -> None:
        if self.eval_num_theorems == 0:
            return

        from lean_dojo_v2.lean_agent.prover.evaluate import (  # Avoid circular import.
            evaluate,
        )

        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last-tmp.ckpt"
        self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}. Evaluating...")
        torch.cuda.empty_cache()

        data_path = self.trainer.datamodule.data_path
        if self.retriever is None:
            acc = evaluate(
                data_path=data_path,
                num_workers=self.eval_num_workers,
                num_gpus=self.eval_num_gpus,
                num_theorems=self.eval_num_theorems,
                ckpt_path=ckpt_path,
            )
        else:
            self.retriever.reindex_corpus(self.trainer.datamodule.eval_batch_size)
            corpus_path = f"{self.trainer.log_dir}/checkpoints/indexed_corpus.pickle"
            pickle.dump(
                IndexedCorpus(
                    self.retriever.corpus, self.retriever.corpus_embeddings.cpu()
                ),
                open(corpus_path, "wb"),
            )
            acc = evaluate(
                data_path=data_path,
                num_workers=self.eval_num_workers,
                num_gpus=self.eval_num_gpus,
                num_theorems=self.eval_num_theorems,
                ckpt_path=ckpt_path,
                indexed_corpus_path=corpus_path,
            )

        self.log("Pass@1_val", acc, on_step=False, on_epoch=True, sync_dist=True)
        logger.info(f"Pass@1: {acc}")

        remove_dir(ckpt_path)

    ##############
    # Prediction #
    ##############

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return self.batch_generate(
            [state], [file_path], [theorem_full_name], [theorem_pos], num_samples
        )[0]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        """
        Generate multiple tactic candidates for a batch of Lean states using a generator model.

        This method processes a batch of Lean theorem states, optionally enhances them with retrieved premises,
        and generates multiple tactic candidates for each state using beam search.

        Args:
            state (List[str]): List of Lean states as strings.
            file_path (List[str]): List of file paths corresponding to each state.
            theorem_full_name (List[str]): List of fully qualified theorem names.
            theorem_pos (List[Pos]): List of position objects indicating locations in source files.
            num_samples (int): Number of tactic candidates to generate per state.

        Returns:
            List[List[Tuple[str, float]]]: A list of lists where each inner list contains tuples of
            (tactic_text, score) for each state. Duplicate tactics are removed.

        Note:
            If a retriever is configured, it will be used to augment states with relevant premises
            before generation.
        """
        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state,
                file_path,
                theorem_full_name,
                theorem_pos,
                self.eval_num_retrieved,
            )
            state = [
                format_augmented_state(s, premises, self.max_inp_seq_len, p_drop=0.0)
                for s, premises in zip_strict(state, retrieved_premises)
            ]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_inp_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # Generate tactic candidates using beam search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_oup_seq_len,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = remove_marks(raw_output_text[j])
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores


class GPT4TacticGenerator(TacticGenerator):
    def __init__(
        self,
        organization: str,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 1024,
        num_retries: int = 3,
        threshold: float = 0.9,
    ):
        super().__init__()
        openai.organization = organization
        openai.api_key = api_key
        self.model = model
        self.default_prompt = "You are an expert in theorem proving in Lean. We are trying to solve the Lean theorem 'THEOREM_FULL_NAME' from the mathlib file 'FILE_PATH'. The current tactic state is: 'TACTIC_STATE'. Suggest exactly NUM_SAMPLES unique tactics to progress in solving 'THEOREM_FULL_NAME', along with their confidence levels as a float between 0 and 1. Rank them in order of effectiveness. Present the tactics and their confidence levels as comma-separated tuples in this format: #(tactic_{1}, confidence_{1})#, #(tactic_{2}, confidence_{2})#, ..., #(tactic_{NUM_SAMPLES}, confidence_{NUM_SAMPLES})#."
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.threshold = threshold

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = (
            self.default_prompt.replace("TACTIC_STATE", state)
            .replace("FILE_PATH", file_path)
            .replace("THEOREM_FULL_NAME", theorem_full_name)
            .replace("NUM_SAMPLES", str(int(num_samples / self.threshold)))
        )
        logger.info(prompt)

        for _ in range(self.num_retries):
            response = None
            # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0,
                    max_tokens=self.max_tokens,
                    # stop="E:" #
                )
            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                logger.info(f"OpenAI API returned an API Error: {e}")
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                logger.info(f"Failed to connect to OpenAI API: {e}")
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                continue
            except Exception as e:
                logger.info(e)
                continue

            if response is None:
                continue

            logger.info(f"GPT-4 response: {response}")
            output = response["choices"][0]["message"]["content"]
            indices = []

            for i, c in enumerate(output):
                if c == "#":
                    indices.append(i)

            tactics_with_scores = []

            for i in range(1, len(indices), 2):
                tactic_and_confidence = output[indices[i - 1] + 1 : indices[i]].strip()

                try:
                    while tactic_and_confidence[0] == "(":
                        tactic_and_confidence = tactic_and_confidence[1:]

                    if tactic_and_confidence[-1] == ")":
                        tactic_and_confidence = tactic_and_confidence[:-1]

                    split_index = tactic_and_confidence.rindex(",")
                    tactic = tactic_and_confidence[:split_index].strip()
                    confidence = float(tactic_and_confidence[split_index + 1 :].strip())
                except Exception as e:
                    logger.info(e)
                    logger.info(
                        f"{self.model} output {output[indices[i-1]+1:indices[i]]} was not formatted correctly and could not be parsed."
                    )
                    continue

                tactics_with_scores.append((tactic, confidence))

            if len(tactics_with_scores) < int(self.threshold * num_samples):
                continue

            tactics_with_scores = sorted(
                tactics_with_scores, key=lambda x: x[1], reverse=True
            )[: min(num_samples, len(tactics_with_scores))]
            logger.debug(f"GPT-4 tactics: {tactics_with_scores}")
            logger.debug(
                f"GPT-4 tactic count requested: {num_samples} / {self.threshold} = {int(num_samples / self.threshold)}"
            )
            logger.debug(
                f"GPT-4 tactic count received and parsed: {len(tactics_with_scores)}"
            )
            return tactics_with_scores

        raise ValueError("GPT-4 outputs are unparsable.")

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, t, p, num_samples)
            for s, f, t, p in zip_strict(
                state, file_path, theorem_full_name, theorem_pos
            )
        ]


class FixedTacticGenerator(TacticGenerator):
    def __init__(self, tactic, module) -> None:
        self.tactic = tactic
        self.module = module

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return [(f"{{ {self.tactic} }}", 1.0)]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, tfn, tp, num_samples)
            for s, f, tfn, tp in zip(state, file_path, theorem_full_name, theorem_pos)
        ]
