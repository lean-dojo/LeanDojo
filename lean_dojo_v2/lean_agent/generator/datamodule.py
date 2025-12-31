"""Data module for the tactic generator."""

import json
import os
import pickle
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, ByT5Tokenizer

from lean_dojo_v2.lean_agent.common import (
    Batch,
    Corpus,
    Example,
    format_augmented_state,
    format_state,
    format_tactic,
)
from lean_dojo_v2.utils.constants import remove_marks


class GeneratorDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing data for a generator model that produces tactics given proof states.

    This dataset handles loading examples from a JSON file, formatting states and tactics,
    and optionally augmenting states with retrieved premises.

    Attributes:
        corpus (Corpus): The corpus containing the proof data.
        keep_marks (bool): Whether to keep markup in the tactics and states.
        preds (List[Dict[str, Any]]): Predictions for augmenting states with retrieved premises.
        max_inp_seq_len (int): Maximum input sequence length for states.
        max_oup_seq_len (int): Maximum output sequence length for tactics.
        p_drop (float): Probability to drop retrieved premises during training.
        tokenizer (ByT5Tokenizer): Tokenizer for encoding states and tactics.
        is_train (bool): Whether this dataset is used for training.
        data (List[Example]): The loaded and processed examples.
    """

    def __init__(
        self,
        data_path: str,
        corpus: Corpus,
        keep_marks: bool,
        preds: List[Dict[str, Any]],
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        normalize_tactics: bool,
        tokenizer: ByT5Tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.keep_marks = keep_marks
        self.preds = preds
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = self._load_data(data_path, normalize_tactics)

    def _load_data(self, data_path: str, normalize_tactics: bool) -> List[Example]:
        data = []
        for thm in tqdm(json.load(open(data_path))):
            for tac in thm["traced_tactics"]:
                if "annotated_tactic" in tac:
                    tactic = format_tactic(*tac["annotated_tactic"], normalize_tactics)
                else:
                    tactic = format_tactic(tac["tactic"], [], normalize_tactics)
                if not self.keep_marks:
                    tactic = remove_marks(tactic)
                data.append(
                    {
                        "url": thm["url"],
                        "commit": thm["commit"],
                        "file_path": thm["file_path"],
                        "full_name": thm["full_name"],
                        "state": format_state(tac["state_before"]),
                        "tactic": tactic,
                    }
                )

        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]

        if self.preds is not None:
            file_path = ex["file_path"]
            pred = self.preds[(file_path, ex["full_name"], ex["state"])]
            ex["state"] = format_augmented_state(
                ex["state"],
                pred["retrieved_premises"],
                self.max_inp_seq_len,
                self.p_drop if self.is_train else 0.0,
            )

        if not self.keep_marks:
            ex["state"] = remove_marks(ex["state"])

        return ex

    def collate(self, examples: List[Example]) -> Batch:
        state = [ex["state"] for ex in examples]
        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_inp_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tactic = [ex["tactic"] for ex in examples]
        tokenized_tactic = self.tokenizer(
            tactic,
            padding="longest",
            max_length=self.max_oup_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tactic_ids = tokenized_tactic.input_ids
        tactic_ids[tactic_ids == self.tokenizer.pad_token_id] = -100

        batch = {}
        batch["state"] = state
        batch["state_ids"] = tokenized_state.input_ids
        batch["state_mask"] = tokenized_state.attention_mask
        batch["tactic"] = tactic
        batch["tactic_ids"] = tactic_ids
        batch["tactic_mask"] = tokenized_tactic.attention_mask

        # Copy other fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        keep_marks: bool,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        normalize_tactics: bool,
        num_workers: int,
        corpus_path: Optional[str] = None,
        preds_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        if corpus_path is not None:
            self.corpus = Corpus(corpus_path)
        else:
            self.corpus = None
        self.keep_marks = keep_marks
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.normalize_tactics = normalize_tactics
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if preds_path is None:
            logger.info("Without retrieval data")
            self.preds = None
        else:
            logger.info("With retrieval data")
            self.preds = {}
            for pred in pickle.load(open(preds_path, "rb")):
                ctx = pred["context"]
                self.preds[ctx.path, ctx.theorem_full_name, ctx.state] = pred

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = GeneratorDataset(
                os.path.join(self.data_path, "train.json"),
                self.corpus,
                self.keep_marks,
                self.preds,
                self.max_inp_seq_len,
                self.max_oup_seq_len,
                self.p_drop,
                self.normalize_tactics,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GeneratorDataset(
                os.path.join(self.data_path, "val.json"),
                self.corpus,
                self.keep_marks,
                self.preds,
                self.max_inp_seq_len,
                self.max_oup_seq_len,
                self.p_drop,
                self.normalize_tactics,
                self.tokenizer,
                is_train=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
