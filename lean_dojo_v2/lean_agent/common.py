import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import networkx as nx
import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from loguru import logger
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from transformers import get_cosine_schedule_with_warmup

from lean_dojo_v2.lean_dojo import Pos
from lean_dojo_v2.utils.common import zip_strict
from lean_dojo_v2.utils.constants import MARK_END_SYMBOL, MARK_START_SYMBOL

Example = Dict[str, Any]
Batch = Dict[str, Any]


@dataclass(unsafe_hash=True)
class Context:
    """Contexts are "queries" in our retrieval setup."""

    path: str
    theorem_full_name: str
    theorem_pos: Pos = field(compare=False)
    state: Optional[str] = None

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.theorem_full_name, str)
        assert isinstance(self.theorem_pos, Pos)
        if self.state is not None:
            if not (
                isinstance(self.state, str)
                and "⊢" in self.state
                and MARK_START_SYMBOL not in self.state
                and MARK_END_SYMBOL not in self.state
            ):
                logger.warning(f"Invalid state: {self.state}")
            assert (
                isinstance(self.state, str)
                and "⊢" in self.state
                and MARK_START_SYMBOL not in self.state
                and MARK_END_SYMBOL not in self.state
            )

    def serialize(self) -> str:
        """Serialize the context into a string for Transformers."""
        if self.state is None:
            return ""
        return self.state


def escape_regex_special_chars(text):
    return re.escape(text)


@dataclass(unsafe_hash=True)
class Premise:
    """Premises are "documents" in our retrieval setup."""

    path: str
    """The ``*.lean`` file this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    start: Pos = field(repr=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """

    code: str = field(compare=False)
    """Raw, human-written code for defining the premise.
    """

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.full_name, str)
        assert (
            isinstance(self.start, Pos)
            and isinstance(self.end, Pos)
            and self.start <= self.end
        )
        assert isinstance(self.code, str) and self.code != ""

    def serialize(self) -> str:
        """Serialize the premise into a string for Transformers."""
        annot_full_name = f"{MARK_START_SYMBOL}{self.full_name}{MARK_END_SYMBOL}"
        code = self.code.replace(f"_root_.{self.full_name}", annot_full_name)
        fields = self.full_name.split(".")

        for i in range(len(fields)):
            prefix = ".".join(fields[i:])
            escaped_prefix = escape_regex_special_chars(prefix)
            pattern = f"(?<=\\s)«?{escaped_prefix}»?"
            new_code = re.sub(pattern, annot_full_name, code)
            if new_code != code:
                code = new_code
                break

        return code


class PremiseSet:
    """A set of premises indexed by their paths and full names."""

    path2premises: Dict[str, Dict[str, Premise]]

    def __init__(self) -> None:
        self.path2premises = {}

    def __iter__(self) -> Generator[Premise, None, None]:
        for _, premises in self.path2premises.items():
            for p in premises.values():
                yield p

    def add(self, p: Premise) -> None:
        if p.path in self.path2premises:
            self.path2premises[p.path][p.full_name] = p
        else:
            self.path2premises[p.path] = {p.full_name: p}

    def update(self, premises: List[Premise]) -> None:
        for p in premises:
            self.add(p)

    def __contains__(self, p: Premise) -> bool:
        return (
            p.path in self.path2premises and p.full_name in self.path2premises[p.path]
        )

    def __len__(self) -> int:
        return sum(len(premises) for premises in self.path2premises.values())


@dataclass(frozen=True)
class File:
    """A file defines 0 or multiple premises."""

    path: str
    """Path of the ``*.lean`` file.
    """

    premises: List[Premise] = field(repr=False, compare=False)
    """A list of premises defined in this file.
    """

    @classmethod
    def from_data(cls, file_data: Dict[str, Any]) -> "File":
        """Construct a :class:`File` object from ``file_data``."""
        path = file_data["path"]
        premises = []
        for p in file_data["premises"]:
            full_name = p["full_name"]
            if full_name is None:
                continue
            if "user__.n" in full_name or p["code"] == "":
                # Ignore ill-formed premises (often due to errors in ASTs).
                continue
            if full_name.startswith("[") and full_name.endswith("]"):
                # Ignore mutual definitions.
                continue
            premises.append(
                Premise(
                    path, p["full_name"], Pos(*p["start"]), Pos(*p["end"]), p["code"]
                )
            )
        return cls(path, premises)

    @property
    def is_empty(self) -> bool:
        """Check whether the file contains no premise."""
        return self.premises == []


class Corpus:
    """Our retrieval corpus is a DAG of files. Each file consists of
    premises (theorems, definitoins, etc.) that can be retrieved.
    """

    transitive_dep_graph: nx.DiGraph
    """Transitive closure of the dependency graph among files. 
    There is an edge from file X to Y iff X import Y (directly or indirectly).
    """

    all_premises: List[Premise]
    """All premises in the entire corpus.
    """

    def __init__(self, jsonl_path: str) -> None:
        """Construct a :class:`Corpus` object from a ``corpus.jsonl`` data file."""
        dep_graph = nx.DiGraph()
        self.all_premises = []

        for line in open(jsonl_path):
            file_data = json.loads(line)
            path = file_data["path"]
            assert not dep_graph.has_node(path)
            file = File.from_data(file_data)

            dep_graph.add_node(path, file=file)
            self.all_premises.extend(file.premises)

            for p in file_data["imports"]:
                assert dep_graph.has_node(p)
                dep_graph.add_edge(path, p)

        assert nx.is_directed_acyclic_graph(dep_graph)
        self.transitive_dep_graph = nx.transitive_closure_dag(dep_graph)

        self.imported_premises_cache = {}
        self.fill_cache()

    def _get_file(self, path: str) -> File:
        # for some reason, the `path` in the parameter starts with ./
        # but the paths in the corpus don't
        # so we need to remove the ./
        if path.startswith("./"):
            path = path[2:]

        if path not in self.transitive_dep_graph:
            logger.warning(f"File {path} not found in dependency graph.")
            # Return an empty file to avoid KeyError
            return File(path=path, premises=[])

        return self.transitive_dep_graph.nodes[path]["file"]

    def __len__(self) -> int:
        return len(self.all_premises)

    def __contains__(self, path: str) -> bool:
        return path in self.transitive_dep_graph

    def __getitem__(self, idx: int) -> Premise:
        return self.all_premises[idx]

    @property
    def files(self) -> List[File]:
        return [self._get_file(p) for p in self.transitive_dep_graph.nodes]

    @property
    def num_files(self) -> int:
        return len(self.files)

    def get_dependencies(self, path: str) -> List[str]:
        """Return a list of (direct and indirect) dependencies of the file ``path``."""
        return list(self.transitive_dep_graph.successors(path))

    def get_premises(self, path: str) -> List[Premise]:
        """Return a list of premises defined in the file ``path``."""
        return self._get_file(path).premises

    def num_premises(self, path: str) -> int:
        """Return the number of premises defined in the file ``path``."""
        return len(self.get_premises(path))

    def locate_premise(self, path: str, pos: Pos) -> Optional[Premise]:
        """Return a premise at position ``pos`` in file ``path``.

        Return None if no such premise can be found.
        """
        for p in self.get_premises(path):
            # accounting for . vs ./ starting in the path
            assert (p.path == path) or (p.path == path[2:])
            if p.start <= pos <= p.end:
                return p
        return None

    def fill_cache(self) -> None:
        for path in self.transitive_dep_graph.nodes:
            self._get_imported_premises(path)

    def _get_imported_premises(self, path: str) -> List[Premise]:
        """Return a list of premises imported in file ``path``. The result is cached."""
        premises = self.imported_premises_cache.get(path, None)
        if premises is not None:
            return premises

        premises = []
        for p in self.transitive_dep_graph.successors(path):
            premises.extend(self._get_file(p).premises)
        self.imported_premises_cache[path] = premises
        return premises

    def get_accessible_premises(self, path: str, pos: Pos) -> PremiseSet:
        """Return the set of premises accessible at position ``pos`` in file ``path``,
        i.e., all premises defined in the (transitively) imported files or earlier in the same file.
        """
        premises = PremiseSet()
        for p in self.get_premises(path):
            if p.end <= pos:
                premises.add(p)
        premises.update(self._get_imported_premises(path))
        return premises

    def get_accessible_premise_indexes(self, path: str, pos: Pos) -> List[int]:
        return [
            i
            for i, p in enumerate(self.all_premises)
            if (p.path == path and p.end <= pos)
            or self.transitive_dep_graph.has_edge(path, p.path)
        ]

    def get_nearest_premises(
        self,
        premise_embeddings: torch.FloatTensor,
        batch_context: List[Context],
        batch_context_emb: torch.Tensor,
        k: int,
    ) -> Tuple[List[List[Premise]], List[List[float]]]:
        """Perform a batch of nearest neighbour search."""
        similarities = batch_context_emb @ premise_embeddings.t()
        idxs_batch = similarities.argsort(dim=1, descending=True).tolist()
        results = [[] for _ in batch_context]
        scores = [[] for _ in batch_context]

        for j, (ctx, idxs) in enumerate(zip(batch_context, idxs_batch)):
            accessible_premises = self.get_accessible_premises(
                ctx.path, ctx.theorem_pos
            )
            for i in idxs:
                p = self.all_premises[i]
                if p in accessible_premises:
                    results[j].append(p)
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        break
            else:
                raise ValueError

        return results, scores


@dataclass(frozen=True)
class IndexedCorpus:
    """A corpus with premise embeddings."""

    corpus: Corpus
    embeddings: torch.FloatTensor

    def __post_init__(self):
        assert self.embeddings.device == torch.device("cpu")
        assert len(self.embeddings) == len(self.corpus)


def get_all_pos_premises(annot_tac, corpus: Corpus) -> List[Premise]:
    """Return a list of all premises that are used in the tactic ``annot_tac``."""
    _, provenances = annot_tac
    all_pos_premises = set()

    for prov in provenances:
        def_path = prov["def_path"]
        p = corpus.locate_premise(def_path, Pos(*prov["def_pos"]))
        if p is not None:
            all_pos_premises.add(p)

    return list(all_pos_premises)


_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)


def normalize_spaces(s: str) -> str:
    """Repalce any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub(" ", s).strip()


def format_tactic(annot_tac: str, provenances, normalize: bool) -> str:
    """Use full names for the all <a>...</a>."""
    if normalize:
        annot_tac = normalize_spaces(annot_tac)
    if len(provenances) == 0:
        return annot_tac

    tac = ""
    marks = list(re.finditer(r"<a>(?P<ident>.+?)</a>", annot_tac))

    for i, (m, prov) in enumerate(zip_strict(marks, provenances)):
        last_end = marks[i - 1].end() if i > 0 else 0
        tac += annot_tac[last_end : m.start()] + "<a>" + prov["full_name"] + "</a>"

    tac += annot_tac[marks[-1].end() :]
    return tac


def format_state(s: str) -> str:
    m = re.match(r"\d+ goals", s)
    if m is not None:
        return s[m.end() :].strip()
    else:
        return s


def format_augmented_state(
    s: str, premises: List[Premise], max_len: int, p_drop: float
) -> str:
    """Format a state with retrieved premises and drop some of them with probability ``p_drop``."""
    s = format_state(s)

    aug_s = ""
    length = 0
    max_premises_len = max_len - len(bytes(s.encode("utf-8")))

    for p in premises:
        if random.random() < p_drop:
            continue
        p_str = f"{p.serialize()}\n\n"
        l = len(bytes(p_str.encode("utf-8")))
        if length + l > max_premises_len:
            continue
        length += l
        aug_s = p_str + aug_s

    aug_s += s
    return aug_s


def get_optimizers(
    parameters, trainer: pl.Trainer, lr: float, warmup_steps: int
) -> Dict[str, Any]:
    """Return an AdamW optimizer with cosine warmup learning rate schedule."""
    strategy = trainer.strategy

    if isinstance(strategy, DeepSpeedStrategy):
        if "offload_optimizer" in strategy.config["zero_optimization"]:
            logger.info("Optimizing with DeepSpeedCPUAdam")
            optimizer = DeepSpeedCPUAdam(parameters, lr=lr, adamw_mode=True)
        else:
            logger.info("Optimizing with FusedAdam")
            optimizer = FusedAdam(parameters, lr=lr, adam_w_mode=True)
    else:
        logger.info("Optimizing with AdamW")
        optimizer = torch.optim.AdamW(parameters, lr=lr)

    if trainer.max_steps != -1:
        max_steps = trainer.max_steps
    else:
        assert trainer.max_epochs is not None
        max_steps = (
            trainer.max_epochs
            * len(trainer.datamodule.train_dataloader())
            // trainer.accumulate_grad_batches
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }
