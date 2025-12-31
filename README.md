# LeanDojo-v2

LeanDojo-v2 is an end-to-end framework for training, evaluating, and deploying AI-assisted theorem provers for Lean 4. It combines repository tracing, lifelong dataset management, retrieval-augmented agents, Hugging Face fine-tuning, and external inference APIs into one toolkit.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Repository Layout](#repository-layout)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Environment Setup](#environment-setup)
7. [Quick Start](#quick-start)
8. [Working with Agents and Trainers](#working-with-agents-and-trainers)
9. [Tracing and Dataset Generation](#tracing-and-dataset-generation)
10. [LeanProgress Step-Prediction](#leanprogress-step-prediction)
11. [Proving Theorems](#proving-theorems)
12. [Testing](#testing)
13. [Troubleshooting & Tips](#troubleshooting--tips)
14. [Contributing](#contributing)
15. [License](#license)

---

## Overview

LeanDojo-v2 extends the original LeanDojo stack with the LeanAgent lifelong learning pipeline. It automates the entire loop of:

1. Cloning Lean repositories (GitHub or local) and tracing them with Lean instrumentation.
2. Storing structured theorem information in a dynamic database.
3. Training agent policies with supervised fine-tuning (SFT), GRPO-style RL, or retrieval objectives.
4. Driving Pantograph-based provers to fill in sorrys or verify solutions.
5. Using HuggingFace API for large model inference.

The codebase is modular: you can reuse the tracing pipeline without the agents, swap in custom trainers, or stand up your own inference service via the external API layer.

---

## Key Features

- **Unified Agent Abstractions**: `BaseAgent` orchestrates repository setup, training, and proving. Concrete implementations (`HFAgent`, `LeanAgent`, and `ExternalAgent`) tailor the workflow to Hugging Face models, retrieval-based provers, or REST-backed models.
- **Powerful Trainers**: `SFTTrainer`, `GRPOTrainer`, and `RetrievalTrainer` cover LoRA-enabled supervised fine-tuning, group-relative policy optimization, and retriever-only curriculum learning.
- **Multi-Modal Provers**: `HFProver`, `RetrievalProver`, and `ExternalProver` run on top of Pantograph’s Lean RPC server to search for tactics, generate whole proofs, or delegate to custom models.
- **Lean Tracing Pipeline**: `lean_dojo` includes the Lean 4 instrumentation (`ExtractData.lean`) and Python utilities to trace commits, normalize ASTs, and cache proof states.
- **Dynamic Repository Database**: `database` tracks repositories, theorems, curriculum difficulty, and sorry status, enabling lifelong training schedules.
- **External API**: The `external_api` folder exposes HTTP endpoints (FastAPI + uvicorn) and Lean frontend snippets so you can query LLMs from Lean editors.

---

## Repository Layout

| Path | Description |
|------|-------------|
| `lean_dojo_v2/agent/` | Base class plus `HFAgent`, `LeanAgent`, and helpers to manage repositories and provers. |
| `lean_dojo_v2/trainer/` | SFT, GRPO, and retrieval trainers with Hugging Face + DeepSpeed integration. |
| `lean_dojo_v2/prover/` | Pantograph-based prover implementations (HF, retrieval, external). |
| `lean_dojo_v2/lean_dojo/` | Lean tracing, dataset generation, caching, and AST utilities. |
| `lean_dojo_v2/lean_agent/` | Lifelong learning pipeline (configs, database, retrieval stack, generator). |
| `lean_dojo_v2/external_api/` | LeanCopilot code (Lean + Python server) to query external models. |
| `lean_dojo_v2/utils/` | Shared helpers for Git, filesystem operations, and constants. |
| `lean_dojo_v2/tests/` | Pytest regression suite. |

For deeper documentation on the lifelong learning component, see `lean_dojo_v2/lean_agent/README.md`.

---

## Requirements

- Python ≥ 3.11.
- CUDA-capable GPU for training and inference (tested with CUDA 12.6).
- Git ≥ 2.25 and `wget`.
- [elan](https://github.com/leanprover/elan) Lean toolchain to trace repositories locally.
- Adequate disk space for the `raid/` working directory (datasets, checkpoints, traces).

Python dependencies are declared in `pyproject.toml` and include PyTorch, PyTorch Lightning, Transformers, DeepSpeed, TRL, PEFT, and more.

---

## Installation

### Option 1: From PyPI

```sh
# Install the core package
pip install lean-dojo-v2

# Pantograph is required for Lean RPC
pip install git+https://github.com/stanford-centaur/PyPantograph

# Install a CUDA-enabled torch build (adjust the index URL for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Option 2: From Source (development)

```sh
git clone https://github.com/lean-dojo/LeanDojo-v2.git
cd LeanDojo-v2
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pip install git+https://github.com/stanford-centaur/PyPantograph
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> Tip: You can use [uv](https://github.com/astral-sh/uv) (`uv pip install lean-dojo-v2`) as an alternative Python package manager.

---

## Environment Setup

1. **GitHub Access Token (required)**  
   The tracing pipeline calls the GitHub API extensively. Create a personal access token and export it before running any agent:

   ```sh
   export GITHUB_ACCESS_TOKEN=<token>
   ```

2. **Hugging Face Token (optional but needed for gated models)**  

   ```sh
   export HF_TOKEN=<hf-token>
   ```

3. **Working directories**  
   By default all datasets, caches, and checkpoints live under `<repo>/raid`. Change the layout by editing `lean_dojo_v2/utils/constants.py` or by pointing `RAID_DIR` to faster storage.

4. **Lean toolchains**  
   Ensure `elan` is configured and Lean 4 (e.g., `leanprover/lean4:nightly`) is available on your `$PATH`. The tracing scripts look under `~/.elan/toolchains/`.

---

## Quick Start

```python
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir="outputs-deepseek",
    epochs_per_repo=1,
    batch_size=2,
    lr=2e-5,
)

agent = HFAgent(trainer=trainer)
agent.setup_github_repository(url=url, commit=commit)
agent.train()
agent.prove()
```

This example:

1. Downloads and traces the target Lean repository + commit.
2. Builds a supervised dataset from sorry theorems.
3. Fine-tunes the specified Hugging Face model (optionally with LoRA).
4. Launches an `HFProver` backed by Pantograph to search for proofs.

## Tracing and Dataset Generation

The `lean_dojo_v2/lean_dojo/data_extraction` package powers repository tracing:

- `lean.py` clones repositories (GitHub, remote, or local), validates Lean versions, and normalizes URLs.
- `trace.py` drives Lean with the custom `ExtractData.lean` instrumented module to capture theorem states.
- `dataset.py` converts traced files to JSONL datasets ready for trainers.
- `cache.py` memoizes repository metadata to avoid redundant downloads.
- `traced_data.py` exposes typed wrappers for traced AST nodes and sorrys.

Typical usage:

```python
from lean_dojo_v2.database import DynamicDatabase

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"

database = DynamicDatabase()

database.setup_github_repository(
    url=url,
    commit=commit,
    build_deps=False,
)
```

The generated artifacts flow into the `DynamicDatabase`, which keeps repositories sorted by difficulty and appends new sorrys without retracing everything.

## Working with Agents and Trainers

### Agents

Agents orchestrate the full workflow of repository setup, training, and theorem proving. Each agent pairs a trainer with a compatible prover.

#### `HFAgent`

Uses Hugging Face models fine-tuned with `SFTTrainer` or `GRPOTrainer` for theorem proving. Loads checkpoints locally and uses `HFProver` for proof search. Ideal for training custom models on your traced repositories. Does not build Lean dependencies by default.

```python
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

trainer = SFTTrainer(model_name="deepseek-ai/DeepSeek-Prover-V2-7B", ...)
agent = HFAgent(trainer=trainer)
agent.setup_github_repository(url, commit)
agent.train()  
agent.prove()   
```

#### `ExternalAgent`

Uses the Hugging Face Inference API to access large models like DeepSeek-Prover-V2-671B without local model loading. Pairs with `ExternalProver` for whole-proof generation or proof search. Best for quick experiments or when you don't have GPU resources for local inference.

```python
from lean_dojo_v2.agent.external_agent import ExternalAgent

agent = ExternalAgent()
agent.setup_github_repository(url, commit)
agent.prove()  
```

#### `LeanAgent`

Implements the lifelong learning pipeline with retrieval-augmented generation. Uses `RetrievalTrainer` to train premise retrievers, then pairs with `RetrievalProver` for retrieval-augmented tactic generation. Maintains repository curricula and builds Lean dependencies by default.

```python
from lean_dojo_v2.agent.lean_agent import LeanAgent

agent = LeanAgent()
agent.setup_github_repository(url, commit)
agent.train()  
agent.prove()   
```

### Trainers

#### Supervised Fine-Tuning (`SFTTrainer`)

- Accepts any Hugging Face causal LM identifier.
- Supports LoRA by passing a `peft.LoraConfig`.
- Key arguments: `epochs_per_repo`, `batch_size`, `max_seq_len`, `lr`, `warmup_steps`, `gradient_checkpointing`.
- Produces checkpoints under `output_dir` that the `HFProver` consumes.

#### GRPO Trainer (`GRPOTrainer`)

- Implements Group Relative Policy Optimization for reinforcement-style refinement.
- Accepts `reference_model`, `reward_weights`, and `kl_beta` settings.
- Useful for improving search policies on curated theorem batches.

#### Retrieval Trainer (`RetrievalTrainer`)

- Trains the dense retriever that scores prior proofs from the corpus.
- Used by `LeanAgent` to build retrieval-augmented generation models.
- Requires indexed corpus and generator checkpoints.

Each agent inherits `BaseAgent`, so you can implement your own by overriding `_get_build_deps()` and `_setup_prover()` to register new trainer/prover pairs.

## LeanProgress Step-Prediction

- Generate a JSONL dataset with remaining-step targets (or replace it with your own LeanProgress export):

  ```sh
  python -m lean_dojo_v2.lean_progress.create_sample_dataset --output raid/data/sample_leanprogress_dataset.jsonl
  ```

- Fine-tune a regression head that predicts `steps_remaining`:

  ```python
  from pathlib import Path

  from lean_dojo_v2.trainer.progress_trainer import ProgressTrainer

  sample_dataset_path = Path("raid/data/sample_leanprogress_dataset.jsonl")

  trainer = ProgressTrainer(
      model_name="bert-base-uncased",
      data_path=str(sample_dataset_path),
      output_dir="outputs-progress",
  )

  trainer.train()
  ```

## Proving Theorems

LeanDojo-v2 provides three prover implementations, each for different use cases:

### `HFProver`

Loads a fine-tuned Hugging Face model from a local checkpoint (supports full models and LoRA adapters) and generates tactics directly, used for locally trained Hugging Face model (e.g. with `SFTTrainer` and `GRPOTrainer`).

### `ExternalProver`

Performs inference with the Hugging Face Inference API to access large models without local GPU resources. Defaults to DeepSeek-Prover-V2-671B. Supports both proof search and whole-proof generation.

### `RetrievalProver`

Used directly with LeanAgent.

### Proof Methods

LeanDojo-v2 supports two methods for theorem proving:

- **Whole-proof generation**: generate complete proof in one forward pass of the prover.

  ```python
  from lean_dojo_v2.prover import ExternalProver

  theorem = "theorem my_and_comm : ∀ {p q : Prop}, And p q → And q p := by"
  prover = ExternalProver()
  proof = prover.generate_whole_proof(theorem)
  ```

- **Proof search**: generate tactics sequentially and update the goal state through interaction with Pantograph until the proof is complete.

  ```python
  from pantograph.server import Server
  from lean_dojo_v2.prover import HFProver

  server = Server()
  prover = HFProver(ckpt_path="outputs-deepseek")

  result, used_tactics = prover.search(
      server=server, goal="∀ {p q : Prop}, p ∧ q → q ∧ p", verbose=False
  )
  ```

## Testing

We use `pytest` for regression coverage.

```sh
pip install -e .[dev]          # make sure dev extras like pytest/trl are present
export GITHUB_ACCESS_TOKEN=<token>
export HF_TOKEN=<hf-token>     # only required for tests touching HF APIs
pytest -v
```

## Troubleshooting & Tips

- **401 Bad Credentials / rate limits**: Ensure `GITHUB_ACCESS_TOKEN` is exported and has `repo` + `read:org` scopes.
- **Lean tracing failures**: Confirm that the repo’s Lean version exists locally (`elan toolchain install <version>`).
- **Missing CUDA libraries**: Install the PyTorch wheel that matches your driver and CUDA version.
- **Dataset location**: The default `raid/` directory can grow large. Point it to high-throughput storage or use symlinks.
- **Pantograph errors**: Reinstall Pantograph from source (`pip install git+https://github.com/stanford-centaur/PyPantograph`) whenever Lean upstream changes.

---

## Contributing

Issues and pull requests are welcome! Please:

1. Open an issue describing the bug or feature.
2. Run formatters (`black`, `isort`) and `pytest` before submitting.
3. Mention if your change touches Lean tracing files so reviewers can re-generate artifacts.

---

## License

LeanDojo-v2 is released under the MIT License. See `LICENSE` for details.
