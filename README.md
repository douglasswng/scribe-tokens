# ScribeTokens

Implementation and experiment pipeline for **ScribeTokens**, the digital-ink tokenization proposed in the paper.

📄 [Paper (PDF)](docs/paper.pdf)

ScribeTokens represents pen trajectories with a fixed base vocabulary of 10 tokens:
- 8 directional unit-step tokens (Freeman-style chain directions)
- 2 pen-state tokens (`[DOWN]`, `[UP]`)

The core idea is to decompose stroke segments into unit pixel steps (via Bresenham decomposition), then apply BPE over this base alphabet. This keeps tokenization OOV-free at the base level while still allowing strong compression and stable cross-entropy training.

## Quick Start

### 1) Install dependencies

```bash
uv sync
```

This project expects Python 3.13 (see `pyproject.toml`).

### 2) Python path

Scripts import modules from `src/`.

- In VS Code integrated terminal, `.vscode/settings.json` already sets `PYTHONPATH` to `${workspaceFolder}/src`.
- Outside VS Code, set it manually:

```bash
export PYTHONPATH="$PWD/src"
```

### 3) Select dataset

Update `DATASET` in:
- `src/constants.py`

Valid values:
- `"iam"`
- `"deepwriting"`

Most scripts read paths from that constant and some preprocess scripts assert a specific dataset value.

## Directory Structure (Brief)

```text
scribe-tokens/
├── data/
│   ├── iam/                # raw, parsed, and split files for IAM
│   └── deepwriting/        # raw, parsed, and split files for DeepWriting
├── models/                 # exported best weights per dataset/task/repr
├── output/
│   ├── results/            # CSV metrics
│   ├── tables/             # LaTeX tables
│   └── figures/            # PDF figures
├── scripts/
│   ├── preprocess/         # parse raw datasets into parsed JSON + splits
│   ├── train/              # model/tokenizer training entrypoints
│   ├── eval/               # tokenizer + HTR/HTG evaluation
│   ├── plot/               # tables + plotting scripts
│   └── utils/              # project utilities
├── src/                    # core library code (models, tokenizers, loaders)
├── tests/                  # unit tests
├── tokenisers/             # trained tokenizers
└── Makefile
```

## Makefile Commands

Run from repo root:

```bash
make <target>
```

| Target | What it does |
|---|---|
| `format` | `ruff format` |
| `check` | `ruff check --fix` |
| `format-check` | Runs both `format` and `check` (this mutates files) |
| `train` | Trains all default model/task combinations (`scripts.train.main --all`) |
| `train-test` | Quick test run (`--all --test`) |
| `train-parallel` | Runs `scripts/train/parallel.sh` |
| `eval` | Evaluates all supported tasks (`scripts.eval.main --all`) |
| `plot` | Generates all figures (`output/figures/`) and tables (`output/tables/`) |
| `kill` | Kills processes matching `scribe-tokens` |
| `check-cuda` | Prints CUDA availability/device via PyTorch |
| `setup-lambda` | Runs Lambda-specific environment setup script |
| `tmux` | Opens/attaches tmux session named `train` |

### Important note on `train-parallel`

`scripts/train/parallel.sh` contains a hard-coded project path:

```bash
cd /home/ubuntu/projects/scribe-tokens
```

Update it for your machine before using `make train-parallel`.

## Script Usage

All commands below assume `PYTHONPATH` is set (automatically in VS Code integrated terminal, or manually via `export PYTHONPATH="$PWD/src"`).

### Preprocessing

#### IAM

Parses IAM raw XML into `data/iam/parsed/*.json` and generates split files.

```bash
uv run python -m scripts.preprocess.parse_iam
```

Expected IAM raw layout (under `data/iam/raw/`):
- `lineStrokes/`
- `original/`
- `trainset.txt`
- `testset_v.txt`
- `testset_t.txt`
- `testset_f.txt`

#### DeepWriting

Parses DeepWriting JSON into parsed word-level samples.

```bash
uv run python -m scripts.preprocess.parse_deepwriting
```

Then create random train/val/test splits:

```bash
uv run python -m scripts.preprocess.split_deepwriting
```

### Tokenizer Analysis

Use this sweep/eval flow to benchmark tokenizers.

```bash
# 1) train tokenizer families (delta/vocab sweep)
uv run python -m scripts.train.tokenisers

# 2) evaluate tokenizer quality metrics
uv run python -m scripts.eval.compression
uv run python -m scripts.eval.oov

# 3) plot tokenizer metrics
uv run python -m scripts.plot.compression
uv run python -m scripts.plot.oov
```

Outputs:
- `output/results/compression.csv`
- `output/results/oov.csv`
- `output/figures/compression.pdf`
- `output/figures/oov.pdf`

### Deep Learning Training

#### Unified trainer

```bash
# Train all default models/tasks
uv run python -m scripts.train.main --all

# Single model
uv run python -m scripts.train.main --task HTR --repr scribe

# Quick test mode
uv run python -m scripts.train.main --all --test

# Optional overrides
uv run python -m scripts.train.main --all --epochs 50 --batch-size 16
```

CLI options (from `scripts/train/main.py`):
- `--all` or `--task <TASK>` (mutually exclusive)
- `--repr <scribe|point5|rel|text>` (required when using `--task`)
- `--test`
- `--experiment-name <name>`
- `--epochs <int>`
- `--batch-size <int>`

Supported tasks:
- `HTR`, `HTG`, `NTP`, `HTR_SFT`, `HTG_SFT`

### Evaluation

#### Unified evaluator

```bash
# All supported tasks
uv run python -m scripts.eval.main --all

# One task
uv run python -m scripts.eval.main --task HTR
```

Writes CSVs to `output/results/` (for example `htr.csv`, `htg.csv`).

#### Direct task evaluators

```bash
uv run python -m scripts.eval.htr
uv run python -m scripts.eval.htg
```

### Tables and Plots

#### CSV -> LaTeX tables

```bash
# all result CSVs in output/results/
uv run python -m scripts.plot.results

# one task
uv run python -m scripts.plot.results --task HTR
```

Writes to `output/tables/*.tex`.

#### Convergence speedup table

```bash
uv run python -m scripts.plot.convergence
```

Reads `output/results/metrics.csv`, writes `output/tables/conv_*.tex`.

#### Discretization figure

```bash
uv run python -m scripts.plot.discretization
```

Writes `output/figures/discretization.pdf`.

#### Double-descent figure

```bash
uv run python -m scripts.plot.double_descent
```

Writes `output/figures/double_descent.pdf`.

#### Attention visualization figure(s)

```bash
uv run python -m scripts.plot.attention
```

Writes attention PDFs under `output/figures/`.

### Utilities

#### Move best checkpoint weights into `models/`

```bash
uv run python -m scripts.utils.move_best_checkpoint
```

#### Fetch SwanLab run metrics to CSV

```bash
uv run python -m scripts.utils.fetch_metrics
```

Writes `output/results/metrics.csv`.

#### Print total compute time from SwanLab runs

```bash
uv run python -m scripts.utils.fetch_compute_time
```

#### Lambda setup helper

```bash
bash scripts/utils/setup_lambda.sh
```

## Typical Workflow

### IAM

```bash
# needed only outside VS Code integrated terminal
export PYTHONPATH="$PWD/src"
# set DATASET="iam" in src/constants.py

uv run python -m scripts.preprocess.parse_iam
uv run python -m scripts.train.tokenisers
uv run python -m scripts.eval.compression
uv run python -m scripts.eval.oov
make train
make eval
uv run python -m scripts.plot.results
```

### DeepWriting

```bash
# needed only outside VS Code integrated terminal
export PYTHONPATH="$PWD/src"
# set DATASET="deepwriting" in src/constants.py

uv run python -m scripts.preprocess.parse_deepwriting
uv run python -m scripts.preprocess.split_deepwriting
uv run python -m scripts.train.tokenisers
uv run python -m scripts.eval.compression
uv run python -m scripts.eval.oov
make train
make eval
uv run python -m scripts.plot.results
```

## Notes

- Run commands from the repository root.
- Existing trained models are skipped by default in training; remove saved model files to force retraining.
- Some utility/plot scripts assume prior artifacts exist (trained models, result CSVs, metrics CSVs).
