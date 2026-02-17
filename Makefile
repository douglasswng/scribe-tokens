PYTHON := uv run

# --- Linting & Formatting ---
format:
	ruff format

check:
	ruff check --fix

format-check:
	make format
	make check

# --- Preprocessing ---
parse-iam:
	$(PYTHON) -m scripts.preprocess.parse_iam

parse-deepwriting:
	$(PYTHON) -m scripts.preprocess.parse_deepwriting

split-deepwriting:
	$(PYTHON) -m scripts.preprocess.split_deepwriting

preprocess-iam: parse-iam
preprocess-deepwriting: parse-deepwriting split-deepwriting

# --- Tokenizer ---
train-tokenisers:
	$(PYTHON) -m scripts.train.tokenisers

eval-compression:
	$(PYTHON) -m scripts.eval.compression

eval-oov:
	$(PYTHON) -m scripts.eval.oov

tokeniser-eval: eval-compression eval-oov

# --- Training ---
train:
	$(PYTHON) -m scripts.train.main --all

train-test:
	$(PYTHON) -m scripts.train.main --all --test

train-parallel:
	bash scripts/train/parallel.sh

# --- Evaluating ---
eval:
	$(PYTHON) -m scripts.eval.main --all

eval-htr:
	$(PYTHON) -m scripts.eval.htr

eval-htg:
	$(PYTHON) -m scripts.eval.htg

# --- Plotting ---
plot:
	$(PYTHON) -m scripts.plot.compression
	$(PYTHON) -m scripts.plot.oov
	$(PYTHON) -m scripts.plot.discretization
	$(PYTHON) -m scripts.plot.double_descent
	$(PYTHON) -m scripts.plot.attention
	$(PYTHON) -m scripts.plot.convergence
	$(PYTHON) -m scripts.plot.results

plot-compression:
	$(PYTHON) -m scripts.plot.compression

plot-oov:
	$(PYTHON) -m scripts.plot.oov

plot-discretization:
	$(PYTHON) -m scripts.plot.discretization

plot-double-descent:
	$(PYTHON) -m scripts.plot.double_descent

plot-attention:
	$(PYTHON) -m scripts.plot.attention

plot-convergence:
	$(PYTHON) -m scripts.plot.convergence

plot-results:
	$(PYTHON) -m scripts.plot.results

# --- Utilities ---
move-checkpoints:
	$(PYTHON) -m scripts.utils.move_best_checkpoint

fetch-metrics:
	$(PYTHON) -m scripts.utils.fetch_metrics

fetch-compute-time:
	$(PYTHON) -m scripts.utils.fetch_compute_time

kill:
	pgrep -f scribe-tokens | xargs kill -9

check-cuda:
	source .venv/bin/activate
	$(PYTHON) -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

tmux:
	tmux new-session -A -s train

-include local.mk
