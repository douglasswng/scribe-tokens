# --- Linting & Formatting ---
format:
	ruff format

check:
	ruff check --fix

format-check:
	make format
	make check

# --- Training ---
train:
	.venv/bin/python -m scripts.train.main --all

train-test:
	.venv/bin/python -m scripts.train.main --all --test

train-parallel:
	bash scripts/train/parallel.sh

# --- Evaluating ---
eval:
	.venv/bin/python -m scripts.eval.main --all

# --- Utilities ---
kill:
	pgrep -f scribe-tokens | xargs kill -9

check-cuda:
	source .venv/bin/activate
	.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

setup-lambda:
	bash scripts/utils/setup_lambda.sh

tmux:
	tmux new-session -A -s train