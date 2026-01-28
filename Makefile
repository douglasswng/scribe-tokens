.PHONY: format check format-check kill test-dist train check-cuda test-lambda train-lambda train-local test-local-dist setup-lambda

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
	uv run python -m scripts.main_train

train-local:
	uv run python -m scripts.main_train

train-lambda:
	.venv/bin/python -m scripts.main_train

# --- Testing ---
test-dist:
	torchrun --nproc_per_node=2 -m scripts.fake_train 

test-local-dist:
	torchrun --nproc_per_node=2 -m scripts.fake_train

test-lambda:
	.venv/bin/python -m scripts.fake_train

# --- Utilities ---
kill:
	pgrep -f scribe-tokens | xargs kill -9

check-cuda:
	source .venv/bin/activate
	uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

setup-lambda:
	bash scripts/setup_lambda.sh