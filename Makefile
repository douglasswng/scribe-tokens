format:
	ruff format

check:
	ruff check --fix

format-check:
	make format
	make check

kill:
	pgrep -f scribe-tokens | xargs kill -9

test-dist:
	torchrun --nproc_per_node=2 -m scripts.fake_train 

train:
	uv run python -m scripts.main_train

check-cuda:
	source .venv/bin/activate
	uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

train-lambda:
	.venv/bin/python -m scripts.main_train

train-local:
	uv run python -m scripts.main_train

test-local-dist:
	torchrun --nproc_per_node=2 -m scripts.fake_train

setup-lambda:
	bash scripts/setup_lambda.sh