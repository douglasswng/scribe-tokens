format:
	ruff format

check:
	ruff check --fix

format-check:
	make format
	make check

train:
	torchrun --nproc_per_node=8 -m train.main

kill:
	pgrep -f scribe_tokens | xargs kill -9

test-dist:
	torchrun --nproc_per_node=2 -m scripts.fake_train 