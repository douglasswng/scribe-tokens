train:
	torchrun --nproc_per_node=8 -m train.main

kill:
	ps -elf | grep "[s]cribe_tokens" | awk '{print $4}' | xargs kill -9