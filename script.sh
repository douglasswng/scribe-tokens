mlflow ui --port 5001 --backend-store-uri ./artifacts/trackers/mlflow
ps -elf | grep "[s]cribe_tokens" | awk '{print $4}' | xargs kill -9
torchrun --nproc_per_node=8 -m train.main
torchrun --nproc_per_node=8 --master_port=29501 -m train.main