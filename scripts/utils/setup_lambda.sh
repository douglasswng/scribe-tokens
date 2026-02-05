#!/bin/bash
# Setup for Lambda instances - uses system CUDA PyTorch

uv venv --python /usr/bin/python3 --system-site-packages
source .venv/bin/activate

# Install everything EXCEPT torch (already in system)
uv pip install "numpy<2" matplotlib mlflow pydantic scipy swanlab tokenizers ujson ruff english-words

echo "✅ Setup complete for Lambda"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"