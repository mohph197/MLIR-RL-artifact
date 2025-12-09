#!/bin/bash

export CONFIG_FILE_PATH=config/example.json

poetry install

echo "--- Running Baseline ---"
poetry run baseline

echo
echo "--- Running MLIR RL ---"
poetry run python paper/mlir_rl.py

echo
echo "--- Running PyTorch and PyTorch Compiler ---"
poetry run python paper/eval_torch.py

echo
echo "--- Generating figures ---"
poetry run python paper/figs.py
