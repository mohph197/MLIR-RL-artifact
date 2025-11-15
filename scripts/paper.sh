#!/bin/bash

export CONFIG_FILE_PATH=config/example.json
poetry install
poetry run python paper/mlir_rl.py
