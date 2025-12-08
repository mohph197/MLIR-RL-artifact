#!/bin/bash

export CONFIG_FILE_PATH=config/example.json

# Default behavior
RUN_BASELINE=true

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --no-baseline    Skip running the baseline model."
    echo "  -h, --help       Show this help message and exit."
    echo
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-baseline)
            RUN_BASELINE=false
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

poetry install

if [ "$RUN_BASELINE" = true ]; then
    echo "--- Running Baseline ---"
    poetry run baseline
else
    echo "--- Skipping Baseline ---"
fi

echo
echo "--- Running MLIR RL ---"
poetry run python paper/mlir_rl.py
