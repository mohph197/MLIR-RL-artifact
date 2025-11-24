"""Core module for MLIR RL.

This module sets up the computation device (CUDA or CPU).
"""

import torch
from mlir_rl_artifact.utils.log import print_info

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print_info("Using device:", device.type)
