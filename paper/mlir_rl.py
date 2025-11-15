# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

# Import modules
import os
import torch
import numpy as np
import random
import re
import json
from typing import Optional
from mlir_rl_artifact.execution import Execution
from mlir_rl_artifact.model import HiearchyModel as Model
from mlir_rl_artifact import device
from mlir_rl_artifact.ppo import evaluate_benchmarks
from mlir_rl_artifact.benchmarks import Benchmarks
from mlir_rl_artifact.utils.dask_manager import DaskManager
from mlir_rl_artifact.utils.gpu_occupier import GPUOccupier
from mlir_rl_artifact.utils.file_logger import FileLogger
from mlir_rl_artifact.utils.config import Config
from mlir_rl_artifact.utils.log import print_info, print_success

eval_dir = 'models'


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize singleton classes
    Config()
    fl = FileLogger()
    fl.disable_logging()
    dm = DaskManager()
    go = GPUOccupier()

    # Start GPU Occupier
    if device.type == "cuda":
        go.start(device)

    # Data loading
    def load_eval_data():
        return Benchmarks(is_training=False)

    def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
        return None

    eval_data = dm.run_and_register_to_workers(load_eval_data)
    main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

    # Initialize execution singleton
    Execution(fl.exec_data_file, main_exec_data)

    # Prepare logging
    print_success(f'Logging to: {fl.run_dir}')

    # Setup torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)

    # Initiate model
    with go.gpu_needed():
        model = Model().to(device)
    print_success("Model initialized")

    # Read the files in the evaluation directory
    eval_files = [f for f in os.listdir(eval_dir) if re.match(r'model_\d+\.(pt|pth)', f)]

    # Order files
    eval_files.sort(key=lambda x: int(re.match(r'model_(\d+)\.(pt|pth)', x).group(1)))

    # Evaluate last file
    model_file = eval_files[-1]
    model_path = os.path.join(eval_dir, model_file)
    if not os.path.exists(model_path):
        print_info(f"Model file {model_path} does not exist. Skipping.")
        return
    print_info(f"Loading model from {model_path}")

    with go.gpu_needed():
        model.load_state_dict(torch.load(model_path, weights_only=True))

    _, bench_speedups = evaluate_benchmarks(model, eval_data)

    out_file = 'paper/results/mlir_rl.json'
    with open(out_file, 'w') as f:
        json.dump(bench_speedups, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
