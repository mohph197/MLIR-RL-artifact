# MLIR-RL-artifact

A deep reinforcement learning system for loop nest optimization in MLIR.

## Table of Contents

- [Installation](#installation)
  - [Method 1: Using Docker](#method-1-using-docker)
  - [Method 2: Without Docker](#method-2-without-docker)
- [Data](#data)
- [Configuration](#configuration)
  - [Configuration Parameters](#configuration-parameters)
  - [Example Configuration](#example-configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Paper Results](#paper-results)
- [Authors](#authors)

## Installation

Start by cloning the repo:

```bash
git clone https://github.com/mohph197/MLIR-RL-artifact.git
```

Before proceeding, follow the instructions in the [Data](#data) section to download and extract the benchmark data files.

### Method 1: Using Docker

Build and run the Docker container:

```bash
cd </path/to/MLIR-RL-artifact>
docker build -t mlir-rl-artifact .
docker run -it mlir-rl-artifact
```

### Method 2: Without Docker

#### Prerequisites

- Conda or Miniconda

#### Steps

1. **Install system dependencies and Conda packages:**

    ```bash
    # Start by activating a Conda environment
    # Install required Conda packages
    conda install -y \
        python=3.11 \
        git=2.51.2 \
        unzip=6.0 \
        cmake=4.1.2 \
        ninja=1.13.1 \
        binutils=2.45 \
        c-compiler=1.11.0 \
        cxx-compiler=1.11.0 \
        clang=21.1.5 \
        clangxx=21.1.5 \
        llvm-openmp=21.1.5 \
        lld=21.1.5 \
        poetry=2.2.1 \
        -c conda-forge
    ```

1. **Clone and build LLVM/MLIR:**

    ```bash
    # Clone LLVM project
    git clone --branch release/19.x --depth 1 https://github.com/llvm/llvm-project.git

    # Build MLIR with Python bindings
    cd <path/to/llvm-project>
    pip install -r mlir/python/requirements.txt
    cmake -S llvm -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD=X86 \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_ENABLE_LLD=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    cmake --build build --target check-mlir -j
    cmake --build build --target check-mlir-python -j
    ```

1. **Set environment variables:**

    ```bash
    export PATH="<path/to/llvm-project>/build/bin:$PATH"
    export PYTHONPATH="</path/to/llvm-project>/build/tools/mlir/python_packages/mlir_core"
    export LLVM_BUILD_PATH="</path/to/llvm-project>/build"
    export MLIR_SHARED_LIBS="</path/to/llvm-project>/build/lib/libmlir_runner_utils.so,</path/to/llvm-project>/build/lib/libmlir_c_runner_utils.so,$CONDA_PREFIX/lib/libomp.so"
    ```

1. **Build custom tools:**

    ```bash
    cd </path/to/MLIR-RL-artifact>/tools
    cmake -S ast_dumper -B ast_dumper/build -G Ninja \
        -DMLIR_DIR=$LLVM_BUILD_PATH/lib/cmake/mlir \
        -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_PATH/bin/llvm-lit \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++
    cmake --build ast_dumper/build -j

    cmake -S pre_vec -B pre_vec/build -G Ninja \
        -DMLIR_DIR=$LLVM_BUILD_PATH/lib/cmake/mlir \
        -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_PATH/bin/llvm-lit \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++
    cmake --build pre_vec/build -j
    ```

1. **Create `.env` file:**

    ```shell
    AST_DUMPER_BIN_PATH="</path/to/MLIR-RL-artifact>/tools/ast_dumper/build/bin/AstDumper"
    PRE_VEC_BIN_PATH="</path/to/MLIR-RL-artifact>/tools/pre_vec/build/bin/PreVec"
    MLIR_SHARED_LIBS="</path/to/llvm-project>/build/lib/libmlir_runner_utils.so,</path/to/llvm-project>/build/lib/libmlir_c_runner_utils.so,</path/to/conda-env>/lib/libomp.so"
    OMP_NUM_THREADS=12
    ```

    The path to conda environment can be found by running `echo $CONDA_PREFIX`.

1. **Install Python dependencies:**

    ```bash
    cd </path/to/MLIR-RL-artifact>
    poetry install
    ```

1. **Enable Execution of Scripts:**

    ```bash
    chmod +x scripts/*.sh
    ```

## Data

Download the benchmarks file from [file link](https://nyu.box.com/shared/static/5y1ilrccu3443dhcr854dt23uv0fysym.zip) and place it in the `data/` directory.

Extract the benchmark files:

```bash
cd data
unzip code_files.zip
cd ..
```

The `data/` directory contains execution time JSON files that specify baseline execution times (in nanoseconds) and determine which benchmarks to use:

- `execution_times_train.json` - Training benchmark set
- `execution_times_eval.json` - Evaluation benchmark set
- `execution_times_eval_full.json` - Full models evaluation benchmarks
- `execution_times_eval_nn.json` - Neural networks single operators evaluation benchmarks
- `execution_times_eval_lqcd.json` - Lattice QCD evaluation benchmarks

These files have the format:

```json
{
  "benchmark_name": baseline_execution_time_ns,
  ...
}
```

## Configuration

Configuration files are located in `config/` and control all aspects of training and evaluation. The system uses the config file specified by the `CONFIG_FILE_PATH` environment variable.

### Configuration Parameters

#### Model Architecture

- `max_num_stores_loads` (int): Maximum number of load/store operations in nested loops
- `max_num_loops` (int): Maximum number of nested loops
- `max_num_load_store_dim` (int): Maximum number of dimensions in load/store buffers
- `num_tile_sizes` (int): Number of tile sizes to consider
- `vect_size_limit` (int): Vectorization size limit to prevent excessive vectorization

#### Action Space

- `order` (list[list[str]]): Enforced sequence of actions. Each inner list specifies allowed actions at that step:
  - Action symbols: `I` (Interchange), `T` (Tiling), `TP` (TiledParallelization), `TF` (TiledFusion), `V` (Vectorization), `NT` (NoTransformation)
  - `!`: Special symbol meaning "Allow everything, except these actions", e.g. `["!", "I", "NT"]` means "Allow everything, except Interchange and NoTransformation"
- `interchange_mode` ("enumerate" | "pointers" | "continuous"): Method for sampling interchange actions

#### Exploration

- `exploration` (list): List of exploration strategies - `["entropy"]` or `["epsilon"]` or both
- `init_epsilon` (float): Initial epsilon value for epsilon-greedy exploration (decays over training)

#### Normalization

- `normalize_bounds` ("none" | "max" | "log"): How to normalize loop bounds in the input
- `normalize_adv` ("none" | "standard" | "max-abs"): Advantage normalization method for PPO

#### Experience Replay

- `reuse_experience` ("none" | "random" | "topk"): Experience replay strategy
- `replay_count` (int): Number of trajectories to keep in replay buffer

#### Training Hyperparameters

- `bench_count` (int): Number of collected benchmarks per training iteration
- `nb_iterations` (int): Total number of training iterations
- `ppo_epochs` (int): Number of PPO update epochs per iteration
- `ppo_batch_size` (int): Batch size for PPO updates
- `value_epochs` (int): Number of value function update epochs (0 to update with policy)
- `value_batch_size` (int): Batch size for value function updates
- `value_coef` (float): Value loss coefficient in combined loss
- `value_clip` (bool): Whether to clip value function loss
- `entropy_coef` (float): Entropy bonus coefficient for exploration
- `lr` (float): Learning rate for Adam optimizer
- `truncate` (int): Maximum number of transformation steps per operation

#### Data Sources

- `benchmarks_folder_path` (str): Path to directory containing `.mlir` benchmark files
- `json_file` (str): Path to training execution times JSON file
- `eval_json_file` (str): Path to evaluation execution times JSON file

#### Logging

- `results_dir` (str): Directory where results will be saved
- `tags` (list[str]): Optional tags for experiment tracking
- `debug` (bool): Enable debug mode
- `main_exec_data_file` (str): Path to global execution cache file (optional)

### Example Configuration

```json
{
  "max_num_stores_loads": 7,
  "max_num_loops": 12,
  "max_num_load_store_dim": 12,
  "num_tile_sizes": 7,
  "vect_size_limit": 512,
  "order": [["I"], ["!", "I", "NT"], ["!", "I"], ["V", "NT"]],
  "interchange_mode": "pointers",
  "exploration": ["entropy"],
  "init_epsilon": 0.0,
  "normalize_bounds": "max",
  "normalize_adv": "standard",
  "reuse_experience": "none",
  "benchmarks_folder_path": "data/code_files",
  "bench_count": 64,
  "replay_count": 0,
  "nb_iterations": 20000,
  "ppo_epochs": 4,
  "ppo_batch_size": 64,
  "value_epochs": 0,
  "value_batch_size": 0,
  "value_coef": 0.5,
  "value_clip": false,
  "entropy_coef": 0.01,
  "lr": 0.001,
  "truncate": 5,
  "json_file": "data/execution_times_train.json",
  "eval_json_file": "data/execution_times_eval.json",
  "tags": [],
  "debug": false,
  "main_exec_data_file": "",
  "results_dir": "results"
}
```

## Usage

### Training

Train the RL model:

```bash
./scripts/train.sh
```

Results will be saved to `results/run_<id>/`:

- **Model snapshots**: `results/run_<id>/models/` - Model checkpoints saved every 5 iterations
- **Logs**: `results/run_<id>/log/` - Training metrics including speedups, losses, rewards, and entropy values

### Evaluation

Evaluate saved models:

```bash
./scripts/evaluate.sh
```

This command will evaluate all saved models in the `models/` directory.

Results will be saved to `results/run_<id>/`:

- **Evaluation logs**: `results/run_<id>/log/` - Speedup metrics for each evaluated model

### Paper Results

Evaluate the latest model in the `models/` directory on the evaluation benchmarks from the paper:

```bash
./scripts/paper.sh
```

Results will be saved to `paper/results/` as a JSON file in the format:

```json
{
  "benchmark_name_1": speedup_value,
  "benchmark_name_2": speedup_value,
  ...
}
```

## Authors

- Mohammed Tirichine (<km_tirichine@esi.dz>)
- Nassim Ameur (<kn_ameur@esi.dz>)
- Iheb Nassim Aouadj (<nassimiheb.aouadj@gmail.com>)
- Nazim Bendib (<jn_bendib@esi.dz>)
- Bouchama Djad (<bouchamadjad@gmail.com>)
- Rafik Bouloudene (<rafikobouloudene@gmail.com>)
- Riyadh Baghdadi (<baghdadi@nyu.edu>)
