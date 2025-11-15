from mlir_rl_artifact.state import BenchmarkFeatures, extract_bench_features_from_code, extract_bench_features_from_file
from mlir_rl_artifact.transforms import transform_img2col
from mlir_rl_artifact.utils.config import Config
from mlir.ir import Context, Module
import json
from tqdm import tqdm
import os

from mlir_rl_artifact.utils.log import print_alert


class Benchmarks:
    """A class that holds benchmarks data"""

    data: list[BenchmarkFeatures]

    def __init__(self, is_training: bool = True):
        """Load benchmarks

        Args:
            is_training (bool): Whether to load train or evaluation set
        """
        cfg = Config()
        # Load benchmark names and execution times from json file
        bench_json_file = cfg.json_file

        # If we are in evaluation mode, use the evaluation json file if provided
        if cfg.eval_json_file and not is_training:
            bench_json_file = cfg.eval_json_file

        with open(bench_json_file) as file:
            benchmarks_json: dict[str, int] = json.load(file)

        # Build benchmark features
        self.data = []
        for bench_name, root_exec_time in tqdm(benchmarks_json.items(), desc="Extracting benchmark features", unit="bench"):
            bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
            benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)
            # NOTE: For now img2col is applied to single operator codes only
            if os.getenv("DISABLE_IMG2COL", "0") != "1" and bench_name.startswith("conv_2d_"):
                modified = False
                with Context():
                    bench_module = Module.parse(benchmark_data.code)
                for op_tag in benchmark_data.operation_tags:
                    if 'conv_2d' not in benchmark_data.operations[op_tag].operation_name:
                        continue
                    try:
                        transform_img2col(bench_module, op_tag)
                    except Exception:
                        print_alert(f"Failed to apply img2col on {bench_name}[{op_tag}]")
                    else:
                        modified = True
                if modified:
                    benchmark_data = extract_bench_features_from_code(bench_name, str(bench_module), root_exec_time)
            self.data.append(benchmark_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
