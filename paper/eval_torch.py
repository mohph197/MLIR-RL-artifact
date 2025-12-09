# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

from statistics import median
import traceback
from typing import Callable
import torch
import torch.nn.functional as F
import time
import json
import torchvision.models as models

# --- 1. Configuration ---
DEVICE = 'cpu'
DTYPE = torch.float64
torch.set_num_threads(12)
torch.set_grad_enabled(False)

# Benchmarking parameters
N_WARMUP = 10
N_RUNS = 11
# --- End Configuration ---


def benchmark_op(op_func: Callable[..., torch.Tensor], op_name: str, inputs_shapes: list[tuple[tuple[int, ...], ...]], name_fn: Callable[..., str]):
    """
    Generic function to benchmark a PyTorch operation in Eager and JIT modes.

    Args:
        op_cls: The operation class to benchmark.
        args: The arguments to pass to the operation.
        op_name: The name of the operation.
        inputs_shapes: The list of input shapes to benchmark.
        name_fn: A function to generate a string representation of the input sizes.
    """
    print(f"\n--- Benchmarking Operator: {op_name} ---")
    results_eager = {}
    results_jit = {}

    for i, shapes in enumerate(inputs_shapes):
        # Create inputs
        dtype = torch.float32 if op_name.startswith('model_') else DTYPE
        inputs_on_device = [torch.zeros(shape, dtype=dtype, device=DEVICE) for shape in shapes]

        # Create a string representation of the input sizes
        size_str = name_fn(shapes)
        print(f"  Size {i + 1}: [{size_str}]")
        bench_name = f"{op_name}_{size_str}" if size_str else op_name

        # --- 1. Eager Mode ---
        try:
            # Warm-up
            for _ in range(N_WARMUP):
                _ = op_func(*inputs_on_device)

            # Timed run
            times: list[int] = []
            for _ in range(N_RUNS):
                start_time = time.time_ns()
                _ = op_func(*inputs_on_device)
                end_time = time.time_ns()
                times.append(end_time - start_time)
            avg_time_eager = median(times)

            results_eager[bench_name] = avg_time_eager
            print(f"    Eager: {avg_time_eager:10.6f} ns")

        except Exception as e:
            print(f"    Eager failed: {e}")
            traceback.print_exc()
            results_eager[bench_name] = -1

        # --- 2. JIT Mode ---
        try:
            with torch.jit.optimized_execution(True):
                # Trace the model
                jit_op = torch.jit.script(op_func)

                # Warm-up
                for _ in range(N_WARMUP):
                    _ = jit_op(*inputs_on_device)

                # Timed run
                times: list[int] = []
                for _ in range(N_RUNS):
                    start_time = time.time_ns()
                    _ = jit_op(*inputs_on_device)
                    end_time = time.time_ns()
                    times.append(end_time - start_time)
                avg_time_jit = median(times)

            results_jit[bench_name] = avg_time_jit

            # Calculate speedup
            speedup = avg_time_eager / avg_time_jit
            print(f"    JIT:   {avg_time_jit:10.6f} ns   (Speedup: {speedup:.2f}x)")

        except Exception as e:
            print(f"    JIT failed: {e}")
            traceback.print_exc()
            results_jit[bench_name] = -1

    return results_eager, results_jit


def get_op_definitions():
    """
    Returns a dictionary of all operations to test, along with their
    input tensor definitions. Tensors are created on CPU with DTYPE.
    """
    all_ops = {}

    # --- Add ---
    def add_func(a, b):
        return a + b
    all_ops['add'] = {
        'op': add_func,
        'name_fn': lambda sizes: "_".join(map(str, sizes[0])),
        'inputs_shapes': [
            ((256, 14, 14, 88), (256, 14, 14, 88)),
            ((256, 42, 42, 168), (256, 42, 42, 168)),
            ((256, 7, 7, 2048), (256, 7, 7, 2048)),
        ]
    }

    def conv2d_name(shapes, s):
        h = shapes[0][2]
        kh = shapes[1][2]
        h_ = ((h - kh) // s) + 1

        return '_'.join(map(str, shapes[0] + (shapes[1][0],) + shapes[1][2:] + (h_, h_)))

    # --- Conv2d ---
    def conv2d_func(img, kern):
        return F.conv2d(img, kern, stride=1, padding=1)
    all_ops['conv_2d_nchw_fchw'] = {
        'op': conv2d_func,
        'name_fn': lambda shapes: conv2d_name(shapes, 1),
        'inputs_shapes': [
            ((256, 128, 28, 28), (512, 128, 1, 1)),
            ((256, 512, 28, 28), (128, 512, 1, 1)),
            ((256, 64, 56, 56), (256, 64, 1, 1)),
        ]
    }

    # --- Matmul ---
    def matmul_func(a, b):
        return torch.matmul(a, b)
    all_ops['matmul'] = {
        'op': matmul_func,
        'name_fn': lambda sizes: "_".join(map(str, sizes[0] + (sizes[1][1],))),
        'inputs_shapes': [
            ((256, 1536), (1536, 1000)),
            ((256, 256), (256, 128)),
            ((256, 256), (256, 512)),
            ((256, 512), (512, 1024)),
        ]
    }

    def pooling_name(sizes):
        h = sizes[0][2]
        kh = 3
        h_ = ((h - kh) // 2) + 1

        return '_'.join(map(str, sizes[0] + (kh, h_, h_)))

    # --- MaxPool2d ---
    def pooling_func(img):
        return F.max_pool2d(img, kernel_size=3, stride=2, padding=1)
    all_ops['pooling_nchw_max'] = {
        'op': pooling_func,
        'name_fn': pooling_name,
        'inputs_shapes': [
            ((256, 336, 43, 43),),
            ((256, 64, 114, 114),),
            ((256, 64, 147, 147),),
        ]
    }

    # --- ReLU ---
    def relu_func(a):
        return F.relu(a)
    all_ops['relu'] = {
        'op': relu_func,
        'name_fn': lambda sizes: "_".join(map(str, sizes[0])),
        'inputs_shapes': [
            ((256, 100),),
            ((256, 512),),
            ((256, 57, 57, 64),),
        ]
    }

    # --- Full Models ---
    def model_name_fn(_):
        return ""
    model_input_shapes = [
        ((1, 3, 224, 224),),
    ]

    all_ops['model_mobile_net_v2'] = {
        'op': models.mobilenet_v2().eval(),
        'name_fn': model_name_fn,
        'inputs_shapes': model_input_shapes,
    }

    all_ops['model_res_net'] = {
        'op': models.resnet18().eval(),
        'name_fn': model_name_fn,
        'inputs_shapes': model_input_shapes,
    }

    all_ops['model_vgg'] = {
        'op': models.vgg16().eval(),
        'name_fn': model_name_fn,
        'inputs_shapes': model_input_shapes,
    }

    return all_ops


def main():
    print(f"Device: {DEVICE}")
    print(f"DType: {DTYPE}")
    print(f"Warm-up runs: {N_WARMUP}, Timed runs: {N_RUNS}")

    # 1. Get all operator definitions
    op_definitions = get_op_definitions()

    with open('data/execution_times_eval.json') as f:
        base_execs: dict[str, int] = json.load(f)

    all_results_eager = {}
    all_results_jit = {}

    # 2. Loop and benchmark
    for op_name, data in op_definitions.items():
        results_eager, results_jit = benchmark_op(
            op_func=data['op'],
            op_name=op_name,
            inputs_shapes=data['inputs_shapes'],
            name_fn=data['name_fn']
        )
        all_results_eager.update({k: (base_execs[k] / v) for k, v in results_eager.items()})
        all_results_jit.update({k: (base_execs[k] / v) for k, v in results_jit.items()})
        with open('paper/results/torch_eager.json', 'w') as f:
            json.dump(all_results_eager, f, indent=2)
        with open('paper/results/torch_jit.json', 'w') as f:
            json.dump(all_results_jit, f, indent=2)

    print("\n--- Benchmark Complete ---")
    print("Results saved to 'paper/results/torch_eager.json' and 'paper/results/torch_jit.json'")


if __name__ == "__main__":
    main()
