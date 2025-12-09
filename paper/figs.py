import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from markdown_pdf import Section, MarkdownPdf
from mlir_rl_artifact.utils.config import Config

cfg = Config()

with open('paper/results/torch_jit.json') as f:
    pytorch_jit_data: dict[str, float] = json.load(f)

with open('paper/results/torch_eager.json') as f:
    pytorch_data: dict[str, float] = json.load(f)

with open('paper/results/mlir_rl.json') as f:
    mlir_rl_data: dict[str, float] = json.load(f)

with open('paper/halide_rl_execs.json') as f:
    halide_rl_execs: dict[str, int] = json.load(f)

with open(cfg.eval_json_file) as f:
    mlir_execs: dict[str, int] = json.load(f)

halide_rl_data = {k: mlir_execs[k] / v for k, v in halide_rl_execs.items()}

# --- Single DNN Plot ---

dnn_benchs_small = ['add_', 'relu_', 'pooling_nchw_max_']
dnn_benchs_large = ['conv_2d_nchw_fchw_', 'matmul_']
dnn_benchs = dnn_benchs_small + dnn_benchs_large


def format_dnn_names(name: str):
    if not any(name.startswith(b) for b in dnn_benchs):
        raise ValueError(f"Unexpected benchmark name: {name}")

    if name.startswith('matmul_'):
        m, k, n = map(int, name.split('matmul_')[1].split('_'))
        return f'matmul\n({m}, {k}) x ({k}, {n})'
    elif name.startswith('add_'):
        return f"add\n({name.split('add_')[1].replace('_', ', ')})"
    elif name.startswith('relu_'):
        return f"relu\n({name.split('relu_')[1].replace('_', ', ')})"
    elif name.startswith('pooling_nchw_max_'):
        n, c, h, w, kh, *_ = map(int, name.split('pooling_nchw_max_')[1].split('_'))
        return f"pooling\n({n}, {c}, {h}, {w}) * ({kh}, {kh})"
    elif name.startswith('conv_2d_nchw_fchw_'):
        n, c, h, w, f, kh, kw, *_ = map(int, name.split('conv_2d_nchw_fchw_')[1].split('_'))
        return f"conv2d\n({n}, {c}, {h}, {w}) * ({f}, {c}, {kh}, {kw})"


# Get DNN keys
dnn_keys_small = []
dnn_keys_large = []
for b in dnn_benchs_small:
    dnn_keys_small.extend([k for k in mlir_rl_data if k.startswith(b)])
for b in dnn_benchs_large:
    dnn_keys_large.extend([k for k in mlir_rl_data if k.startswith(b)])
dnn_keys = dnn_keys_small + dnn_keys_large


def get_values(keys, data):
    return [data[k] if k in data else 0.0 for k in keys]


mlir = [get_values(dnn_keys_small, mlir_rl_data), get_values(dnn_keys_large, mlir_rl_data)]
pytorch = [get_values(dnn_keys_small, pytorch_data), get_values(dnn_keys_large, pytorch_data)]
jit = [get_values(dnn_keys_small, pytorch_jit_data), get_values(dnn_keys_large, pytorch_jit_data)]
halide = [get_values(dnn_keys_small, halide_rl_data), get_values(dnn_keys_large, halide_rl_data)]

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(27, 10))
width = 0.15

for i, keys in enumerate([dnn_keys_small, dnn_keys_large]):
    x = np.arange(len(keys))
    ax = axes[i]
    ax.bar(x - width * 3 / 2, mlir[i], width, label='MLIR RL (ours)')
    ax.bar(x - width / 2, pytorch[i], width, label='PyTorch')
    ax.bar(x + width / 2, jit[i], width, label='PyTorch JIT')
    ax.bar(x + width * 3 / 2, halide[i], width, label='Halide RL')

    ax.set_ylabel('Speedup over baseline', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([format_dnn_names(k) for k in keys], fontdict={'fontsize': 14})
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

axes[1].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.3), fontsize=16)

plt.tight_layout()
plt.savefig('paper/figures/figure_dnn.pdf')

# --- Full Models Table ---

models = ["model_res_net", "model_mobile_net_v2", "model_vgg"]

table_rows = []
for model in models:
    # Get the raw values
    v1 = mlir_rl_data[model]
    v2 = pytorch_data[model]
    v3 = pytorch_jit_data[model]

    # Find the max value
    max_val = max(v1, v2, v3)

    # Helper function to format and highlight
    def fmt(val, is_max):
        s = f"{val:.4f}"  # Formatting to 4 decimal places
        if is_max:
            return f"**{s}**"  # Markdown bold syntax
        return s

    table_rows.append({
        "Model": model.split("_", 1)[1].replace("_", " "),
        "MLIR RL": fmt(v1, v1 == max_val),
        "PyTorch": fmt(v2, v2 == max_val),
        "PyTorch JIT": fmt(v3, v3 == max_val)
    })

df = pd.DataFrame(table_rows)
md = df.to_markdown(index=False)

md_pdf = MarkdownPdf()
css = """th, td {
    border: 1px solid black;
    padding: 3px 5px;
}"""
md_pdf.add_section(Section(md), user_css=css)
md_pdf.save("paper/figures/table_full_models.pdf")
