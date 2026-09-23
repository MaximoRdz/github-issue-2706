"""
Plot nnU-Net inference-benchmark comparisons across one or more GPUs.

Each positional argument is `<gpu-label>=<path>`, where <path> is either:
  - a directory containing an `all_results.json` (as produced by run_all.py), or
  - a path directly to an `all_results.json` (or any JSON list of result
    dicts with the same shape: dataset/configuration/mode/mean_ms/...).

Produces one grouped bar-chart panel per `configuration` (2d, 3d_fullres),
with one bar group per `mode` and one bar per GPU within each group.

Usage
-----
python plot_results.py \
    dgx-spark=./bench_results_dgx_spark \
    nvidia-A40=./bench_results_a40 \
    --output gpu_comparison.png

python plot_results.py \
    dgx-spark=./bench_results_dgx_spark/all_results.json \
    nvidia-A40=./bench_results_a40/all_results.json \
    --metric p95_ms --log-scale --csv-output combined.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

METRIC_CHOICES = ["mean_ms", "median_ms", "p90_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"]


def parse_results_arg(arg: str) -> Tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Expected '<gpu-label>=<path>', got '{arg}'"
        )
    label, path_str = arg.split("=", 1)
    path = Path(path_str)
    if path.is_dir():
        path = path / "all_results.json"
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Results file not found: {path}")
    return label, path


def load_results(label: str, path: Path) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a JSON list of result dicts")
    for r in data:
        r["gpu"] = label
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", type=parse_results_arg,
                    help="One or more '<gpu-label>=<path>' pairs, e.g. dgx-spark=./bench_results_dgx")
    p.add_argument("--metric", default="mean_ms", choices=METRIC_CHOICES,
                    help="Latency metric plotted as bar height (default: mean_ms)")
    p.add_argument("--error-metric", default="std_ms", choices=["std_ms", "none"],
                    help="Metric drawn as error bars (default: std_ms, use 'none' to disable)")
    p.add_argument("--log-scale", action="store_true", help="Log scale on the latency axis")
    p.add_argument("--output", type=Path, default=Path("gpu_comparison.png"))
    p.add_argument("--csv-output", type=Path, default=None,
                    help="Optional: also dump the combined raw results to this CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_rows: List[dict] = []
    for label, path in args.results:
        rows = load_results(label, path)
        print(f"Loaded {len(rows)} result(s) for '{label}' from {path}")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No results loaded.")

    configurations = sorted({r["configuration"] for r in all_rows})
    modes = sorted({r["mode"] for r in all_rows})
    gpus = list(dict.fromkeys(label for label, _ in args.results))  # de-dupe, keep CLI order

    fig, axes = plt.subplots(1, len(configurations), figsize=(7 * len(configurations), 6), squeeze=False)
    axes = axes[0]

    bar_width = 0.8 / max(len(gpus), 1)
    x = np.arange(len(modes))

    for ax, configuration in zip(axes, configurations):
        for gi, gpu in enumerate(gpus):
            heights, errors = [], []
            for mode in modes:
                match = [r for r in all_rows
                         if r["configuration"] == configuration and r["mode"] == mode and r["gpu"] == gpu]
                if match:
                    heights.append(match[0][args.metric])
                    errors.append(match[0].get(args.error_metric, 0.0) if args.error_metric != "none" else 0.0)
                else:
                    heights.append(np.nan)  # missing combo for this GPU: leave a gap, don't fabricate a bar
                    errors.append(0.0)
            offsets = x + (gi - (len(gpus) - 1) / 2) * bar_width
            ax.bar(offsets, heights, width=bar_width, yerr=errors, capsize=3, label=gpu)

        ax.set_title(f"configuration = {configuration}")
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=30, ha="right")
        ax.set_ylabel(f"{args.metric} (ms)")
        if args.log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="GPU")

    fig.suptitle("nnU-Net sliding-window inference benchmark")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    if args.csv_output:
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with open(args.csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved combined CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
