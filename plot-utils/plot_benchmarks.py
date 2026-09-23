"""
Plot TensorRT / PyTorch inference-benchmark results.

Expects benchmark results as JSON files (list of records in the format you showed,
with at least: dataset, configuration, mode, mean_ms, std_ms), organized per
dataset and per GPU.

IMPORTANT ASSUMPTION: your sample record has no "gpu" field, and GPU is only
encoded by how you've organized the files. This script infers GPU from the
JSON filename (a trailing "__<gpu>" suffix) or, failing that, from the parent
directory name. Edit `infer_gpu()` below if that doesn't match your real layout
(or just add a "gpu" key to each record yourself and this script will use it).

Usage:
    python plot_results.py /path/to/results_dir --outdir plots/
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODE_ORDER = [
    "github-issue",
    "pytorch",
    "pytorch-compile",
    "pytorch-compile-nhwc-split-conv",
    "pytorch-nhwc",
    "pytorch-nhwc-split-conv",
    "trt-solution-fp32",
    "trt-solution-autocast",
    "trt-solution-fp16",
    "trt-solution-fp16-nhwc",
    "trt-solution-fp16-nhwc-split-conv",
]

FP32_MODES = {"github-issue", "trt-solution-fp32"}


def mode_family(mode: str) -> str:
    if mode == "github-issue":
        return "github-issue"
    if mode.startswith("pytorch"):
        return "pytorch"
    if mode.startswith("trt-solution"):
        return "trt"
    return "other"


def infer_gpu(path: Path) -> str:
    """Best-effort GPU name inference. EDIT THIS to match your real naming.
    Tries a trailing '__<gpu>' in the filename, else falls back to the
    parent directory name (e.g. results/<gpu>/<dataset>.json)."""
    stem = path.stem
    m = re.search(r"__([A-Za-z0-9\-]+)$", stem)
    if m:
        return m.group(1)
    return path.parent.name or "unknown-gpu"


def load_data(input_dir: Path) -> pd.DataFrame:
    records = []
    for f in sorted(input_dir.rglob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        gpu_guess = infer_gpu(f)
        for rec in data:
            rec = dict(rec)
            rec.setdefault("gpu", gpu_guess)
            records.append(rec)
    if not records:
        raise SystemExit(f"No JSON records found under {input_dir}")
    df = pd.DataFrame(records)
    df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
    df["family"] = df["mode"].map(mode_family)
    return df


def _ordered_modes(sub: pd.DataFrame):
    return [m for m in MODE_ORDER if m in set(sub["mode"])]


def plot_fp32_tax(df: pd.DataFrame, outdir: Path):
    """Point: github-issue and trt-solution-fp32 both sit an order of magnitude
    above everything else, because both are effectively running fp32."""
    for (dataset, config), sub in df.groupby(["dataset", "configuration"]):
        gpus = sorted(sub["gpu"].unique())
        modes = _ordered_modes(sub)
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(modes))
        width = 0.8 / max(len(gpus), 1)

        for i, gpu in enumerate(gpus):
            g = sub[sub["gpu"] == gpu].set_index("mode").reindex(modes)
            ax.bar(x + i * width, g["mean_ms"], width, yerr=g["std_ms"],
                   capsize=2, label=gpu, alpha=0.85)

        # Shade the fp32 modes and color their tick labels to call them out
        for m in modes:
            if m in FP32_MODES:
                xi = modes.index(m)
                ax.axvspan(xi - 0.5, xi + 0.5, color="red", alpha=0.06, zorder=0)

        ax.set_yscale("log")
        ax.set_xticks(x + width * (len(gpus) - 1) / 2)
        labels = ax.set_xticklabels(modes, rotation=45, ha="right")
        for lbl, m in zip(labels, modes):
            if m in FP32_MODES:
                lbl.set_color("#d62728")
                lbl.set_fontweight("bold")

        ax.set_ylabel("mean latency (ms, log scale)")
        ax.set_title(
            f"{dataset} / {config}: fp32 modes (github-issue, trt-solution-fp32) "
            "dominate runtime"
        )
        ax.legend(title="GPU")
        fig.tight_layout()
        fig.savefig(outdir / f"fp32_tax_{dataset}_{config}.png", dpi=150)
        plt.close(fig)


def plot_ablation(df: pd.DataFrame, outdir: Path):
    """Point: full ablation across every configuration, as speedup relative to
    plain eager pytorch, one chart per dataset/configuration/gpu."""
    for (dataset, config, gpu), sub in df.groupby(["dataset", "configuration", "gpu"]):
        modes = _ordered_modes(sub)
        sub = sub.set_index("mode").reindex(modes)
        baseline = sub.loc["pytorch", "mean_ms"] if "pytorch" in sub.index else sub["mean_ms"].max()
        speedup = baseline / sub["mean_ms"]

        colors = [
            "#7f7f7f" if m == "github-issue"
            else "#d62728" if m == "trt-solution-fp32"
            else "#2ca02c" if m.startswith("trt-solution")
            else "#1f77b4"
            for m in modes
        ]

        fig, ax = plt.subplots(figsize=(9, 6))
        y = np.arange(len(modes))
        ax.barh(y, speedup, color=colors)
        ax.axvline(1.0, color="black", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(modes)
        ax.invert_yaxis()
        ax.set_xlabel("speedup vs plain pytorch (baseline mean_ms / mean_ms)")
        ax.set_title(f"{dataset} / {config} / {gpu}: configuration ablation")
        for yi, v in zip(y, speedup):
            if pd.notna(v):
                ax.text(v, yi, f" {v:.2f}x", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"ablation_{dataset}_{config}_{gpu}.png", dpi=150)
        plt.close(fig)


def plot_trt_not_magic(df: pd.DataFrame, outdir: Path):
    """Point: trt-solution-autocast is not automatically the fastest option —
    flag cases where it's beaten by torch.compile (or even eager pytorch)."""
    modes = ["pytorch", "pytorch-compile", "trt-solution-autocast"]
    sub = df[df["mode"].isin(modes)].copy()
    sub["group"] = (
        sub["dataset"].astype(str) + " / " + sub["configuration"].astype(str)
        + " / " + sub["gpu"].astype(str)
    )
    groups = sorted(sub["group"].unique())
    colors = {"pytorch": "#1f77b4", "pytorch-compile": "#ff7f0e", "trt-solution-autocast": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(groups)), 6))
    width = 0.25
    x = np.arange(len(groups))

    for i, mode in enumerate(modes):
        vals = [sub[(sub["group"] == g) & (sub["mode"] == mode)]["mean_ms"].mean() for g in groups]
        ax.bar(x + (i - 1) * width, vals, width, label=mode, color=colors[mode])

    for xi, g in zip(x, groups):
        pc = sub[(sub["group"] == g) & (sub["mode"] == "pytorch-compile")]["mean_ms"].mean()
        ta = sub[(sub["group"] == g) & (sub["mode"] == "trt-solution-autocast")]["mean_ms"].mean()
        if pd.notna(pc) and pd.notna(ta) and ta > pc:
            ax.annotate("autocast TRT\nslower than compile!", xy=(xi, ta), xytext=(xi, ta * 1.2),
                        ha="center", fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("mean latency (ms, log scale)")
    ax.set_title("TensorRT is not magic: autocast TRT sometimes loses to torch.compile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "trt_not_magic.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", type=Path, help="Directory with benchmark JSON files (searched recursively)")
    ap.add_argument("--outdir", type=Path, default=Path("plots"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input_dir)
    plot_fp32_tax(df, args.outdir)
    plot_ablation(df, args.outdir)
    plot_trt_not_magic(df, args.outdir)
    print(f"Saved plots to {args.outdir}/")


if __name__ == "__main__":
    main()
