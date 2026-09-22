#!/usr/bin/env python3
"""
plot_bones_profile.py

Visualize a TensorRT "bones" layer-timing profile (the JSON produced by
trtexec / TensorRT's --exportProfile) for an nnU-Net FullRes-style
encoder/decoder network.

It plots the average time (ms) of every layer/kernel *in the order the
tensor actually flows through the engine* (i.e. the order layers appear
in the JSON), color-coded by operation category, so you can see where
time is spent as data travels through encoder stages, the bottleneck,
and back up through the decoder/transpose-conv stages.

Usage
-----
    python plot_bones_profile.py bones_profile.json
    python plot_bones_profile.py bones_profile.json -o profile.png --metric averageMs
    python plot_bones_profile.py bones_profile.json --top 15

Outputs
-------
    - A PNG (or whatever extension you pass to -o) with two panels:
        1. Bar chart of every layer in execution order, colored by category,
           with vertical bands marking encoder/decoder stage boundaries.
        2. Cumulative time curve over the same x-axis, so you can read off
           "how much time has elapsed by the time the tensor reaches here".
    - A printed summary table of total time per category and the top-N
      slowest individual layers.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Parsing / categorization
# --------------------------------------------------------------------------

# Order matters: more specific patterns first.
CATEGORY_PATTERNS = [
    ("Convolution",   re.compile(r"\[CONVOLUTION\]")),
    ("Deconvolution", re.compile(r"\[DECONVOLUTION\]")),
    ("GroupNorm/Act (fused foreign node)", re.compile(r"ForeignNode")),
    ("Reformat/Copy", re.compile(r"^Reformatting")),
    ("Concat",        re.compile(r"__myl_Conc")),
    ("Transpose",     re.compile(r"__myl_Tran")),
    ("Move/Reorder",  re.compile(r"__myl_Move")),
    ("Norm+Act (Inor)", re.compile(r"__myl_Inor")),
    ("MulAddMulMax (Norm+Act fused)", re.compile(r"__myl_MulAdd")),
]


def categorize(name: str) -> str:
    for label, pattern in CATEGORY_PATTERNS:
        if pattern.search(name):
            return label
    return "Other"


# Pull out a short, human-readable tag from the verbose layer name, e.g.
#   "[CONVOLUTION]-...-[encoder.stages.1.0.convs.0.../convolution_2]_myl0_9"
#     -> "enc.s1.convs.0 (conv_2)"
#   "[DECONVOLUTION]-...-[decoder.transpconvs.2/convolution_18]"
#     -> "dec.transpconv.2"
MODULE_RE = re.compile(
    r"(?P<side>encoder|decoder)\.(?P<rest>[a-zA-Z0-9_.]+)/(?P<op>[a-zA-Z_]+(?:_\d+)?)"
)


def short_label(name: str) -> str:
    m = MODULE_RE.search(name)
    if not m:
        # fused/reformat/internal kernels: just shorten the __myl_ tag
        myl = re.search(r"__myl_[A-Za-z]+(?:_myl\d+_\d+)?", name)
        if myl:
            return myl.group(0).replace("__myl_", "")
        return name[:24]
    side = "enc" if m.group("side") == "encoder" else "dec"
    rest = m.group("rest").replace("stages.", "s").replace("convs.", "c").replace(
        ".all_modules.0", ""
    ).replace(".transpconvs.", "transpconv.").replace(".seg_layers.", "seg.")
    op = m.group("op")
    return f"{side}.{rest} ({op})"


def stage_key(name: str):
    """Group key used to draw encoder/decoder stage boundary bands."""
    m = re.search(r"(encoder|decoder)\.stages\.(\d+)", name)
    if m:
        return f"{'enc' if m.group(1) == 'encoder' else 'dec'} stage {m.group(2)}"
    m = re.search(r"decoder\.transpconvs\.(\d+)", name)
    if m:
        return f"dec upsample {m.group(1)}"
    m = re.search(r"decoder\.seg_layers", name)
    if m:
        return "seg head"
    return None


def load_profile(path: Path):
    data = json.loads(path.read_text())
    # First element is typically {"count": N} metadata, skip anything without a name.
    layers = [d for d in data if "name" in d]
    return layers


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_profile(layers, metric="averageMs", top_n=10, out_path="bones_profile.png",
                  title="TensorRT layer timing — nnU-Net FullRes engine"):
    names = [l["name"] for l in layers]
    values = np.array([l[metric] for l in layers], dtype=float)
    categories = [categorize(n) for n in names]
    labels = [short_label(n) for n in names]

    cats_in_order = sorted(set(categories), key=lambda c: -sum(
        v for v, c2 in zip(values, categories) if c2 == c
    ))
    cmap = plt.get_cmap("tab10" if len(cats_in_order) <= 10 else "tab20")
    color_of = {c: cmap(i % cmap.N) for i, c in enumerate(cats_in_order)}
    bar_colors = [color_of[c] for c in categories]

    x = np.arange(len(layers))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(14, len(layers) * 0.16), 9),
        sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )

    # --- Panel 1: per-layer bar chart in execution order ---
    ax1.bar(x, values, color=bar_colors, width=0.9, edgecolor="none")
    ax1.set_ylabel(f"{metric} (ms)")
    ax1.set_title(title)

    # Stage boundary shading (labels drawn rotated, near the bottom of the
    # panel so they don't collide with each other on narrow stages)
    stage_keys = [stage_key(n) for n in names]
    prev_key = None
    band_start = 0
    toggle = False
    ymax = values.max()
    for i, k in enumerate(stage_keys + [None]):
        if k != prev_key:
            if prev_key is not None:
                toggle = not toggle
                if toggle:
                    ax1.axvspan(band_start - 0.5, i - 0.5, color="gray", alpha=0.06, zorder=0)
                if prev_key:
                    ax1.text((band_start + i - 1) / 2, ymax * 0.02,
                             prev_key, rotation=90, ha="center", va="bottom",
                             fontsize=6, color="dimgray")
            band_start = i
            prev_key = k

    # Legend for categories
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_of[c]) for c in cats_in_order]
    ax1.legend(handles, cats_in_order, loc="upper right", fontsize=8, ncol=1,
               framealpha=0.9, title="Operation type")

    # Annotate top-N slowest layers
    top_idx = np.argsort(values)[::-1][:top_n]
    for i in top_idx:
        ax1.annotate(f"{values[i]:.2f}", (x[i], values[i]),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=7, color="black")

    # --- Panel 2: cumulative time as tensor flows through the network ---
    cumulative = np.cumsum(values)
    ax2.plot(x, cumulative, color="black", lw=1.5)
    ax2.fill_between(x, cumulative, color="steelblue", alpha=0.15)
    ax2.set_ylabel(f"cumulative {metric} (ms)")
    ax2.set_xlabel("layer / kernel (execution order, encoder → bottleneck → decoder)")

    # Sparse x tick labels (too many layers to label all of them)
    step = max(1, len(layers) // 40)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([labels[i] for i in range(0, len(layers), step)],
                         rotation=90, fontsize=6)

    ax1.grid(axis="y", alpha=0.3)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot -> {out_path}")

    return names, values, categories, labels


def print_summary(names, values, categories, labels, metric, top_n=10):
    total = values.sum()
    print(f"\nTotal {metric} across {len(values)} layers: {total:.2f} ms\n")

    print("By category:")
    cat_totals = {}
    for c, v in zip(categories, values):
        cat_totals[c] = cat_totals.get(c, 0.0) + v
    for c, v in sorted(cat_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<32} {v:8.2f} ms  ({100 * v / total:5.1f}%)")

    print(f"\nTop {top_n} slowest individual layers:")
    order = np.argsort(values)[::-1][:top_n]
    for rank, i in enumerate(order, 1):
        print(f"  {rank:2d}. {values[i]:8.3f} ms  [{categories[i]}]  {labels[i]}")
        print(f"      {names[i]}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", type=Path, help="path to the bones_profile.json file")
    ap.add_argument("-o", "--out", default="bones_profile.png",
                     help="output image path (default: bones_profile.png)")
    ap.add_argument("--metric", default="averageMs",
                     choices=["averageMs", "medianMs", "timeMs", "percentage"],
                     help="which field to plot per layer (default: averageMs)")
    ap.add_argument("--top", type=int, default=10,
                     help="number of slowest layers to annotate/report (default: 10)")
    args = ap.parse_args()

    layers = load_profile(args.json_path)
    if not layers:
        sys.exit("No layer entries found in JSON (expected objects with a 'name' field).")

    names, values, categories, labels = plot_profile(
        layers, metric=args.metric, top_n=args.top, out_path=str(args.out)
    )
    print_summary(names, values, categories, labels, args.metric, top_n=args.top)


if __name__ == "__main__":
    main()
