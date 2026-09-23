#!/usr/bin/env python3
"""
plot_pytorch_profile.py

Visualize a hierarchical PyTorch module timing profile JSON.

This is intended for profiles where entries look like:

    {
        "name": "ResidualEncoderUNet",
        "depth": 0,
        "depthIdx": 1,
        "timeMs": 23.297,
        "averageMs": 23.297,
        "medianMs": 23.297,
        "percentage": 99.78
    }

Unlike a TensorRT layer profile, this profile is hierarchical:
parents include the time of their children.

Therefore, DO NOT sum entries from different depths. Doing so would
double-count execution time.

By default this script uses --leaves, which keeps only the deepest
nodes that do not have children.

Examples
--------
    # Recommended: plot leaf nodes only
    python plot_pytorch_profile.py pytorch_profile.json --leaves

    # Plot a specific hierarchy level
    python plot_pytorch_profile.py pytorch_profile.json --depth 3

    # Plot depth 2
    python plot_pytorch_profile.py pytorch_profile.json --depth 2 \
        -o depth2.png

    # Use median instead of average
    python plot_pytorch_profile.py pytorch_profile.json --leaves \
        --metric medianMs

    # Annotate/report the 15 slowest nodes
    python plot_pytorch_profile.py pytorch_profile.json --leaves --top 15

    # Inspect all nodes (WARNING: cumulative time is not an execution
    # total when multiple depths are included)
    python plot_pytorch_profile.py pytorch_profile.json --depth all

Outputs
-------
    - PNG (or whatever extension is passed to -o)
    - Per-node timing plot in hierarchy/execution order
    - Cumulative timing curve
    - Printed summary by module/category
    - Top-N slowest selected nodes

Notes
-----
The --leaves mode is generally the most appropriate mode for obtaining
a flat execution-time breakdown without parent/child double counting.

If the profile contains nodes with zero time, they are retained unless
--drop-zero is specified.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Categorization
# --------------------------------------------------------------------------

CATEGORY_PATTERNS = [
    # nnU-Net / architecture-level names
    ("UNet", re.compile(r"UNet", re.IGNORECASE)),
    ("Encoder", re.compile(r"Encoder", re.IGNORECASE)),
    ("Decoder", re.compile(r"Decoder", re.IGNORECASE)),
    ("ConvBlock", re.compile(r"Conv(Block|Blocks)", re.IGNORECASE)),
    ("StackedConv", re.compile(r"StackedConv", re.IGNORECASE)),
    ("Residual", re.compile(r"Residual", re.IGNORECASE)),
    ("Sequential", re.compile(r"Sequential", re.IGNORECASE)),
    ("Normalization", re.compile(
        r"(Norm|BatchNorm|InstanceNorm|GroupNorm|LayerNorm)",
        re.IGNORECASE,
    )),
    ("Activation", re.compile(
        r"(ReLU|LeakyReLU|GELU|SiLU|Sigmoid|Softmax|Activation)",
        re.IGNORECASE,
    )),
    ("Convolution", re.compile(
        r"(Conv\d*d?|Convolution)",
        re.IGNORECASE,
    )),
    ("Pooling", re.compile(
        r"(Pool|Pooling|MaxPool|AvgPool)",
        re.IGNORECASE,
    )),
    ("Upsample", re.compile(
        r"(Upsample|Interpolate|Transpose)",
        re.IGNORECASE,
    )),
    ("Concat", re.compile(r"Concat", re.IGNORECASE)),
]


def categorize(name: str) -> str:
    for label, pattern in CATEGORY_PATTERNS:
        if pattern.search(name):
            return label

    return "Other"


# --------------------------------------------------------------------------
# Label handling
# --------------------------------------------------------------------------

def short_name(name: str, max_len: int = 40) -> str:
    """
    Convert a potentially verbose module name into a readable label.
    """

    name = str(name)

    replacements = {
        "ResidualEncoderUNet": "ResidualEncoderUNet",
        "ResidualEncoder": "ResidualEncoder",
        "UNetDecoder": "UNetDecoder",
        "StackedConvBlocks": "StackedConvBlocks",
    }

    if name in replacements:
        return replacements[name]

    if len(name) <= max_len:
        return name

    return name[: max_len - 3] + "..."


def make_labels(layers, show_depth=True):
    """
    Generate x-axis labels.

    Example:
        d3 StackedConvBlocks
        d3 Sequential
    """

    labels = []

    for layer in layers:
        name = short_name(layer["name"])

        if show_depth:
            labels.append(f"d{layer['depth']} {name}")
        else:
            labels.append(name)

    return labels


# --------------------------------------------------------------------------
# JSON parsing
# --------------------------------------------------------------------------

def load_profile(path: Path):
    """
    Load the profile JSON.

    Entries without a 'name' field are treated as metadata and ignored.
    """

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"Invalid JSON: {e}")

    if not isinstance(data, list):
        sys.exit("Expected the JSON root to be a list.")

    layers = []

    for entry in data:
        if not isinstance(entry, dict):
            continue

        if "name" not in entry:
            continue

        if "depth" not in entry:
            continue

        layers.append(entry)

    return layers


# --------------------------------------------------------------------------
# Hierarchy handling
# --------------------------------------------------------------------------

def find_leaf_nodes(layers):
    """
    Identify nodes that have no children.

    The profile is assumed to be ordered in depth-first/tree order.

    A node at depth D has a child if the following entry has depth > D
    before the tree returns to depth <= D.
    """

    leaves = []

    for i, layer in enumerate(layers):
        depth = int(layer["depth"])

        has_child = False

        if i + 1 < len(layers):
            next_depth = int(layers[i + 1]["depth"])

            if next_depth > depth:
                has_child = True

        if not has_child:
            leaves.append(layer)

    return leaves


def filter_layers(layers, depth=None, leaves=False, drop_zero=False):
    """
    Select which hierarchy nodes should be plotted.

    Exactly one of depth/leaves should normally be used.
    """

    if leaves:
        selected = find_leaf_nodes(layers)

    elif depth is not None and depth != "all":
        selected = [
            layer
            for layer in layers
            if int(layer["depth"]) == int(depth)
        ]

    else:
        selected = list(layers)

    if drop_zero:
        selected = [
            layer
            for layer in selected
            if float(layer.get("averageMs", layer.get("timeMs", 0.0))) != 0.0
        ]

    return selected


# --------------------------------------------------------------------------
# Parent hierarchy utilities
# --------------------------------------------------------------------------

def build_parent_paths(layers):
    """
    Build a textual parent path for every node.

    Example:

        ResidualEncoderUNet
        ResidualEncoderUNet / ResidualEncoder
        ResidualEncoderUNet / ResidualEncoder / StackedConvBlocks

    Returns:
        list[str]
    """

    stack = []
    paths = []

    for layer in layers:
        depth = int(layer["depth"])
        name = str(layer["name"])

        # Keep entries up to the parent depth.
        while len(stack) > depth:
            stack.pop()

        if len(stack) == depth:
            stack.append(name)
        else:
            # Defensive handling if the profile has unusual depth jumps.
            while len(stack) < depth:
                stack.append("?")

            stack.append(name)

        paths.append(" / ".join(stack))

    return paths


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_profile(
    layers,
    metric="averageMs",
    top_n=10,
    out_path="pytorch_profile.png",
    title="PyTorch module timing profile",
    show_depth=True,
):
    """
    Plot selected PyTorch profile nodes.

    IMPORTANT:
    The caller is responsible for selecting a non-overlapping hierarchy
    level, e.g. --leaves or --depth 3.
    """

    names = [str(l["name"]) for l in layers]

    values = np.array(
        [float(l.get(metric, 0.0)) for l in layers],
        dtype=float,
    )

    depths = np.array(
        [int(l["depth"]) for l in layers],
        dtype=int,
    )

    categories = [categorize(n) for n in names]
    labels = make_labels(layers, show_depth=show_depth)

    if len(values) == 0:
        sys.exit("No layers selected for plotting.")

    # ------------------------------------------------------------------
    # Category colors
    # ------------------------------------------------------------------

    cat_totals = {}

    for c, v in zip(categories, values):
        cat_totals[c] = cat_totals.get(c, 0.0) + v

    cats_in_order = sorted(
        cat_totals,
        key=lambda c: -cat_totals[c],
    )

    cmap = plt.get_cmap(
        "tab10" if len(cats_in_order) <= 10 else "tab20"
    )

    color_of = {
        c: cmap(i % cmap.N)
        for i, c in enumerate(cats_in_order)
    }

    bar_colors = [color_of[c] for c in categories]

    x = np.arange(len(layers))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig_width = max(14, len(layers) * 0.18)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3]},
    )

    # ------------------------------------------------------------------
    # Panel 1: per-node timing
    # ------------------------------------------------------------------

    ax1.bar(
        x,
        values,
        color=bar_colors,
        width=0.9,
        edgecolor="none",
    )

    ax1.set_ylabel(f"{metric} (ms)")
    ax1.set_title(title)

    # ------------------------------------------------------------------
    # Depth bands
    #
    # Useful when plotting --depth all.
    # For --leaves/depth N there is usually only one depth.
    # ------------------------------------------------------------------

    prev_depth = None
    band_start = 0
    toggle = False

    ymax = max(values.max(), 1e-9)

    for i, depth in enumerate(list(depths) + [None]):

        if depth != prev_depth:

            if prev_depth is not None:

                if toggle:
                    ax1.axvspan(
                        band_start - 0.5,
                        i - 0.5,
                        color="gray",
                        alpha=0.06,
                        zorder=0,
                    )

                ax1.text(
                    (band_start + i - 1) / 2,
                    ymax * 0.02,
                    f"depth {prev_depth}",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="dimgray",
                )

                toggle = not toggle

            band_start = i
            prev_depth = depth

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=color_of[c],
        )
        for c in cats_in_order
    ]

    ax1.legend(
        handles,
        cats_in_order,
        loc="upper right",
        fontsize=8,
        ncol=1,
        framealpha=0.9,
        title="Module type",
    )

    # ------------------------------------------------------------------
    # Top-N annotations
    # ------------------------------------------------------------------

    positive_indices = np.argsort(values)[::-1][:top_n]

    for i in positive_indices:

        if values[i] <= 0:
            continue

        ax1.annotate(
            f"{values[i]:.2f}",
            (x[i], values[i]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7,
            color="black",
        )

    # ------------------------------------------------------------------
    # Panel 2: cumulative time
    # ------------------------------------------------------------------

    cumulative = np.cumsum(values)

    ax2.plot(
        x,
        cumulative,
        color="black",
        lw=1.5,
    )

    ax2.fill_between(
        x,
        cumulative,
        alpha=0.15,
    )

    ax2.set_ylabel(
        f"cumulative {metric} (ms)"
    )

    ax2.set_xlabel(
        "module / operation (profile order)"
    )

    # ------------------------------------------------------------------
    # X-axis labels
    # ------------------------------------------------------------------

    step = max(
        1,
        len(layers) // 40,
    )

    ax2.set_xticks(x[::step])

    ax2.set_xticklabels(
        [labels[i] for i in range(0, len(layers), step)],
        rotation=90,
        fontsize=6,
    )

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    ax1.grid(
        axis="y",
        alpha=0.3,
    )

    ax2.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot -> {out_path}")

    return (
        names,
        values,
        categories,
        labels,
        depths,
    )


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def print_summary(
    layers,
    names,
    values,
    categories,
    labels,
    metric,
    top_n=10,
):
    """
    Print timing summary for the selected, non-overlapping nodes.
    """

    total = values.sum()

    print()
    print(
        f"Total {metric} across {len(values)} selected nodes: "
        f"{total:.3f} ms"
    )
    print()

    print("Selected nodes:")

    for i, layer in enumerate(layers):
        print(
            f"  d{layer['depth']:<2} "
            f"{values[i]:8.3f} ms  "
            f"[{categories[i]:<16}] "
            f"{labels[i]}"
        )

    # ------------------------------------------------------------------
    # Category summary
    # ------------------------------------------------------------------

    print()
    print("By category:")

    cat_totals = {}

    for c, v in zip(categories, values):
        cat_totals[c] = cat_totals.get(c, 0.0) + v

    for c, v in sorted(
        cat_totals.items(),
        key=lambda kv: -kv[1],
    ):

        if total > 0:
            pct = 100.0 * v / total
        else:
            pct = 0.0

        print(
            f"  {c:<24} "
            f"{v:8.3f} ms  "
            f"({pct:5.1f}%)"
        )

    # ------------------------------------------------------------------
    # Top N
    # ------------------------------------------------------------------

    print()
    print(
        f"Top {top_n} slowest selected nodes:"
    )

    order = np.argsort(values)[::-1][:top_n]

    for rank, i in enumerate(order, 1):

        layer = layers[i]

        print(
            f"  {rank:2d}. "
            f"{values[i]:8.3f} ms  "
            f"[d{layer['depth']}] "
            f"[{categories[i]}] "
            f"{labels[i]}"
        )


# --------------------------------------------------------------------------
# Hierarchy diagnostic
# --------------------------------------------------------------------------

def print_hierarchy(layers):
    """
    Print the complete hierarchy without doing timing aggregation.

    Useful for understanding what the JSON contains.
    """

    print()
    print("Profile hierarchy:")
    print()

    for layer in layers:

        depth = int(layer["depth"])
        name = str(layer["name"])
        value = float(
            layer.get(
                "averageMs",
                layer.get("timeMs", 0.0),
            )
        )

        indent = "  " * depth

        print(
            f"{indent}d{depth} "
            f"{name} "
            f"({value:.3f} ms)"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "json_path",
        type=Path,
        help="path to the PyTorch profile JSON",
    )

    ap.add_argument(
        "-o",
        "--out",
        default="",
        help="output image path (default: pytorch_profile.png)",
    )

    ap.add_argument(
        "--metric",
        default="averageMs",
        choices=[
            "averageMs",
            "medianMs",
            "timeMs",
            "percentage",
        ],
        help=(
            "which field to plot "
            "(default: averageMs)"
        ),
    )

    selection = ap.add_mutually_exclusive_group()

    selection.add_argument(
        "--depth",
        default=None,
        help=(
            "plot only nodes at this depth. "
            "Use --depth all to plot the complete hierarchy."
        ),
    )

    selection.add_argument(
        "--leaves",
        action="store_true",
        help=(
            "plot only leaf nodes. "
            "This is the recommended mode for avoiding "
            "parent/child double counting."
        ),
    )

    ap.add_argument(
        "--top",
        type=int,
        default=10,
        help=(
            "number of slowest nodes to annotate/report "
            "(default: 10)"
        ),
    )

    ap.add_argument(
        "--drop-zero",
        action="store_true",
        help="remove nodes whose selected metric is zero",
    )

    ap.add_argument(
        "--no-depth-label",
        action="store_true",
        help="do not prefix x-axis labels with the depth",
    )

    ap.add_argument(
        "--show-hierarchy",
        action="store_true",
        help="print the complete profile hierarchy before plotting",
    )

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    layers = load_profile(args.json_path)

    if not layers:
        sys.exit(
            "No layer entries found in JSON "
            "(expected objects with 'name' and 'depth')."
        )

    # ------------------------------------------------------------------
    # Optional hierarchy dump
    # ------------------------------------------------------------------

    if args.show_hierarchy:
        print_hierarchy(layers)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    if args.leaves:
        selection_description = "leaf nodes"

    elif args.depth is not None:

        if args.depth == "all":
            selection_description = "all hierarchy levels"
        else:
            try:
                int(args.depth)
            except ValueError:
                sys.exit(
                    "--depth must be an integer or 'all'"
                )

            selection_description = (
                f"depth {args.depth}"
            )

    else:
        # Default: leaves.
        args.leaves = True
        selection_description = "leaf nodes (default)"

    selected = filter_layers(
        layers,
        depth=args.depth,
        leaves=args.leaves,
        drop_zero=args.drop_zero,
    )

    if not selected:
        sys.exit(
            "No nodes selected. "
            "Check the requested depth or filtering options."
        )

    print()
    print(
        f"Loaded {len(layers)} profile nodes."
    )

    print(
        f"Plotting {len(selected)} nodes: "
        f"{selection_description}."
    )

    if args.depth == "all":
        print(
            "WARNING: --depth all includes parents and children. "
            "The resulting cumulative curve is NOT an execution "
            "time total because hierarchical times overlap."
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    names, values, categories, labels, depths = plot_profile(
        selected,
        metric=args.metric,
        top_n=args.top,
        out_path=str(args.out) if str(args.out) else str(args.json_path.name).replace(".json", ".png"),
        title=(
            "PyTorch module timing — "
            f"{selection_description}"
        ),
        show_depth=not args.no_depth_label,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print_summary(
        selected,
        names,
        values,
        categories,
        labels,
        args.metric,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()

