#!/usr/bin/env bash
# Profile ONE (configuration, mode) experiment with Nsight Systems and
# dump a kernel-time summary next to the raw trace.
#
# Under the hood this just runs run_single.py (with a small default
# iteration count, since a profiling trace doesn't need 20+ iterations
# and huge traces are painful to load) inside `nsys profile`, then runs
# `nsys stats` against the resulting .nsys-rep.
#
# common.py wraps warmup calls and each measured call in its own NVTX
# range ("warmup_0", "measured_iter_0", "measured_iter_1", ...) plus
# ranges around network build / input prep / finalize, so the trace
# timeline is directly readable instead of one undifferentiated blob
# of kernels.
#
# Usage
# -----
#   ./profile_experiment.sh <configuration> <mode> [-- extra run_single.py args]
#
# Example
# -------
#   ./profile_experiment.sh 3d_fullres trt-solution -- \
#       --dataset-name Dataset306_BONE_TUMOR_EXTENDED \
#       --nnunet-preprocessed /lustre/.../nnUNet_preprocessed/Dataset306_BONE_TUMOR_EXTENDED \
#       --plans-filename nnUNetPlans.json \
#       --compiled-engines-dir ./Dataset306_BONE_TUMOR_EXTENDED
#
# Env overrides
# -------------
#   WARMUP_ITERATIONS  (default 2)
#   ITERATIONS         (default 5)
#   OUTPUT_DIR         (default ./profiling)

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <configuration> <mode> [-- extra run_single.py args]" >&2
    exit 1
fi

CONFIGURATION="$1"; shift
MODE="$1"; shift

# allow an optional leading "--" separator before pass-through args
if [[ "${1:-}" == "--" ]]; then
    shift
fi

WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-2}"
ITERATIONS="${ITERATIONS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-./profiling}"
mkdir -p "$OUTPUT_DIR"

RUN_TAG="${CONFIGURATION}_${MODE}"
REPORT_PREFIX="$OUTPUT_DIR/nsys-${RUN_TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Profiling configuration=${CONFIGURATION} mode=${MODE} " \
     "(warmup=${WARMUP_ITERATIONS}, measured=${ITERATIONS})"

nsys profile \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --pytorch=autograd-shapes-nvtx \
    --python-backtrace=cuda \
    --python-sampling=true \
    --force-overwrite=true \
    -o "$REPORT_PREFIX" \
    python "$SCRIPT_DIR/run_single.py" \
        --configuration "$CONFIGURATION" \
        --mode "$MODE" \
        --warmup-iterations "$WARMUP_ITERATIONS" \
        --iterations "$ITERATIONS" \
        --verbose \
        --output-json "$OUTPUT_DIR/${RUN_TAG}.json" \
        "$@"

echo ">>> Extracting kernel summary..."
nsys stats \
    --report cuda_gpu_kern_sum \
    --force-export=true \
    "${REPORT_PREFIX}.nsys-rep" \
    > "$OUTPUT_DIR/${RUN_TAG}_kernels.txt"

echo ">>> Extracting NVTX range summary (correlates kernels back to warmup_N / measured_iter_N)..."
nsys stats \
    --report nvtx_sum \
    --force-export=true \
    "${REPORT_PREFIX}.nsys-rep" \
    > "$OUTPUT_DIR/${RUN_TAG}_nvtx.txt"

echo ""
echo "Done. Wrote:"
echo "  ${REPORT_PREFIX}.nsys-rep              (open in the Nsight Systems GUI for the full timeline)"
echo "  $OUTPUT_DIR/${RUN_TAG}_kernels.txt   (per-kernel GPU time summary)"
echo "  $OUTPUT_DIR/${RUN_TAG}_nvtx.txt      (time spent per NVTX range: warmup_0, measured_iter_3, ...)"
echo "  $OUTPUT_DIR/${RUN_TAG}.json          (the usual run_single.py latency summary)"
