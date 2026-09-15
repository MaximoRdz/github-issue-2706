"""
Run exactly ONE (configuration, mode) benchmarking experiment in a fresh
Python process and write the result to a JSON file.

Running each experiment in its own process is what actually gives you a
"reset to scratch" GPU/allocator/cuDNN/TensorRT state between
configurations and modes -- there is no reliable way to fully undo
TensorRT engine / CUDA graph state within a single long-lived process.

Example
-------
python run_single.py \\
    --nnunet-preprocessed /lustre/.../nnUNet_preprocessed/Dataset027_ACDC \\
    --configuration 3d_fullres \\
    --mode trt-cuda-graphs-solution \\
    --output-json ./bench_results/3d_fullres_trt-cuda-graphs-solution.json
"""
import argparse
import json
from pathlib import Path

from common import (
    CONFIGURATIONS,
    DEFAULT_COMPILE_CONFIGS_PATH,
    ExperimentConfig,
    run_experiment,
    WARMUP_ITERATIONS,
    ITERATIONS
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-name", default="Dataset027_ACDC")
    p.add_argument("--nnunet-preprocessed", type=Path, required=True,
                    help="Path to nnUNet_preprocessed/<dataset_name>")
    p.add_argument("--plans-filename", default="nnUNetResEncUNetLPlans.json")
    p.add_argument("--measure-scope", default="full-inference")
    p.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    p.add_argument("--mode", required=True,
                    help="'pytorch', 'pytorch-cuda-graphs-solution', or any mode name defined in "
                         "--compile-configs (e.g. 'github-issue', 'trt-solution', or your own "
                         "custom recipe name). If that mode's engine .ep doesn't exist yet, it's "
                         "compiled automatically before benchmarking (see --no-auto-compile).")
    p.add_argument("--compile-configs", type=Path, default=DEFAULT_COMPILE_CONFIGS_PATH,
                    help="JSON registry of named TensorRT compile recipes (default: "
                         "compile_configs.json next to this script).")
    p.add_argument("--no-auto-compile", action="store_true",
                    help="Fail with FileNotFoundError instead of auto-compiling a missing engine.")
    p.add_argument("--dry-run-compile", action="store_true",
                    help="If an engine needs auto-compiling, pass dryrun=True to torch_tensorrt.compile "
                         "(partitioning report only, NOT a real usable engine -- see compile_and_save.py).")
    p.add_argument("--compiled-engines-dir", type=Path, default=None,
                    help="Directory containing/receiving trt_compiled_<config>_<engine_suffix>.ep "
                         "(defaults to ./<dataset_name>)")
    p.add_argument("--warmup-iterations", type=int, default=WARMUP_ITERATIONS)
    p.add_argument("--iterations", type=int, default=ITERATIONS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--patient-files", nargs="+", type=Path, default=None,
                    help="One or more preprocessed-order image files for a REAL patient "
                         "case (e.g. --patient-files case_0000.nii.gz case_0001.nii.gz), "
                         "in dataset_json channel order. Runs nnU-Net's own preprocessing "
                         "(resampling + normalization) and then benchmarks the whole "
                         "resulting volume's sliding-window pass -- batch stays 1 always. "
                         "If omitted, a synthetic single-patch tensor is used instead "
                         "(a fast microbenchmark of one tile, not a whole case).")
    p.add_argument("--postprocess", action="store_true",
                    help="Resample the output back to the ORIGINAL patient geometry "
                         "(e.g. 512x512 instead of the network's resampled 437x429). "
                         "Untimed, runs after the benchmark loop. Requires --patient-files.")
    p.add_argument("--save-segmentation", type=Path, default=None,
                    help="Write the final label-map segmentation to this path (e.g. a "
                         ".nii.gz), aligned to the original patient geometry. Implies "
                         "--postprocess. Requires --patient-files.")
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compiled_engines_dir = args.compiled_engines_dir or (Path(".") / args.dataset_name)

    cfg = ExperimentConfig(
        dataset_name=args.dataset_name,
        nnunet_preprocessed_dir=args.nnunet_preprocessed,
        plans_filename=args.plans_filename,
        configuration_name=args.configuration,
        mode=args.mode,
        compiled_engines_dir=compiled_engines_dir,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        device=args.device,
        verbose=args.verbose,
        patient_files=args.patient_files,
        postprocess_to_original_shape=args.postprocess,
        save_segmentation_path=args.save_segmentation,
        compile_configs_path=args.compile_configs,
        auto_compile=not args.no_auto_compile,
        dry_run_compile=args.dry_run_compile,
        measure_scope=args.measure_scope,
    )

    print(f"=== configuration={cfg.configuration_name} mode={cfg.mode} ===")
    result = run_experiment(cfg)
    print(json.dumps(result, indent=2))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
