"""
Driver that runs every (configuration, mode) combination as an
independent subprocess -- each one gets a fresh CUDA/cuDNN/TensorRT
context, see run_single.py -- collects the JSON result of each, and
prints/saves a comparison table.

Example
-------
python run_all.py \\
    --nnunet-preprocessed /lustre/.../nnUNet_preprocessed/Dataset027_ACDC \\
    --results-dir ./bench_results
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import (
    CONFIGURATIONS,
    DEFAULT_COMPILE_CONFIGS_PATH,
    build_mode_registry,
    load_compile_configs,
    WARMUP_ITERATIONS,
    ITERATIONS
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-name", default="Dataset027_ACDC")
    p.add_argument("--nnunet-preprocessed", type=Path, required=True)
    p.add_argument("--plans-filename", default="nnUNetResEncUNetLPlans.json")
    p.add_argument("--measure-scope", default="full-inference")
    p.add_argument("--compile-configs", type=Path, default=DEFAULT_COMPILE_CONFIGS_PATH,
                    help="JSON registry of named TensorRT compile recipes (default: "
                         "compile_configs.json next to this script). Also determines the "
                         "default --modes list (every mode defined there, plus the pytorch "
                         "baselines) if --modes isn't given explicitly.")
    p.add_argument("--no-auto-compile", action="store_true",
                    help="Fail instead of auto-compiling any missing engine.")
    p.add_argument("--dry-run-compile", action="store_true",
                    help="If an engine needs auto-compiling, pass dryrun=True (partitioning "
                         "report only, NOT a real usable engine).")
    p.add_argument("--compiled-engines-dir", type=Path, default=None,
                    help="defaults to ./<dataset_name>")
    p.add_argument("--configurations", nargs="+", default=list(CONFIGURATIONS), choices=CONFIGURATIONS)
    p.add_argument("--modes", nargs="+", default=None,
                    help="Mode name(s) to sweep. Defaults to every mode in --compile-configs "
                         "plus the pytorch baselines.")
    p.add_argument("--warmup-iterations", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--patient-files", nargs="+", type=Path, default=None,
                    help="Optional: real patient case files (see run_single.py --help). "
                         "The SAME case is used for every (configuration, mode) run in the sweep.")
    p.add_argument("--results-dir", type=Path, default=Path("./bench_results"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    compiled_engines_dir = args.compiled_engines_dir or (Path(".") / args.dataset_name)
    run_single_py = Path(__file__).parent / "run_single.py"

    modes = args.modes
    if modes is None:
        registry = build_mode_registry(load_compile_configs(args.compile_configs))
        modes = sorted(registry)
        print(f"--modes not given, using every mode in the registry: {modes}")

    results = []
    for configuration in args.configurations:
        for mode in modes:
            out_json = args.results_dir / f"{configuration}__{mode}.json"
            cmd = [
                sys.executable, str(run_single_py),
                "--dataset-name", args.dataset_name,
                "--nnunet-preprocessed", str(args.nnunet_preprocessed),
                "--plans-filename", args.plans_filename,
                "--configuration", configuration,
                "--mode", mode,
                "--compile-configs", str(args.compile_configs),
                "--compiled-engines-dir", str(compiled_engines_dir),
                "--warmup-iterations", str(args.warmup_iterations),
                "--iterations", str(args.iterations),
                "--device", args.device,
                "--output-json", str(out_json),
                "--measure-scope", args.measure_scope,
            ]
            if args.no_auto_compile:
                cmd += ["--no-auto-compile"]
            if args.verbose:
                cmd += ["--verbose"]
            if args.dry_run_compile:
                cmd += ["--dry-run-compile"]
            if args.patient_files:
                cmd += ["--patient-files", *[str(f) for f in args.patient_files]]
            print(f"\n>>> Running: configuration={configuration} mode={mode}")
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"!!! Experiment FAILED (configuration={configuration}, mode={mode}) -- "
                      f"skipping it in the comparison table")
                continue
            with open(out_json) as f:
                results.append(json.load(f))

    print_table(results)

    combined_path = args.results_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved combined results to {combined_path}")


def print_table(results: list) -> None:
    if not results:
        print("\nNo successful results to report.")
        return
    header = f"{'configuration':<14}{'mode':<32}{'mean_ms':>10}{'median_ms':>12}{'p95_ms':>10}{'std_ms':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: (r["configuration"], r["mode"])):
        print(f"{r['configuration']:<14}{r['mode']:<32}{r['mean_ms']:>10.2f}"
              f"{r['median_ms']:>12.2f}{r['p95_ms']:>10.2f}{r['std_ms']:>10.2f}")


if __name__ == "__main__":
    main()
