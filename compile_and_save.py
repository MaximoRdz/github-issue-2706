"""
Compile and save TensorRT engines ahead of time, driven by named recipes
in compile_configs.json.

This is the BULK/OFFLINE counterpart to the auto-compile-on-first-use
behavior in common.build_network(): run_single.py / run_all.py will
already compile a missing engine automatically the first time an
experiment needs it, so you don't have to run this script at all if
you're fine paying the compile cost inline on first use. Use this
script when you want to pre-warm every engine for a sweep ahead of
time (e.g. before an overnight run_all.py), or to (re)compile without
running any benchmark.

Usage
-----
# compile one mode, both configurations (default)
python compile_and_save.py \\
    --dataset-name Dataset306_BONE_TUMOR_EXTENDED \\
    --nnunet-preprocessed /path/to/nnUNet_preprocessed/Dataset306_BONE_TUMOR_EXTENDED \\
    --plans-filename nnUNetPlans.json \\
    --mode trt-solution

# compile every TRT mode in the registry, 2d only
python compile_and_save.py ... --modes all --configurations 2d

# force-recompile even if the .ep already exists
python compile_and_save.py ... --modes all --force

# dry-run: torch_tensorrt partitioning report only, no real usable engine saved
python compile_and_save.py ... --mode github-issue --dry-run
"""
import argparse
import json
from pathlib import Path

import torch

from common import (CONFIGURATIONS, DEFAULT_COMPILE_CONFIGS_PATH, PlansManager, build_mode_registry,
                     compile_engine, export_raw_trt_engine, load_compile_configs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--nnunet-preprocessed", type=Path, required=True,
                    help="Path to nnUNet_preprocessed/<dataset_name>")
    p.add_argument("--plans-filename", default="nnUNetPlans.json")
    p.add_argument("--compile-configs", type=Path, default=DEFAULT_COMPILE_CONFIGS_PATH)
    p.add_argument("--modes", nargs="+", default=["all"],
                    help="Mode name(s) from --compile-configs to (re)compile, or 'all' (default) "
                         "to compile every TRT mode defined there. Modes that share an "
                         "engine_suffix (e.g. a plain mode and its cuda-graphs twin) are only "
                         "compiled once.")
    p.add_argument("--configurations", nargs="+", default=list(CONFIGURATIONS), choices=CONFIGURATIONS)
    p.add_argument("--compiled-engines-dir", type=Path, default=None,
                    help="defaults to ./<dataset_name>")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="Recompile even if the engine file already exists.")
    p.add_argument("--dry-run", action="store_true",
                    help="torch_tensorrt dryrun=True -- prints the partitioning report but does NOT "
                         "produce a real, usable compiled engine.")
    p.add_argument("--export-raw-engine", action="store_true",
                    help="Also export a standalone .engine file (trt_raw_<config>_<suffix>.engine) "
                         "loadable directly by trtexec, alongside the normal .ep artifact. Ignores "
                         "--force for this part -- a raw engine is always (re)built when requested, "
                         "since it's cheap relative to the .ep compile and there's no separate "
                         "existence check to skip.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    compiled_engines_dir = args.compiled_engines_dir or (Path(".") / args.dataset_name)

    with open(args.nnunet_preprocessed / args.plans_filename) as f:
        plans = json.load(f)
    with open(args.nnunet_preprocessed / "dataset.json") as f:
        dataset_json = json.load(f)
    plans_manager = PlansManager(plans)
    num_input_channels = len(dataset_json["channel_names"])

    compile_configs = load_compile_configs(args.compile_configs)
    mode_registry = build_mode_registry(compile_configs)

    requested = list(mode_registry) if args.modes == ["all"] else args.modes
    unknown = [m for m in requested if m not in mode_registry]
    if unknown:
        raise SystemExit(f"Unknown mode(s) {unknown}. Available: {sorted(mode_registry)}")
    modes = [m for m in requested if mode_registry[m].compile_kwargs is not None]
    skipped_no_recipe = [m for m in requested if m not in modes]
    if skipped_no_recipe:
        print(f"Skipping (no compile_kwargs, e.g. a pytorch baseline): {skipped_no_recipe}")
    if not modes:
        raise SystemExit("Nothing to compile.")

    compiled_this_run = set()      # (configuration_name, engine_suffix) -- avoid recompiling a shared .ep twice
    raw_exported_this_run = set()  # (configuration_name, engine_suffix) -- avoid re-exporting a shared raw engine twice
    for configuration_name in args.configurations:
        configuration_manager = plans_manager.get_configuration(configuration_name)
        for mode in modes:
            spec = mode_registry[mode]
            key = (configuration_name, spec.engine_suffix)
            engine_path = compiled_engines_dir / f"trt_compiled_{configuration_name}_{spec.engine_suffix}.ep"

            if key in compiled_this_run:
                print(f"[skip] mode='{mode}' shares engine_suffix='{spec.engine_suffix}' with an "
                      f"already-compiled mode this run -- {engine_path} is up to date")
            elif engine_path.exists() and not args.force:
                print(f"[skip] {engine_path} already exists (use --force to recompile)")
                compiled_this_run.add(key)
            else:
                compile_engine(
                    mode, spec,
                    plans_manager=plans_manager, configuration_manager=configuration_manager,
                    configuration_name=configuration_name, dataset_json=dataset_json,
                    num_input_channels=num_input_channels, device=device,
                    compiled_engines_dir=compiled_engines_dir, dry_run=args.dry_run,
                )
                compiled_this_run.add(key)

            if args.export_raw_engine and not args.dry_run and key not in raw_exported_this_run:
                export_raw_trt_engine(
                    mode, spec,
                    plans_manager=plans_manager, configuration_manager=configuration_manager,
                    configuration_name=configuration_name, dataset_json=dataset_json,
                    num_input_channels=num_input_channels, device=device,
                    compiled_engines_dir=compiled_engines_dir,
                )
                raw_exported_this_run.add(key)

    print(f"\nDone. Compiled/verified {len(compiled_this_run)} .ep engine(s) in {compiled_engines_dir}")
    if args.export_raw_engine:
        print(f"Exported {len(raw_exported_this_run)} raw .engine file(s) for trtexec.")


if __name__ == "__main__":
    main()
