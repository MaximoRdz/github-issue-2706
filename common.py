"""
Shared utilities for the nnU-Net sliding-window inference benchmark suite.

Design decisions
----------------
1. Timing isolation
   The GPU timer wraps ONLY the core sliding-window inference call
   (`nnUNetPredictor._internal_predict_sliding_window_return_logits`).
   Padding and slicer computation are input-dependent, not mode-dependent,
   so they run exactly ONCE (`BenchPredictor.prepare`) before the timed
   loop instead of being repeated -- and measured -- on every iteration.

2. Process isolation
   torch, cuDNN, TensorRT and CUDA graphs all keep internal state
   (autotuned-algorithm cache, captured device-memory addresses, engine
   contexts) that `del` + `torch.cuda.empty_cache()` cannot reliably undo
   within a single process. To guarantee every (configuration, mode)
   experiment starts from an identical "cold" GPU/allocator state, each
   experiment is meant to be run in its own subprocess -- see run_all.py.
   `cleanup_gpu()` here is only a best-effort in-process cleanup for
   quick/manual runs.

3. No branching duplication
   Each mode is one entry in MODE_REGISTRY. Adding a new mode means
   adding one ModeSpec, not copy-pasting a block.
"""

from __future__ import annotations

import gc
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Dict, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.nvtx as nvtx
from torch import nn

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from acvl_utils.cropping_and_padding.padding import pad_nd_image

# torch_tensorrt registers the custom ops needed to deserialize/run TRT-compiled
# torch.export programs. Import it unconditionally (as the original script did)
# so "github-issue" / "trt-*" modes work even though "pytorch" mode doesn't need it.
import torch_tensorrt  # noqa: F401

cudnn.benchmark = True

CONFIGURATIONS = ("2d", "3d_fullres")

# Torch dtype strings usable inside compile_configs.json (JSON can't hold
# torch.dtype objects directly, so recipes spell them as strings and we
# resolve them here before handing kwargs to torch_tensorrt.compile).
DTYPE_MAP = {
    "float32": torch.float32, "float": torch.float32,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}

DEFAULT_COMPILE_CONFIGS_PATH = Path(__file__).resolve().parent / "compile_configs.json"


# --------------------------------------------------------------------------- #
# GPU timing
# --------------------------------------------------------------------------- #

@dataclass
class TimingResult:
    gpu_latencies_ms: np.ndarray
    wall_latencies_ms: np.ndarray

    def summary(self) -> Dict[str, float]:
        arr = self.gpu_latencies_ms
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "std_ms": float(np.std(arr)),
        }

    def print_summary(self, name: str = "GPU latency (cuda events)") -> None:
        s = self.summary()
        print(f"\n{name}:")
        for k, v in s.items():
            print(f"  {k:10s}: {v:.3f}")


def time_callable(fn: Callable[[], object], warmup_iterations: int, iterations: int) -> TimingResult:
    """
    Times a zero-arg callable with CUDA events (GPU time) and wall clock.

    `fn` should do exactly the work you want measured and nothing else --
    callers keep any setup (padding, slicer computation, context-manager
    entry, moving the network to device, ...) OUTSIDE of `fn`.

    Every warmup call and every measured iteration is wrapped in its own
    NVTX range (visible in `nsys profile --trace=...,nvtx ...`), so a
    profile trace clearly separates "warmup_0", "warmup_1", ...,
    "measured_iter_0", "measured_iter_1", ... instead of showing one
    undifferentiated blob of kernels.
    """
    nvtx.range_push("warmup")
    for i in range(warmup_iterations):
        nvtx.range_push(f"warmup_{i}")
        fn()
        nvtx.range_pop()
    torch.cuda.synchronize()
    nvtx.range_pop()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    wall_times_s = np.empty(iterations)

    nvtx.range_push("measured")
    for i in range(iterations):
        nvtx.range_push(f"measured_iter_{i}")
        torch.cuda.synchronize()  # drain the previous iteration before starting the clock
        wall_start = time.perf_counter()
        start_events[i].record()
        fn()
        end_events[i].record()
        torch.cuda.synchronize()
        wall_times_s[i] = time.perf_counter() - wall_start
        nvtx.range_pop()
    nvtx.range_pop()

    gpu_latencies_ms = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    return TimingResult(gpu_latencies_ms=gpu_latencies_ms, wall_latencies_ms=wall_times_s * 1000.0)


def cleanup_gpu(device: torch.device) -> None:
    """
    Best-effort in-process cleanup. Does NOT guarantee a fully "cold"
    CUDA/cuDNN/TensorRT state -- use process isolation (run_all.py) for
    that guarantee between experiments.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------- #
# Predictor: split into prepare() (untimed) / infer() (timed) / finalize()
# --------------------------------------------------------------------------- #

@dataclass
class PreparedInput:
    data: torch.Tensor
    slicers: list
    slicer_revert_padding: tuple


class BenchPredictor(nnUNetPredictor):
    """
    Same sliding-window logic as nnUNetPredictor, but split so that only
    the actual per-patch network inference is inside the timed region.
    """

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                               configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                               dataset_json: dict, trainer_name: str,
                               inference_allowed_mirroring_axes=(0, 1)) -> None:
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

    @torch.inference_mode()
    def prepare(self, input_image: torch.Tensor) -> PreparedInput:
        """Padding + slicer computation. Input-dependent only, so this
        runs ONCE before the timed loop, never inside it."""
        assert isinstance(input_image, torch.Tensor)
        assert input_image.ndim == 4, "input_image must be a 4D tensor (c, x, y, z)"
        data, slicer_revert_padding = pad_nd_image(
            input_image, self.configuration_manager.patch_size,
            "constant", {"value": 0}, True, None,
        )
        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
        if self.verbose:
            print(f"Input shape: {input_image.shape}")
            print(f"there are {len(slicers)} slicers, e.g. {slicers[0]}")
        return PreparedInput(data=data, slicers=slicers, slicer_revert_padding=slicer_revert_padding)

    @torch.inference_mode()
    def infer(self, prepared: PreparedInput, perform_everything_on_device: bool) -> torch.Tensor:
        """
        THIS is what the benchmark times: a one-to-one wrapper around
        `_internal_predict_sliding_window_return_logits`, with no padding,
        slicer, or device-transfer overhead attached.
        """
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            return self._internal_predict_sliding_window_return_logits(
                prepared.data, prepared.slicers, perform_everything_on_device,
            )

    def finalize(self, prepared: PreparedInput, predicted_logits: torch.Tensor) -> torch.Tensor:
        """Revert the padding applied in prepare(). Not timed."""
        return predicted_logits[(slice(None), *prepared.slicer_revert_padding[1:])]

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
        """Untimed convenience path kept for correctness checks / parity
        with the original single-call API."""
        empty_cache(self.device)
        self.network = self.network.to(self.device)
        prepared = self.prepare(input_image)
        perform_on_device = self.perform_everything_on_device and self.device.type != "cpu"
        try:
            predicted_logits = self.infer(prepared, perform_on_device)
        except RuntimeError:
            print("Prediction on device was unsuccessful (likely OOM). Falling back to CPU result tensors.")
            empty_cache(self.device)
            predicted_logits = self.infer(prepared, False)
        empty_cache(self.device)
        return self.finalize(prepared, predicted_logits)


def get_dummy_network(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                       dataset_json: dict, num_input_channels: int,
                       enable_deep_supervision: bool = False) -> nn.Module:
    """
    Build a raw nnU-Net architecture straight from the plans/configuration,
    with randomly initialized weights -- exactly what nnUNetTrainer.initialize()
    calls internally, without needing a trainer/checkpoint/fingerprint.
    """
    label_manager = plans_manager.get_label_manager(dataset_json)
    return nnUNetTrainer.build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=enable_deep_supervision,
    )

def _apply_precision(
    network: nn.Module,
    input_tensor: torch.Tensor,
    precision: str,
) -> tuple[nn.Module, torch.Tensor]:

    precision = precision.lower()

    if precision == "fp32":
        network = network.float()
        input_tensor = input_tensor.float()

    elif precision == "fp16":
        network = network.half()
        input_tensor = input_tensor.half()

    elif precision == "autocast":
        # Leave model/input in their normal dtype.
        pass

    else:
        raise ValueError(
            f"Unknown compile precision '{precision}'. "
            "Expected: fp32, fp16, autocast"
        )

    return network, input_tensor

# --------------------------------------------------------------------------- #
# Mode registry -- modes come from TWO sources:
#   1. A couple of fixed, always-available baseline modes (plain pytorch,
#      and pytorch wrapped in cuda-graphs) that don't need any compiled
#      artifact.
#   2. Named TensorRT compile recipes loaded from compile_configs.json --
#      add a new TRT variant to compare by adding a JSON entry, not by
#      editing this file. See compile_configs.json for the schema.
# --------------------------------------------------------------------------- #

@dataclass
class ModeSpec:
    name: str
    needs_compiled_engine: bool          # load a torch.export .ep artifact instead of building a fresh network
    engine_suffix: Optional[str] = None  # trt_compiled_<config>_<engine_suffix>.ep -- several modes may share one
    uses_cuda_graphs: bool = False       # wrap the network with torch_tensorrt.runtime.enable_cudagraphs
    precision: str = "autocast"
    compile_kwargs: Optional[dict] = None  # raw torch_tensorrt.compile() kwargs, used only if the engine is missing


PYTORCH_MODE_SPECS: Dict[str, ModeSpec] = {
    "pytorch": ModeSpec("pytorch", needs_compiled_engine=False),
    "pytorch-cuda-graphs-solution": ModeSpec("pytorch-cuda-graphs-solution",
                                              needs_compiled_engine=False, uses_cuda_graphs=True),
}


def load_compile_configs(path: Path) -> dict:
    """
    Loads the named TensorRT compile recipes. See compile_configs.json next
    to this file for the schema and worked examples (github-issue /
    trt-solution / trt-cuda-graphs-solution reproduce the original
    hand-written compile_as_issue / compile_as_solution functions).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Compile-configs registry not found: {path}\n"
            f"Pass --compile-configs to point at your compile_configs.json, "
            f"or create one (see the shipped compile_configs.json for the schema)."
        )
    with open(path) as f:
        return json.load(f)


def build_mode_registry(compile_configs: dict) -> Dict[str, ModeSpec]:
    """
    Merges the fixed pytorch baseline modes with one ModeSpec per entry in
    compile_configs.json. The JSON key IS the mode name used everywhere
    (--mode, run_all.py --modes, result JSON "mode" field) -- e.g.
    "github-issue", "trt-solution", "best_config", "test_whatever_config".
    """
    registry = dict(PYTORCH_MODE_SPECS)
    for mode_name, entry in compile_configs.items():
        if mode_name.startswith("_"):
            continue  # convention: keys starting with "_" are comments/metadata, not modes
        if mode_name in registry:
            raise ValueError(f"compile_configs.json mode name '{mode_name}' collides with a "
                              f"built-in mode name ({list(PYTORCH_MODE_SPECS)}) -- rename it.")
        registry[mode_name] = ModeSpec(
            name=mode_name,
            needs_compiled_engine=True,
            engine_suffix=entry.get("engine_suffix", mode_name),
            uses_cuda_graphs=entry.get("cuda_graphs", False),
            precision=entry.get(
                "precision",
                entry.get("compile_kwargs", {}).get("precision", "autocast")
            ),
            compile_kwargs=entry.get("compile_kwargs"),
        )
    return registry


def _resolve_dtype_fields(d: dict) -> dict:
    """Recursively turns known dtype-bearing string fields ("float16", ...)
    into actual torch dtypes, so compile_configs.json can stay plain JSON."""
    d = dict(d)
    if "enabled_precisions" in d:
        d["enabled_precisions"] = {DTYPE_MAP[p] for p in d["enabled_precisions"]}
    if "autocast_low_precision_type" in d and isinstance(d["autocast_low_precision_type"], str):
        d["autocast_low_precision_type"] = DTYPE_MAP[d["autocast_low_precision_type"]]
    if "options" in d and isinstance(d["options"], dict):
        d["options"] = _resolve_dtype_fields(d["options"])
    return d


def compile_engine(mode: str, spec: ModeSpec, *, plans_manager: PlansManager,
                    configuration_manager: ConfigurationManager, configuration_name: str,
                    dataset_json: dict, num_input_channels: int, device: torch.device,
                    compiled_engines_dir: Path, dry_run: bool = False) -> Path:
    """
    Builds a fresh dummy network and compiles it with TensorRT per
    `spec.compile_kwargs` (sourced from compile_configs.json), saving the
    result to compiled_engines_dir / trt_compiled_<configuration_name>_<engine_suffix>.ep.

    Shared by:
      - build_network()'s auto-compile-if-missing path (lazy, one engine
        at a time, whatever an experiment actually needs), and
      - compile_and_save.py (eager/bulk pre-compilation of a whole sweep).

    Note: mirrors the original compile_as_issue/compile_as_solution
    functions' use of torch_tensorrt.dynamo.Debugger + torch_tensorrt.compile
    + torch_tensorrt.save. If your torch_tensorrt version's API differs,
    the resulting error will point at what's expected.
    """
    if spec.compile_kwargs is None:
        raise ValueError(f"Mode '{mode}' has no compile_kwargs in the compile-configs registry "
                          f"-- can't auto-compile it (only modes loaded from compile_configs.json can be).")

    engine_path = compiled_engines_dir / f"trt_compiled_{configuration_name}_{spec.engine_suffix}.ep"
    print(f"[compile] '{engine_path.name}' not found -- compiling now (mode='{mode}', dry_run={dry_run})")

    network = get_dummy_network(plans_manager, configuration_manager, dataset_json, num_input_channels).to(device)
    input_tensor = build_input_tensor(configuration_manager, num_input_channels, device)



    if configuration_name == "3d_fullres": input_tensor = input_tensor.unsqueeze(0)

    kwargs = _resolve_dtype_fields(spec.compile_kwargs)

    precision = kwargs.pop("precision", "autocast")

    network, input_tensor = _apply_precision(network, input_tensor, precision)

    use_debugger = kwargs.pop("use_debugger", True)
    debugger_log_level = kwargs.pop("debugger_log_level", "error")
    debugger_ctx = torch_tensorrt.dynamo.Debugger(log_level=debugger_log_level) if use_debugger \
        else nullcontext()

    with debugger_ctx:
        trt_gm = torch_tensorrt.compile(
            network, ir="dynamo", inputs=[input_tensor], backend="torch_tensorrt",
            dryrun=dry_run, **kwargs,
        )
        compiled_engines_dir.mkdir(parents=True, exist_ok=True)
        torch_tensorrt.save(trt_gm, str(engine_path), inputs=[input_tensor])

    del network, input_tensor, trt_gm
    cleanup_gpu(device)
    print(f"[compile] saved {engine_path}")
    return engine_path

def export_raw_trt_engine(mode: str, spec: ModeSpec, *, plans_manager: PlansManager,
                           configuration_manager: ConfigurationManager, configuration_name: str,
                           dataset_json: dict, num_input_channels: int, device: torch.device,
                           compiled_engines_dir: Path) -> Path:
    """
    Exports a RAW, standalone serialized TensorRT engine (.engine) for the
    given mode -- loadable directly by `trtexec` or the plain TensorRT
    Python/C++ runtime, with none of the torch.export/torch_tensorrt
    wrapping the normal .ep artifacts (from compile_engine()) carry.

    Uses the same compile_kwargs (from compile_configs.json) as the .ep
    path, via the lower-level torch_tensorrt.dynamo.trace() +
    torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine()
    pair -- the same two steps torch_tensorrt.compile() performs
    internally, just stopping short of re-wrapping the result as a
    torch.export artifact.

    Output: compiled_engines_dir / trt_raw_<configuration_name>_<engine_suffix>.engine

    IMPORTANT, same as the .ep engines: a raw TensorRT .engine file is tied
    to the exact TensorRT version (and typically GPU architecture) it was
    built on -- it is NOT portable across machines/TensorRT installs. Build
    and profile it with trtexec on the same box.

    Also note: this traces the network the same way compile_engine() does,
    so any mode that fails to trace via torch.export will fail identically
    here -- this doesn't route around a tracing failure, only around the
    torch.export re-wrapping step.

    Note: torch_tensorrt's public API surface for this two-step path has
    shifted across versions (`torch_tensorrt.dynamo.trace` /
    `convert_exported_program_to_serialized_trt_engine`). If your installed
    version's signature differs, the resulting error will point at what's
    expected -- paste it back and I'll adjust to match your version.
    """
    if spec.compile_kwargs is None:
        raise ValueError(f"Mode '{mode}' has no compile_kwargs -- can't export a raw engine for it.")

    raw_engine_path = compiled_engines_dir / f"trt_raw_{configuration_name}_{spec.engine_suffix}.engine"
    print(f"[export-raw] building standalone TensorRT engine for mode='{mode}' -> {raw_engine_path}")

    network = get_dummy_network(plans_manager, configuration_manager, dataset_json, num_input_channels).to(device)
    input_tensor = build_input_tensor(configuration_manager, num_input_channels, device)
    if configuration_name == "3d_fullres":
        # same fix as compile_engine() -- see the comment there. The raw
        # nn.Module.forward() being traced needs a genuine batch axis;
        # build_input_tensor deliberately omits one for 3D to match the
        # sliding-window PREPARE contract instead.
        input_tensor = input_tensor.unsqueeze(0)

    kwargs = _resolve_dtype_fields(spec.compile_kwargs)
    precision = kwargs.pop("precision", "autocast")
    network, input_tensor = _apply_precision(network, input_tensor, precision)

    use_debugger = kwargs.pop("use_debugger", True)
    debugger_log_level = kwargs.pop("debugger_log_level", "error")
    # torch_tensorrt.compile() accepts a nested options={...} dict (used by
    # e.g. the github-issue recipe); the lower-level dynamo functions take
    # flat kwargs only, so flatten it back out here.
    if "options" in kwargs:
        kwargs.update(kwargs.pop("options"))
    # 'dynamic' isn't a recognized kwarg on convert_exported_program_to_serialized_trt_engine
    # (dynamism is fully determined by the traced ExportedProgram's shapes instead) --
    # drop it here, same static-shape behavior either way since we always trace
    # against a single fixed input_tensor.
    kwargs.pop("dynamic", None)

    debugger_ctx = torch_tensorrt.dynamo.Debugger(log_level=debugger_log_level) if use_debugger \
        else nullcontext()

    with debugger_ctx:
        exported_program = torch_tensorrt.dynamo.trace(network, [input_tensor])
        serialized_engine = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported_program, inputs=[input_tensor], **kwargs,
        )
        compiled_engines_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_engine_path, "wb") as f:
            f.write(bytes(serialized_engine))

    size_mb = raw_engine_path.stat().st_size / 1e6
    del network, input_tensor, exported_program, serialized_engine
    cleanup_gpu(device)
    print(f"[export-raw] saved {raw_engine_path} ({size_mb:.1f} MB)")
    return raw_engine_path

def build_network(mode: str, *, mode_registry: Dict[str, ModeSpec], plans_manager: PlansManager,
                   configuration_manager: ConfigurationManager, configuration_name: str, dataset_json: dict,
                   num_input_channels: int, device: torch.device, compiled_engines_dir: Path,
                   auto_compile: bool = True, dry_run_compile: bool = False) -> nn.Module:
    if mode not in mode_registry:
        raise KeyError(f"Unknown mode '{mode}'. Available modes: {sorted(mode_registry)}")
    spec = mode_registry[mode]
    if spec.needs_compiled_engine:
        engine_path = compiled_engines_dir / f"trt_compiled_{configuration_name}_{spec.engine_suffix}.ep"
        if not engine_path.exists():
            if not auto_compile:
                raise FileNotFoundError(f"Compiled engine not found: {engine_path} "
                                         f"(auto-compile is disabled -- pass auto_compile=True / drop "
                                         f"--no-auto-compile to build it automatically)")
            compile_engine(
                mode, spec, plans_manager=plans_manager, configuration_manager=configuration_manager,
                configuration_name=configuration_name, dataset_json=dataset_json,
                num_input_channels=num_input_channels, device=device,
                compiled_engines_dir=compiled_engines_dir, dry_run=dry_run_compile,
            )
        return torch.export.load(engine_path).module()
    network = get_dummy_network(plans_manager, configuration_manager, dataset_json, num_input_channels)
    return network.to(device)


def runtime_context(mode: str, mode_registry: Dict[str, ModeSpec], network: nn.Module) -> ContextManager[nn.Module]:
    """
    Context manager yielding the nn.Module to actually run inference with:
    identity for plain modes, the CUDA-graph-captured wrapper for
    `uses_cuda_graphs` modes. Callers must do prepare()/infer() calls
    *inside* this `with` block for cuda-graph modes.
    """
    spec = mode_registry[mode]
    if spec.uses_cuda_graphs:
        return torch_tensorrt.runtime.enable_cudagraphs(network)
    return nullcontext(network)


# --------------------------------------------------------------------------- #
# Experiment config + runner
# --------------------------------------------------------------------------- #

@dataclass
class ExperimentConfig:
    dataset_name: str
    nnunet_preprocessed_dir: Path
    plans_filename: str
    configuration_name: str
    mode: str
    compiled_engines_dir: Path
    warmup_iterations: int = 5
    iterations: int = 20
    device: str = "cuda"
    verbose: bool = False
    patient_files: Optional[List[Path]] = None
    """Real patient case, in dataset_json channel order (e.g.
    [case_0000.nii.gz]). If None (default), a synthetic single-patch
    tensor is used instead -- see build_input_tensor / load_case_from_files."""
    postprocess_to_original_shape: bool = False
    """If True and patient_files is set, also resample the output back to
    the ORIGINAL patient geometry (untimed, after the benchmark loop) --
    see postprocess_to_original_geometry."""
    save_segmentation_path: Optional[Path] = None
    """If set (implies postprocess_to_original_shape), write the final
    label-map segmentation to this path (e.g. a .nii.gz) using the
    dataset's own reader/writer, aligned to the original patient geometry."""
    compile_configs_path: Path = DEFAULT_COMPILE_CONFIGS_PATH
    """Path to the JSON registry of named TensorRT compile recipes."""
    auto_compile: bool = True
    """If a TRT mode's engine file doesn't exist yet, compile it on the
    fly (using its compile_kwargs from compile_configs.json) instead of
    raising FileNotFoundError."""
    dry_run_compile: bool = False
    """Passed through to torch_tensorrt.compile(dryrun=...) if an engine
    needs to be auto-compiled -- True gives a partitioning report only,
    not a real usable engine (see the compile_and_save.py docstring)."""

    def load_plans_and_dataset(self):
        with open(self.nnunet_preprocessed_dir / self.plans_filename) as f:
            plans = json.load(f)
        with open(self.nnunet_preprocessed_dir / "dataset.json") as f:
            dataset_json = json.load(f)
        return plans, dataset_json


def build_input_tensor(configuration_manager: ConfigurationManager, num_input_channels: int,
                        device: torch.device) -> torch.Tensor:
    """
    Synthetic single-tile input: a random tensor with NO batch dimension,
    sized to exactly one patch. Used when no real patient case is given
    (see `load_case_from_files` for the real-data path). The exact axis
    semantics don't matter for random data, only getting ndim right:

    - 2D configurations: patch_size is (y, x) -- 2 spatial dims, so a
      singleton leading axis is needed to reach 4D: (1, C, Y, X).
    - 3D configurations: patch_size is (z, y, x) -- 3 spatial dims, so
      channel alone is enough to reach 4D, with NO extra leading batch
      axis: (C, Z, Y, X).
    """
    patch_size = configuration_manager.patch_size
    if len(patch_size) == 2:
        y, x = patch_size
        shape = (1, num_input_channels, y, x)
    else:
        z, y, x = patch_size
        shape = (num_input_channels, z, y, x)
    return torch.randn(shape, device=device)


def load_case_from_files(image_files: List[Path], plans_manager: PlansManager,
                          configuration_manager: ConfigurationManager, dataset_json: dict,
                          verbose: bool = False):
    """
    Loads and preprocesses a REAL patient case using nnU-Net's OWN
    preprocessing pipeline -- the exact code path real inference uses
    (resampling to the configuration's target spacing, intensity
    normalization, cropping, ...) -- rather than hand-building a tensor.
    That sidesteps having to guess axis ordering per configuration: the
    official preprocessor already returns the convention the predictor
    expects for whichever `configuration_manager` (2D or 3D) you pass in.

    Always a single case -- no batch dimension is added, batch stays 1
    for real data just like it does for the synthetic path.

    `image_files` must be given in dataset_json channel order, e.g.
    [".../case_0000.nii.gz"] for a single-modality dataset, or
    [".../case_0000.nii.gz", ".../case_0001.nii.gz"] for multi-modal.

    Note: this relies on `ConfigurationManager.preprocessor_class` and
    `<PreprocessorClass>.run_case(...)` from the standard nnU-Net v2 API.
    If your installed nnunetv2 version's signature differs, the resulting
    error will point at exactly what's expected -- happy to adjust if so.
    """
    expected_channels = len(dataset_json["channel_names"])
    if len(image_files) != expected_channels:
        raise ValueError(
            f"This dataset expects {expected_channels} channel(s) "
            f"({list(dataset_json['channel_names'].values())}), but got "
            f"{len(image_files)} file(s): {image_files}"
        )
    for f in image_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Patient image file not found: {f}")

    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    data, seg, properties = preprocessor.run_case(
        image_files=[str(f) for f in image_files],
        seg_file=None,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        dataset_json=dataset_json,
    )

    data_tensor = torch.from_numpy(data).float()
    if verbose:
        print(f"Loaded real case: {[str(f) for f in image_files]}")
        print(f"  preprocessed shape (c, ...spatial), batch=1: {tuple(data_tensor.shape)}")
        print(f"  original spacing:  {properties.get('spacing')}")
        print(f"  original shape:    {properties.get('shape_before_cropping', properties.get('original_size_of_raw_data'))}")

    return data_tensor, properties


def postprocess_to_original_geometry(predictor: "BenchPredictor", predicted_logits: torch.Tensor,
                                      properties: dict, return_probabilities: bool = True):
    """
    Undo the RESAMPLING (not just the sliding-window padding `finalize()`
    undoes) so the result is aligned back to the ORIGINAL patient geometry
    -- e.g. (76, 512, 512) instead of the resampled (76, 437, 429) network
    space. This mirrors what real nnU-Net inference does after
    predict_sliding_window_return_logits, via
    nnUNetPredictor.convert_predicted_logits_to_segmentation_with_correct_shape.

    This is CPU-bound resampling, not GPU inference -- call it OUTSIDE the
    timed benchmark loop. It's a correctness/visualization step, not a
    latency measurement.

    return_probabilities=True  -> float array (num_classes, *original_shape)
    return_probabilities=False -> integer label map (*original_shape,)

    Note: relies on the standard nnU-Net v2
    `convert_predicted_logits_to_segmentation_with_correct_shape` API.
    If your installed version's signature/return contract differs, the
    resulting error will point at what's expected.
    """
    logits_np = predicted_logits.detach().cpu().numpy() if isinstance(predicted_logits, torch.Tensor) \
        else predicted_logits
    result = predictor.convert_predicted_logits_to_segmentation_with_correct_shape(
        logits_np, predictor.plans_manager, predictor.configuration_manager, predictor.label_manager,
        properties, return_probabilities=return_probabilities,
    )
    if return_probabilities:
        # some nnU-Net versions return (segmentation, probabilities), others just probabilities
        return result[1] if isinstance(result, tuple) else result
    return result


def describe_dtype(network, input_tensor, precision):
    if isinstance(network, torch.nn.Module):
        try:
            network_dtype = next(network.parameters()).dtype
        except StopIteration:
            network_dtype = "no parameters"
        backend = "PyTorch"
    else:
        network_dtype = getattr(network, "dtype", "N/A")
        backend = type(network).__name__

    print(
        f"INFO: precision={precision}, "
        f"backend={backend}, "
        f"network_dtype={network_dtype}, "
        f"input_dtype={input_tensor.dtype}"
    )

def run_experiment(cfg: ExperimentConfig) -> dict:
    """
    Runs ONE (configuration, mode) experiment end to end and returns a
    result dict. Intended to be called from a fresh process per
    experiment (see run_single.py) so no state leaks in from a previous
    mode/configuration.
    """
    if (cfg.postprocess_to_original_shape or cfg.save_segmentation_path) and not cfg.patient_files:
        raise ValueError("--postprocess / --save-segmentation require --patient-files "
                          "(there's no 'original geometry' for a synthetic patch).")

    device = torch.device(cfg.device)
    plans, dataset_json = cfg.load_plans_and_dataset()
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(cfg.configuration_name)
    num_input_channels = len(dataset_json["channel_names"])

    compile_configs = load_compile_configs(cfg.compile_configs_path)
    mode_registry = build_mode_registry(compile_configs)
    if cfg.mode not in mode_registry:
        raise KeyError(f"Unknown mode '{cfg.mode}'. Available modes: {sorted(mode_registry)} "
                        f"(pytorch baselines are always available; TRT modes come from "
                        f"{cfg.compile_configs_path})")

    empty_cache(device)
    nvtx.range_push("build_network")
    network = build_network(
        cfg.mode,
        mode_registry=mode_registry,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        configuration_name=cfg.configuration_name,
        dataset_json=dataset_json,
        num_input_channels=num_input_channels,
        device=device,
        compiled_engines_dir=cfg.compiled_engines_dir,
        auto_compile=cfg.auto_compile,
        dry_run_compile=cfg.dry_run_compile,
    )
    nvtx.range_pop()

    if cfg.patient_files:
        nvtx.range_push("load_real_case")
        input_tensor, case_properties = load_case_from_files(
            cfg.patient_files, plans_manager, configuration_manager, dataset_json, verbose=cfg.verbose,
        )
        input_tensor = input_tensor.to(device)
        nvtx.range_pop()
        print(f"Using REAL patient case {[str(f) for f in cfg.patient_files]}")
        print(f"Preprocessed input shape (batch=1): {tuple(input_tensor.shape)}")
    else:
        case_properties = None
        input_tensor = build_input_tensor(configuration_manager, num_input_channels, device)
        print(f"Using SYNTHETIC single-patch input shape: {tuple(input_tensor.shape)}")

    network, input_tensor = _apply_precision(network, input_tensor, mode_registry[cfg.mode].precision)

    describe_dtype(
        network,
        input_tensor,
        mode_registry[cfg.mode].precision,
    )

    predictor = BenchPredictor(device=device, verbose=cfg.verbose, allow_tqdm=False)
    predictor.manual_initialization(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        parameters=None,
        dataset_json=dataset_json,
        trainer_name="nnUNetTrainer",
        inference_allowed_mirroring_axes=(0, 1),
    )

    with runtime_context(cfg.mode, mode_registry, predictor.network) as active_network:
        predictor.network = active_network

        # Setup (padding + slicer computation) happens ONCE, outside the
        # timed loop -- it doesn't depend on the mode being benchmarked.
        nvtx.range_push("prepare_input")
        prepared = predictor.prepare(input_tensor)
        nvtx.range_pop()

        # Resolve perform_everything_on_device once so every timed
        # iteration takes an identical code path.
        perform_on_device = predictor.perform_everything_on_device and device.type != "cpu"
        nvtx.range_push("oom_probe_call")
        try:
            predictor.infer(prepared, perform_on_device)  # also serves as an extra warmup call
        except RuntimeError:
            print("Prediction on device was unsuccessful (likely OOM). Falling back to CPU result tensors.")
            perform_on_device = False
            empty_cache(device)
        nvtx.range_pop()

        timing = time_callable(
            lambda: predictor.infer(prepared, perform_on_device),
            warmup_iterations=cfg.warmup_iterations,
            iterations=cfg.iterations,
        )
        timing.print_summary()

        nvtx.range_push("extra_infer_for_output_shape")
        predicted_logits = predictor.infer(prepared, perform_on_device)  # untimed, just to report output shape
        nvtx.range_pop()

        nvtx.range_push("finalize")
        output = predictor.finalize(prepared, predicted_logits)
        nvtx.range_pop()

    result = {
        "dataset": cfg.dataset_name,
        "configuration": cfg.configuration_name,
        "mode": cfg.mode,
        "input_source": "real" if cfg.patient_files else "synthetic",
        "patient_files": [str(f) for f in cfg.patient_files] if cfg.patient_files else None,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(output.shape),
        "warmup_iterations": cfg.warmup_iterations,
        "iterations": cfg.iterations,
        **timing.summary(),
    }

    do_postprocess = cfg.patient_files and (cfg.postprocess_to_original_shape or cfg.save_segmentation_path)
    if do_postprocess:
        nvtx.range_push("postprocess_to_original_geometry")
        probabilities = postprocess_to_original_geometry(
            predictor, output, case_properties, return_probabilities=True,
        )
        nvtx.range_pop()
        original_shape = tuple(probabilities.shape[1:])  # drop the channel dim
        print(f"Postprocessed (original-geometry) probabilities shape: {tuple(probabilities.shape)}")
        result["original_patient_shape"] = list(original_shape)
        result["postprocessed_output_shape"] = list(probabilities.shape)

        if cfg.save_segmentation_path:
            import numpy as _np
            from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
            segmentation = _np.argmax(probabilities, axis=0).astype(_np.uint8)
            rw_class = determine_reader_writer_from_dataset_json(dataset_json, str(cfg.patient_files[0]))
            rw = rw_class()
            cfg.save_segmentation_path.parent.mkdir(parents=True, exist_ok=True)
            rw.write_seg(segmentation, str(cfg.save_segmentation_path), case_properties)
            print(f"Wrote segmentation to {cfg.save_segmentation_path}")
            result["saved_segmentation_path"] = str(cfg.save_segmentation_path)

    del predicted_logits, output, predictor, network
    cleanup_gpu(device)
    return result
