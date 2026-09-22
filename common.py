"""
Shared utilities for the nnU-Net sliding-window inference benchmark suite.

Design decisions
----------------
1. Timing isolation

2. Process isolation

3. No branching duplication
"""

from __future__ import annotations

import itertools
import gc
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager, Dict, List, Optional

import numpy as np
import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import torch.backends.cudnn as cudnn
import torch.cuda.nvtx as nvtx
from torch import nn
from torch.profiler import profile, ProfilerActivity
from torch.autograd import DeviceType

from torchinfo import summary


# torch._logging.set_logs(graph_code=True)

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from acvl_utils.cropping_and_padding.padding import pad_nd_image

import torch_tensorrt  # noqa: F401

from no_cat_network import build_no_cat_network, test_no_cat_network

cudnn.benchmark = True

CONFIGURATIONS = ("2d", "3d_fullres")

DTYPE_MAP = {
    "float32": torch.float32, "float": torch.float32,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
}

DEFAULT_COMPILE_CONFIGS_PATH = Path(__file__).resolve().parent / "compile_configs.json"

WARMUP_ITERATIONS = 20
ITERATIONS = 100

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
        torch.cuda.synchronize()
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


def get_torchinfo_profile_layers(model, sample_input, max_depth=3):
    info = summary(
        model,
        input_data=sample_input,
        verbose=0,
        depth=max_depth,
        device=str(sample_input.device),
    )

    layers = []

    for layer_info in info.summary_list:

        depth = getattr(layer_info, "depth", None)

        if depth is None or depth > max_depth:
            continue

        module = getattr(layer_info, "module", None)

        if module is None:
            continue

        class_name = getattr( layer_info, "class_name", module.__class__.__name__,)

        layers.append( { "layer_info": layer_info, "module": module, "class_name": class_name, "depth": depth, "depth_idx": getattr( layer_info, "depth_index", None,), })

    return layers

def profile_torchinfo_blocks(
    model,
    sample_input,
    output_json_path,
    warmup_iterations=5,
    max_depth=3,
):
    model.eval()

    profile_layers = get_torchinfo_profile_layers(model, sample_input, max_depth)

    print("\n" + "=" * 100)
    print("Torchinfo profiling blocks")
    print("=" * 100)

    for i, layer in enumerate(profile_layers):
        print(
            f"{i:4d} "
            f"{layer['depth']}-{layer['depth_idx']} "
            f"{layer['class_name']:<30} "
            f"{layer['module'].__class__.__name__}"
        )

    print("=" * 100)
    module_to_layer_ids = {}

    for layer_id, layer in enumerate(profile_layers):
        module = layer["module"]

        module_to_layer_ids.setdefault( id(module), [],).append(layer_id)

    execution_events = { layer_id: [] for layer_id in range(len(profile_layers)) }

    active_calls = {}

    def pre_hook(module, inputs):
        module_id = id(module)

        layer_ids = module_to_layer_ids.get( module_id, [],)

        if not layer_ids:
            return

        layer_id = layer_ids[0]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        active_calls.setdefault( module_id, [],).append( ( layer_id, start, end,))

    def post_hook(module, inputs, output):
        module_id = id(module)

        calls = active_calls.get(module_id)

        if not calls:
            return

        layer_id, start, end = calls.pop()

        end.record()

        execution_events[layer_id].append( ( start, end,))

    handles = []

    hooked_modules = set()

    for layer in profile_layers:
        module = layer["module"]

        if id(module) in hooked_modules:
            continue

        hooked_modules.add(id(module))

        handles.append( module.register_forward_pre_hook(pre_hook))

        handles.append( module.register_forward_hook(post_hook))

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            with torch.autocast( device_type=sample_input.device.type, enabled=sample_input.device.type == "cuda",):
                model(sample_input)

    torch.cuda.synchronize()

    execution_events = { layer_id: [] for layer_id in range(len(profile_layers)) }

    active_calls.clear()

    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)

    total_start.record()

    with torch.inference_mode():
        with torch.autocast( device_type=sample_input.device.type, enabled=sample_input.device.type == "cuda",):
            model(sample_input)

    total_end.record()

    torch.cuda.synchronize()

    total_time_ms = total_start.elapsed_time(total_end)

    profile_records = [ { "count": 1, } ]

    for layer_id, layer in enumerate(profile_layers):

        events = execution_events[layer_id]

        if not events:
            time_ms = 0.0
        else:
            time_ms = sum( start.elapsed_time(end) for start, end in events)

        percentage = ( 100.0 * time_ms / total_time_ms if total_time_ms > 0 else 0.0)

        depth = layer["depth"]
        depth_idx = layer["depth_idx"]

        profile_records.append(
            {
                "name": layer["class_name"],
                "depth": depth,
                "depthIdx": depth_idx,
                "timeMs": time_ms,
                "averageMs": time_ms,
                "medianMs": time_ms,
                "percentage": percentage,
            }
        )

    for handle in handles:
        handle.remove()

    with open(output_json_path, "w") as f:
        json.dump( profile_records, f, indent=2,)

    print("\n" + "=" * 100)
    print("PyTorch torchinfo architecture profile")
    print("=" * 100)
    print(f"Total forward : {total_time_ms:.3f} ms")
    print(f"Profile blocks: {len(profile_layers)}")
    print(f"Saved         : {output_json_path}")
    print("-" * 100)

    for record in profile_records[1:]:
        print( f"{record['name']:<30} " f"{record['timeMs']:>10.3f} ms " f"{record['percentage']:>8.2f}%")

    print("=" * 100)

    return profile_records


def cleanup_gpu(device: torch.device) -> None:
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
                      enable_deep_supervision: bool = False, split_conv_mode: bool = False) -> nn.Module:
    label_manager = plans_manager.get_label_manager(dataset_json)
    network = nnUNetTrainer.build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=enable_deep_supervision,
    )

    if not split_conv_mode: return network

    return build_no_cat_network(network, stage_indices=[2, 3, 4])

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


def save_fx_model_with_shapes(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_path: str | Path,
) -> None:
    model.eval()

    gm = symbolic_trace(model)

    with torch.no_grad():
        ShapeProp(gm).propagate(example_input)

    nodes = []

    for idx, node in enumerate(gm.graph.nodes):
        entry = {
            "index": idx,
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": str(node.args),
            "kwargs": str(node.kwargs),
        }

        tensor_meta = node.meta.get("tensor_meta")

        if tensor_meta is not None:
            entry["shape"] = list(tensor_meta.shape)
            entry["dtype"] = str(tensor_meta.dtype)
            entry["requires_grad"] = getattr(
                tensor_meta, "requires_grad", None
            )

            if hasattr(tensor_meta, "stride"):
                entry["stride"] = list(tensor_meta.stride)

            if hasattr(tensor_meta, "memory_format"):
                entry["memory_format"] = str(tensor_meta.memory_format)

        nodes.append(entry)

    result = {
        "graph": nodes,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"[FX] saved characterized graph to {output_path}")

# --------------------------------------------------------------------------- #
# Mode registry
# --------------------------------------------------------------------------- #

@dataclass
class ModeSpec:
    name: str
    needs_compiled_engine: bool          # load a torch.export .ep artifact instead of building a fresh network
    engine_suffix: Optional[str] = None  # trt_compiled_<config>_<engine_suffix>.ep -- several modes may share one
    uses_cuda_graphs: bool = False       # wrap the network with torch_tensorrt.runtime.enable_cudagraphs
    precision: str = "autocast"
    compile_kwargs: Optional[dict] = None  # raw torch_tensorrt.compile() kwargs, used only if the engine is missing
    split_conv_mode: bool = False        # optional: conv(cat(A, B)) = conv1(A) + conv2(B)
    use_channels_last: bool = False      # defaul: NCHW
    torch_compile: bool = False
    force_recompile: bool = False


PYTORCH_MODE_SPECS: Dict[str, ModeSpec] = {
    "pytorch": ModeSpec("pytorch", needs_compiled_engine=False),
    "pytorch-compile": ModeSpec("pytorch", needs_compiled_engine=False, torch_compile=True),
    "pytorch-nhwc": ModeSpec("pytorch-nhwc", needs_compiled_engine=False, use_channels_last=True),
    "pytorch-nhwc-split-conv": ModeSpec("pytorch-nhwc-split-conv", needs_compiled_engine=False, use_channels_last=True,
                                        split_conv_mode=True),
    "pytorch-compile-nhwc-split-conv": ModeSpec("pytorch-nhwc-split-conv", needs_compiled_engine=False, use_channels_last=True,
                                        split_conv_mode=True, torch_compile=True),
    # "pytorch-cuda-graphs-solution": ModeSpec("pytorch-cuda-graphs-solution",
    #                                           needs_compiled_engine=False, uses_cuda_graphs=True),
}


def load_compile_configs(path: Path) -> dict:
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
            split_conv_mode=entry.get("split_conv_mode", False),
            use_channels_last=entry.get("use_channels_last", False),
            force_recompile=entry.get("force_recompile", False)
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
    if spec.compile_kwargs is None:
        raise ValueError(f"Mode '{mode}' has no compile_kwargs in the compile-configs registry "
                          f"-- can't auto-compile it (only modes loaded from compile_configs.json can be).")

    engine_path = compiled_engines_dir / f"trt_compiled_{configuration_name}_{spec.engine_suffix}.ep"
    print(f"[compile] '{engine_path.name}' not found -- compiling now (mode='{mode}', dry_run={dry_run})")

    network = get_dummy_network(plans_manager,
                                configuration_manager,
                                dataset_json,
                                num_input_channels,
                                split_conv_mode=spec.split_conv_mode).to(device)

    input_tensor = build_input_tensor(configuration_manager, num_input_channels, device)

    if configuration_name == "3d_fullres":
        input_tensor = input_tensor.unsqueeze(0)

    if spec.use_channels_last:
        mem_format = (torch.channels_last_3d if input_tensor.ndim == 5 else torch.channels_last)

        network = network.to(memory_format=mem_format)
        input_tensor = input_tensor.to(memory_format=mem_format)

    kwargs = _resolve_dtype_fields(spec.compile_kwargs)
    precision = kwargs.pop("precision", "autocast")

    network, input_tensor = _apply_precision(network, input_tensor, precision)

    trt_input = torch_tensorrt.Input(
        shape=input_tensor.shape,
        dtype=input_tensor.dtype,
        format=torch.contiguous_format if not spec.use_channels_last else mem_format,
    )

    use_debugger = kwargs.pop("use_debugger", True)
    debugger_log_level = kwargs.pop("debugger_log_level", "error")
    debugger_ctx = torch_tensorrt.dynamo.Debugger(log_level=debugger_log_level) if use_debugger \
        else nullcontext()

    print(f"INFO: kwargs getting into compile engine:")
    for k, v in kwargs.items():
        print(f"\t{k}: {v}")

    with debugger_ctx:
        trt_gm = torch_tensorrt.compile(
            network,
            ir="dynamo",
            inputs=[trt_input],
            backend="torch_tensorrt",
            dryrun=dry_run,
            **kwargs,
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
    if spec.compile_kwargs is None:
        raise ValueError(f"Mode '{mode}' has no compile_kwargs -- can't export a raw engine for it.")

    raw_engine_path = compiled_engines_dir / f"trt_raw_{configuration_name}_{spec.engine_suffix}.engine"
    print(f"[export-raw] building standalone TensorRT engine for mode='{mode}' -> {raw_engine_path}")

    print("WARNING: using hardcoded split conv net approach")
    network = get_dummy_network(plans_manager, configuration_manager, dataset_json, num_input_channels, split_conv_mode=spec.split_conv_mode).to(device)
    input_tensor = build_input_tensor(configuration_manager, num_input_channels, device)

    if configuration_name == "3d_fullres": input_tensor = input_tensor.unsqueeze(0)

    if spec.use_channels_last:
        mem_format = (
            torch.channels_last_3d
            if input_tensor.ndim == 5
            else torch.channels_last
        )
        network = network.to(memory_format=mem_format)
        input_tensor = input_tensor.to(memory_format=mem_format)

    kwargs = _resolve_dtype_fields(spec.compile_kwargs)
    precision = kwargs.pop("precision", "autocast")

    network, input_tensor = _apply_precision(network, input_tensor, precision)

    print("INFO: constructing fx graph of raw pytorch layers!")
    # somehow add distinction of datasetname to avoid collisions when saving files
    save_fx_model_with_shapes(
        network,
        input_tensor,
        f"pytorch_fx_graph_{configuration_name}.json",
    )

    use_debugger = kwargs.pop("use_debugger", True)
    debugger_log_level = kwargs.pop("debugger_log_level", "error")
    if "options" in kwargs:
        kwargs.update(kwargs.pop("options"))
    kwargs.pop("dynamic", None)

    # 2. Define explicit TensorRT Input spec preserving memory format
    trt_input = torch_tensorrt.Input(
        shape=input_tensor.shape,
        dtype=input_tensor.dtype,
        format=torch.contiguous_format if not spec.use_channels_last else mem_format,
    )

    debugger_ctx = torch_tensorrt.dynamo.Debugger(log_level=debugger_log_level) if use_debugger \
        else nullcontext()

    print(f"INFO: kwargs getting into engine creation:")
    for k, v in kwargs.items():
        print(f"\t{k}: {v}")

    with debugger_ctx:
        exported_program = torch_tensorrt.dynamo.trace(network, [input_tensor])

        serialized_engine = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exported_program, inputs=[trt_input], **kwargs,
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
        if spec.force_recompile or not engine_path.exists():
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
    network = get_dummy_network(plans_manager, configuration_manager, dataset_json, num_input_channels, split_conv_mode=spec.split_conv_mode)

    if "pytorch" in mode and spec.torch_compile:
        print(f"INFO: mode {mode} using torch.compile()")

        compiled = torch.compile(
            network.to(device),
            # backend="inductor",
            # # mode="max-autotune",   # or "reduce-overhead" if launch overhead dominates for your patch size
            # mode="reduce-overhead",   # or "reduce-overhead" if launch overhead dominates for your patch size
            fullgraph=True,
        )
        return compiled

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
    measure_scope: str = "full-inference"

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
        backend = "torch.nn.Module"
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

    if mode_registry[cfg.mode].use_channels_last:
        mem_format = (
            torch.channels_last_3d
            if cfg.configuration_name == "3d_fullres"
            else torch.channels_last
        )
        network = network.to(memory_format=mem_format)
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

        nvtx.range_push("prepare_input")
        print(f"INFOOOOO: input tensor to prepare: {input_tensor.shape}")
        prepared = predictor.prepare(input_tensor)

        nvtx.range_pop()

        perform_on_device = predictor.perform_everything_on_device and device.type != "cpu"
        nvtx.range_push("oom_probe_call")
        try:
            predictor.infer(prepared, perform_on_device)  # also serves as an extra warmup call
        except RuntimeError:
            print("Prediction on device was unsuccessful (likely OOM). Falling back to CPU result tensors.")
            perform_on_device = False
            empty_cache(device)
        nvtx.range_pop()

        single_patch = prepared.data[prepared.slicers[0]][None]

        if mode_registry[cfg.mode].use_channels_last:
            single_patch = single_patch.to(memory_format=mem_format)

        print(f"INFO: running measure-scope: [{cfg.measure_scope}]")
        timing = {}

        if cfg.measure_scope == "full-inference":
            timing = time_callable(
                lambda: predictor.infer(prepared, perform_on_device),
                warmup_iterations=cfg.warmup_iterations,
                iterations=cfg.iterations,
            )

        elif cfg.measure_scope == "single-forward":
            print("INFO: single forward input tensor info ", prepared.data[prepared.slicers[0]][None].shape, prepared.data.device)
            print(f"INFO: single forward is_contiguous: {single_patch.is_contiguous()}")

            def single_forward():
                with torch.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                ):
                    return predictor.network(single_patch)

            timing = time_callable(
                lambda: single_forward(),
                warmup_iterations=cfg.warmup_iterations,
                iterations=cfg.iterations,
            )

        elif cfg.measure_scope == "single-forward-profile-pytorch":
            print("INFO: single forward input tensor info ", prepared.data[prepared.slicers[0]][None].shape, prepared.data.device)
            print(f"INFO: single forward is_contiguous: {single_patch.is_contiguous()}")

            def single_forward():
                with torch.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                ):
                    return predictor.network(single_patch)

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                timing = time_callable(
                    lambda: single_forward(),
                    warmup_iterations=cfg.warmup_iterations,
                    iterations=cfg.iterations,
                )

            events = prof.events()

            pytorch_ops = [ e for e in events if e.device_type == DeviceType.CPU and e.cuda_time_total > 0 ]
            pytorch_ops.sort(key=lambda e: e.time_range.start)

            pytorch_json = [ { "count": cfg.iterations, "name": e.name, "timeMs": e.cuda_time_total / 1000.0, "averageMs": e.cuda_time_total / e.count / 1000.0, } for e in pytorch_ops if "nn.Module" in e.name]

            with open(f"{cfg.configuration_name}_pytorch_ops_profile.json", "w") as f:
                json.dump(pytorch_json, f, indent=2)

            cuda_kernels = [
                e for e in events
                if e.device_type == DeviceType.CUDA
            ]

            cuda_kernels.sort(key=lambda e: e.time_range.start)

            cuda_json = [ { "count": cfg.iterations, "name": e.name, "timeMs": e.cuda_time_total / 1000.0, "averageMs": e.cuda_time_total / e.count / 1000.0, } for e in cuda_kernels ]

            with open(f"{cfg.configuration_name}_pytorch_cuda_profile.json", "w") as f:
                json.dump(cuda_json, f, indent=2)

        elif cfg.measure_scope == "single-forward-profile-blocks":
            profile_model = predictor.network
            _ = profile_torchinfo_blocks(profile_model, single_patch, f"{cfg.configuration_name}_{cfg.mode}_{cfg.dataset_name}_pytorch_blocks_profile.json", warmup_iterations=5, max_depth=3)

        elif cfg.measure_scope == "tta-inference":
            def tta_forward():
                with torch.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                ):
                    mirror_axes = predictor.allowed_mirroring_axes if predictor.use_mirroring else None
                    prediction = predictor.network(single_patch)

                    if mirror_axes is not None:
                        assert max(mirror_axes) <= single_patch.ndim - 3, 'mirror_axes does not match the dimension of the input!'

                        mirror_axes = [m + 2 for m in mirror_axes]
                        axes_combinations = [
                            c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
                        ]
                        for axes in axes_combinations:
                            prediction += torch.flip(predictor.network(torch.flip(single_patch, axes)), axes)
                        prediction /= (len(axes_combinations) + 1)
                    return prediction

            timing = time_callable(
                lambda: tta_forward(),
                warmup_iterations=cfg.warmup_iterations,
                iterations=cfg.iterations,
            )
        else:
            raise ValueError(
                f"Not implemented cfg.measure_scope: [{cfg.measure_scope}]"
            )

        if timing: timing.print_summary()

        nvtx.range_push("extra_infer_for_output_shape")
        predicted_logits = predictor.infer(prepared, perform_on_device)  # untimed, just to report output shape
        nvtx.range_pop()

        nvtx.range_push("finalize")
        output = predictor.finalize(prepared, predicted_logits)
        nvtx.range_pop()

    if timing:
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

