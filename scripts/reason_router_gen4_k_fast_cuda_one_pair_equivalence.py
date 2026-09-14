from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import reason_router_gen4_k_directional_alignment_transport_core as core
from scripts import reason_router_gen4_k_directional_alignment_transport_runner as parent
from scripts import reason_router_gen4_k_directional_alignment_transport_runtime as transport_runtime
from scripts import reason_router_gen4_native_mamba_state_extraction as extraction
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement


ROOT = Path(__file__).resolve().parents[1]
BASE_HEAD = "8496ece911e0d461f0abbdf1a0fa619f8a2f22ab"
EXPECTED_BRANCH = "gen4-k-fast-cuda-equivalence"

KERNELS_VERSION = "0.10.2"
MAMBA_REV = "c8ffc584c147878a6eb978ae0e8db4d116c93a8c"
CONV_REV = "f2651e776f66069cdcf842840db637583def1223"
BUILD_VARIANT = "torch210-cxx11-cu128-x86_64-linux"
MAMBA_BINARY_SHA256 = "dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587"
CONV_BINARY_SHA256 = "6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6"

EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "torch": "2.10.0+cu128",
    "transformers": "5.0.0",
}
EXPECTED_CUDA_RUNTIME = "12.8"
EXPECTED_DEVICE_NAME = "Tesla T4"
EXPECTED_CAPABILITY = (7, 5)

FORWARDS_PER_BACKEND = 8
TOTAL_MODEL_FORWARDS = 16

# Frozen before observing scientific model outputs.
STATE_ATOL = 1e-4
STATE_RTOL = 1e-4
GEOMETRY_ATOL = 1e-4
GEOMETRY_RTOL = 1e-4
PE_ATOL = 1e-4

BRANCH_LABELS = (
    "baseline_tp",
    "baseline_tm",
    "baseline_rp",
    "baseline_rm",
    "alignment_tp",
    "alignment_tm",
    "magnitude_tp",
    "magnitude_tm",
)

GEOMETRY_FIELDS = (
    "target_A",
    "target_B",
    "target_C",
    "reference_A",
    "reference_B",
    "reference_C",
    "alignment_realized_A",
    "alignment_realized_B",
    "alignment_target_cosine",
    "alignment_realized_cosine",
    "magnitude_target_A",
    "magnitude_target_B",
    "magnitude_realized_A",
    "magnitude_realized_B",
    "magnitude_baseline_cosine",
    "magnitude_realized_cosine",
)

PE_FIELDS = (
    "baseline_plus_path_efficiency",
    "baseline_minus_path_efficiency",
    "alignment_plus_path_efficiency",
    "alignment_minus_path_efficiency",
    "magnitude_plus_path_efficiency",
    "magnitude_minus_path_efficiency",
    "delta_baseline",
    "delta_alignment",
    "delta_magnitude",
    "R_ALIGN",
    "R_MAG",
    "ALIGNMENT_SPECIFICITY",
)

EXACT_FIELDS = (
    "schema_version",
    "source_pair_id",
    "source_block",
    "target_residual_layer",
    "intervention_layer",
    "relative_coordinate",
    "target_plus_cell",
    "target_minus_cell",
    "reference_plus_cell",
    "reference_minus_cell",
    "anchor_name",
    "target_plus_anchor",
    "target_minus_anchor",
    "reference_plus_anchor",
    "reference_minus_anchor",
    "target_plus_intervention_token",
    "target_minus_intervention_token",
    "reference_plus_geometry_token",
    "reference_minus_geometry_token",
    "frozen_plus_path_efficiency",
    "frozen_minus_path_efficiency",
)


class EquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise EquivalenceError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EquivalenceError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repo(expected_head: str) -> None:
    require(git("branch", "--show-current") == EXPECTED_BRANCH, "BRANCH_MISMATCH")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", BASE_HEAD, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "BASE_HEAD_NOT_ANCESTOR")

    frozen = (
        "scripts/reason_router_gen4_k_directional_alignment_transport_core.py",
        "scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py",
        "scripts/reason_router_gen4_k_directional_alignment_transport_runner.py",
        "scripts/reason_router_gen4_native_mamba_state_measurement.py",
    )
    rc = subprocess.call(
        ["git", "diff", "--quiet", BASE_HEAD, expected_head, "--", *frozen],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_PARENT_CODE_CHANGED")


def runtime_gate() -> None:
    import transformers

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    require(observed == EXPECTED_RUNTIME, f"RUNTIME_MISMATCH:{observed}")
    require(importlib.metadata.version("kernels") == KERNELS_VERSION, "KERNELS_VERSION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= 1, "CUDA_DEVICE_COUNT")
    require(torch.version.cuda == EXPECTED_CUDA_RUNTIME, "CUDA_RUNTIME")
    torch.cuda.set_device(0)
    require(torch.cuda.get_device_name(0) == EXPECTED_DEVICE_NAME, "CUDA_DEVICE_NAME")
    require(tuple(torch.cuda.get_device_capability(0)) == EXPECTED_CAPABILITY, "CUDA_CAPABILITY")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _single_binary_sha(module: Any, label: str) -> str:
    root = Path(module.__file__).parent
    require(root.name == BUILD_VARIANT, f"{label}_BUILD_VARIANT")
    binaries = sorted(root.glob("*.so"))
    require(len(binaries) == 1, f"{label}_BINARY_COUNT")
    return sha256_file(binaries[0])


def load_exact_fast_kernels() -> dict[str, Any]:
    from kernels import get_kernel
    import transformers.models.mamba.modeling_mamba as mm

    mamba = get_kernel("kernels-community/mamba-ssm", revision=MAMBA_REV)
    conv = get_kernel("kernels-community/causal-conv1d", revision=CONV_REV)

    require(mamba is not None, "MAMBA_KERNEL_LOAD")
    require(conv is not None, "CONV_KERNEL_LOAD")
    require(MAMBA_REV in Path(mamba.__file__).parts, "MAMBA_REVISION_PATH")
    require(CONV_REV in Path(conv.__file__).parts, "CONV_REVISION_PATH")
    require(_single_binary_sha(mamba, "MAMBA") == MAMBA_BINARY_SHA256, "MAMBA_BINARY_SHA256")
    require(_single_binary_sha(conv, "CONV") == CONV_BINARY_SHA256, "CONV_BINARY_SHA256")

    functions = {
        "selective_scan_fn": getattr(mamba, "selective_scan_fn", None),
        "selective_state_update": getattr(mamba, "selective_state_update", None),
        "mamba_inner_fn": getattr(mamba, "mamba_inner_fn", None),
        "causal_conv1d_fn": getattr(conv, "causal_conv1d_fn", None),
        "causal_conv1d_update": getattr(conv, "causal_conv1d_update", None),
    }
    require(all(callable(v) for v in functions.values()), "KERNEL_FUNCTION_SURFACE")

    for name, fn in functions.items():
        setattr(mm, name, fn)

    return {"mamba": mamba, "conv": conv, **functions}


@contextmanager
def parent_runtime_rebind():
    original_versions = dict(measurement.EXPECTED_VERSIONS)
    original_tol = parent.BASELINE_REPRO_TOL
    measurement.EXPECTED_VERSIONS = {
        **original_versions,
        "torch": EXPECTED_RUNTIME["torch"],
    }
    parent.BASELINE_REPRO_TOL = PE_ATOL
    try:
        yield
    finally:
        measurement.EXPECTED_VERSIONS = original_versions
        parent.BASELINE_REPRO_TOL = original_tol


def _flatten_state(state: torch.Tensor) -> np.ndarray:
    out = (
        state.detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .numpy()
        .reshape(-1)
        .copy()
    )
    require(out.size == core.INTERMEDIATE_SIZE * core.STATE_SIZE, "STATE_WIDTH")
    require(bool(np.isfinite(out).all()), "STATE_NONFINITE")
    return out


def _five_state_window(states: Sequence[Any] | None, anchor: int):
    if states is None:
        return None
    require(anchor + 4 < len(states), "STATE_WINDOW_RANGE")
    return [
        np.ascontiguousarray(np.asarray(states[t], dtype=np.float32)).copy()
        for t in range(anchor, anchor + 5)
    ]


def _logging_capture(base_capture: Any, records: list[dict[str, Any]]):
    counter = {"value": 0}

    def wrapped(*args, **kwargs):
        i = counter["value"]
        require(i < len(BRANCH_LABELS), "CAPTURE_CALL_OVERFLOW")
        counter["value"] += 1
        result = base_capture(*args, **kwargs)
        anchor = int(result["anchor"])
        records.append(
            {
                "label": BRANCH_LABELS[i],
                "anchor": anchor,
                "target_abs": int(result["target_abs"]),
                "states": _five_state_window(result["states"], anchor),
            }
        )
        return result

    return wrapped


def _make_fast_capture(kernels: Mapping[str, Any]):
    import transformers.models.mamba.modeling_mamba as mm

    kernel_scan = kernels["selective_scan_fn"]
    kernel_update = kernels["selective_state_update"]

    def capture_branch(
        model: Any,
        runtime_ctx: Mapping[str, Any],
        *,
        trace_code: Any,
        trace_line: int,
        input_ids: torch.Tensor,
        anchor: int,
        budget: Any,
        capture_states: bool,
        delta_h: Any | None = None,
        plus_branch: bool | None = None,
    ) -> dict[str, Any]:
        del trace_code, trace_line

        target_abs = int(anchor) + core.TARGET_OFFSET
        require(
            tuple(input_ids.shape) == (1, extraction.MAX_MODEL_SEQUENCE_LENGTH),
            "INPUT_SHAPE",
        )
        input_ids = input_ids.detach().to("cuda:0").contiguous()

        layer15 = runtime_ctx["layer15"]
        norm17 = runtime_ctx["norm17"]
        mixer17 = runtime_ctx["mixer17"]

        holders: dict[str, Any] = {}
        counts = {"r15": 0, "y15": 0, "r17": 0, "x17": 0}

        def take(full: Any, label: str) -> torch.Tensor:
            value = parent._finite_tensor(full, label)
            require(0 <= target_abs < value.shape[1], f"{label}_TARGET_RANGE")
            return value[0, target_abs, :].contiguous().clone()

        def layer15_pre(_module, args):
            counts["r15"] += 1
            require(counts["r15"] == 1 and len(args) >= 1, "R15_HOOK")
            holders["R"] = take(args[0], "R15_FULL")

        def layer15_post(_module, _args, output):
            counts["y15"] += 1
            require(counts["y15"] == 1, "Y15_HOOK")
            holders["Y"] = take(output, "Y15_FULL")

        def norm17_pre(_module, args):
            counts["r17"] += 1
            require(counts["r17"] == 1 and len(args) == 1, "R17_HOOK")
            holders["R17"] = take(args[0], "R17_FULL")

        def norm17_post(_module, _args, output):
            counts["x17"] += 1
            require(counts["x17"] == 1, "X17_HOOK")
            holders["X"] = take(output, "X17_FULL")

        handles = [
            layer15.register_forward_pre_hook(layer15_pre),
            layer15.mixer.register_forward_hook(layer15_post),
            norm17.register_forward_pre_hook(norm17_pre),
            norm17.register_forward_hook(norm17_post),
        ]

        intervention_audit = None
        if delta_h is not None:
            require(plus_branch is not None, "INTERVENTION_BRANCH_REQUIRED")
            intervention_audit = {}
            handles.append(
                transport_runtime.install_inproj_hook(
                    mixer17,
                    token_index=target_abs,
                    strong_mask=runtime_ctx["strong_mask"],
                    delta_h=delta_h,
                    plus_branch=bool(plus_branch),
                    audit=intervention_audit,
                )
            )

        active = {"value": False}
        captured: list[tuple[torch.Tensor, ...]] = []

        def scan_wrapper(*args, **kwargs):
            if active["value"]:
                require(len(captured) == 0, "LAYER17_SCAN_DUPLICATE")
                require(len(args) >= 8, "SCAN_ARG_COUNT")
                captured.append(tuple(v.detach().clone() for v in args[:8]))
            return kernel_scan(*args, **kwargs)

        original_scan = mm.selective_scan_fn
        original_cuda = mixer17.cuda_kernels_forward

        def cuda_wrapper(
            _self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            require(not active["value"], "LAYER17_ACTIVE_REENTRY")
            active["value"] = True
            try:
                return original_cuda(
                    hidden_states,
                    cache_params,
                    cache_position,
                    attention_mask,
                )
            finally:
                active["value"] = False

        mm.selective_scan_fn = scan_wrapper
        mixer17.cuda_kernels_forward = types.MethodType(cuda_wrapper, mixer17)

        budget.consume()
        try:
            model.mamba.eval()
            with torch.inference_mode():
                _ = model.mamba(input_ids=input_ids)
            torch.cuda.synchronize()
        finally:
            for handle in reversed(handles):
                handle.remove()
            mm.selective_scan_fn = original_scan
            if "cuda_kernels_forward" in mixer17.__dict__:
                del mixer17.__dict__["cuda_kernels_forward"]

        require(
            counts == {"r15": 1, "y15": 1, "r17": 1, "x17": 1},
            "HOOK_COUNT_FAILURE",
        )
        require(set(holders) == {"R", "Y", "R17", "X"}, "HOOK_CAPTURE_MISSING")
        require(len(captured) == 1, "LAYER17_SCAN_CAPTURE_COUNT")

        r17 = holders["R17"].to(torch.float64)
        eps = float(norm17.variance_epsilon)
        scale = float(torch.rsqrt(r17.pow(2).mean() + eps).item())
        require(math.isfinite(scale) and scale > 0.0, "RMS_SCALE")

        states = None
        if capture_states:
            (
                u,
                delta,
                a_matrix,
                b_scan,
                c_scan,
                d_vector,
                gate,
                delta_bias,
            ) = captured[0]

            prefix_end = int(anchor) + 1
            require(prefix_end + 4 <= u.shape[-1], "FAST_POST4_RANGE")

            _, state = kernel_scan(
                u[..., :prefix_end].contiguous(),
                delta[..., :prefix_end].contiguous(),
                a_matrix,
                b_scan[..., :prefix_end].contiguous(),
                c_scan[..., :prefix_end].contiguous(),
                d_vector,
                gate[..., :prefix_end].contiguous(),
                delta_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            window = [_flatten_state(state)]

            for token in range(prefix_end, prefix_end + 4):
                _ = kernel_update(
                    state,
                    u[..., token],
                    delta[..., token],
                    a_matrix,
                    b_scan[..., token],
                    c_scan[..., token],
                    d_vector,
                    gate[..., token],
                    delta_bias,
                    dt_softplus=True,
                )
                window.append(_flatten_state(state))

            filler = window[0]
            token_count = int(input_ids.shape[1])
            states = [filler.copy() for _ in range(token_count)]
            for offset, vector in enumerate(window):
                states[int(anchor) + offset] = vector

        if delta_h is not None:
            require(
                intervention_audit is not None and bool(intervention_audit),
                "INTERVENTION_HOOK_NOT_OBSERVED",
            )
            require(
                intervention_audit["token_index"] == target_abs,
                "INTERVENTION_TOKEN_AUDIT",
            )

        return {
            "geometry_branch": {
                "R": holders["R"],
                "Y": holders["Y"],
                "X": holders["X"],
                "rms_scale": scale,
            },
            "states": states,
            "intervention_audit": intervention_audit,
            "anchor": int(anchor),
            "target_abs": target_abs,
        }

    return capture_branch


def _compare_scalar(
    observed: float,
    reference: float,
    *,
    atol: float,
    rtol: float = 0.0,
    label: str,
) -> float:
    a = float(observed)
    b = float(reference)
    require(math.isfinite(a) and math.isfinite(b), f"NONFINITE:{label}")
    diff = abs(a - b)
    limit = atol + rtol * abs(b)
    require(diff <= limit, f"EQUIVALENCE_FAILURE:{label}:{diff}:{limit}")
    return diff


def compare_pair(
    cpu_item: Mapping[str, Any],
    gpu_item: Mapping[str, Any],
    cpu_records: Sequence[Mapping[str, Any]],
    gpu_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(set(cpu_item) == set(gpu_item), "ITEM_SCHEMA")

    for field in EXACT_FIELDS:
        require(cpu_item[field] == gpu_item[field], f"EXACT_FIELD:{field}")

    max_geometry = 0.0
    for field in GEOMETRY_FIELDS:
        max_geometry = max(
            max_geometry,
            _compare_scalar(
                gpu_item[field],
                cpu_item[field],
                atol=GEOMETRY_ATOL,
                rtol=GEOMETRY_RTOL,
                label=f"geometry:{field}",
            ),
        )

    max_pe = 0.0
    for field in PE_FIELDS:
        max_pe = max(
            max_pe,
            _compare_scalar(
                gpu_item[field],
                cpu_item[field],
                atol=PE_ATOL,
                label=f"pe:{field}",
            ),
        )

    require(
        len(cpu_records) == len(BRANCH_LABELS)
        and len(gpu_records) == len(BRANCH_LABELS),
        "CAPTURE_RECORD_COUNT",
    )

    max_state_abs = 0.0
    compared_windows = 0

    for cpu_record, gpu_record in zip(cpu_records, gpu_records, strict=True):
        require(cpu_record["label"] == gpu_record["label"], "CAPTURE_LABEL")
        require(cpu_record["anchor"] == gpu_record["anchor"], "CAPTURE_ANCHOR")
        require(cpu_record["target_abs"] == gpu_record["target_abs"], "CAPTURE_TARGET")

        cpu_states = cpu_record["states"]
        gpu_states = gpu_record["states"]
        require((cpu_states is None) == (gpu_states is None), "CAPTURE_STATE_PRESENCE")
        if cpu_states is None:
            continue

        require(len(cpu_states) == 5 and len(gpu_states) == 5, "CAPTURE_WINDOW_WIDTH")
        compared_windows += 1

        for offset, (cpu_state, gpu_state) in enumerate(zip(cpu_states, gpu_states, strict=True)):
            require(cpu_state.shape == gpu_state.shape, "STATE_SHAPE_MISMATCH")
            cpu64 = cpu_state.astype(np.float64)
            gpu64 = gpu_state.astype(np.float64)
            max_abs = float(np.max(np.abs(gpu64 - cpu64)))
            scale = float(np.max(np.abs(cpu64)))
            limit = STATE_ATOL + STATE_RTOL * scale
            require(
                max_abs <= limit,
                f"STATE_EQUIVALENCE_FAILURE:{cpu_record['label']}:{offset}:{max_abs}:{limit}",
            )
            max_state_abs = max(max_state_abs, max_abs)

    require(compared_windows == 6, "STATE_WINDOW_COUNT")

    return {
        "max_state_abs_diff": max_state_abs,
        "max_geometry_abs_diff": max_geometry,
        "max_pe_abs_diff": max_pe,
    }


def run_one_pair(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    runtime_gate()
    require(not output.exists(), "OUTPUT_COLLISION")

    with parent_runtime_rebind():
        rows, encoded, event_rows = extraction.load_scientific_inputs(tokenizer_snapshot)
        pairs = list(extraction.canonical_pair_order(rows))
        require(len(pairs) == parent.SOURCE_PAIR_COUNT, "PAIR_COUNT")
        pair = pairs[0]

        events = parent.event_lookup(event_rows)
        parent.validate_transport_event_plan(pairs, events)
        row_index = parent.build_row_index(rows)
        frozen_endpoint = parent.load_frozen_layer17_endpoints()
        trace_code, trace_line = measurement._resolve_and_validate_runtime_binding()

        cpu_model, _ = parent.load_representative_model_external(
            model_snapshot=model_snapshot,
            checkpoint_path=checkpoint_path,
        )
        cpu_ctx = transport_runtime.validate_runtime_components(cpu_model)

        gpu_model, _ = parent.load_representative_model_external(
            model_snapshot=model_snapshot,
            checkpoint_path=checkpoint_path,
        )
        gpu_ctx = transport_runtime.validate_runtime_components(gpu_model)

        original_capture = parent.capture_branch

        cpu_records: list[dict[str, Any]] = []
        parent.capture_branch = _logging_capture(original_capture, cpu_records)
        cpu_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        try:
            cpu_item = parent.run_pair(
                pair,
                model=cpu_model,
                runtime_ctx=cpu_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                frozen_endpoint=frozen_endpoint,
                budget=cpu_budget,
            )
        finally:
            parent.capture_branch = original_capture
        cpu_budget.assert_exact()

        kernels = load_exact_fast_kernels()
        gpu_model.to(torch.device("cuda:0"))
        gpu_model.eval()
        require(
            all(p.device.type == "cuda" for p in gpu_model.mamba.parameters()),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = _make_fast_capture(kernels)
        gpu_records: list[dict[str, Any]] = []
        parent.capture_branch = _logging_capture(fast_capture, gpu_records)
        gpu_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        try:
            gpu_item = parent.run_pair(
                pair,
                model=gpu_model,
                runtime_ctx=gpu_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                frozen_endpoint=frozen_endpoint,
                budget=gpu_budget,
            )
        finally:
            parent.capture_branch = original_capture
        gpu_budget.assert_exact()
        torch.cuda.synchronize()

    comparison = compare_pair(cpu_item, gpu_item, cpu_records, gpu_records)

    cpu_baseline_residual = max(
        float(cpu_item["baseline_plus_reproduction_abs_residual"]),
        float(cpu_item["baseline_minus_reproduction_abs_residual"]),
    )
    gpu_baseline_residual = max(
        float(gpu_item["baseline_plus_reproduction_abs_residual"]),
        float(gpu_item["baseline_minus_reproduction_abs_residual"]),
    )

    require(cpu_baseline_residual <= PE_ATOL, "CPU_FROZEN_BASELINE_REPRO")
    require(gpu_baseline_residual <= PE_ATOL, "GPU_FROZEN_BASELINE_REPRO")

    report = {
        "schema_version": "gen4-k-fast-cuda-one-pair-equivalence-v1",
        "result": "PASS_FAST_CUDA_ONE_PAIR_EQUIVALENCE",
        "base_head": BASE_HEAD,
        "execution_head": expected_head,
        "source_pair_id": str(cpu_item["source_pair_id"]),
        "cpu_model_forward_count": FORWARDS_PER_BACKEND,
        "gpu_model_forward_count": FORWARDS_PER_BACKEND,
        "total_model_forward_count": TOTAL_MODEL_FORWARDS,
        "scientific_conclusion": None,
        "kernels_version": KERNELS_VERSION,
        "mamba_revision": MAMBA_REV,
        "mamba_binary_sha256": MAMBA_BINARY_SHA256,
        "causal_conv_revision": CONV_REV,
        "causal_conv_binary_sha256": CONV_BINARY_SHA256,
        "build_variant": BUILD_VARIANT,
        "state_atol": STATE_ATOL,
        "state_rtol": STATE_RTOL,
        "geometry_atol": GEOMETRY_ATOL,
        "geometry_rtol": GEOMETRY_RTOL,
        "pe_atol": PE_ATOL,
        "max_cpu_frozen_baseline_abs_residual": cpu_baseline_residual,
        "max_gpu_frozen_baseline_abs_residual": gpu_baseline_residual,
        **comparison,
        "raw_vectors_persisted": False,
        "endpoint_values_persisted": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--tokenizer-snapshot", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_one_pair(
        expected_head=args.expected_head,
        model_snapshot=Path(args.model_snapshot),
        tokenizer_snapshot=Path(args.tokenizer_snapshot),
        checkpoint_path=Path(args.checkpoint),
        output=Path(args.output),
    )
    print("RESULT =", report["result"])
    print("SOURCE_PAIR_ID =", report["source_pair_id"])
    print("CPU_MODEL_FORWARD_COUNT =", report["cpu_model_forward_count"])
    print("GPU_MODEL_FORWARD_COUNT =", report["gpu_model_forward_count"])
    print("TOTAL_MODEL_FORWARD_COUNT =", report["total_model_forward_count"])
    print("MAX_STATE_ABS_DIFF =", report["max_state_abs_diff"])
    print("MAX_GEOMETRY_ABS_DIFF =", report["max_geometry_abs_diff"])
    print("MAX_PE_ABS_DIFF =", report["max_pe_abs_diff"])
    print("SCIENTIFIC_CONCLUSION = NONE")


if __name__ == "__main__":
    main()
