#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_generator_family_prevalence_kernel_compat
    as kernel_compat,
)
from scripts import (
    reason_router_gen4_mamba14b_adjacent_geometry_prepare_fast_cuda
    as adjacent,
)
from scripts import (
    reason_router_gen4_mamba14b_geometry_prepare_fast_cuda
    as canonical,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
DESIGN_FREEZE_COMMIT = "3ee2d7600de649a711752b8293687047c3c5ec4e"
STATIC_IMPLEMENTATION_COMMIT = "b6c1c7bdb65f1edaeed78044d0b699578552ac5f"

FAMILIES = ("xg2", "xg4")
PAIR_BY_FAMILY = {
    "xg2": "xg2_fact_301",
    "xg4": "xg4_fact_301",
}

GEOMETRY_ATOL = 1.0e-4
GEOMETRY_RTOL = 1.0e-4

FORWARDS_PER_FAMILY_PER_BACKEND = 4
CPU_MODEL_FORWARD_COUNT = len(FAMILIES) * FORWARDS_PER_FAMILY_PER_BACKEND
GPU_MODEL_FORWARD_COUNT = len(FAMILIES) * FORWARDS_PER_FAMILY_PER_BACKEND
TOTAL_MODEL_FORWARD_COUNT = CPU_MODEL_FORWARD_COUNT + GPU_MODEL_FORWARD_COUNT

REPORT_FILE = "adjacent_geometry_one_pair_equivalence_report.json"
SUMS_FILE = "SHA256SUMS.txt"
REPORT_SCHEMA = "gen4-mamba14b-adjacent-geometry-one-pair-equivalence-v1"
RESULT_PASS = "PASS_MAMBA14B_ADJACENT_GEOMETRY_ONE_PAIR_EQUIVALENCE"


class AdjacentGeometryEquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AdjacentGeometryEquivalenceError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AdjacentGeometryEquivalenceError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def authenticate_repo(expected_head: str) -> None:
    adjacent._configure_canonical_module()
    adjacent.validate_protocol_constants()

    require(
        git("branch", "--show-current") == EXPECTED_BRANCH,
        "BRANCH_MISMATCH",
    )
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (DESIGN_FREEZE_COMMIT, "DESIGN"),
        (STATIC_IMPLEMENTATION_COMMIT, "STATIC_IMPLEMENTATION"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _take_token(full: Any, target_abs: int, label: str) -> torch.Tensor:
    require(torch.is_tensor(full), f"{label}_NOT_TENSOR")
    value = full.detach()
    require(value.ndim == 3 and value.shape[0] == 1, f"{label}_SHAPE")
    require(0 <= target_abs < value.shape[1], f"{label}_TARGET_RANGE")
    result = value[0, target_abs, :].detach().cpu().contiguous().clone()
    require(bool(torch.isfinite(result).all().item()), f"{label}_NONFINITE")
    return result


def capture_geometry_branch_cpu_slow(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    input_ids: torch.Tensor,
    anchor: int,
) -> dict[str, Any]:
    target_abs = int(anchor) + canonical.TARGET_OFFSET
    require(tuple(input_ids.shape) == (1, 128), "INPUT_SHAPE")
    require(
        all(parameter.device.type == "cpu" for parameter in model.mamba.parameters()),
        "CPU_MODEL_DEVICE",
    )

    source_layer = runtime_ctx["source_layer"]
    source_mixer = runtime_ctx["source_mixer"]
    norm = runtime_ctx["intervention_norm"]

    holders: dict[str, torch.Tensor] = {}
    counts = {"r": 0, "y": 0, "r_int": 0, "x": 0}

    def source_pre(_module, args):
        counts["r"] += 1
        require(counts["r"] == 1 and len(args) >= 1, "SOURCE_PRE_HOOK")
        holders["R"] = _take_token(args[0], target_abs, "R_FULL")

    def source_post(_module, _args, output):
        counts["y"] += 1
        require(counts["y"] == 1, "SOURCE_POST_HOOK")
        holders["Y"] = _take_token(output, target_abs, "Y_FULL")

    def norm_pre(_module, args):
        counts["r_int"] += 1
        require(counts["r_int"] == 1 and len(args) == 1, "NORM_PRE_HOOK")
        holders["R_INT"] = _take_token(args[0], target_abs, "R_INT_FULL")

    def norm_post(_module, _args, output):
        counts["x"] += 1
        require(counts["x"] == 1, "NORM_POST_HOOK")
        holders["X"] = _take_token(output, target_abs, "X_FULL")

    handles = [
        source_layer.register_forward_pre_hook(source_pre),
        source_mixer.register_forward_hook(source_post),
        norm.register_forward_pre_hook(norm_pre),
        norm.register_forward_hook(norm_post),
    ]

    try:
        model.mamba.eval()
        with torch.inference_mode():
            _ = model.mamba(
                input_ids=input_ids.detach().cpu().contiguous()
            )
    finally:
        for handle in reversed(handles):
            handle.remove()

    require(
        counts == {"r": 1, "y": 1, "r_int": 1, "x": 1},
        f"HOOK_COUNTS:{counts}",
    )
    require(set(holders) == {"R", "Y", "R_INT", "X"}, "HOOK_CAPTURE")

    r_int = holders["R_INT"].to(torch.float64)
    eps = float(norm.variance_epsilon)
    scale = float(torch.rsqrt(r_int.pow(2).mean() + eps).item())
    require(math.isfinite(scale) and scale > 0.0, "RMS_SCALE")

    return {
        "R": holders["R"],
        "Y": holders["Y"],
        "X": holders["X"],
        "rms_scale": scale,
        "target_abs": target_abs,
        "fast_path_calls": 0,
    }


def _family_inputs(
    family: str,
    snapshot: Path,
) -> dict[str, Any]:
    rows, encoded, events, tokenizer_provenance = canonical.load_family_inputs(
        family,
        snapshot,
    )
    pairs = canonical._pair_order(family, rows)
    pair = PAIR_BY_FAMILY[family]
    require(pairs[0] == pair, f"PAIR_SELECTION:{family}")
    row_index = canonical._row_index(rows)

    cells = {
        "tp": canonical.TARGET_PLUS_CELL,
        "tm": canonical.TARGET_MINUS_CELL,
        "rp": canonical.REFERENCE_PLUS_CELL,
        "rm": canonical.REFERENCE_MINUS_CELL,
    }

    anchors: dict[str, int] = {}
    inputs: dict[str, torch.Tensor] = {}
    for role, cell in cells.items():
        event = events[(pair, cell, canonical.ANCHOR_NAME)]
        anchors[role] = int(event["absolute_anchor_token_index"])
        index = row_index[(pair, cell)]
        inputs[role] = encoded["input_ids"][index].unsqueeze(0).contiguous()

    return {
        "family": family,
        "pair": pair,
        "anchors": anchors,
        "inputs": inputs,
        "tokenizer_provenance": tokenizer_provenance,
    }


def _derived(
    *,
    family_input: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    runtime_ctx: Mapping[str, Any],
) -> dict[str, Any]:
    target = canonical.reconstruct_pair_geometry(
        branches["tp"],
        branches["tm"],
        gamma=runtime_ctx["gamma"],
        w_hidden=runtime_ctx["w_hidden"],
        strong_mask=runtime_ctx["strong_mask"],
    )
    reference = canonical.reconstruct_pair_geometry(
        branches["rp"],
        branches["rm"],
        gamma=runtime_ctx["gamma"],
        w_hidden=runtime_ctx["w_hidden"],
        strong_mask=runtime_ctx["strong_mask"],
    )
    delta_norm, alignment = canonical.alignment_delta(
        target["x"],
        target["y"],
        float(reference["C"]),
    )
    delta_h = (
        float(target["d"]) * delta_norm
    ).detach().cpu().to(torch.float64).contiguous()

    require(
        int(delta_h.numel())
        == int(runtime_ctx["strong_mask"].sum().item()),
        "PLAN_WIDTH",
    )
    require(bool(torch.isfinite(delta_h).all().item()), "PLAN_NONFINITE")

    return {
        "family": str(family_input["family"]),
        "source_pair_id": str(family_input["pair"]),
        "anchors": dict(family_input["anchors"]),
        "branches": branches,
        "target": target,
        "reference": reference,
        "alignment": alignment,
        "delta_h": delta_h,
    }


def _compare_tensor(
    cpu_value: Any,
    gpu_value: Any,
    *,
    label: str,
) -> float:
    require(torch.is_tensor(cpu_value), f"CPU_TENSOR:{label}")
    require(torch.is_tensor(gpu_value), f"GPU_TENSOR:{label}")
    cpu = cpu_value.detach().cpu().to(torch.float64).contiguous()
    gpu = gpu_value.detach().cpu().to(torch.float64).contiguous()
    require(cpu.shape == gpu.shape, f"SHAPE:{label}")
    require(
        bool(torch.isfinite(cpu).all().item())
        and bool(torch.isfinite(gpu).all().item()),
        f"NONFINITE:{label}",
    )
    diff = torch.abs(gpu - cpu)
    limit = GEOMETRY_ATOL + GEOMETRY_RTOL * torch.abs(cpu)
    require(
        bool(torch.all(diff <= limit).item()),
        (
            f"EQUIVALENCE_FAILURE:{label}:"
            f"{float(diff.max().item())}:"
            f"{float(limit.max().item())}"
        ),
    )
    return float(diff.max().item()) if diff.numel() else 0.0


def _compare_scalar(
    cpu_value: Any,
    gpu_value: Any,
    *,
    label: str,
) -> float:
    cpu = float(cpu_value)
    gpu = float(gpu_value)
    require(math.isfinite(cpu) and math.isfinite(gpu), f"NONFINITE:{label}")
    diff = abs(gpu - cpu)
    limit = GEOMETRY_ATOL + GEOMETRY_RTOL * abs(cpu)
    require(
        diff <= limit,
        f"EQUIVALENCE_FAILURE:{label}:{diff}:{limit}",
    )
    return diff


def compare_family(
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> dict[str, Any]:
    require(cpu["family"] == gpu["family"], "FAMILY")
    require(cpu["source_pair_id"] == gpu["source_pair_id"], "PAIR")
    require(cpu["anchors"] == gpu["anchors"], "ANCHORS")

    max_branch = 0.0
    max_rms = 0.0
    for role in ("tp", "tm", "rp", "rm"):
        cb = cpu["branches"][role]
        gb = gpu["branches"][role]
        require(cb["target_abs"] == gb["target_abs"], f"TARGET_ABS:{role}")
        require(cb["fast_path_calls"] == 0, f"CPU_FAST_CALL:{role}")
        require(gb["fast_path_calls"] == 1, f"GPU_FAST_CALL:{role}")
        for field in ("R", "Y", "X"):
            max_branch = max(
                max_branch,
                _compare_tensor(
                    cb[field],
                    gb[field],
                    label=f"{role}:{field}",
                ),
            )
        max_rms = max(
            max_rms,
            _compare_scalar(
                cb["rms_scale"],
                gb["rms_scale"],
                label=f"{role}:rms_scale",
            ),
        )

    max_geometry_vector = 0.0
    max_geometry_scalar = 0.0
    for side in ("target", "reference"):
        ca = cpu[side]
        ga = gpu[side]
        for field in ("x", "y"):
            max_geometry_vector = max(
                max_geometry_vector,
                _compare_tensor(
                    ca[field],
                    ga[field],
                    label=f"{side}:{field}",
                ),
            )
        for field in ("d", "A", "B", "C", "I"):
            max_geometry_scalar = max(
                max_geometry_scalar,
                _compare_scalar(
                    ca[field],
                    ga[field],
                    label=f"{side}:{field}",
                ),
            )

    max_delta_h = _compare_tensor(
        cpu["delta_h"],
        gpu["delta_h"],
        label="delta_h",
    )

    for key in (
        "baseline_A",
        "baseline_B",
        "baseline_C",
        "target_C",
        "realized_C",
        "A_abs_residual",
        "B_abs_residual",
        "C_abs_residual",
    ):
        if key in cpu["alignment"] or key in gpu["alignment"]:
            require(
                key in cpu["alignment"] and key in gpu["alignment"],
                f"ALIGNMENT_SCHEMA:{key}",
            )
            max_geometry_scalar = max(
                max_geometry_scalar,
                _compare_scalar(
                    cpu["alignment"][key],
                    gpu["alignment"][key],
                    label=f"alignment:{key}",
                ),
            )

    return {
        "family_key": cpu["family"],
        "source_pair_id": cpu["source_pair_id"],
        "source_pair_selection": "first_fixed_family_pair_outcome_blind",
        "cpu_model_forward_count": FORWARDS_PER_FAMILY_PER_BACKEND,
        "gpu_model_forward_count": FORWARDS_PER_FAMILY_PER_BACKEND,
        "max_branch_tensor_abs_diff": max_branch,
        "max_rms_scale_abs_diff": max_rms,
        "max_geometry_vector_abs_diff": max_geometry_vector,
        "max_geometry_scalar_abs_diff": max_geometry_scalar,
        "max_delta_h_abs_diff": max_delta_h,
        "equivalence_atol": GEOMETRY_ATOL,
        "equivalence_rtol": GEOMETRY_RTOL,
        "result": "PASS",
    }


def _partition_exact(
    cpu_ctx: Mapping[str, Any],
    gpu_ctx: Mapping[str, Any],
) -> None:
    cpu = cpu_ctx["partition"]
    gpu = gpu_ctx["partition"]
    for key in (
        "strong_indices",
        "strong_count",
        "weak_count",
        "equal_count",
        "strong_index_sha256",
    ):
        require(cpu[key] == gpu[key], f"PARTITION:{key}")
    require(
        float(cpu["mu_k2"]) == float(gpu["mu_k2"]),
        "PARTITION:mu_k2",
    )
    require(
        torch.equal(
            cpu_ctx["strong_mask"].detach().cpu(),
            gpu_ctx["strong_mask"].detach().cpu(),
        ),
        "PARTITION:mask",
    )


def run_gate(
    *,
    expected_head: str,
    snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    canonical.validate_fast_runtime_for_device(0)
    canonical.validate_snapshot(snapshot)
    require(
        compact_checkpoint.resolve()
        == (ROOT / canonical.COMPACT_CHECKPOINT_REL).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        canonical.sha256_file(compact_checkpoint)
        == canonical.COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )

    family_inputs = {
        family: _family_inputs(family, snapshot)
        for family in FAMILIES
    }

    model, kernels, model_provenance = canonical.reconstruct_model(
        snapshot=snapshot,
        compact_checkpoint=compact_checkpoint,
        gpu_id=0,
    )
    kernel_compat.validate_transformers_kernel_bindings(kernels)

    model.to(torch.device("cpu"))
    model.eval()
    require(
        all(
            parameter.device.type == "cpu"
            for parameter in model.mamba.parameters()
        ),
        "CPU_MODEL_DEVICE",
    )
    torch.cuda.empty_cache()

    cpu_ctx = canonical.runtime_components(model)
    cpu_results: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        fi = family_inputs[family]
        branches = {
            role: capture_geometry_branch_cpu_slow(
                model=model,
                runtime_ctx=cpu_ctx,
                input_ids=fi["inputs"][role],
                anchor=fi["anchors"][role],
            )
            for role in ("tp", "tm", "rp", "rm")
        }
        cpu_results[family] = _derived(
            family_input=fi,
            branches=branches,
            runtime_ctx=cpu_ctx,
        )

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    require(
        all(
            parameter.device == device
            for parameter in model.mamba.parameters()
        ),
        "GPU_MODEL_DEVICE",
    )

    gpu_ctx = canonical.runtime_components(model)
    _partition_exact(cpu_ctx, gpu_ctx)

    gpu_results: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        fi = family_inputs[family]
        branches = {
            role: canonical.capture_geometry_branch(
                model=model,
                runtime_ctx=gpu_ctx,
                input_ids=fi["inputs"][role],
                anchor=fi["anchors"][role],
                device=device,
            )
            for role in ("tp", "tm", "rp", "rm")
        }
        gpu_results[family] = _derived(
            family_input=fi,
            branches=branches,
            runtime_ctx=gpu_ctx,
        )

    torch.cuda.synchronize(device)

    family_reports = {
        family: compare_family(
            cpu_results[family],
            gpu_results[family],
        )
        for family in FAMILIES
    }

    report = {
        "schema_version": REPORT_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "static_implementation_commit": STATIC_IMPLEMENTATION_COMMIT,
        "site": {
            "canonical_triplet": [33, 34, 35],
            "adjacent_triplet": [34, 35, 36],
            "adjacent_shift": 1,
            "source_block": canonical.SOURCE_BLOCK,
            "target_residual_layer": canonical.TARGET_RESIDUAL_LAYER,
            "intervention_layer": canonical.INTERVENTION_LAYER,
            "target_offset": canonical.TARGET_OFFSET,
        },
        "families": family_reports,
        "tolerances": {
            "geometry_atol": GEOMETRY_ATOL,
            "geometry_rtol": GEOMETRY_RTOL,
            "source": "frozen Gen4 native-state fast CUDA runbook",
        },
        "forward_accounting": {
            "cpu_model_forward_count": CPU_MODEL_FORWARD_COUNT,
            "gpu_model_forward_count": GPU_MODEL_FORWARD_COUNT,
            "total_model_forward_count": TOTAL_MODEL_FORWARD_COUNT,
            "scientific_model_forward_count": 0,
        },
        "model_provenance": model_provenance,
        "kernel_provenance": {
            "kernels_version": kernel_compat.KERNELS_VERSION,
            "build_variant": kernel_compat.BUILD_VARIANT,
            "mamba_scientific_revision":
                kernel_compat.MAMBA_SPEC.scientific_revision,
            "mamba_binary_sha256":
                kernel_compat.MAMBA_SPEC.binary_sha256,
            "mamba_transport_revision":
                kernels["mamba_transport_revision"],
            "mamba_transport_repo_type":
                kernels["mamba_transport_repo_type"],
            "mamba_transport_source":
                kernels["mamba_transport_source"],
            "causal_conv_scientific_revision":
                kernel_compat.CONV_SPEC.scientific_revision,
            "causal_conv_binary_sha256":
                kernel_compat.CONV_SPEC.binary_sha256,
            "causal_conv_transport_revision":
                kernels["causal_conv_transport_revision"],
            "causal_conv_transport_repo_type":
                kernels["causal_conv_transport_repo_type"],
            "causal_conv_transport_source":
                kernels["causal_conv_transport_source"],
            "kernel_transport_identity_status":
                kernels["transport_identity_status"],
        },
        "boundary": {
            "xg1_accessed": False,
            "causal_response_observed": False,
            "plane_selection_performed": False,
            "control_selection_performed": False,
            "statistical_testing_performed": False,
            "training_executed": False,
            "backward_executed": False,
            "scientific_conclusion": None,
            "full_adjacent_geometry_execution_authorized_by_this_artifact": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / REPORT_FILE
    sums_path = output_dir / SUMS_FILE

    report_path.write_text(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sums_path.write_text(
        f"{sha256_file(report_path)}  {REPORT_FILE}\n",
        encoding="utf-8",
        newline="\n",
    )

    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded CPU-slow versus CUDA-fast equivalence gate for the "
            "frozen Mamba-1.4B one-shot adjacent geometry site (34,35,36). "
            "Runs first fixed XG2 and XG4 pair only; no XG1 response."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=ROOT / canonical.COMPACT_CHECKPOINT_REL,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_gate(
        expected_head=args.expected_head,
        snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )

    print("RESULT=" + str(report["result"]))
    print("ADJACENT_TRIPLET=34,35,36")
    for family in FAMILIES:
        item = report["families"][family]
        print(f"{family.upper()}_PAIR=" + item["source_pair_id"])
        print(
            f"{family.upper()}_MAX_BRANCH_TENSOR_ABS_DIFF="
            + repr(item["max_branch_tensor_abs_diff"])
        )
        print(
            f"{family.upper()}_MAX_GEOMETRY_VECTOR_ABS_DIFF="
            + repr(item["max_geometry_vector_abs_diff"])
        )
        print(
            f"{family.upper()}_MAX_GEOMETRY_SCALAR_ABS_DIFF="
            + repr(item["max_geometry_scalar_abs_diff"])
        )
        print(
            f"{family.upper()}_MAX_DELTA_H_ABS_DIFF="
            + repr(item["max_delta_h_abs_diff"])
        )
    print("CPU_MODEL_FORWARD_COUNT=8")
    print("GPU_MODEL_FORWARD_COUNT=8")
    print("TOTAL_MODEL_FORWARD_COUNT=16")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("EXACT_FAST_KERNEL_PREFLIGHT=PASS")
    print("XG1_ACCESSED=False")
    print("CAUSAL_RESPONSE_OBSERVED=False")
    print("STATISTICAL_TESTING_PERFORMED=False")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
