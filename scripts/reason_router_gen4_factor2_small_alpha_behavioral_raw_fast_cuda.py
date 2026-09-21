#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_factor2_small_alpha_runtime_inputs
    as inputs,
)
from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as bridge,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "069b48dee6cb39909623defd862965b3f940e9d9"

SCALES = ("mamba370m", "mamba14b")
SCALE_TO_PHYSICAL_GPU = {
    "mamba370m": 0,
    "mamba14b": 1,
}
TARGET_CELLS = ("C0_SHAM", "C2_NAME")
LABEL_ID_BY_CELL = {
    "C0_SHAM": 2,
    "C2_NAME": 1,
}
LABEL_NAME_BY_CELL = {
    "C0_SHAM": "SUPPORT",
    "C2_NAME": "NOT_ENTITLED",
}

PAIR_FIRST = 5701
PAIR_LAST = 6000
PAIR_COUNT = 300
PAIR_IDS = tuple(f"xg1_fact_{i}" for i in range(PAIR_FIRST, PAIR_LAST + 1))

ALPHAS = (0.25, 0.125, 0.0625, 0.03125)
PRIMARY_CURVE = "K(alpha)"
CONDITIONS = (
    "restored_native",
    "control_alpha_0_25",
    "control_alpha_0_125",
    "control_alpha_0_0625",
    "control_alpha_0_03125",
)
ROWS_PER_SCALE = PAIR_COUNT * len(TARGET_CELLS)
FORWARDS_PER_ROW = len(CONDITIONS)
FORWARDS_PER_SCALE = ROWS_PER_SCALE * FORWARDS_PER_ROW
TOTAL_FORWARD_BUDGET = len(SCALES) * FORWARDS_PER_SCALE
STATE_ROWS_PER_SCALE = ROWS_PER_SCALE
TOTAL_STATE_ROWS = len(SCALES) * STATE_ROWS_PER_SCALE

BEHAVIOR_ITEM_FILE = "factor2_small_alpha_behavioral_items.jsonl"
BEHAVIOR_SUMMARY_FILE = "raw_behavioral_summary.json"
BEHAVIOR_MANIFEST_FILE = "artifact_manifest.json"
STATE_FILE = "factor2_small_alpha_states.npz"
STATE_INDEX_FILE = "capture_index.jsonl"
STATE_MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

BEHAVIOR_ITEM_SCHEMA = "gen4-factor2-small-alpha-behavior-row-v1"
BEHAVIOR_SUMMARY_SCHEMA = "gen4-factor2-small-alpha-behavior-summary-v1"
BEHAVIOR_MANIFEST_SCHEMA = "gen4-factor2-small-alpha-behavior-manifest-v1"
STATE_INDEX_SCHEMA = "gen4-factor2-small-alpha-state-index-v1"
STATE_MANIFEST_SCHEMA = "gen4-factor2-small-alpha-state-manifest-v1"

TECHNICAL_GATE_RESULT = "PASS_GEN4_FACTOR2_SMALL_ALPHA_BEHAVIORAL_TECHNICAL_GATE"
RAW_RESULT = "PASS_GEN4_FACTOR2_SMALL_ALPHA_BEHAVIORAL_RAW"
STATE_RESULT = "PASS_GEN4_FACTOR2_SMALL_ALPHA_STATE_CAPTURE_RAW"


class Factor2SmallAlphaBehavioralRawError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Factor2SmallAlphaBehavioralRawError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Factor2SmallAlphaBehavioralRawError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR_MISSING")


def validate_protocol() -> None:
    require(PAIR_FIRST == 5701 and PAIR_LAST == 6000, "PAIR_RANGE")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(len(PAIR_IDS) == PAIR_COUNT, "PAIR_IDS")
    require(TARGET_CELLS == ("C0_SHAM", "C2_NAME"), "TARGET_CELLS")
    require(ALPHAS == (0.25, 0.125, 0.0625, 0.03125), "ALPHAS")
    require(PRIMARY_CURVE == "K(alpha)", "PRIMARY_CURVE")
    require(CONDITIONS == (
        "restored_native",
        "control_alpha_0_25",
        "control_alpha_0_125",
        "control_alpha_0_0625",
        "control_alpha_0_03125",
    ), "CONDITIONS")
    require(ROWS_PER_SCALE == 600, "ROWS_PER_SCALE")
    require(FORWARDS_PER_ROW == 5, "FORWARDS_PER_ROW")
    require(FORWARDS_PER_SCALE == 3000, "FORWARDS_PER_SCALE")
    require(TOTAL_FORWARD_BUDGET == 6000, "TOTAL_FORWARD_BUDGET")
    require(TOTAL_STATE_ROWS == 1200, "TOTAL_STATE_ROWS")
    require(
        SCALE_TO_PHYSICAL_GPU == {
            "mamba370m": 0,
            "mamba14b": 1,
        },
        "GPU_MAPPING",
    )


def scale_spec(scale: str) -> Mapping[str, Any]:
    require(scale in SCALES, f"SCALE:{scale}")
    return inputs.scale_spec(scale)


def load_population() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    return inputs.load_population()


def validate_behavioral_rows(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    return inputs.validate_target_rows(facts, rows)


def build_input_state(
    *,
    spec: Mapping[str, Any],
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Any],
]:
    return inputs.build_input_state(spec=spec, snapshot=snapshot)


def row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    return inputs.row_index(rows)


def _native_capture_hook(
    mixer: Any,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    spec: Mapping[str, Any],
    frozen: Mapping[str, Any],
    capture: dict[str, Any],
):
    geom = spec["geom"]
    confirmation = spec["confirmation"]
    intermediate_size = int(geom.INTERMEDIATE_SIZE)
    dim = int(spec["dim"])
    mask_cpu = strong_mask.detach().cpu().bool().contiguous()
    require(
        mask_cpu.numel() == intermediate_size
        and int(mask_cpu.sum().item()) == dim,
        "STRONG_MASK",
    )

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * intermediate_size,
            "INPROJ_SHAPE",
        )
        require(0 <= token_index < output.shape[1], "TOKEN_INDEX")

        native_full = (
            output.detach()[0, token_index, :intermediate_size]
            .cpu()
            .clone()
            .contiguous()
        )
        native_strong = native_full[mask_cpu].contiguous()

        info = bridge.condition_correction(
            native_strong,
            condition="dominant_control",
            planes=frozen["planes"],
            selected_plane=str(spec["selected_plane"]),
            control_plane=str(spec["control_plane"]),
            dim=dim,
            tol=float(confirmation.TOL),
        )
        correction64 = info["correction"].detach().cpu().to(torch.float64).contiguous()
        require(tuple(correction64.shape) == (dim,), "CORRECTION_STRONG_SHAPE")

        unscaled_full = torch.zeros(
            intermediate_size,
            dtype=native_full.dtype,
        )
        unscaled_full[mask_cpu] = correction64.to(dtype=native_full.dtype)
        require(bool(torch.isfinite(unscaled_full).all().item()), "UNSCALED_NONFINITE")

        cast_residual = float(
            torch.max(
                torch.abs(
                    unscaled_full[mask_cpu].to(torch.float64)
                    - correction64
                )
            ).item()
        )

        capture.clear()
        capture.update({
            "native_full": native_full,
            "native_strong": native_strong,
            "unscaled_delta_full": unscaled_full.contiguous(),
            "unscaled_delta_strong":
                unscaled_full[mask_cpu].contiguous(),
            "selected_a": float(info["a"]),
            "selected_b": float(info["b"]),
            "selected_component_l2":
                float(info["selected_component_l2"]),
            "unscaled_correction_l2":
                float(torch.linalg.vector_norm(unscaled_full.to(torch.float64)).item()),
            "unscaled_cast_max_abs_residual": cast_residual,
        })
        return output

    return mixer.in_proj.register_forward_hook(hook)


def scaled_full_correction(
    unscaled_full: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    require(alpha in ALPHAS, f"ALPHA:{alpha}")
    require(unscaled_full.ndim == 1, "UNSCALED_FULL_SHAPE")
    out = (unscaled_full * alpha).contiguous()
    require(bool(torch.isfinite(out).all().item()), "SCALED_NONFINITE")
    return out


def _scaled_control_hook(
    mixer: Any,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    native_full: torch.Tensor,
    unscaled_delta_full: torch.Tensor,
    alpha: float,
    intermediate_size: int,
    cast_tol: float,
    audit: dict[str, Any],
):
    mask_cpu = strong_mask.detach().cpu().bool().contiguous()
    require(mask_cpu.numel() == intermediate_size, "MASK_WIDTH")
    intended_cpu = scaled_full_correction(unscaled_delta_full, alpha)

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * intermediate_size,
            "INPROJ_SHAPE",
        )
        require(0 <= token_index < output.shape[1], "TOKEN_INDEX")

        before = output.detach().clone()
        current_native = (
            before[0, token_index, :intermediate_size]
            .detach().cpu().to(native_full.dtype).contiguous()
        )
        replay_residual = float(
            torch.max(torch.abs(
                current_native.to(torch.float64)
                - native_full.to(torch.float64)
            )).item()
        )
        require(replay_residual <= cast_tol, f"NATIVE_REPLAY:{replay_residual}")

        intended = intended_cpu.to(
            device=output.device,
            dtype=output.dtype,
        )
        out = output.clone()
        out[0, token_index, :intermediate_size] += intended

        require(
            torch.equal(
                out[:, :, intermediate_size:],
                before[:, :, intermediate_size:],
            ),
            "GATE_CHANGED",
        )
        if token_index:
            require(
                torch.equal(out[:, :token_index, :], before[:, :token_index, :]),
                "EARLIER_TOKEN_CHANGED",
            )
        if token_index + 1 < out.shape[1]:
            require(
                torch.equal(
                    out[:, token_index + 1:, :],
                    before[:, token_index + 1:, :],
                ),
                "LATER_TOKEN_CHANGED",
            )

        nonstrong = ~mask_cpu.to(output.device)
        require(
            torch.equal(
                out[0, token_index, :intermediate_size][nonstrong],
                before[0, token_index, :intermediate_size][nonstrong],
            ),
            "NONSTRONG_CHANGED",
        )

        applied = (
            out[0, token_index, :intermediate_size]
            - before[0, token_index, :intermediate_size]
        ).detach().cpu().to(torch.float64)
        intended64 = intended.detach().cpu().to(torch.float64)
        applied_residual = float(
            torch.max(torch.abs(applied - intended64)).item()
        )
        require(applied_residual <= cast_tol, f"APPLIED_RESIDUAL:{applied_residual}")

        scaling_residual = float(
            torch.max(
                torch.abs(
                    intended_cpu.to(torch.float64)
                    - (
                        unscaled_delta_full.to(torch.float64)
                        * alpha
                    )
                )
            ).item()
        )
        require(scaling_residual <= cast_tol, f"SCALING_RESIDUAL:{scaling_residual}")

        audit.clear()
        audit.update({
            "alpha": alpha,
            "native_replay_max_abs_residual": replay_residual,
            "applied_correction_max_abs_residual": applied_residual,
            "scaling_identity_max_abs_residual": scaling_residual,
            "applied_correction_l2":
                float(torch.linalg.vector_norm(applied).item()),
        })
        return out

    return mixer.in_proj.register_forward_hook(hook)


def _serialize_model_output(
    output: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    item = bridge.serialize_output(
        output,
        row=row,
        scale=str(spec["scale"]),
        checkpoint_sha256=str(spec["checkpoint_sha256"]),
    )
    cell = str(row["contrast_cell_id"])
    label_id = LABEL_ID_BY_CELL[cell]
    item.update({
        "schema_version": BEHAVIOR_ITEM_SCHEMA,
        "correct_label_id": label_id,
        "correct_label": LABEL_NAME_BY_CELL[cell],
        "is_correct": item["prediction_id"] == label_id,
        "correct_class_logit_margin":
            bridge.correct_margin(item["final_logits"], label_id),
        "selected_plane": str(spec["selected_plane"]),
        "control_plane": str(spec["control_plane"]),
        "primary_inference_executed": False,
        "scientific_conclusion": None,
    })
    return item


def run_five_forward_row(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index_value: int,
    anchor_index: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    geom = spec["geom"]
    confirmation = spec["confirmation"]
    adapter = spec["adapter"]
    target_index = int(anchor_index + geom.TARGET_OFFSET)
    strong_mask = runtime_ctx["strong_mask"].detach().cpu().bool().contiguous()
    intermediate_size = int(geom.INTERMEDIATE_SIZE)
    dim = int(spec["dim"])
    require(
        strong_mask.numel() == intermediate_size
        and int(strong_mask.sum().item()) == dim,
        "STRONG_MASK",
    )

    capture: dict[str, Any] = {}
    handle = _native_capture_hook(
        runtime_ctx["intervention_mixer"],
        token_index=target_index,
        strong_mask=strong_mask,
        spec=spec,
        frozen=frozen,
        capture=capture,
    )
    try:
        with torch.inference_mode():
            native_output = adapter.historical_forward(
                model,
                bridge.feature_batch(encoded, row_index_value, device),
                arm=geom.ARM,
            )
    finally:
        handle.remove()

    require(
        {
            "native_full",
            "native_strong",
            "unscaled_delta_full",
            "unscaled_delta_strong",
        } <= set(capture),
        "NATIVE_CAPTURE",
    )

    items: list[dict[str, Any]] = []
    native_item = _serialize_model_output(
        native_output,
        row=row,
        spec=spec,
    )
    native_item.update({
        "condition": "restored_native",
        "alpha": 0.0,
        "absolute_anchor_token_index": int(anchor_index),
        "target_intervention_token_index": target_index,
        "intervention_audit": None,
        "scientific_full_model_forward_count": 1,
    })
    items.append(native_item)

    max_replay_residual = 0.0
    max_applied_residual = 0.0
    max_scaling_residual = 0.0
    for alpha, condition in zip(
        ALPHAS,
        CONDITIONS[1:],
        strict=True,
    ):
        audit: dict[str, Any] = {}
        handle = _scaled_control_hook(
            runtime_ctx["intervention_mixer"],
            token_index=target_index,
            strong_mask=strong_mask,
            native_full=capture["native_full"],
            unscaled_delta_full=capture["unscaled_delta_full"],
            alpha=alpha,
            intermediate_size=intermediate_size,
            cast_tol=float(confirmation.transport_runtime.RUNTIME_CAST_TOL),
            audit=audit,
        )
        try:
            with torch.inference_mode():
                output = adapter.historical_forward(
                    model,
                    bridge.feature_batch(encoded, row_index_value, device),
                    arm=geom.ARM,
                )
        finally:
            handle.remove()

        item = _serialize_model_output(
            output,
            row=row,
            spec=spec,
        )
        item.update({
            "condition": condition,
            "alpha": alpha,
            "absolute_anchor_token_index": int(anchor_index),
            "target_intervention_token_index": target_index,
            "intervention_audit": dict(audit),
            "scientific_full_model_forward_count": 1,
        })
        items.append(item)
        max_replay_residual = max(
            max_replay_residual,
            float(audit["native_replay_max_abs_residual"]),
        )
        max_applied_residual = max(
            max_applied_residual,
            float(audit["applied_correction_max_abs_residual"]),
        )
        max_scaling_residual = max(
            max_scaling_residual,
            float(audit["scaling_identity_max_abs_residual"]),
        )

    require(len(items) == FORWARDS_PER_ROW, "ROW_FORWARD_COUNT")

    native_full = capture["native_full"].detach().cpu().contiguous()
    native_strong = capture["native_strong"].detach().cpu().contiguous()
    delta_full = capture["unscaled_delta_full"].detach().cpu().contiguous()
    require(bool(torch.isfinite(native_full).all().item()), "NATIVE_FULL_NONFINITE")
    require(bool(torch.isfinite(native_strong).all().item()), "NATIVE_STRONG_NONFINITE")
    require(bool(torch.isfinite(delta_full).all().item()), "DELTA_FULL_NONFINITE")

    arrays = {
        "h_native_full": native_full.numpy(),
        "h_native_strong": native_strong.numpy(),
        "delta_control_full": delta_full.numpy(),
    }
    capture_meta = {
        "schema_version": STATE_INDEX_SCHEMA,
        "scale": str(spec["scale"]),
        "source_pair_id": str(row["source_pair_id"]),
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "checkpoint_sha256": str(spec["checkpoint_sha256"]),
        "selected_plane": str(spec["selected_plane"]),
        "control_plane": str(spec["control_plane"]),
        "absolute_anchor_token_index": int(anchor_index),
        "target_intervention_token_index": target_index,
        "intermediate_size": intermediate_size,
        "strong_dim": dim,
        "activation_dtype": str(native_full.dtype).replace("torch.", ""),
        "unscaled_delta_dtype": str(delta_full.dtype).replace("torch.", ""),
        "unscaled_delta_cast_max_abs_residual":
            float(capture["unscaled_cast_max_abs_residual"]),
        "behavioral_alphas": list(ALPHAS),
        "alpha1_behavioral_forward_executed": False,
        "negative_alpha_arm_executed": False,
        "adaptive_rerun_executed": False,
        "manifold_metric_computed": False,
        "inference_performed": False,
    }
    gate = {
        "finite_native_pass": True,
        "finite_unscaled_correction_pass": True,
        "strong_indices_valid_pass": True,
        "five_forward_execution_pass": True,
        "all_four_alpha_scaling_pass":
            max_scaling_residual <= float(
                confirmation.transport_runtime.RUNTIME_CAST_TOL
            ),
        "native_replay_pass":
            max_replay_residual <= float(
                confirmation.transport_runtime.RUNTIME_CAST_TOL
            ),
        "applied_correction_pass":
            max_applied_residual <= float(
                confirmation.transport_runtime.RUNTIME_CAST_TOL
            ),
    }
    return items, arrays, capture_meta, gate


def _prepare_scale(
    *,
    scale: str,
    model_snapshot: Path,
    physical_device: int,
) -> tuple[
    Mapping[str, Any],
    torch.nn.Module,
    Mapping[str, Any],
    Mapping[str, Any],
    Sequence[Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[tuple[str, str, str], Mapping[str, Any]],
    Mapping[tuple[str, str], int],
    torch.device,
    Mapping[str, Any],
]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    spec = scale_spec(scale)
    confirmation = spec["confirmation"]
    geom = spec["geom"]

    device = confirmation.runtime_gate_single_visible_gpu(physical_device)
    snapshot = Path(model_snapshot)
    checkpoint = ROOT / spec["checkpoint_rel"]

    geom.validate_snapshot(snapshot)
    require(
        sha256_file(checkpoint) == spec["checkpoint_sha256"],
        "CHECKPOINT_SHA",
    )
    confirmation.load_frozen_selection()
    frozen = confirmation.load_frozen_geometry()
    rows, encoded, events, population_provenance = build_input_state(
        spec=spec,
        snapshot=snapshot,
    )
    lookup = row_index(rows)

    model, kernels, model_provenance = geom.reconstruct_model(
        snapshot=snapshot,
        compact_checkpoint=checkpoint,
        gpu_id=0,
    )
    confirmation.kernel_compat.validate_transformers_kernel_bindings(kernels)
    runtime_ctx = geom.runtime_components(model)
    confirmation.validate_runtime_geometry(runtime_ctx, frozen)
    model.eval()

    strong_mask = runtime_ctx["strong_mask"].detach().cpu().bool().contiguous()
    require(int(strong_mask.sum().item()) == int(spec["dim"]), "STRONG_DIM")

    return (
        spec,
        model,
        runtime_ctx,
        frozen,
        rows,
        encoded,
        events,
        lookup,
        device,
        {
            "population": population_provenance,
            "model": model_provenance,
        },
    )


def _worker_paths(temp_dir: Path, scale: str) -> dict[str, Path]:
    return {
        "behavior": temp_dir / f"{scale}_behavior.jsonl",
        "capture": temp_dir / f"{scale}_capture.npz",
        "capture_index": temp_dir / f"{scale}_capture_index.jsonl",
        "meta": temp_dir / f"{scale}_meta.json",
        "gate": temp_dir / f"{scale}_gate.json",
        "error": temp_dir / f"{scale}_error.txt",
    }


def technical_gate_worker(
    *,
    scale: str,
    model_snapshot: str,
    temp_dir: str,
) -> None:
    paths = _worker_paths(Path(temp_dir), scale)
    try:
        (
            spec,
            model,
            runtime_ctx,
            frozen,
            rows,
            encoded,
            events,
            lookup,
            device,
            _provenance,
        ) = _prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )

        pair = PAIR_IDS[0]
        cell = TARGET_CELLS[0]
        idx = lookup[(pair, cell)]
        anchor = int(
            events[(pair, cell, "A_IDENTITY")][
                "absolute_anchor_token_index"
            ]
        )
        _items, _arrays, _capture_meta, gate = run_five_forward_row(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            frozen=frozen,
            encoded=encoded,
            row=rows[idx],
            row_index_value=idx,
            anchor_index=anchor,
            device=device,
        )
        torch.cuda.synchronize(device)

        paths["gate"].write_bytes(pretty_json_bytes({
            "scale": scale,
            **gate,
            "numeric_margin_retained": False,
            "numeric_behavioral_endpoint_retained": False,
            "numeric_displacement_metric_retained": False,
            "scientific_conclusion": None,
        }))
    except BaseException:
        paths["error"].write_text(traceback.format_exc(), encoding="utf-8")
        raise


def raw_worker(
    *,
    scale: str,
    model_snapshot: str,
    temp_dir: str,
    expected_head: str,
) -> None:
    paths = _worker_paths(Path(temp_dir), scale)
    try:
        authenticate_repo(expected_head)
        (
            spec,
            model,
            runtime_ctx,
            frozen,
            rows,
            encoded,
            events,
            lookup,
            device,
            provenance,
        ) = _prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )

        behavior_rows: list[dict[str, Any]] = []
        capture_index: list[dict[str, Any]] = []
        arrays_by_name: dict[str, list[np.ndarray]] = {
            "h_native_full": [],
            "h_native_strong": [],
            "delta_control_full": [],
        }
        strong_indices = np.flatnonzero(
            runtime_ctx["strong_mask"]
            .detach().cpu().numpy().astype(bool)
        ).astype(np.int64)
        require(
            strong_indices.shape == (int(spec["dim"]),),
            "STRONG_INDEX_SHAPE",
        )

        scale_row_index = 0
        for pair in PAIR_IDS:
            for cell in TARGET_CELLS:
                idx = lookup[(pair, cell)]
                anchor = int(
                    events[(pair, cell, "A_IDENTITY")][
                        "absolute_anchor_token_index"
                    ]
                )
                items, arrays, cmeta, gate = run_five_forward_row(
                    spec=spec,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    frozen=frozen,
                    encoded=encoded,
                    row=rows[idx],
                    row_index_value=idx,
                    anchor_index=anchor,
                    device=device,
                )
                require(all(bool(v) for v in gate.values()), "ROW_TECHNICAL_GATE")
                for item in items:
                    item["scale_row_index"] = scale_row_index
                    item["physical_device"] = SCALE_TO_PHYSICAL_GPU[scale]
                    behavior_rows.append(item)
                cmeta["scale_row_index"] = scale_row_index
                capture_index.append(cmeta)
                for key in arrays_by_name:
                    arrays_by_name[key].append(arrays[key])
                scale_row_index += 1

        require(len(behavior_rows) == FORWARDS_PER_SCALE, "BEHAVIOR_ROW_COUNT")
        require(len(capture_index) == STATE_ROWS_PER_SCALE, "CAPTURE_INDEX_COUNT")
        torch.cuda.synchronize(device)

        paths["behavior"].write_bytes(jsonl_bytes(behavior_rows))
        paths["capture_index"].write_bytes(jsonl_bytes(capture_index))
        np.savez(
            paths["capture"],
            strong_indices=strong_indices,
            **{
                key: np.stack(values, axis=0)
                for key, values in arrays_by_name.items()
            },
        )
        manifest = provenance["population"]["manifest"]
        paths["meta"].write_bytes(pretty_json_bytes({
            "scale": scale,
            "execution_head": expected_head,
            "pair_first": PAIR_IDS[0],
            "pair_last": PAIR_IDS[-1],
            "pair_count": PAIR_COUNT,
            "target_cells": list(TARGET_CELLS),
            "behavior_row_count": len(behavior_rows),
            "state_row_count": len(capture_index),
            "scientific_full_model_forward_count": FORWARDS_PER_SCALE,
            "backward_count": 0,
            "training_executed": False,
            "alpha1_behavioral_forward_executed": False,
            "negative_alpha_arm_executed": False,
            "adaptive_rerun_executed": False,
            "inferential_test_performed": False,
            "scientific_conclusion": None,
            "physical_device": SCALE_TO_PHYSICAL_GPU[scale],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "selected_plane": spec["selected_plane"],
            "control_plane": spec["control_plane"],
            "structural_source_sha256": manifest["source_file_sha256"],
            "structural_rows_sha256": manifest["row_file_sha256"],
            "tokenizer": provenance["population"]["tokenizer"],
            "model_provenance": provenance["model"],
        }))
    except BaseException:
        paths["error"].write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _run_two_processes(
    *,
    target: Any,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    temp_dir: Path,
    expected_head: str | None,
) -> None:
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= 2, "PHYSICAL_GPU_COUNT")

    ctx = mp.get_context("spawn")
    processes: list[tuple[str, mp.Process]] = []
    snapshots = {
        "mamba370m": mamba370m_snapshot,
        "mamba14b": mamba14b_snapshot,
    }
    for scale in SCALES:
        kwargs: dict[str, Any] = {
            "scale": scale,
            "model_snapshot": str(snapshots[scale]),
            "temp_dir": str(temp_dir),
        }
        if expected_head is not None:
            kwargs["expected_head"] = expected_head
        process = ctx.Process(
            target=target,
            kwargs=kwargs,
            name=f"factor2-smallalpha-behavior-{scale}",
        )
        process.start()
        processes.append((scale, process))

    for scale, process in processes:
        process.join()
        if process.exitcode != 0:
            paths = _worker_paths(temp_dir, scale)
            detail = (
                paths["error"].read_text(encoding="utf-8")
                if paths["error"].is_file()
                else "NO_WORKER_ERROR_FILE"
            )
            raise Factor2SmallAlphaBehavioralRawError(
                f"WORKER_FAILED:{scale}:\n{detail}"
            )


def run_technical_gate(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)

    with tempfile.TemporaryDirectory(
        prefix="gen4_factor2_smallalpha_behavior_gate_"
    ) as tmp:
        temp_dir = Path(tmp)
        _run_two_processes(
            target=technical_gate_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=temp_dir,
            expected_head=None,
        )
        for scale in SCALES:
            gate = json.loads(
                _worker_paths(temp_dir, scale)["gate"].read_text(
                    encoding="utf-8"
                )
            )
            for key in (
                "finite_native_pass",
                "finite_unscaled_correction_pass",
                "strong_indices_valid_pass",
                "five_forward_execution_pass",
                "all_four_alpha_scaling_pass",
                "native_replay_pass",
                "applied_correction_pass",
            ):
                require(gate[key] is True, f"GATE:{scale}:{key}")
            require(gate["numeric_margin_retained"] is False, "GATE_MARGIN")
            require(
                gate["numeric_behavioral_endpoint_retained"] is False,
                "GATE_ENDPOINT",
            )
            require(
                gate["numeric_displacement_metric_retained"] is False,
                "GATE_DISPLACEMENT",
            )

    print("RESULT=" + TECHNICAL_GATE_RESULT)
    print("MAMBA370M=PASS")
    print("MAMBA14B=PASS")
    print("FIVE_FORWARD_TECHNICAL_ROW_PER_SCALE=PASS")
    print("NUMERIC_MARGIN_RETAINED=False")
    print("NUMERIC_BEHAVIORAL_ENDPOINT_RETAINED=False")
    print("NUMERIC_DISPLACEMENT_METRIC_RETAINED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def _write_sums(root: Path, names: Sequence[str]) -> None:
    (root / CHECKSUM_FILE).write_text(
        "".join(
            f"{sha256_file(root / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_raw(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    behavioral_output_dir: Path,
    state_output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    require(not behavioral_output_dir.exists(), "BEHAVIOR_OUTPUT_COLLISION")
    require(not state_output_dir.exists(), "STATE_OUTPUT_COLLISION")

    with tempfile.TemporaryDirectory(
        prefix="gen4_factor2_smallalpha_behavior_raw_"
    ) as tmp:
        temp_dir = Path(tmp)
        _run_two_processes(
            target=raw_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=temp_dir,
            expected_head=expected_head,
        )

        all_behavior: list[dict[str, Any]] = []
        all_capture_index: list[dict[str, Any]] = []
        worker_meta: dict[str, Any] = {}
        state_arrays: dict[str, np.ndarray] = {}
        global_capture_index = 0

        for scale in SCALES:
            paths = _worker_paths(temp_dir, scale)
            scale_behavior = [
                json.loads(line)
                for line in paths["behavior"].read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            scale_index = [
                json.loads(line)
                for line in paths["capture_index"].read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            require(
                len(scale_behavior) == FORWARDS_PER_SCALE,
                f"BEHAVIOR_COUNT:{scale}",
            )
            require(
                len(scale_index) == STATE_ROWS_PER_SCALE,
                f"STATE_COUNT:{scale}",
            )
            all_behavior.extend(scale_behavior)
            for row in scale_index:
                row["global_capture_index"] = global_capture_index
                global_capture_index += 1
                all_capture_index.append(row)

            with np.load(paths["capture"], allow_pickle=False) as data:
                for key in data.files:
                    state_arrays[f"{scale}__{key}"] = data[key].copy()

            worker_meta[scale] = json.loads(
                paths["meta"].read_text(encoding="utf-8")
            )

    require(len(all_behavior) == TOTAL_FORWARD_BUDGET, "TOTAL_BEHAVIOR_COUNT")
    require(len(all_capture_index) == TOTAL_STATE_ROWS, "TOTAL_STATE_COUNT")

    behavioral_output_dir.mkdir(parents=True, exist_ok=False)
    behavior_raw = jsonl_bytes(all_behavior)
    (behavioral_output_dir / BEHAVIOR_ITEM_FILE).write_bytes(behavior_raw)

    behavior_summary = {
        "schema_version": BEHAVIOR_SUMMARY_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "population": "xg1_fact_5701..xg1_fact_6000",
        "pair_count": PAIR_COUNT,
        "cells": list(TARGET_CELLS),
        "scales": list(SCALES),
        "behavioral_alphas": list(ALPHAS),
        "primary_curve_reserved_for_static_analysis": PRIMARY_CURVE,
        "condition_order": list(CONDITIONS),
        "behavior_row_count": TOTAL_FORWARD_BUDGET,
        "state_capture_row_count": TOTAL_STATE_ROWS,
        "scientific_full_model_forward_count": TOTAL_FORWARD_BUDGET,
        "backward_count": 0,
        "alpha1_behavioral_forward_executed": False,
        "negative_alpha_arm_executed": False,
        "adaptive_rerun_executed": False,
        "primary_inference_executed": False,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
        "training_executed": False,
        "parameter_update_executed": False,
        "rescue_performed": False,
        "workers": worker_meta,
    }
    (behavioral_output_dir / BEHAVIOR_SUMMARY_FILE).write_bytes(
        pretty_json_bytes(behavior_summary)
    )
    behavior_manifest = {
        "schema_version": BEHAVIOR_MANIFEST_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "item_sha256": sha256_bytes(behavior_raw),
        "summary_sha256":
            sha256_file(behavioral_output_dir / BEHAVIOR_SUMMARY_FILE),
        "behavior_row_count": TOTAL_FORWARD_BUDGET,
        "scientific_full_model_forward_count": TOTAL_FORWARD_BUDGET,
        "primary_p_value_count_executed": 0,
        "negative_alpha_arm_executed": False,
        "adaptive_rerun_executed": False,
        "scientific_conclusion": None,
    }
    (behavioral_output_dir / BEHAVIOR_MANIFEST_FILE).write_bytes(
        pretty_json_bytes(behavior_manifest)
    )
    _write_sums(
        behavioral_output_dir,
        (
            BEHAVIOR_ITEM_FILE,
            BEHAVIOR_SUMMARY_FILE,
            BEHAVIOR_MANIFEST_FILE,
        ),
    )

    state_output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(state_output_dir / STATE_FILE, **state_arrays)
    state_index_raw = jsonl_bytes(all_capture_index)
    (state_output_dir / STATE_INDEX_FILE).write_bytes(state_index_raw)
    state_manifest = {
        "schema_version": STATE_MANIFEST_SCHEMA,
        "result": STATE_RESULT,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "population": "xg1_fact_5701..xg1_fact_6000",
        "scales": list(SCALES),
        "cells": list(TARGET_CELLS),
        "behavioral_alphas": list(ALPHAS),
        "state_row_count": TOTAL_STATE_ROWS,
        "state_file_sha256": sha256_file(state_output_dir / STATE_FILE),
        "index_file_sha256": sha256_bytes(state_index_raw),
        "additional_model_forward_count": 0,
        "additional_backward_count": 0,
        "alpha1_geometric_reference_permitted_static_only": True,
        "alpha1_behavioral_forward_executed": False,
        "negative_alpha_arm_executed": False,
        "adaptive_rerun_executed": False,
        "manifold_metric_computed": False,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
    }
    (state_output_dir / STATE_MANIFEST_FILE).write_bytes(
        pretty_json_bytes(state_manifest)
    )
    _write_sums(
        state_output_dir,
        (
            STATE_FILE,
            STATE_INDEX_FILE,
            STATE_MANIFEST_FILE,
        ),
    )

    print("RESULT=" + RAW_RESULT)
    print("STATE_RESULT=" + STATE_RESULT)
    print("PAIR_RANGE=xg1_fact_5701..xg1_fact_6000")
    print("PAIR_COUNT=300")
    print("BEHAVIOR_ROW_COUNT=6000")
    print("STATE_CAPTURE_ROW_COUNT=1200")
    print("MAMBA370M_FORWARD_COUNT=3000")
    print("MAMBA14B_FORWARD_COUNT=3000")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT_TOTAL=6000")
    print("BACKWARD_COUNT=0")
    print("ALPHA1_BEHAVIORAL_FORWARD_EXECUTED=False")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_EXECUTED=0")
    print("MANIFOLD_METRIC_COMPUTED=False")
    print("TRAINING_EXECUTED=False")
    print("RESCUE_PERFORMED=False")


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gen4 Mamba-370M/1.4B factor-2 small-alpha behavioral raw runner. "
            "Technical-gate mode retains no numeric behavioral result. "
            "Raw mode performs exactly 6000 full-model forwards and no backward."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("technical-gate", "raw"),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--mamba370m-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mamba14b-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--behavioral-output-dir",
        type=Path,
    )
    parser.add_argument(
        "--state-output-dir",
        type=Path,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    if args.mode == "technical-gate":
        require(
            args.behavioral_output_dir is None
            and args.state_output_dir is None,
            "TECHNICAL_GATE_OUTPUT_ARGUMENT",
        )
        run_technical_gate(
            expected_head=args.expected_head,
            mamba370m_snapshot=args.mamba370m_snapshot,
            mamba14b_snapshot=args.mamba14b_snapshot,
        )
        return 0

    require(
        args.behavioral_output_dir is not None,
        "BEHAVIOR_OUTPUT_REQUIRED",
    )
    require(
        args.state_output_dir is not None,
        "STATE_OUTPUT_REQUIRED",
    )
    run_raw(
        expected_head=args.expected_head,
        mamba370m_snapshot=args.mamba370m_snapshot,
        mamba14b_snapshot=args.mamba14b_snapshot,
        behavioral_output_dir=args.behavioral_output_dir,
        state_output_dir=args.state_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
