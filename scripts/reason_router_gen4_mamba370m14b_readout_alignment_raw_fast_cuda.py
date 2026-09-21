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
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as bridge,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

B_DESIGN_PATH = Path(
    "reports/reason_router_gen4_study_b_readout_alignment_prospective_design.md"
)
B_DESIGN_SHA256 = "4a3f1db427eddbf8c19aa94ffc2bd21c344b80a2a87f8407aa16a0f540708288"
C_DESIGN_PATH = Path(
    "reports/reason_router_gen4_study_c_manifold_audit_prospective_design.md"
)
C_DESIGN_SHA256 = "dfb21320b3ac024c9be705353146f0ef63117ca48c62c7b202532575aa284047"

SCALES = ("mamba370m", "mamba14b")
SCALE_TO_PHYSICAL_GPU = {"mamba370m": 0, "mamba14b": 1}
TARGET_CELLS = ("C0_SHAM", "C2_NAME")
LABEL_ID_BY_CELL = {"C0_SHAM": 2, "C2_NAME": 1}
PAIR_IDS = tuple(bridge.PAIR_IDS)
PAIR_COUNT = 300
ROWS_PER_SCALE = PAIR_COUNT * len(TARGET_CELLS)
TOTAL_ROWS = ROWS_PER_SCALE * len(SCALES)

PRIMARY_ENDPOINT = "R=Delta_L_370M-Delta_L_1.4B"
PRIMARY_TEST = "paired_one_sample_student_t_greater"
PRIMARY_ALPHA = 0.05
PRIMARY_P_VALUE_COUNT = 1
SIGN_GATE_370M = "mean(Delta_L_370M)>0"
SIGN_GATE_14B = "mean(Delta_L_1.4B)<0"

B_ITEM_FILE = "readout_alignment_items.jsonl"
B_SUMMARY_FILE = "raw_readout_alignment_summary.json"
B_MANIFEST_FILE = "artifact_manifest.json"
B_SUMS_FILE = "SHA256SUMS.txt"

C_STATE_FILE = "manifold_capture_states.npz"
C_INDEX_FILE = "manifold_capture_index.jsonl"
C_MANIFEST_FILE = "artifact_manifest.json"
C_SUMS_FILE = "SHA256SUMS.txt"

B_ITEM_SCHEMA = "gen4-cross-scale-readout-alignment-item-v1"
B_SUMMARY_SCHEMA = "gen4-cross-scale-readout-alignment-raw-summary-v1"
B_MANIFEST_SCHEMA = "gen4-cross-scale-readout-alignment-raw-manifest-v1"
C_INDEX_SCHEMA = "gen4-intervention-manifold-capture-index-v1"
C_MANIFEST_SCHEMA = "gen4-intervention-manifold-capture-manifest-v1"

RAW_RESULT = "PASS_GEN4_CROSS_SCALE_READOUT_ALIGNMENT_RAW_WITH_MANIFOLD_CAPTURE"
GATE_RESULT = "PASS_GEN4_CROSS_SCALE_READOUT_ALIGNMENT_TECHNICAL_GATE"

class ReadoutAlignmentError(RuntimeError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReadoutAlignmentError(message)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

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
        raise ReadoutAlignmentError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc

def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE_DIRTY")

def validate_protocol() -> None:
    require(PAIR_IDS == tuple(f"xg1_fact_{i}" for i in range(4801, 5101)), "PAIR_IDS")
    require(PAIR_COUNT == 300 and ROWS_PER_SCALE == 600 and TOTAL_ROWS == 1200, "COUNTS")
    require(TARGET_CELLS == ("C0_SHAM", "C2_NAME"), "TARGET_CELLS")
    require(bridge.scale_spec("mamba370m")["selected_plane"] == "P3", "370_SELECTED")
    require(bridge.scale_spec("mamba370m")["control_plane"] == "P5", "370_CONTROL")
    require(bridge.scale_spec("mamba14b")["selected_plane"] == "P5", "14_SELECTED")
    require(bridge.scale_spec("mamba14b")["control_plane"] == "P4", "14_CONTROL")
    require(PRIMARY_ENDPOINT == "R=Delta_L_370M-Delta_L_1.4B", "PRIMARY_ENDPOINT")
    require(PRIMARY_TEST == "paired_one_sample_student_t_greater", "PRIMARY_TEST")
    require(PRIMARY_ALPHA == 0.05 and PRIMARY_P_VALUE_COUNT == 1, "PRIMARY_FAMILY")
    require(sha256_file(ROOT / B_DESIGN_PATH) == B_DESIGN_SHA256, "B_DESIGN_SHA")
    require(sha256_file(ROOT / C_DESIGN_PATH) == C_DESIGN_SHA256, "C_DESIGN_SHA")

def _worker_paths(temp_dir: Path, scale: str) -> dict[str, Path]:
    return {
        "items": temp_dir / f"{scale}_items.jsonl",
        "capture": temp_dir / f"{scale}_capture.npz",
        "capture_index": temp_dir / f"{scale}_capture_index.jsonl",
        "meta": temp_dir / f"{scale}_meta.json",
        "gate": temp_dir / f"{scale}_gate.json",
        "error": temp_dir / f"{scale}_error.txt",
    }

def _cosine_from_dot(dot: float, grad_norm: float, component_norm: float) -> float | None:
    if component_norm == 0.0 or grad_norm == 0.0:
        return None
    value = dot / (grad_norm * component_norm)
    require(math.isfinite(value), "COS_NONFINITE")
    require(-1.0 - 1e-9 <= value <= 1.0 + 1e-9, "COS_RANGE")
    return float(min(1.0, max(-1.0, value)))

def _active_margin(logits: torch.Tensor, label_id: int) -> tuple[torch.Tensor, int]:
    require(torch.is_tensor(logits) and tuple(logits.shape) == (1, 3), "LOGITS")
    require(label_id in (0, 1, 2), "LABEL_ID")
    wrong_ids = [i for i in range(3) if i != label_id]
    wrong = logits[0, wrong_ids]
    # Exact tie is a prospectively blocking technical condition.
    require(
        float(wrong[0].detach().item()) != float(wrong[1].detach().item()),
        "WRONG_CLASS_EXACT_TIE",
    )
    local = int(torch.argmax(wrong).item())
    wrong_id = int(wrong_ids[local])
    margin = logits[0, label_id] - logits[0, wrong_id]
    require(bool(torch.isfinite(margin.detach()).item()), "MARGIN_NONFINITE")
    return margin, wrong_id

def _install_local_leaf_hook(
    mixer: Any,
    *,
    token_index: int,
    intermediate_size: int,
    strong_mask: torch.Tensor,
    capture: dict[str, Any],
):
    mask_cpu = strong_mask.detach().cpu().bool().contiguous()
    require(mask_cpu.numel() == intermediate_size, "MASK_WIDTH")

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * intermediate_size,
            "INPROJ_SHAPE",
        )
        require(0 <= token_index < output.shape[1], "TOKEN_INDEX")

        # Cut all upstream autograd history at this exact intervention boundary.
        before = output.detach()
        native_full = (
            before[0, token_index, :intermediate_size]
            .clone()
            .contiguous()
        )
        leaf = native_full.clone().requires_grad_(True)
        gate = (
            before[0, token_index, intermediate_size:]
            .clone()
            .contiguous()
        )
        token = torch.cat([leaf, gate], dim=0).view(1, 1, -1)
        out = torch.cat(
            [
                before[:, :token_index, :],
                token,
                before[:, token_index + 1:, :],
            ],
            dim=1,
        )
        require(tuple(out.shape) == tuple(output.shape), "HOOK_OUTPUT_SHAPE")
        require(torch.equal(out.detach(), before), "HOOK_VALUE_DRIFT")

        capture.clear()
        capture["leaf"] = leaf
        capture["native_full"] = native_full.detach().cpu().contiguous()
        capture["strong_mask"] = mask_cpu
        return out

    return mixer.in_proj.register_forward_hook(hook)

def _plane_components(
    native_strong: torch.Tensor,
    *,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    selected_plane: str,
    control_plane: str,
    dim: int,
) -> dict[str, Any]:
    value = native_strong.detach().cpu().to(torch.float64).contiguous()
    require(tuple(value.shape) == (dim,), "NATIVE_STRONG_SHAPE")
    sel = bridge.plane_component(
        value,
        plane=selected_plane,
        planes=planes,
        dim=dim,
    )
    a = float(sel["a"])
    b = float(sel["b"])
    selected_component = sel["component"].contiguous()
    control_component = (
        a * planes[control_plane]["plus"]
        + b * planes[control_plane]["minus"]
    ).contiguous()
    require(tuple(control_component.shape) == (dim,), "CONTROL_COMPONENT_SHAPE")
    require(bool(torch.isfinite(control_component).all().item()), "CONTROL_NONFINITE")
    return {
        "a": a,
        "b": b,
        "selected_component": selected_component,
        "control_component": control_component,
        "delta_neutralized": (-selected_component).contiguous(),
        "delta_control": (-selected_component + control_component).contiguous(),
    }

def _scatter_applied(
    strong: torch.Tensor,
    *,
    strong_mask: torch.Tensor,
    intermediate_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    value = strong.detach().cpu().to(torch.float64).contiguous()
    mask = strong_mask.detach().cpu().bool().contiguous()
    require(int(mask.sum().item()) == value.numel(), "SCATTER_DIM")
    full64 = torch.zeros(intermediate_size, dtype=torch.float64)
    full64[mask] = value
    applied = full64.to(dtype=dtype).contiguous()
    cast_residual = float(
        torch.max(
            torch.abs(applied.to(torch.float64)[mask] - value)
        ).item()
    )
    return applied, cast_residual

def run_native_gradient_row(
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
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    geom = spec["geom"]
    adapter = spec["adapter"]
    intermediate_size = int(geom.INTERMEDIATE_SIZE)
    dim = int(spec["dim"])
    strong_mask = runtime_ctx["strong_mask"].detach().cpu().bool().contiguous()
    require(int(strong_mask.sum().item()) == dim, "STRONG_DIM")

    capture: dict[str, Any] = {}
    handle = _install_local_leaf_hook(
        runtime_ctx["intervention_mixer"],
        token_index=anchor_index + geom.TARGET_OFFSET,
        intermediate_size=intermediate_size,
        strong_mask=strong_mask,
        capture=capture,
    )
    try:
        output = adapter.historical_forward(
            model,
            bridge.feature_batch(encoded, row_index_value, device),
            arm=geom.ARM,
        )
    finally:
        handle.remove()

    require("leaf" in capture and "native_full" in capture, "HOOK_CAPTURE")
    leaf = capture["leaf"]
    label_id = LABEL_ID_BY_CELL[str(row["contrast_cell_id"])]
    margin, active_wrong_id = _active_margin(output["logits"], label_id)
    grad = torch.autograd.grad(
        margin,
        leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    require(tuple(grad.shape) == (intermediate_size,), "GRAD_SHAPE")
    require(bool(torch.isfinite(grad).all().item()), "GRAD_NONFINITE")

    # No parameter gradients are permitted or needed.
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "PARAMETER_GRAD_CREATED",
    )

    native_full = capture["native_full"]
    native_strong = native_full[strong_mask]
    grad_full64 = grad.detach().cpu().to(torch.float64).contiguous()
    grad_strong64 = grad_full64[strong_mask]
    comp = _plane_components(
        native_strong,
        planes=frozen["planes"],
        selected_plane=str(spec["selected_plane"]),
        control_plane=str(spec["control_plane"]),
        dim=dim,
    )

    sel = comp["selected_component"]
    ctrl = comp["control_component"]
    selected_coords = [
        float(torch.dot(grad_strong64, frozen["planes"][str(spec["selected_plane"])]["plus"]).item()),
        float(torch.dot(grad_strong64, frozen["planes"][str(spec["selected_plane"])]["minus"]).item()),
    ]
    control_coords = [
        float(torch.dot(grad_strong64, frozen["planes"][str(spec["control_plane"])]["plus"]).item()),
        float(torch.dot(grad_strong64, frozen["planes"][str(spec["control_plane"])]["minus"]).item()),
    ]

    grad_norm = float(torch.linalg.vector_norm(grad_full64).item())
    selected_projection_norm = float(math.hypot(*selected_coords))
    control_projection_norm = float(math.hypot(*control_coords))
    selected_norm = float(torch.linalg.vector_norm(sel).item())
    control_norm = float(torch.linalg.vector_norm(ctrl).item())
    l_selected = float(torch.dot(grad_strong64, sel).item())
    l_control = float(torch.dot(grad_strong64, ctrl).item())
    delta_l = l_selected - l_control

    require(
        all(
            math.isfinite(v)
            for v in (
                grad_norm,
                selected_projection_norm,
                control_projection_norm,
                selected_norm,
                control_norm,
                l_selected,
                l_control,
                delta_l,
            )
        ),
        "READOUT_NONFINITE",
    )

    delta_neutralized_applied, neutral_cast_residual = _scatter_applied(
        comp["delta_neutralized"],
        strong_mask=strong_mask,
        intermediate_size=intermediate_size,
        dtype=native_full.dtype,
    )
    delta_control_applied, control_cast_residual = _scatter_applied(
        comp["delta_control"],
        strong_mask=strong_mask,
        intermediate_size=intermediate_size,
        dtype=native_full.dtype,
    )

    item = {
        "schema_version": B_ITEM_SCHEMA,
        "scale": str(spec["scale"]),
        "source_pair_id": str(row["source_pair_id"]),
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "correct_label_id": label_id,
        "active_wrong_class_id": active_wrong_id,
        "native_correct_class_margin": float(margin.detach().cpu().item()),
        "checkpoint_sha256": str(spec["checkpoint_sha256"]),
        "selected_plane": str(spec["selected_plane"]),
        "control_plane": str(spec["control_plane"]),
        "absolute_anchor_token_index": int(anchor_index),
        "target_intervention_token_index": int(anchor_index + geom.TARGET_OFFSET),
        "gradient_full_l2": grad_norm,
        "selected_projection_l2": selected_projection_norm,
        "control_projection_l2": control_projection_norm,
        "selected_projection_fraction": selected_projection_norm / max(grad_norm, 1e-12),
        "control_projection_fraction": control_projection_norm / max(grad_norm, 1e-12),
        "selected_directional_coordinates": selected_coords,
        "control_directional_coordinates": control_coords,
        "selected_component_l2": selected_norm,
        "control_component_l2": control_norm,
        "L_selected": l_selected,
        "L_control": l_control,
        "Delta_L_row": delta_l,
        "cosine_gradient_selected_component":
            _cosine_from_dot(l_selected, grad_norm, selected_norm),
        "cosine_gradient_control_component":
            _cosine_from_dot(l_control, grad_norm, control_norm),
        "parameter_gradient_created": False,
        "training_executed": False,
        "parameter_update_executed": False,
        "study_a_response_accessed": False,
        "experiment1_D_BEH_accessed": False,
        "inferential_test_performed": False,
    }

    arrays = {
        "h_native_full": native_full.numpy(),
        "h_native_strong": native_strong.contiguous().numpy(),
        "delta_neutralized_full": delta_neutralized_applied.numpy(),
        "delta_control_full": delta_control_applied.numpy(),
    }
    capture_meta = {
        "schema_version": C_INDEX_SCHEMA,
        "scale": str(spec["scale"]),
        "source_pair_id": str(row["source_pair_id"]),
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "checkpoint_sha256": str(spec["checkpoint_sha256"]),
        "selected_plane": str(spec["selected_plane"]),
        "control_plane": str(spec["control_plane"]),
        "absolute_anchor_token_index": int(anchor_index),
        "target_intervention_token_index": int(anchor_index + geom.TARGET_OFFSET),
        "intermediate_size": intermediate_size,
        "strong_dim": dim,
        "activation_dtype": str(native_full.dtype).replace("torch.", ""),
        "neutralized_cast_max_abs_residual": neutral_cast_residual,
        "control_cast_max_abs_residual": control_cast_residual,
        "study_b_result_accessed": False,
        "study_a_response_accessed": False,
        "manifold_metric_computed": False,
    }
    return item, arrays, capture_meta

def _prepare_scale(
    *,
    scale: str,
    model_snapshot: Path,
    physical_device: int,
) -> tuple[Mapping[str, Any], torch.nn.Module, Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any], Mapping[tuple[str, str, str], Mapping[str, Any]], Mapping[tuple[str, str], int], torch.device, Mapping[str, Any]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    spec = bridge.scale_spec(scale)
    confirmation = spec["confirmation"]
    geom = spec["geom"]

    device = confirmation.runtime_gate_single_visible_gpu(physical_device)
    snapshot = Path(model_snapshot)
    checkpoint = ROOT / spec["checkpoint_rel"]
    geom.validate_snapshot(snapshot)
    require(sha256_file(checkpoint) == spec["checkpoint_sha256"], "CHECKPOINT_SHA")

    confirmation.load_frozen_selection()
    frozen = confirmation.load_frozen_geometry()
    rows, encoded, events, population_provenance = bridge.build_input_state(
        spec=spec,
        snapshot=snapshot,
    )
    lookup = bridge.row_index(rows)

    model, kernels, model_provenance = geom.reconstruct_model(
        snapshot=snapshot,
        compact_checkpoint=checkpoint,
        gpu_id=0,
    )
    confirmation.kernel_compat.validate_transformers_kernel_bindings(kernels)
    runtime_ctx = geom.runtime_components(model)
    confirmation.validate_runtime_geometry(runtime_ctx, frozen)

    # Freeze every parameter, including downstream heads. This preserves
    # d margin / d activation while guaranteeing no parameter gradients.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    require(not any(p.requires_grad for p in model.parameters()), "PARAM_REQUIRES_GRAD")
    model.eval()

    return (
        spec, model, runtime_ctx, frozen, rows, encoded, events,
        lookup, device, {
            "population": population_provenance,
            "model": model_provenance,
        },
    )

def technical_gate_worker(
    *,
    scale: str,
    model_snapshot: str,
    temp_dir: str,
) -> None:
    paths = _worker_paths(Path(temp_dir), scale)
    try:
        (
            spec, model, runtime_ctx, frozen, rows, encoded, events,
            lookup, device, provenance,
        ) = _prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )

        pair = PAIR_IDS[0]
        cell = TARGET_CELLS[0]
        idx = lookup[(pair, cell)]
        anchor = int(events[(pair, cell, "A_IDENTITY")]["absolute_anchor_token_index"])
        _item, _arrays, _capture = run_native_gradient_row(
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

        # Deliberately surface capability booleans only; no numeric gradient,
        # Delta_L, or manifold value is retained.
        paths["gate"].write_bytes(pretty_json_bytes({
            "scale": scale,
            "model_reconstruction_pass": True,
            "local_leaf_hook_pass": True,
            "finite_margin_pass": True,
            "finite_gradient_pass": True,
            "parameter_gradient_absent_pass": True,
            "frozen_plane_identity_pass": True,
            "numeric_gradient_retained": False,
            "numeric_alignment_retained": False,
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
            spec, model, runtime_ctx, frozen, rows, encoded, events,
            lookup, device, provenance,
        ) = _prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )

        items: list[dict[str, Any]] = []
        capture_index: list[dict[str, Any]] = []
        arrays_by_name: dict[str, list[np.ndarray]] = {
            "h_native_full": [],
            "h_native_strong": [],
            "delta_neutralized_full": [],
            "delta_control_full": [],
        }
        strong_indices = np.flatnonzero(
            runtime_ctx["strong_mask"].detach().cpu().numpy().astype(bool)
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
                item, arrays, cmeta = run_native_gradient_row(
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
                item["scale_row_index"] = scale_row_index
                cmeta["scale_row_index"] = scale_row_index
                items.append(item)
                capture_index.append(cmeta)
                for key in arrays_by_name:
                    arrays_by_name[key].append(arrays[key])
                scale_row_index += 1

        require(len(items) == ROWS_PER_SCALE, "WORKER_ITEM_COUNT")
        require(len(capture_index) == ROWS_PER_SCALE, "WORKER_CAPTURE_INDEX_COUNT")
        torch.cuda.synchronize(device)

        paths["items"].write_bytes(jsonl_bytes(items))
        paths["capture_index"].write_bytes(jsonl_bytes(capture_index))
        np.savez(
            paths["capture"],
            strong_indices=strong_indices,
            **{
                key: np.stack(values, axis=0)
                for key, values in arrays_by_name.items()
            },
        )
        paths["meta"].write_bytes(pretty_json_bytes({
            "scale": scale,
            "execution_head": expected_head,
            "pair_first": PAIR_IDS[0],
            "pair_last": PAIR_IDS[-1],
            "pair_count": PAIR_COUNT,
            "row_count": ROWS_PER_SCALE,
            "physical_device": SCALE_TO_PHYSICAL_GPU[scale],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "selected_plane": spec["selected_plane"],
            "control_plane": spec["control_plane"],
            "scientific_full_model_forward_count": ROWS_PER_SCALE,
            "local_backward_count": ROWS_PER_SCALE,
            "additional_forward_count_for_manifold_capture": 0,
            "training_executed": False,
            "parameter_update_executed": False,
            "inferential_test_performed": False,
            "study_a_response_accessed": False,
            "experiment1_D_BEH_accessed": False,
            "model_provenance": provenance["model"],
            "tokenizer": provenance["population"]["tokenizer"],
        }))
    except BaseException:
        paths["error"].write_text(traceback.format_exc(), encoding="utf-8")
        raise

def _run_two_processes(
    *,
    target,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    temp_dir: Path,
    expected_head: str | None,
) -> None:
    ctx = mp.get_context("spawn")
    processes = []
    snapshots = {
        "mamba370m": mamba370m_snapshot,
        "mamba14b": mamba14b_snapshot,
    }
    for scale in SCALES:
        kwargs = {
            "scale": scale,
            "model_snapshot": str(snapshots[scale]),
            "temp_dir": str(temp_dir),
        }
        if expected_head is not None:
            kwargs["expected_head"] = expected_head
        p = ctx.Process(
            target=target,
            kwargs=kwargs,
            name=f"readout-alignment-{scale}",
        )
        p.start()
        processes.append((scale, p))

    for scale, p in processes:
        p.join()
        if p.exitcode != 0:
            paths = _worker_paths(temp_dir, scale)
            detail = (
                paths["error"].read_text(encoding="utf-8")
                if paths["error"].is_file()
                else "NO_WORKER_ERROR"
            )
            raise ReadoutAlignmentError(
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
    with tempfile.TemporaryDirectory(prefix="gen4_readout_gate_") as tmp:
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
                _worker_paths(temp_dir, scale)["gate"].read_text(encoding="utf-8")
            )
            require(all(
                gate[key] is True
                for key in (
                    "model_reconstruction_pass",
                    "local_leaf_hook_pass",
                    "finite_margin_pass",
                    "finite_gradient_pass",
                    "parameter_gradient_absent_pass",
                    "frozen_plane_identity_pass",
                )
            ), f"GATE:{scale}")
            require(gate["numeric_gradient_retained"] is False, "GATE_NUMERIC_GRAD")
            require(gate["numeric_alignment_retained"] is False, "GATE_NUMERIC_ALIGN")

    print("RESULT=" + GATE_RESULT)
    print("MAMBA370M=PASS")
    print("MAMBA14B=PASS")
    print("NUMERIC_GRADIENT_RETAINED=False")
    print("NUMERIC_ALIGNMENT_RETAINED=False")
    print("SCIENTIFIC_CONCLUSION=None")

def _write_sums(root: Path, names: Sequence[str]) -> None:
    (root / "SHA256SUMS.txt").write_text(
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
    output_dir: Path,
    manifold_output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "B_OUTPUT_COLLISION")
    require(not manifold_output_dir.exists(), "C_OUTPUT_COLLISION")

    with tempfile.TemporaryDirectory(prefix="gen4_readout_raw_") as tmp:
        temp_dir = Path(tmp)
        _run_two_processes(
            target=raw_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=temp_dir,
            expected_head=expected_head,
        )

        all_items: list[dict[str, Any]] = []
        all_capture_index: list[dict[str, Any]] = []
        worker_meta: dict[str, Any] = {}
        c_arrays: dict[str, np.ndarray] = {}

        global_capture_index = 0
        for scale in SCALES:
            paths = _worker_paths(temp_dir, scale)
            scale_items = [
                json.loads(line)
                for line in paths["items"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            scale_index = [
                json.loads(line)
                for line in paths["capture_index"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            require(len(scale_items) == ROWS_PER_SCALE, f"ITEM_COUNT:{scale}")
            require(len(scale_index) == ROWS_PER_SCALE, f"INDEX_COUNT:{scale}")

            for item in scale_items:
                all_items.append(item)
            for row in scale_index:
                row["global_capture_index"] = global_capture_index
                global_capture_index += 1
                all_capture_index.append(row)

            with np.load(paths["capture"], allow_pickle=False) as data:
                for key in data.files:
                    c_arrays[f"{scale}__{key}"] = data[key].copy()
            worker_meta[scale] = json.loads(paths["meta"].read_text(encoding="utf-8"))

        require(len(all_items) == TOTAL_ROWS, "TOTAL_ITEM_COUNT")
        require(len(all_capture_index) == TOTAL_ROWS, "TOTAL_CAPTURE_INDEX_COUNT")

    output_dir.mkdir(parents=True, exist_ok=False)
    b_items_raw = jsonl_bytes(all_items)
    (output_dir / B_ITEM_FILE).write_bytes(b_items_raw)

    summary = {
        "schema_version": B_SUMMARY_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "population": "xg1_fact_4801..xg1_fact_5100",
        "pair_count": PAIR_COUNT,
        "cells": list(TARGET_CELLS),
        "scales": list(SCALES),
        "row_count": TOTAL_ROWS,
        "primary_endpoint_reserved_for_static_analysis": PRIMARY_ENDPOINT,
        "primary_test_reserved_for_static_analysis": PRIMARY_TEST,
        "primary_alpha": PRIMARY_ALPHA,
        "primary_p_value_count": PRIMARY_P_VALUE_COUNT,
        "sign_gates": [SIGN_GATE_370M, SIGN_GATE_14B],
        "scientific_full_model_forward_count": TOTAL_ROWS,
        "local_backward_count": TOTAL_ROWS,
        "training_executed": False,
        "parameter_update_executed": False,
        "inferential_test_performed": False,
        "study_a_response_accessed": False,
        "experiment1_D_BEH_accessed": False,
        "workers": worker_meta,
    }
    (output_dir / B_SUMMARY_FILE).write_bytes(pretty_json_bytes(summary))
    b_manifest = {
        "schema_version": B_MANIFEST_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "B_design_sha256": B_DESIGN_SHA256,
        "C_design_sha256": C_DESIGN_SHA256,
        "item_sha256": sha256_bytes(b_items_raw),
        "summary_sha256": sha256_file(output_dir / B_SUMMARY_FILE),
        "row_count": TOTAL_ROWS,
        "primary_p_value_count_executed": 0,
        "scientific_conclusion": None,
    }
    (output_dir / B_MANIFEST_FILE).write_bytes(pretty_json_bytes(b_manifest))
    _write_sums(output_dir, (B_ITEM_FILE, B_SUMMARY_FILE, B_MANIFEST_FILE))

    manifold_output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(manifold_output_dir / C_STATE_FILE, **c_arrays)
    c_index_raw = jsonl_bytes(all_capture_index)
    (manifold_output_dir / C_INDEX_FILE).write_bytes(c_index_raw)
    c_manifest = {
        "schema_version": C_MANIFEST_SCHEMA,
        "result": "PASS_GEN4_INTERVENTION_MANIFOLD_RAW_CAPTURE",
        "execution_head": expected_head,
        "C_design_sha256": C_DESIGN_SHA256,
        "population": "xg1_fact_4801..xg1_fact_5100",
        "scales": list(SCALES),
        "cells": list(TARGET_CELLS),
        "row_count": TOTAL_ROWS,
        "state_file_sha256": sha256_file(manifold_output_dir / C_STATE_FILE),
        "index_file_sha256": sha256_bytes(c_index_raw),
        "additional_model_forward_count": 0,
        "additional_backward_count": 0,
        "manifold_metric_computed": False,
        "scientific_conclusion": None,
        "study_a_response_accessed": False,
        "study_b_result_accessed": False,
    }
    (manifold_output_dir / C_MANIFEST_FILE).write_bytes(pretty_json_bytes(c_manifest))
    _write_sums(manifold_output_dir, (C_STATE_FILE, C_INDEX_FILE, C_MANIFEST_FILE))

    print("RESULT=" + RAW_RESULT)
    print("PAIR_RANGE=xg1_fact_4801..xg1_fact_5100")
    print("PAIR_COUNT=300")
    print("ROW_COUNT=1200")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT=1200")
    print("LOCAL_BACKWARD_COUNT=1200")
    print("MANIFOLD_ADDITIONAL_FORWARD_COUNT=0")
    print("MANIFOLD_ADDITIONAL_BACKWARD_COUNT=0")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("MANIFOLD_METRIC_COMPUTED=False")
    print("TRAINING_EXECUTED=False")

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gen4 prospective Study-B native task-gradient/readout-alignment raw "
            "measurement with Study-C manifold state capture piggyback. No p-value."
        )
    )
    parser.add_argument("--mode", choices=("gate", "raw"), required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--mamba370m-snapshot", type=Path, required=True)
    parser.add_argument("--mamba14b-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifold-output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "raw":
        require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
        require(args.manifold_output_dir is not None, "MANIFOLD_OUTPUT_DIR_REQUIRED")
    else:
        require(args.output_dir is None, "GATE_OUTPUT_DIR_FORBIDDEN")
        require(args.manifold_output_dir is None, "GATE_MANIFOLD_OUTPUT_DIR_FORBIDDEN")
    return args

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "gate":
        run_technical_gate(
            expected_head=args.expected_head,
            mamba370m_snapshot=args.mamba370m_snapshot,
            mamba14b_snapshot=args.mamba14b_snapshot,
        )
    else:
        run_raw(
            expected_head=args.expected_head,
            mamba370m_snapshot=args.mamba370m_snapshot,
            mamba14b_snapshot=args.mamba14b_snapshot,
            output_dir=args.output_dir,
            manifold_output_dir=args.manifold_output_dir,
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
