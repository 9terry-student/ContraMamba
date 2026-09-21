#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_seed181_behavioral_restoration_bridge_fast_cuda
    as bridge,
)

ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "788f448d22fbc1f91b438bf50c4692375c01eb4d"

PLAN_PATH = Path(
    "reports/reason_router_gen4_mamba130m_readout_alignment_prospective_plan.md"
)
PLAN_SHA256 = "a8a2cb2af1ed35b82f404a713c15ebf176cd645f653c6eacf73722b6e34058b3"

MAMBA130_REPO = "state-spaces/mamba-130m-hf"
MAMBA130_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
MAMBA130_REQUIRED_SNAPSHOT_SHA256 = {
    "config.json":
        "784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a",
    "tokenizer.json":
        "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
    "tokenizer_config.json":
        "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
    "special_tokens_map.json":
        "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
}

CHECKPOINT_SHA256 = (
    "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
)
CHECKPOINT_BYTES = 518270455
GEOMETRY_JSON_SHA256 = (
    "e6e9db909eb7d2c6bbdb493a4efeca8c18e4d474cf99a943be0f3f7b9dee1012"
)
GEOMETRY_PT_SHA256 = (
    "de3ae6a450c2ba0a85b4f53919e3765e6e7dfb6dba3676535554c437b1647a1c"
)

SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"
SELECTED_PLUS = "pp3_plus"
SELECTED_MINUS = "pp3_minus"
CONTROL_PLUS = "pp5_plus"
CONTROL_MINUS = "pp5_minus"
INTERVENTION_LAYER = 17
TARGET_OFFSET = 2
DIM = 395

PAIR_IDS = tuple(f"xg1_fact_{i:03d}" for i in range(2701, 3001))
PAIR_COUNT = 300
TARGET_CELLS = ("C0_SHAM", "C2_NAME")
LABEL_ID_BY_CELL = {"C0_SHAM": 2, "C2_NAME": 1}
SHARD_RANGES = {
    0: (2701, 2850),
    1: (2851, 3000),
}
ROWS_PER_SHARD = 300
TOTAL_ROWS = 600
SCIENTIFIC_FORWARD_COUNT = 600
LOCAL_BACKWARD_COUNT = 600

PRIMARY_ENDPOINT = "Delta_L_130M"
PRIMARY_TEST = "one_sample_student_t_greater"
PRIMARY_ALPHA = 0.05
PRIMARY_P_VALUE_COUNT = 1
SIGN_GATE = "mean(Delta_L_130M)>0"

ITEM_FILE = "readout_alignment_items.jsonl"
SUMMARY_FILE = "raw_readout_alignment_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-mamba130m-readout-alignment-item-v1"
SUMMARY_SCHEMA = "gen4-mamba130m-readout-alignment-raw-summary-v1"
MANIFEST_SCHEMA = "gen4-mamba130m-readout-alignment-raw-manifest-v1"

GATE_RESULT = "PASS_GEN4_MAMBA130M_READOUT_ALIGNMENT_TECHNICAL_GATE"
RAW_RESULT = "PASS_GEN4_MAMBA130M_READOUT_ALIGNMENT_RAW"


class Mamba130ReadoutAlignmentError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Mamba130ReadoutAlignmentError(message)


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
    return b"".join(canonical_json_bytes(x) for x in rows)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Mamba130ReadoutAlignmentError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    head = git("rev-parse", "HEAD")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "PLAN_COMMIT_NOT_ANCESTOR")


def validate_snapshot(snapshot: Path) -> None:
    require(snapshot.is_dir(), f"SNAPSHOT_MISSING:{snapshot}")
    require(snapshot.name == MAMBA130_REVISION, f"SNAPSHOT_REVISION_PATH:{snapshot.name}")
    for name, digest in MAMBA130_REQUIRED_SNAPSHOT_SHA256.items():
        path = snapshot / name
        require(path.is_file(), f"SNAPSHOT_FILE_MISSING:{name}")
        require(sha256_file(path) == digest, f"SNAPSHOT_SHA256:{name}")


def validate_protocol() -> None:
    require(PAIR_IDS == tuple(f"xg1_fact_{i:03d}" for i in range(2701, 3001)), "PAIR_IDS")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(TARGET_CELLS == ("C0_SHAM", "C2_NAME"), "TARGET_CELLS")
    require(SHARD_RANGES == {0: (2701, 2850), 1: (2851, 3000)}, "SHARD_RANGES")
    require(ROWS_PER_SHARD == 300 and TOTAL_ROWS == 600, "ROW_COUNTS")
    require(SCIENTIFIC_FORWARD_COUNT == 600, "FORWARD_COUNT")
    require(LOCAL_BACKWARD_COUNT == 600, "BACKWARD_COUNT")
    require(SELECTED_PLANE == "P3" and CONTROL_PLANE == "P5", "PLANE_IDS")
    require(INTERVENTION_LAYER == 17 and TARGET_OFFSET == 2, "INTERVENTION_SITE")
    require(DIM == 395, "DIM")
    require(PRIMARY_ENDPOINT == "Delta_L_130M", "PRIMARY_ENDPOINT")
    require(PRIMARY_TEST == "one_sample_student_t_greater", "PRIMARY_TEST")
    require(PRIMARY_ALPHA == 0.05 and PRIMARY_P_VALUE_COUNT == 1, "PRIMARY_FAMILY")
    require(sha256_file(ROOT / PLAN_PATH) == PLAN_SHA256, "PLAN_SHA256")

    require(bridge.CHECKPOINT_SHA256 == CHECKPOINT_SHA256, "BRIDGE_CHECKPOINT_SHA")
    require(bridge.CHECKPOINT_BYTES == CHECKPOINT_BYTES, "BRIDGE_CHECKPOINT_BYTES")
    require(bridge.GEOMETRY_JSON_SHA256 == GEOMETRY_JSON_SHA256, "BRIDGE_GEOMETRY_JSON_SHA")
    require(bridge.GEOMETRY_PT_SHA256 == GEOMETRY_PT_SHA256, "BRIDGE_GEOMETRY_PT_SHA")
    require(bridge.TARGET_CELLS == TARGET_CELLS, "BRIDGE_TARGET_CELLS")
    require(bridge.SHARD_RANGES == SHARD_RANGES, "BRIDGE_SHARD_RANGES")
    require(bridge.restoration.DIM == DIM, "BRIDGE_DIM")
    core = bridge.restoration.holdout.phase1.base.prevalence_eq.core
    require(core.TARGET_OFFSET == TARGET_OFFSET, "BRIDGE_TARGET_OFFSET")


def shard_pairs(shard_index: int) -> tuple[str, ...]:
    require(shard_index in SHARD_RANGES, f"SHARD_INDEX:{shard_index}")
    first, last = SHARD_RANGES[shard_index]
    pairs = tuple(f"xg1_fact_{i:03d}" for i in range(first, last + 1))
    require(len(pairs) == 150, f"SHARD_PAIR_COUNT:{shard_index}")
    return pairs


def _worker_paths(temp_dir: Path, shard_index: int) -> dict[str, Path]:
    prefix = f"shard{shard_index}"
    return {
        "items": temp_dir / f"{prefix}_items.jsonl",
        "meta": temp_dir / f"{prefix}_meta.json",
        "gate": temp_dir / f"{prefix}_gate.json",
        "error": temp_dir / f"{prefix}_error.txt",
    }


def _active_margin(
    logits: torch.Tensor,
    label_id: int,
) -> tuple[torch.Tensor, int]:
    require(torch.is_tensor(logits) and tuple(logits.shape) == (1, 3), "LOGITS_SHAPE")
    require(label_id in (0, 1, 2), "LABEL_ID")
    wrong_ids = [i for i in range(3) if i != label_id]
    wrong = logits[0, wrong_ids]
    require(
        float(wrong[0].detach().item()) != float(wrong[1].detach().item()),
        "WRONG_CLASS_EXACT_TIE",
    )
    local = int(torch.argmax(wrong).item())
    wrong_id = int(wrong_ids[local])
    margin = logits[0, label_id] - logits[0, wrong_id]
    require(bool(torch.isfinite(margin.detach()).item()), "MARGIN_NONFINITE")
    return margin, wrong_id


def _cosine_from_dot(
    dot: float,
    grad_norm: float,
    component_norm: float,
) -> float | None:
    if grad_norm == 0.0 or component_norm == 0.0:
        return None
    value = dot / (grad_norm * component_norm)
    require(math.isfinite(value), "COSINE_NONFINITE")
    require(-1.0 - 1e-9 <= value <= 1.0 + 1e-9, "COSINE_RANGE")
    return float(min(1.0, max(-1.0, value)))


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
    require(int(mask_cpu.sum().item()) == DIM, "MASK_STRONG_DIM")

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * intermediate_size,
            "INPROJ_SHAPE",
        )
        require(0 <= token_index < output.shape[1], "TOKEN_INDEX")

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
    planes: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    h = native_strong.detach().cpu().to(torch.float64).contiguous()
    require(tuple(h.shape) == (DIM,), "NATIVE_STRONG_SHAPE")

    u3p = planes[SELECTED_PLUS].detach().cpu().to(torch.float64).contiguous()
    u3m = planes[SELECTED_MINUS].detach().cpu().to(torch.float64).contiguous()
    u5p = planes[CONTROL_PLUS].detach().cpu().to(torch.float64).contiguous()
    u5m = planes[CONTROL_MINUS].detach().cpu().to(torch.float64).contiguous()

    for name, value in (
        (SELECTED_PLUS, u3p),
        (SELECTED_MINUS, u3m),
        (CONTROL_PLUS, u5p),
        (CONTROL_MINUS, u5m),
    ):
        require(tuple(value.shape) == (DIM,), f"PLANE_SHAPE:{name}")

    a = float(torch.dot(h, u3p).item())
    b = float(torch.dot(h, u3m).item())
    c3 = (a * u3p + b * u3m).contiguous()
    c5 = (a * u5p + b * u5m).contiguous()
    require(bool(torch.isfinite(c3).all().item()), "C3_NONFINITE")
    require(bool(torch.isfinite(c5).all().item()), "C5_NONFINITE")
    mismatch = abs(
        float(torch.linalg.vector_norm(c3).item())
        - float(torch.linalg.vector_norm(c5).item())
    )
    require(mismatch <= bridge.restoration.TOL, f"MATCHED_COMPONENT_NORM:{mismatch}")
    return {
        "a": a,
        "b": b,
        "selected_component": c3,
        "control_component": c5,
        "delta_selected_minus_control": (c3 - c5).contiguous(),
    }


def run_native_gradient_row(
    *,
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    planes: Mapping[str, torch.Tensor],
    encoded: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index: int,
    anchor_index: int,
    device: torch.device,
) -> dict[str, Any]:
    core = bridge.restoration.holdout.phase1.base.prevalence_eq.core
    intermediate_size = int(core.INTERMEDIATE_SIZE)
    strong_mask = runtime_ctx["strong_mask"].detach().cpu().bool().contiguous()
    require(strong_mask.numel() == intermediate_size, "STRONG_MASK_WIDTH")
    require(int(strong_mask.sum().item()) == DIM, "STRONG_MASK_DIM")

    capture: dict[str, Any] = {}
    handle = _install_local_leaf_hook(
        runtime_ctx["mixer17"],
        token_index=anchor_index + TARGET_OFFSET,
        intermediate_size=intermediate_size,
        strong_mask=strong_mask,
        capture=capture,
    )
    try:
        output = bridge.adapter.historical_forward(
            model,
            bridge.feature_batch(encoded, row_index, device),
            arm=bridge.ARM,
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
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "PARAMETER_GRAD_CREATED",
    )

    native_full = capture["native_full"]
    native_strong = native_full[strong_mask]
    grad_full = grad.detach().cpu().to(torch.float64).contiguous()
    grad_strong = grad_full[strong_mask]
    comp = _plane_components(native_strong, planes=planes)

    c3 = comp["selected_component"]
    c5 = comp["control_component"]
    u3p = planes[SELECTED_PLUS].detach().cpu().to(torch.float64).contiguous()
    u3m = planes[SELECTED_MINUS].detach().cpu().to(torch.float64).contiguous()
    u5p = planes[CONTROL_PLUS].detach().cpu().to(torch.float64).contiguous()
    u5m = planes[CONTROL_MINUS].detach().cpu().to(torch.float64).contiguous()

    selected_coords = [
        float(torch.dot(grad_strong, u3p).item()),
        float(torch.dot(grad_strong, u3m).item()),
    ]
    control_coords = [
        float(torch.dot(grad_strong, u5p).item()),
        float(torch.dot(grad_strong, u5m).item()),
    ]

    grad_norm = float(torch.linalg.vector_norm(grad_full).item())
    selected_projection_norm = float(math.hypot(*selected_coords))
    control_projection_norm = float(math.hypot(*control_coords))
    selected_component_norm = float(torch.linalg.vector_norm(c3).item())
    control_component_norm = float(torch.linalg.vector_norm(c5).item())
    l_selected = float(torch.dot(grad_strong, c3).item())
    l_control = float(torch.dot(grad_strong, c5).item())
    delta_l = l_selected - l_control

    numeric = (
        grad_norm,
        selected_projection_norm,
        control_projection_norm,
        selected_component_norm,
        control_component_norm,
        l_selected,
        l_control,
        delta_l,
    )
    require(all(math.isfinite(x) for x in numeric), "READOUT_NONFINITE")

    return {
        "schema_version": ITEM_SCHEMA,
        "scale": "mamba130m",
        "source_pair_id": str(row["source_pair_id"]),
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "correct_label_id": label_id,
        "active_wrong_class_id": active_wrong_id,
        "native_correct_class_margin": float(margin.detach().cpu().item()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_plane": SELECTED_PLANE,
        "control_plane": CONTROL_PLANE,
        "intervention_layer": INTERVENTION_LAYER,
        "anchor_name": "A_IDENTITY",
        "absolute_anchor_token_index": int(anchor_index),
        "target_intervention_token_index": int(anchor_index + TARGET_OFFSET),
        "native_selected_coordinates": [float(comp["a"]), float(comp["b"])],
        "gradient_full_l2": grad_norm,
        "selected_projection_l2": selected_projection_norm,
        "control_projection_l2": control_projection_norm,
        "selected_projection_fraction":
            selected_projection_norm / max(grad_norm, 1e-12),
        "control_projection_fraction":
            control_projection_norm / max(grad_norm, 1e-12),
        "selected_directional_coordinates": selected_coords,
        "control_directional_coordinates": control_coords,
        "selected_component_l2": selected_component_norm,
        "control_component_l2": control_component_norm,
        "L_selected": l_selected,
        "L_control": l_control,
        "Delta_L_row": delta_l,
        "cosine_gradient_selected_component":
            _cosine_from_dot(l_selected, grad_norm, selected_component_norm),
        "cosine_gradient_control_component":
            _cosine_from_dot(l_control, grad_norm, control_component_norm),
        "parameter_gradient_created": False,
        "intervention_forward_executed": False,
        "training_executed": False,
        "parameter_update_executed": False,
        "behavioral_bridge_D_BEH_accessed": False,
        "cross_scale_readout_result_accessed": False,
        "inferential_test_performed": False,
    }


@contextlib.contextmanager
def _worker_environment(
    *,
    shard_index: int,
    physical_device: int,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint: Path,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    validate_snapshot(model_snapshot)
    validate_snapshot(tokenizer_snapshot)
    bridge.authenticate_checkpoint(checkpoint)

    physical_gpu_count = bridge.physical_gpu_inventory_count()
    device = bridge.validate_cuda_partition(shard_index, physical_device)

    facts, all_rows, structural_manifest = bridge.validate_fresh_data()
    selected_rows, facts_by_id = bridge.validate_label_semantics(facts, all_rows)
    events, encoded, tokenizer_provenance = bridge.build_anchor_and_encoding(
        selected_rows,
        facts_by_id,
        tokenizer_snapshot,
    )
    planes = bridge.load_seed181_planes()

    row_index = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"])): i
        for i, row in enumerate(selected_rows)
    }
    require(len(row_index) == TOTAL_ROWS, "ROW_INDEX_COUNT")

    runtime = bridge.restoration.holdout.phase1.base.prevalence_eq
    runtime.backend.runtime_gate()
    kernels = runtime.kernel_compat.load_exact_fast_kernels()

    with bridge.seed181.seed181_binding():
        with runtime.backend.parent_runtime_rebind():
            with runtime.kernel_compat.exact_transformers_kernel_loader(
                kernels
            ) as calls:
                model, checkpoint_sha = (
                    runtime.parent.load_representative_model_external(
                        model_snapshot=model_snapshot,
                        checkpoint_path=checkpoint,
                    )
                )
                require(
                    checkpoint_sha == CHECKPOINT_SHA256,
                    "MODEL_CHECKPOINT_SHA",
                )
                runtime_ctx = (
                    runtime.transport_runtime.validate_runtime_components(model)
                )

            counts = Counter(calls)
            require(
                set(counts) == {"causal-conv1d", "mamba-ssm"}
                and counts["causal-conv1d"] > 0
                and counts["causal-conv1d"] == counts["mamba-ssm"],
                "KERNEL_CONSTRUCTOR",
            )
            runtime.kernel_compat.validate_transformers_kernel_bindings(kernels)

            model.to(device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            require(
                not any(parameter.requires_grad for parameter in model.parameters()),
                "PARAM_REQUIRES_GRAD",
            )
            model.eval()

            yield {
                "model": model,
                "runtime_ctx": runtime_ctx,
                "planes": planes,
                "rows": selected_rows,
                "encoded": encoded,
                "events": events,
                "row_index": row_index,
                "device": device,
                "physical_gpu_count": physical_gpu_count,
                "structural_manifest": structural_manifest,
                "tokenizer_provenance": tokenizer_provenance,
                "kernel_constructor_calls": dict(counts),
            }


def technical_gate_worker(
    *,
    shard_index: int,
    physical_device: int,
    model_snapshot: str,
    tokenizer_snapshot: str,
    checkpoint: str,
    temp_dir: str,
) -> None:
    paths = _worker_paths(Path(temp_dir), shard_index)
    try:
        with _worker_environment(
            shard_index=shard_index,
            physical_device=physical_device,
            model_snapshot=Path(model_snapshot),
            tokenizer_snapshot=Path(tokenizer_snapshot),
            checkpoint=Path(checkpoint),
        ) as env:
            pair = shard_pairs(shard_index)[0]
            cell = TARGET_CELLS[0]
            idx = env["row_index"][(pair, cell)]
            anchor = int(
                env["events"][(pair, cell, "A_IDENTITY")][
                    "absolute_anchor_token_index"
                ]
            )
            _ = run_native_gradient_row(
                model=env["model"],
                runtime_ctx=env["runtime_ctx"],
                planes=env["planes"],
                encoded=env["encoded"],
                row=env["rows"][idx],
                row_index=idx,
                anchor_index=anchor,
                device=env["device"],
            )
            torch.cuda.synchronize(env["device"])

        paths["gate"].write_bytes(pretty_json_bytes({
            "shard_index": shard_index,
            "physical_device": physical_device,
            "model_reconstruction_pass": True,
            "checkpoint_identity_pass": True,
            "frozen_geometry_identity_pass": True,
            "population_identity_pass": True,
            "anchor_identity_pass": True,
            "local_leaf_hook_pass": True,
            "finite_margin_pass": True,
            "finite_gradient_pass": True,
            "parameter_gradient_absent_pass": True,
            "numeric_margin_retained": False,
            "numeric_gradient_retained": False,
            "numeric_alignment_retained": False,
            "scientific_conclusion": None,
        }))
    except BaseException:
        paths["error"].write_text(traceback.format_exc(), encoding="utf-8")
        raise


def raw_worker(
    *,
    shard_index: int,
    physical_device: int,
    model_snapshot: str,
    tokenizer_snapshot: str,
    checkpoint: str,
    temp_dir: str,
    expected_head: str,
) -> None:
    paths = _worker_paths(Path(temp_dir), shard_index)
    try:
        authenticate_repo(expected_head)
        pairs = shard_pairs(shard_index)

        with _worker_environment(
            shard_index=shard_index,
            physical_device=physical_device,
            model_snapshot=Path(model_snapshot),
            tokenizer_snapshot=Path(tokenizer_snapshot),
            checkpoint=Path(checkpoint),
        ) as env:
            items: list[dict[str, Any]] = []
            shard_row_index = 0
            for pair in pairs:
                for cell in TARGET_CELLS:
                    idx = env["row_index"][(pair, cell)]
                    anchor = int(
                        env["events"][(pair, cell, "A_IDENTITY")][
                            "absolute_anchor_token_index"
                        ]
                    )
                    item = run_native_gradient_row(
                        model=env["model"],
                        runtime_ctx=env["runtime_ctx"],
                        planes=env["planes"],
                        encoded=env["encoded"],
                        row=env["rows"][idx],
                        row_index=idx,
                        anchor_index=anchor,
                        device=env["device"],
                    )
                    item["shard_index"] = shard_index
                    item["physical_device"] = physical_device
                    item["shard_row_index"] = shard_row_index
                    items.append(item)
                    shard_row_index += 1

            require(len(items) == ROWS_PER_SHARD, "SHARD_ITEM_COUNT")
            torch.cuda.synchronize(env["device"])

            paths["items"].write_bytes(jsonl_bytes(items))
            paths["meta"].write_bytes(pretty_json_bytes({
                "shard_index": shard_index,
                "physical_device": physical_device,
                "execution_head": expected_head,
                "pair_first": pairs[0],
                "pair_last": pairs[-1],
                "pair_count": len(pairs),
                "row_count": ROWS_PER_SHARD,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "geometry_json_sha256": GEOMETRY_JSON_SHA256,
                "geometry_pt_sha256": GEOMETRY_PT_SHA256,
                "selected_plane": SELECTED_PLANE,
                "control_plane": CONTROL_PLANE,
                "intervention_layer": INTERVENTION_LAYER,
                "scientific_full_model_forward_count": ROWS_PER_SHARD,
                "local_backward_count": ROWS_PER_SHARD,
                "intervention_condition_forward_count": 0,
                "training_executed": False,
                "parameter_update_executed": False,
                "inferential_test_performed": False,
                "behavioral_bridge_D_BEH_accessed": False,
                "cross_scale_readout_result_accessed": False,
                "physical_gpu_inventory_count": env["physical_gpu_count"],
                "logical_device": "cuda:0",
                "logical_device_name": torch.cuda.get_device_name(0),
                "kernel_constructor_calls": env["kernel_constructor_calls"],
                "structural_source_sha256":
                    env["structural_manifest"]["source_file_sha256"],
                "structural_rows_sha256":
                    env["structural_manifest"]["row_file_sha256"],
                "tokenizer": env["tokenizer_provenance"],
            }))
    except BaseException:
        paths["error"].write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _run_two_shards(
    *,
    target,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint: Path,
    temp_dir: Path,
    expected_head: str | None,
) -> None:
    ctx = mp.get_context("spawn")
    processes = []
    for shard_index in (0, 1):
        kwargs = {
            "shard_index": shard_index,
            "physical_device": shard_index,
            "model_snapshot": str(model_snapshot),
            "tokenizer_snapshot": str(tokenizer_snapshot),
            "checkpoint": str(checkpoint),
            "temp_dir": str(temp_dir),
        }
        if expected_head is not None:
            kwargs["expected_head"] = expected_head
        p = ctx.Process(
            target=target,
            kwargs=kwargs,
            name=f"mamba130-readout-shard{shard_index}",
        )
        p.start()
        processes.append((shard_index, p))

    for shard_index, p in processes:
        p.join()
        if p.exitcode != 0:
            paths = _worker_paths(temp_dir, shard_index)
            detail = (
                paths["error"].read_text(encoding="utf-8")
                if paths["error"].is_file()
                else "NO_WORKER_ERROR"
            )
            raise Mamba130ReadoutAlignmentError(
                f"WORKER_FAILED:{shard_index}:\n{detail}"
            )


def run_technical_gate(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    validate_snapshot(model_snapshot)
    validate_snapshot(tokenizer_snapshot)
    bridge.authenticate_checkpoint(checkpoint)

    with tempfile.TemporaryDirectory(prefix="gen4_m130_readout_gate_") as tmp:
        temp_dir = Path(tmp)
        _run_two_shards(
            target=technical_gate_worker,
            model_snapshot=model_snapshot,
            tokenizer_snapshot=tokenizer_snapshot,
            checkpoint=checkpoint,
            temp_dir=temp_dir,
            expected_head=None,
        )
        for shard_index in (0, 1):
            gate = json.loads(
                _worker_paths(temp_dir, shard_index)["gate"]
                .read_text(encoding="utf-8")
            )
            for key in (
                "model_reconstruction_pass",
                "checkpoint_identity_pass",
                "frozen_geometry_identity_pass",
                "population_identity_pass",
                "anchor_identity_pass",
                "local_leaf_hook_pass",
                "finite_margin_pass",
                "finite_gradient_pass",
                "parameter_gradient_absent_pass",
            ):
                require(gate[key] is True, f"GATE:{shard_index}:{key}")
            require(gate["numeric_margin_retained"] is False, "GATE_MARGIN_RETAINED")
            require(gate["numeric_gradient_retained"] is False, "GATE_GRAD_RETAINED")
            require(gate["numeric_alignment_retained"] is False, "GATE_ALIGN_RETAINED")
            require(gate["scientific_conclusion"] is None, "GATE_CONCLUSION")

    print("RESULT=" + GATE_RESULT)
    print("SHARD0=PASS")
    print("SHARD1=PASS")
    print("TECHNICAL_FULL_MODEL_FORWARD_COUNT=2")
    print("TECHNICAL_LOCAL_BACKWARD_COUNT=2")
    print("NUMERIC_MARGIN_RETAINED=False")
    print("NUMERIC_GRADIENT_RETAINED=False")
    print("NUMERIC_ALIGNMENT_RETAINED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def _write_sums(root: Path, names: Sequence[str]) -> None:
    (root / SUMS_FILE).write_text(
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
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint: Path,
    output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    validate_snapshot(model_snapshot)
    validate_snapshot(tokenizer_snapshot)
    bridge.authenticate_checkpoint(checkpoint)
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    with tempfile.TemporaryDirectory(prefix="gen4_m130_readout_raw_") as tmp:
        temp_dir = Path(tmp)
        _run_two_shards(
            target=raw_worker,
            model_snapshot=model_snapshot,
            tokenizer_snapshot=tokenizer_snapshot,
            checkpoint=checkpoint,
            temp_dir=temp_dir,
            expected_head=expected_head,
        )

        all_items: list[dict[str, Any]] = []
        worker_meta: dict[str, Any] = {}
        for shard_index in (0, 1):
            paths = _worker_paths(temp_dir, shard_index)
            shard_items = [
                json.loads(line)
                for line in paths["items"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            require(len(shard_items) == ROWS_PER_SHARD, f"SHARD_ITEMS:{shard_index}")
            all_items.extend(shard_items)
            worker_meta[f"shard{shard_index}"] = json.loads(
                paths["meta"].read_text(encoding="utf-8")
            )

    require(len(all_items) == TOTAL_ROWS, "TOTAL_ITEM_COUNT")
    expected_tuples = {
        (pair, cell)
        for pair in PAIR_IDS
        for cell in TARGET_CELLS
    }
    observed_tuples = {
        (str(item["source_pair_id"]), str(item["contrast_cell_id"]))
        for item in all_items
    }
    require(observed_tuples == expected_tuples, "PAIR_CELL_COVERAGE")
    require(len(observed_tuples) == TOTAL_ROWS, "PAIR_CELL_UNIQUENESS")

    output_dir.mkdir(parents=True, exist_ok=False)
    item_raw = jsonl_bytes(all_items)
    (output_dir / ITEM_FILE).write_bytes(item_raw)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "population": "xg1_fact_2701..xg1_fact_3000",
        "pair_count": PAIR_COUNT,
        "cells": list(TARGET_CELLS),
        "scale": "mamba130m",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "geometry_json_sha256": GEOMETRY_JSON_SHA256,
        "geometry_pt_sha256": GEOMETRY_PT_SHA256,
        "selected_plane": SELECTED_PLANE,
        "control_plane": CONTROL_PLANE,
        "intervention_layer": INTERVENTION_LAYER,
        "anchor_name": "A_IDENTITY",
        "target_offset": TARGET_OFFSET,
        "row_count": TOTAL_ROWS,
        "primary_endpoint_reserved_for_static_analysis": PRIMARY_ENDPOINT,
        "primary_test_reserved_for_static_analysis": PRIMARY_TEST,
        "primary_alpha": PRIMARY_ALPHA,
        "primary_p_value_count_reserved": PRIMARY_P_VALUE_COUNT,
        "sign_gate_reserved_for_static_analysis": SIGN_GATE,
        "scientific_full_model_forward_count": SCIENTIFIC_FORWARD_COUNT,
        "local_backward_count": LOCAL_BACKWARD_COUNT,
        "intervention_condition_forward_count": 0,
        "training_executed": False,
        "parameter_update_executed": False,
        "inferential_test_performed": False,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
        "behavioral_bridge_D_BEH_accessed": False,
        "cross_scale_readout_result_accessed": False,
        "workers": worker_meta,
    }
    (output_dir / SUMMARY_FILE).write_bytes(pretty_json_bytes(summary))

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "plan_sha256": PLAN_SHA256,
        "item_sha256": sha256_bytes(item_raw),
        "summary_sha256": sha256_file(output_dir / SUMMARY_FILE),
        "row_count": TOTAL_ROWS,
        "scientific_full_model_forward_count": SCIENTIFIC_FORWARD_COUNT,
        "local_backward_count": LOCAL_BACKWARD_COUNT,
        "intervention_condition_forward_count": 0,
        "primary_p_value_count_executed": 0,
        "scientific_conclusion": None,
        "training_executed": False,
        "parameter_update_executed": False,
        "behavioral_bridge_D_BEH_accessed": False,
        "cross_scale_readout_result_accessed": False,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))
    _write_sums(output_dir, (ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE))

    print("RESULT=" + RAW_RESULT)
    print("PAIR_RANGE=xg1_fact_2701..xg1_fact_3000")
    print("PAIR_COUNT=300")
    print("ROW_COUNT=600")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT=600")
    print("LOCAL_BACKWARD_COUNT=600")
    print("INTERVENTION_CONDITION_FORWARD_COUNT=0")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_EXECUTED=0")
    print("BEHAVIORAL_BRIDGE_D_BEH_ACCESSED=False")
    print("CROSS_SCALE_READOUT_RESULT_ACCESSED=False")
    print("TRAINING_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gen4 Mamba-130M prospective native task-gradient/readout-alignment "
            "raw runner. Two T4 pair shards, 600 native forwards/backwards, "
            "zero intervention-condition forwards, no p-value."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("technical-gate", "raw"),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "raw":
        require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
    else:
        require(args.output_dir is None, "GATE_OUTPUT_DIR_FORBIDDEN")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "technical-gate":
        run_technical_gate(
            expected_head=args.expected_head,
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
            checkpoint=args.checkpoint,
        )
    else:
        run_raw(
            expected_head=args.expected_head,
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
