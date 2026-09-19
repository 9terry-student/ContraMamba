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

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    build_reason_router_gen4_mamba370m14b_behavioral_bridge_holdout
    as holdout,
)
from scripts import (
    reason_router_gen4_mamba370m_confirmation_fast_cuda
    as c370,
)
from scripts import (
    reason_router_gen4_mamba14b_confirmation_fast_cuda
    as c14,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "74ccb17ef168176f61b7c1946178dbadc7a1b685"

SCALE_ORDER = ("mamba370m", "mamba14b")
CONDITIONS = (
    "native",
    "dominant_neutralized",
    "dominant_restored",
    "dominant_control",
)
TARGET_CELLS = ("C0_SHAM", "C2_NAME")
LABEL_ID_BY_CELL = {"C0_SHAM": 2, "C2_NAME": 1}
LABEL_NAME_BY_CELL = {
    "C0_SHAM": "SUPPORT",
    "C2_NAME": "NOT_ENTITLED",
}

PAIR_FIRST = holdout.FIRST_PAIR
PAIR_LAST = holdout.LAST_PAIR
PAIR_COUNT = holdout.PAIR_COUNT
PAIR_IDS = tuple(holdout.expected_pair_ids())
ROWS_PER_PAIR = len(TARGET_CELLS) * len(CONDITIONS)
FORWARDS_PER_PAIR = ROWS_PER_PAIR
TOTAL_FORWARD_BUDGET = PAIR_COUNT * FORWARDS_PER_PAIR
GPU_COUNT = 2
PAIRS_PER_SHARD = PAIR_COUNT // GPU_COUNT
FORWARDS_PER_SHARD = PAIRS_PER_SHARD * FORWARDS_PER_PAIR

SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start_index": 0,
        "end_index": 150,
        "pair_first": "xg1_fact_4801",
        "pair_last": "xg1_fact_4950",
        "pair_count": 150,
        "forward_budget": FORWARDS_PER_SHARD,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_4951",
        "pair_last": "xg1_fact_5100",
        "pair_count": 150,
        "forward_budget": FORWARDS_PER_SHARD,
    },
)

ROW_FILE = "behavioral_rows.jsonl"
SUMMARY_FILE = "shard_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ROW_SCHEMA = "gen4-mamba370m14b-behavioral-bridge-row-v1"
SUMMARY_SCHEMA = "gen4-mamba370m14b-behavioral-bridge-shard-summary-v1"
RESULT_PASS = "PASS_RAW_CROSS_SCALE_BEHAVIORAL_SHARD"


class BehavioralBridgeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise BehavioralBridgeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        raise BehavioralBridgeError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH_MISMATCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR_MISSING")


def scale_spec(scale: str) -> dict[str, Any]:
    require(scale in SCALE_ORDER, f"SCALE:{scale}")
    if scale == "mamba370m":
        return {
            "scale": scale,
            "confirmation": c370,
            "geom": c370.geom,
            "adapter": c370.adapter,
            "tokenizer_gate": c370.tokenizer_gate,
            "selected_plane": "P3",
            "control_plane": "P5",
            "dim": c370.DIM,
            "checkpoint_rel": c370.geom.COMPACT_CHECKPOINT_REL,
            "checkpoint_sha256": c370.geom.COMPACT_CHECKPOINT_SHA256,
            "hf_repo": c370.geom.HF_REPO,
            "hf_revision": c370.geom.HF_REVISION,
            "geometry_freeze_commit": c370.GEOMETRY_FREEZE_COMMIT,
            "discovery_freeze_commit": c370.DISCOVERY_FREEZE_COMMIT,
            "discovery_selection_sha256": c370.DISCOVERY_SELECTION_SHA256,
        }
    return {
        "scale": scale,
        "confirmation": c14,
        "geom": c14.geom,
        "adapter": c14.adapter,
        "tokenizer_gate": c14.tokenizer_gate,
        "selected_plane": "P5",
        "control_plane": "P4",
        "dim": c14.DIM,
        "checkpoint_rel": c14.geom.COMPACT_CHECKPOINT_REL,
        "checkpoint_sha256": c14.geom.COMPACT_CHECKPOINT_SHA256,
        "hf_repo": c14.geom.HF_REPO,
        "hf_revision": c14.geom.HF_REVISION,
        "geometry_freeze_commit": c14.GEOMETRY_FREEZE_COMMIT,
        "discovery_freeze_commit": c14.DISCOVERY_FREEZE_COMMIT,
        "discovery_selection_sha256": c14.DISCOVERY_SELECTION_SHA256,
    }


def validate_protocol() -> None:
    require(PAIR_FIRST == 4801 and PAIR_LAST == 5100, "PAIR_RANGE")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(CONDITIONS == (
        "native",
        "dominant_neutralized",
        "dominant_restored",
        "dominant_control",
    ), "CONDITIONS")
    require(TARGET_CELLS == ("C0_SHAM", "C2_NAME"), "TARGET_CELLS")
    require(FORWARDS_PER_PAIR == 8, "FORWARDS_PER_PAIR")
    require(FORWARDS_PER_SHARD == 1200, "FORWARDS_PER_SHARD")
    require(TOTAL_FORWARD_BUDGET == 2400, "TOTAL_FORWARD_BUDGET")
    require(len(SHARDS) == GPU_COUNT == 2, "SHARD_COUNT")

    covered: list[int] = []
    for expected_id, shard in enumerate(SHARDS):
        require(shard["shard_id"] == expected_id, "SHARD_ID")
        require(shard["physical_device"] == expected_id, "SHARD_DEVICE")
        require(
            shard["end_index"] - shard["start_index"]
            == shard["pair_count"],
            "SHARD_PAIR_COUNT",
        )
        require(
            shard["pair_count"] * FORWARDS_PER_PAIR
            == shard["forward_budget"],
            "SHARD_FORWARD_BUDGET",
        )
        covered.extend(range(shard["start_index"], shard["end_index"]))
    require(covered == list(range(PAIR_COUNT)), "SHARD_COVERAGE")


def load_population() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    root = ROOT / holdout.OUTPUT_DIR
    manifest = holdout.validate_written(root)
    require(manifest["result"] == holdout.RESULT, "HOLDOUT_RESULT")
    require(
        manifest["shared_population_across_scales"] is True,
        "HOLDOUT_SHARED",
    )

    facts = holdout._read_jsonl(root / holdout.SOURCE_FILE)
    rows = holdout._read_jsonl(root / holdout.ROW_FILE)
    require(len(facts) == 300 and len(rows) == 1800, "HOLDOUT_COUNTS")
    return facts, rows, manifest


def validate_behavioral_rows(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    holdout.validate_semantic_labels(facts, rows)
    facts_by_id = {str(row["pair_id"]): row for row in facts}
    by_key = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"])): row
        for row in rows
    }
    selected: list[dict[str, Any]] = []
    for pair in PAIR_IDS:
        selected.extend([
            dict(by_key[(pair, "C0_SHAM")]),
            dict(by_key[(pair, "C2_NAME")]),
        ])
    require(len(selected) == 600, "SELECTED_ROW_COUNT")
    return selected, facts_by_id


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
    facts, all_rows, manifest = load_population()
    selected, facts_by_id = validate_behavioral_rows(facts, all_rows)

    geom = spec["geom"]
    tokenizer_gate = spec["tokenizer_gate"]
    adapter = spec["adapter"]

    tokenizer, tokenizer_provenance = geom.load_tokenizer(snapshot)
    encoded = adapter.encode_gen4_rows(selected, tokenizer)
    require(
        tuple(encoded["input_ids"].shape) == (600, 128),
        "ENCODED_INPUT_SHAPE",
    )

    events: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in selected:
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        analyzed = tokenizer_gate.analyze_required_anchors_for_row(
            row,
            facts_by_id[pair],
            tokenizer,
        )
        names = {str(event["anchor_name"]) for event in analyzed}
        require(
            "A_IDENTITY" in names,
            f"ANCHOR_SET:{pair}:{cell}",
        )
        for event in analyzed:
            key = (pair, cell, str(event["anchor_name"]))
            require(key not in events, f"ANCHOR_DUPLICATE:{key}")
            require(
                bool(event["post4_eligible"]),
                f"ANCHOR_INELIGIBLE:{key}",
            )
            events[key] = dict(event)


    return selected, encoded, events, {
        "manifest": manifest,
        "tokenizer": tokenizer_provenance,
    }


def row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in out, f"DUPLICATE_ROW:{key}")
        out[key] = index
    require(len(out) == 600, "ROW_INDEX_COUNT")
    return out


def feature_batch(
    encoded: Mapping[str, Any],
    index: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in (
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
    ):
        tensor = encoded[key]
        require(torch.is_tensor(tensor), f"ENCODED_TENSOR:{key}")
        out[key] = (
            tensor[index:index + 1]
            .detach()
            .to(device)
            .contiguous()
        )
    return out


def plane_component(
    h: torch.Tensor,
    *,
    plane: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    dim: int,
) -> dict[str, Any]:
    value = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(value.shape) == (dim,), "H_SHAPE")
    require(plane in planes, f"PLANE:{plane}")

    plus = planes[plane]["plus"]
    minus = planes[plane]["minus"]
    require(
        tuple(plus.shape) == (dim,)
        and tuple(minus.shape) == (dim,),
        f"PLANE_SHAPE:{plane}",
    )
    a = float(torch.dot(value, plus).item())
    b = float(torch.dot(value, minus).item())
    component = (a * plus + b * minus).contiguous()
    return {
        "a": a,
        "b": b,
        "component": component,
        "component_l2": float(
            torch.linalg.vector_norm(component).item()
        ),
    }


def condition_correction(
    h: torch.Tensor,
    *,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    selected_plane: str,
    control_plane: str,
    dim: int,
    tol: float,
) -> dict[str, Any]:
    require(condition in CONDITIONS[1:], f"CONDITION:{condition}")
    value = h.detach().cpu().to(torch.float64).contiguous()
    selected = plane_component(
        value,
        plane=selected_plane,
        planes=planes,
        dim=dim,
    )
    selected_component = selected["component"]

    matched_norm_mismatch = None
    selected_post_max = None

    if condition == "dominant_neutralized":
        correction = (-selected_component).contiguous()

    elif condition == "dominant_restored":
        correction = (
            -selected_component + selected_component
        ).contiguous()
        require(
            float(torch.max(torch.abs(correction)).item()) <= tol,
            "RESTORATION_ROUND_TRIP",
        )

    else:
        control_plus = planes[control_plane]["plus"]
        control_minus = planes[control_plane]["minus"]
        control_component = (
            float(selected["a"]) * control_plus
            + float(selected["b"]) * control_minus
        ).contiguous()
        matched_norm_mismatch = abs(
            float(torch.linalg.vector_norm(selected_component).item())
            - float(torch.linalg.vector_norm(control_component).item())
        )
        require(
            matched_norm_mismatch <= tol,
            f"MATCHED_CONTROL_NORM:{matched_norm_mismatch}",
        )
        correction = (
            -selected_component + control_component
        ).contiguous()

    post = (value + correction).contiguous()
    selected_post = [
        float(torch.dot(post, planes[selected_plane]["plus"]).item()),
        float(torch.dot(post, planes[selected_plane]["minus"]).item()),
    ]

    if condition in {
        "dominant_neutralized",
        "dominant_control",
    }:
        selected_post_max = max(abs(v) for v in selected_post)
        require(
            selected_post_max <= tol,
            f"SELECTED_NEUTRALIZATION:{selected_post_max}",
        )

    require(
        bool(torch.isfinite(correction).all().item()),
        "CORRECTION_NONFINITE",
    )
    return {
        "condition": condition,
        "a": float(selected["a"]),
        "b": float(selected["b"]),
        "selected_component_l2":
            float(selected["component_l2"]),
        "correction": correction,
        "correction_l2":
            float(torch.linalg.vector_norm(correction).item()),
        "selected_post_max_abs_projection": selected_post_max,
        "matched_control_norm_abs_difference":
            matched_norm_mismatch,
    }


def behavior_hook(
    output: torch.Tensor,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    selected_plane: str,
    control_plane: str,
    dim: int,
    intermediate_size: int,
    tol: float,
    cast_tol: float,
    audit: dict[str, Any],
) -> torch.Tensor:
    require(condition in CONDITIONS[1:], f"HOOK_CONDITION:{condition}")
    require(
        output.ndim == 3
        and output.shape[0] == 1
        and output.shape[-1] == 2 * intermediate_size,
        "INPROJ_SHAPE",
    )
    require(0 <= token_index < output.shape[1], "TOKEN_INDEX")

    before = output.detach().clone()
    mask = strong_mask.detach().cpu().bool().contiguous()
    require(
        mask.numel() == intermediate_size
        and int(mask.sum().item()) == dim,
        "STRONG_MASK",
    )
    mask_device = mask.to(before.device)

    h = (
        before[0, token_index, :intermediate_size][mask_device]
        .detach().cpu().to(torch.float64).contiguous()
    )

    info = condition_correction(
        h,
        condition=condition,
        planes=planes,
        selected_plane=selected_plane,
        control_plane=control_plane,
        dim=dim,
        tol=tol,
    )
    correction = info["correction"]

    out = output.clone()
    intended = correction.to(
        device=out.device,
        dtype=out.dtype,
    )
    out[
        0,
        token_index,
        :intermediate_size,
    ][mask_device] += intended

    require(
        torch.equal(
            out[:, :, intermediate_size:],
            before[:, :, intermediate_size:],
        ),
        "GATE_CHANGED",
    )
    require(
        torch.equal(
            out[:, :, :intermediate_size][:, :, ~mask_device],
            before[:, :, :intermediate_size][:, :, ~mask_device],
        ),
        "NONSTRONG_CHANGED",
    )
    if token_index:
        require(
            torch.equal(
                out[:, :token_index, :],
                before[:, :token_index, :],
            ),
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

    applied = (
        out[0, token_index, :intermediate_size][mask_device]
        - before[0, token_index, :intermediate_size][mask_device]
    ).detach().cpu().to(torch.float64)

    residual = float(
        torch.max(
            torch.abs(
                applied
                - intended.detach().cpu().to(torch.float64)
            )
        ).item()
    )
    require(
        residual <= cast_tol,
        f"APPLIED_RESIDUAL:{residual}",
    )

    audit.clear()
    audit.update({
        "condition": condition,
        "selected_plane": selected_plane,
        "control_plane": control_plane,
        "token_index": int(token_index),
        "native_selected_a": float(info["a"]),
        "native_selected_b": float(info["b"]),
        "selected_component_l2":
            float(info["selected_component_l2"]),
        "correction_l2": float(info["correction_l2"]),
        "selected_post_max_abs_projection":
            info["selected_post_max_abs_projection"],
        "matched_control_norm_abs_difference":
            info["matched_control_norm_abs_difference"],
        "applied_correction_max_abs_residual": residual,
        "probe_correction_l2": 0.0,
    })
    return out


def install_behavior_hook(
    mixer: Any,
    **kwargs: Any,
):
    def hook(_module, _args, output):
        return behavior_hook(output, **kwargs)

    return mixer.in_proj.register_forward_hook(hook)


def correct_margin(
    logits: Sequence[float],
    label_id: int,
) -> float:
    require(
        len(logits) == 3 and label_id in {0, 1, 2},
        "MARGIN_INPUT",
    )
    correct = float(logits[label_id])
    wrong = max(
        float(logits[index])
        for index in range(3)
        if index != label_id
    )
    value = correct - wrong
    require(math.isfinite(value), "MARGIN_NONFINITE")
    return value


def serialize_output(
    output: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    scale: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    for key in ("logits", "q_authorized", "entitlement_prob"):
        require(key in output, f"MODEL_OUTPUT_KEY:{key}")

    logits = output["logits"]
    q = output["q_authorized"]
    entitlement = output["entitlement_prob"]

    require(
        torch.is_tensor(logits)
        and tuple(logits.shape) == (1, 3),
        "LOGIT_SHAPE",
    )
    require(
        torch.is_tensor(q) and tuple(q.shape) == (1,),
        "Q_AUTHORIZED_SHAPE",
    )
    require(
        torch.is_tensor(entitlement)
        and tuple(entitlement.shape) == (1,),
        "ENTITLEMENT_SHAPE",
    )

    values = [
        float(value)
        for value in logits.detach().cpu()[0].tolist()
    ]
    require(
        all(math.isfinite(value) for value in values),
        "LOGITS_NONFINITE",
    )
    prediction_id = int(torch.argmax(logits, dim=-1).item())

    return {
        "scale": scale,
        "checkpoint_sha256": checkpoint_sha256,
        "source_pair_id": str(row["source_pair_id"]),
        "row_id": str(row["row_id"]),
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "refute_logit": values[0],
        "not_entitled_logit": values[1],
        "support_logit": values[2],
        "final_logits": values,
        "prediction_id": prediction_id,
        "prediction": (
            "REFUTE",
            "NOT_ENTITLED",
            "SUPPORT",
        )[prediction_id],
        "q_authorized": float(q.detach().cpu().item()),
        "entitlement_prob":
            float(entitlement.detach().cpu().item()),
    }


def run_condition(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index_value: int,
    anchor_index: int,
    condition: str,
    device: torch.device,
) -> dict[str, Any]:
    geom = spec["geom"]
    confirmation = spec["confirmation"]
    adapter = spec["adapter"]

    audit: dict[str, Any] | None = None
    handle = None

    if condition != "native":
        audit = {}
        handle = install_behavior_hook(
            runtime_ctx["intervention_mixer"],
            token_index=anchor_index + geom.TARGET_OFFSET,
            strong_mask=runtime_ctx["strong_mask"],
            condition=condition,
            planes=frozen["planes"],
            selected_plane=str(spec["selected_plane"]),
            control_plane=str(spec["control_plane"]),
            dim=int(spec["dim"]),
            intermediate_size=int(geom.INTERMEDIATE_SIZE),
            tol=float(confirmation.TOL),
            cast_tol=float(confirmation.transport_runtime.RUNTIME_CAST_TOL),
            audit=audit,
        )

    try:
        with torch.inference_mode():
            output = adapter.historical_forward(
                model,
                feature_batch(
                    encoded,
                    row_index_value,
                    device,
                ),
                arm=geom.ARM,
            )
    finally:
        if handle is not None:
            handle.remove()

    serialized = serialize_output(
        output,
        row=row,
        scale=str(spec["scale"]),
        checkpoint_sha256=str(spec["checkpoint_sha256"]),
    )
    cell = str(row["contrast_cell_id"])
    label_id = LABEL_ID_BY_CELL[cell]
    serialized.update({
        "schema_version": ROW_SCHEMA,
        "condition": condition,
        "correct_label_id": label_id,
        "correct_label": LABEL_NAME_BY_CELL[cell],
        "is_correct": serialized["prediction_id"] == label_id,
        "correct_class_logit_margin":
            correct_margin(serialized["final_logits"], label_id),
        "anchor_name": "A_IDENTITY",
        "absolute_anchor_token_index": int(anchor_index),
        "target_intervention_token_index":
            int(anchor_index + geom.TARGET_OFFSET),
        "selected_dominant_candidate":
            str(spec["selected_plane"]),
        "response_blind_control_plane":
            str(spec["control_plane"]),
        "intervention_audit": audit,
        "scientific_full_model_forward_count": 1,
    })
    return serialized


def _worker_paths(
    temp_dir: Path,
    shard_id: int,
) -> dict[str, Path]:
    return {
        "rows": temp_dir / f"shard_{shard_id}_rows.jsonl",
        "summary": temp_dir / f"shard_{shard_id}_summary.json",
        "error": temp_dir / f"shard_{shard_id}.error.txt",
    }


def worker_run(
    *,
    scale: str,
    shard: Mapping[str, Any],
    model_snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
    expected_head: str,
) -> None:
    shard_id = int(shard["shard_id"])
    physical_device = int(shard["physical_device"])
    paths = _worker_paths(Path(temp_dir), shard_id)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        spec = scale_spec(scale)
        confirmation = spec["confirmation"]
        geom = spec["geom"]

        device = confirmation.runtime_gate_single_visible_gpu(
            physical_device
        )
        snapshot = Path(model_snapshot)
        checkpoint = Path(compact_checkpoint)

        geom.validate_snapshot(snapshot)
        require(
            checkpoint.resolve()
            == (ROOT / spec["checkpoint_rel"]).resolve(),
            "COMPACT_CHECKPOINT_PATH",
        )
        require(
            sha256_file(checkpoint)
            == spec["checkpoint_sha256"],
            "COMPACT_CHECKPOINT_SHA",
        )

        confirmation.load_frozen_selection()
        frozen = confirmation.load_frozen_geometry()
        rows, encoded, events, population_provenance = (
            build_input_state(
                spec=spec,
                snapshot=snapshot,
            )
        )
        lookup = row_index(rows)

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        confirmation.kernel_compat.validate_transformers_kernel_bindings(
            kernels
        )
        runtime_ctx = geom.runtime_components(model)
        confirmation.validate_runtime_geometry(
            runtime_ctx,
            frozen,
        )

        output_rows: list[dict[str, Any]] = []

        for global_index in range(
            int(shard["start_index"]),
            int(shard["end_index"]),
        ):
            pair = PAIR_IDS[global_index]
            for cell in TARGET_CELLS:
                index = lookup[(pair, cell)]
                row = rows[index]
                anchor = int(
                    events[(pair, cell, "A_IDENTITY")][
                        "absolute_anchor_token_index"
                    ]
                )
                for condition in CONDITIONS:
                    item = run_condition(
                        spec=spec,
                        model=model,
                        runtime_ctx=runtime_ctx,
                        frozen=frozen,
                        encoded=encoded,
                        row=row,
                        row_index_value=index,
                        anchor_index=anchor,
                        condition=condition,
                        device=device,
                    )
                    item["shard_index"] = shard_id
                    item["physical_device"] = physical_device
                    output_rows.append(item)

        torch.cuda.synchronize(device)

        require(
            len(output_rows) == int(shard["forward_budget"]),
            "WORKER_ROW_COUNT",
        )
        expected = {
            (pair, cell, condition)
            for pair in PAIR_IDS[
                int(shard["start_index"]):
                int(shard["end_index"])
            ]
            for cell in TARGET_CELLS
            for condition in CONDITIONS
        }
        observed = {
            (
                str(row["source_pair_id"]),
                str(row["contrast_cell_id"]),
                str(row["condition"]),
            )
            for row in output_rows
        }
        require(observed == expected, "WORKER_TUPLE_COVERAGE")

        paths["rows"].write_bytes(jsonl_bytes(output_rows))
        manifest = population_provenance["manifest"]

        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "result": RESULT_PASS,
            "execution_head": expected_head,
            "scale": scale,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "hf_repo": spec["hf_repo"],
            "hf_revision": spec["hf_revision"],
            "geometry_freeze_commit":
                spec["geometry_freeze_commit"],
            "discovery_freeze_commit":
                spec["discovery_freeze_commit"],
            "discovery_selection_sha256":
                spec["discovery_selection_sha256"],
            "selected_dominant_candidate":
                spec["selected_plane"],
            "response_blind_control_plane":
                spec["control_plane"],
            "selection_uses_behavioral_response": False,
            "population_first": shard["pair_first"],
            "population_last": shard["pair_last"],
            "source_pair_count": shard["pair_count"],
            "shared_population_first": "xg1_fact_4801",
            "shared_population_last": "xg1_fact_5100",
            "structural_source_sha256":
                manifest["source_file_sha256"],
            "structural_rows_sha256":
                manifest["row_file_sha256"],
            "target_cells": list(TARGET_CELLS),
            "condition_order": list(CONDITIONS),
            "label_contract": {
                "C0_SHAM": "SUPPORT",
                "C2_NAME": "NOT_ENTITLED",
            },
            "shard_index": shard_id,
            "physical_device": physical_device,
            "cuda_visible_devices":
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": "cuda:0",
            "logical_device_name":
                torch.cuda.get_device_name(0),
            "tokenizer": population_provenance["tokenizer"],
            "model_provenance": model_provenance,
            "scientific_full_model_forward_count_this_run":
                int(shard["forward_budget"]),
            "primary_inference_executed": False,
            "scientific_conclusion": None,
            "selection_reopened": False,
            "confirmation_inference_accessed": False,
            "rescue_performed": False,
            "training_executed": False,
            "backward_executed": False,
        }
        paths["summary"].write_bytes(pretty_json_bytes(summary))

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def write_shard_dir(
    output_dir: Path,
    *,
    shard_id: int,
    rows_raw: bytes,
    summary_raw: bytes,
) -> None:
    shard_dir = output_dir / f"shard{shard_id}"
    require(not shard_dir.exists(), f"SHARD_OUTPUT_COLLISION:{shard_id}")
    shard_dir.mkdir(parents=True, exist_ok=False)
    (shard_dir / ROW_FILE).write_bytes(rows_raw)
    (shard_dir / SUMMARY_FILE).write_bytes(summary_raw)

    hashes = {
        ROW_FILE: hashlib.sha256(rows_raw).hexdigest(),
        SUMMARY_FILE: hashlib.sha256(summary_raw).hexdigest(),
    }
    (shard_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_scale(
    *,
    scale: str,
    expected_head: str,
    model_snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
    authenticate_checkout: bool = True,
) -> None:
    validate_protocol()
    if authenticate_checkout:
        authenticate_repo(expected_head)
    spec = scale_spec(scale)
    geom = spec["geom"]
    confirmation = spec["confirmation"]

    geom.validate_snapshot(model_snapshot)
    require(
        compact_checkpoint.resolve()
        == (ROOT / spec["checkpoint_rel"]).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        sha256_file(compact_checkpoint)
        == spec["checkpoint_sha256"],
        "COMPACT_CHECKPOINT_SHA",
    )
    confirmation.load_frozen_selection()
    confirmation.load_frozen_geometry()
    load_population()

    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= 2, "PHYSICAL_GPU_COUNT")

    with tempfile.TemporaryDirectory(
        prefix=f"gen4_{scale}_behavioral_bridge_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for shard in SHARDS:
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "scale": scale,
                    "shard": shard,
                    "model_snapshot": str(model_snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                    "expected_head": expected_head,
                },
                name=(
                    f"{scale}-behavioral-bridge-"
                    f"shard{shard['shard_id']}"
                ),
            )
            process.start()
            processes.append(process)

        for shard, process in zip(
            SHARDS,
            processes,
            strict=True,
        ):
            process.join()
            if process.exitcode != 0:
                paths = _worker_paths(
                    temp_dir,
                    int(shard["shard_id"]),
                )
                detail = (
                    paths["error"].read_text(encoding="utf-8")
                    if paths["error"].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise BehavioralBridgeError(
                    "WORKER_FAILED:"
                    f"{shard['shard_id']}:\n{detail}"
                )

        output_dir.mkdir(parents=True, exist_ok=False)
        for shard in SHARDS:
            shard_id = int(shard["shard_id"])
            paths = _worker_paths(temp_dir, shard_id)
            require(paths["rows"].is_file(), "WORKER_ROWS_MISSING")
            require(
                paths["summary"].is_file(),
                "WORKER_SUMMARY_MISSING",
            )
            write_shard_dir(
                output_dir,
                shard_id=shard_id,
                rows_raw=paths["rows"].read_bytes(),
                summary_raw=paths["summary"].read_bytes(),
            )

    total = 0
    for shard in SHARDS:
        summary = json.loads(
            (
                output_dir
                / f"shard{shard['shard_id']}"
                / SUMMARY_FILE
            ).read_text(encoding="utf-8")
        )
        total += int(
            summary[
                "scientific_full_model_forward_count_this_run"
            ]
        )
    require(total == TOTAL_FORWARD_BUDGET, "TOTAL_FORWARD_SUM")

    print("RESULT=PASS_RAW_CROSS_SCALE_BEHAVIORAL_RUN")
    print("SCALE=" + scale)
    print(
        "SELECTED_DOMINANT_CANDIDATE="
        + str(spec["selected_plane"])
    )
    print(
        "RESPONSE_BLIND_CONTROL_PLANE="
        + str(spec["control_plane"])
    )
    print("PAIR_RANGE=xg1_fact_4801..xg1_fact_5100")
    print("SOURCE_PAIR_COUNT=300")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT=2400")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("CONFIRMATION_INFERENCE_ACCESSED=False")
    print("SELECTION_REOPENED=False")
    print("RESCUE_PERFORMED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")



def run_both(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run_scale(
            scale="mamba370m",
            expected_head=expected_head,
            model_snapshot=mamba370m_snapshot,
            compact_checkpoint=(
                ROOT / scale_spec("mamba370m")["checkpoint_rel"]
            ),
            output_dir=output_dir / "mamba370m",
            authenticate_checkout=False,
        )
        run_scale(
            scale="mamba14b",
            expected_head=expected_head,
            model_snapshot=mamba14b_snapshot,
            compact_checkpoint=(
                ROOT / scale_spec("mamba14b")["checkpoint_rel"]
            ),
            output_dir=output_dir / "mamba14b",
            authenticate_checkout=False,
        )
    except BaseException:
        raise

    print("RESULT=PASS_RAW_CROSS_SCALE_BEHAVIORAL_BOTH_SCALES")
    print("PAIR_RANGE=xg1_fact_4801..xg1_fact_5100")
    print("MAMBA370M_FORWARD_COUNT=2400")
    print("MAMBA14B_FORWARD_COUNT=2400")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT_TOTAL=4800")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 370M/1.4B full-model behavioral bridge "
            "on the shared XG1 4801..5100 cohort. Raw outputs only."
        )
    )
    parser.add_argument(
        "--scale",
        choices=("mamba370m", "mamba14b", "both"),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--mamba370m-snapshot", type=Path)
    parser.add_argument("--mamba14b-snapshot", type=Path)
    parser.add_argument("--compact-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.scale == "both":
        require(
            args.mamba370m_snapshot is not None,
            "MAMBA370M_SNAPSHOT_REQUIRED",
        )
        require(
            args.mamba14b_snapshot is not None,
            "MAMBA14B_SNAPSHOT_REQUIRED",
        )
        require(
            args.model_snapshot is None,
            "MODEL_SNAPSHOT_NOT_ALLOWED_FOR_BOTH",
        )
        require(
            args.compact_checkpoint is None,
            "COMPACT_CHECKPOINT_NOT_ALLOWED_FOR_BOTH",
        )
    else:
        require(
            args.model_snapshot is not None,
            "MODEL_SNAPSHOT_REQUIRED",
        )
        if args.compact_checkpoint is None:
            spec = scale_spec(args.scale)
            args.compact_checkpoint = ROOT / spec["checkpoint_rel"]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.scale == "both":
        run_both(
            expected_head=args.expected_head,
            mamba370m_snapshot=args.mamba370m_snapshot,
            mamba14b_snapshot=args.mamba14b_snapshot,
            output_dir=args.output_dir,
        )
        return

    run_scale(
        scale=args.scale,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
