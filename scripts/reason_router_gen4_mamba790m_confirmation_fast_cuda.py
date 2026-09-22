#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import (
    build_reason_router_gen4_mamba370m_xg1_holdouts
    as holdout_builder,
)
from scripts import (
    reason_router_gen4_generator_family_prevalence_kernel_compat
    as kernel_compat,
)
from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend
from scripts import (
    reason_router_gen4_k_directional_alignment_transport_runtime
    as transport_runtime,
)
from scripts import (
    reason_router_gen4_mamba790m_discovery_fast_cuda
    as discovery,
)
from scripts import (
    reason_router_gen4_mamba790m_geometry_prepare_fast_cuda
    as geom,
)
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement
from scripts import (
    reason_router_gen4_six_cell_tier2_inference_adapter
    as adapter,
)
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"

GEOMETRY_FREEZE_COMMIT = "ba4c73aa9a5e63108f3582e053678b6344990d2c"
CONFIRMATION_HOLDOUT_FREEZE_COMMIT = "6c0989a45382db31c9052b89b088e999b3e7cf59"
DISCOVERY_FREEZE_COMMIT = "e6ba3c466d3063ff69112ca8f57f0d569ee63d06"

DISCOVERY_SELECTION_PATH = Path(
    "reports/reason_router_gen4_mamba790m_discovery_runs/"
    "g4k-mamba790m-discovery-xg1-6901-7200-2gpu-e951584-workcache/"
    "discovery_selection.json"
)
DISCOVERY_SELECTION_SHA256 = (
    "854324f4799d9011bcde734fb8b4959f2ff1aa765a9f82572aa9ede1ea3c188e"
)

CONFIRMATION_ROOT = Path(
    "data/reason_router_gen4_mamba790m_xg1_confirmation_v1"
)
CONFIRMATION_SOURCE_SHA256 = (
    "8da2bf45693ee4d2224db3d3379af1f17ec082f8210da8b988a84559ef96671b"
)
CONFIRMATION_ROWS_SHA256 = (
    "e1e1e93272d244b4db91bcd7dc7eeb20a48575e36853509a6b5ff9944c11e011"
)
CONFIRMATION_MANIFEST_SHA256 = (
    "0db2bb28c41dd40037466da25892cdb6b6e5cc268bf7819fd7f1a9642b23aaf9"
)

PAIR_FIRST = 7201
PAIR_LAST = 7500
PAIR_COUNT = 300
ROWS = 1800
PAIR_IDS = tuple(
    f"xg1_fact_{index:03d}"
    for index in range(PAIR_FIRST, PAIR_LAST + 1)
)

PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")
SELECTED_PLANE = "P2"
CONTROL_PLANE = "P5"

K = 5
EPS = 0.025
DIM = 975
STATE_WIDTH = geom.INTERMEDIATE_SIZE * geom.STATE_SIZE

DIRECTION_ORDER = tuple(
    [f"xg2_{index}" for index in range(K)]
    + [f"xg4_{index}" for index in range(K)]
)

CONDITION_ORDER = (
    "dominant_restored",
    "dominant_control",
)

TARGET_PLUS_CELL = geom.TARGET_PLUS_CELL
TARGET_MINUS_CELL = geom.TARGET_MINUS_CELL
REFERENCE_PLUS_CELL = geom.REFERENCE_PLUS_CELL
REFERENCE_MINUS_CELL = geom.REFERENCE_MINUS_CELL
ANCHOR_NAME = "A_IDENTITY"

FORWARDS_PER_SIGNED = 2
FORWARDS_PER_DIRECTION = 4
FORWARDS_PER_CONDITION = len(DIRECTION_ORDER) * FORWARDS_PER_DIRECTION
FORWARDS_PER_PAIR = len(CONDITION_ORDER) * FORWARDS_PER_CONDITION
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
        "pair_first": "xg1_fact_7201",
        "pair_last": "xg1_fact_7350",
        "pair_count": 150,
        "forward_budget": FORWARDS_PER_SHARD,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_7351",
        "pair_last": "xg1_fact_7500",
        "pair_count": 150,
        "forward_budget": FORWARDS_PER_SHARD,
    },
)

ALPHA = 0.05
SUCCESS_LABEL = "MAMBA790M_CORE_CONFIRMATION_SUPPORTED"
FAILURE_LABEL = "MAMBA790M_CORE_CONFIRMATION_NOT_SUPPORTED"

ITEM_FILE = "confirmation_items.jsonl"
INFERENCE_FILE = "confirmation_inference.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

RESULT_PASS = "PASS_MAMBA790M_CORE_CONFIRMATION"
ITEM_SCHEMA = "gen4-mamba790m-core-confirmation-item-v1"
INFERENCE_SCHEMA = "gen4-mamba790m-core-confirmation-inference-v1"
MANIFEST_SCHEMA = "gen4-mamba790m-core-confirmation-manifest-v1"

TOL = 1.0e-12
PLANE_TOL = 1.0e-9


class ConfirmationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ConfirmationError(message)


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
        raise ConfirmationError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(status == "", "WORKTREE_NOT_CLEAN")
    for ancestor, label in (
        (GEOMETRY_FREEZE_COMMIT, "GEOMETRY"),
        (CONFIRMATION_HOLDOUT_FREEZE_COMMIT, "CONFIRMATION_HOLDOUT"),
        (DISCOVERY_FREEZE_COMMIT, "DISCOVERY"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_FREEZE_NOT_ANCESTOR")


def validate_protocol() -> None:
    require(geom.SOURCE_BLOCK == 33, "SOURCE_BLOCK")
    require(geom.TARGET_RESIDUAL_LAYER == 34, "TARGET_LAYER")
    require(geom.INTERVENTION_LAYER == 35, "INTERVENTION_LAYER")
    require(geom.TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(K == 5, "K")
    require(EPS == 0.025, "EPS")
    require(DIM == 975, "DIM")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(SELECTED_PLANE == "P2", "SELECTED_PLANE")
    require(CONTROL_PLANE == "P5", "CONTROL_PLANE")
    require(FORWARDS_PER_CONDITION == 40, "FORWARDS_PER_CONDITION")
    require(FORWARDS_PER_PAIR == 80, "FORWARDS_PER_PAIR")
    require(TOTAL_FORWARD_BUDGET == 24000, "TOTAL_FORWARD_BUDGET")
    require(FORWARDS_PER_SHARD == 12000, "FORWARDS_PER_SHARD")
    require(len(SHARDS) == GPU_COUNT == 2, "SHARD_COUNT")
    require(ALPHA == 0.05, "ALPHA")
    covered = []
    for expected_id, shard in enumerate(SHARDS):
        require(shard["shard_id"] == expected_id, "SHARD_ID")
        require(shard["physical_device"] == expected_id, "SHARD_DEVICE")
        require(shard["end_index"] - shard["start_index"] == shard["pair_count"], "SHARD_PAIR_COUNT")
        require(shard["pair_count"] * FORWARDS_PER_PAIR == shard["forward_budget"], "SHARD_FORWARD_BUDGET")
        covered.extend(range(shard["start_index"], shard["end_index"]))
    require(covered == list(range(PAIR_COUNT)), "SHARD_COVERAGE")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        output.append(value)
    return output


def load_frozen_selection() -> dict[str, Any]:
    path = ROOT / DISCOVERY_SELECTION_PATH
    require(path.is_file(), "DISCOVERY_SELECTION_MISSING")
    require(sha256_file(path) == DISCOVERY_SELECTION_SHA256, "DISCOVERY_SELECTION_SHA")
    selection = json.loads(path.read_text(encoding="utf-8-sig"))
    require(selection["result"] == "PASS_MAMBA790M_DISCOVERY_SELECTION_FREEZE", "DISCOVERY_SELECTION_RESULT")
    require(selection["selected_dominant_candidate"] == SELECTED_PLANE, "DISCOVERY_SELECTED_PLANE")
    require(selection["response_blind_control_plane"] == CONTROL_PLANE, "DISCOVERY_CONTROL_PLANE")
    require(selection["selection_unique"] is True, "DISCOVERY_SELECTION_UNIQUE")
    require(selection["positivity_gate_applied"] is False, "DISCOVERY_POSITIVITY_GATE")
    require(selection["control_selection_uses_response"] is False, "DISCOVERY_CONTROL_RESPONSE")
    require(selection["confirmation_accessed"] is False, "DISCOVERY_CONFIRMATION_ALREADY_ACCESSED")
    require(selection["confirmation_response_accessed"] is False, "DISCOVERY_CONFIRMATION_RESPONSE_ALREADY_ACCESSED")
    require(selection["scientific_conclusion_established"] is False, "DISCOVERY_CONCLUSION_PREMATURE")
    require(selection["geometry_freeze_commit"] == GEOMETRY_FREEZE_COMMIT, "DISCOVERY_GEOMETRY_FREEZE")
    return selection


def load_confirmation_population() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = CONFIRMATION_ROOT / holdout_builder.SOURCE_FILE
    rows = CONFIRMATION_ROOT / holdout_builder.ROW_FILE
    manifest_path = CONFIRMATION_ROOT / "structural_manifest.json"
    require(source.is_file(), "CONFIRMATION_SOURCE_MISSING")
    require(rows.is_file(), "CONFIRMATION_ROWS_MISSING")
    require(manifest_path.is_file(), "CONFIRMATION_MANIFEST_MISSING")
    require(sha256_file(source) == CONFIRMATION_SOURCE_SHA256, "CONFIRMATION_SOURCE_SHA")
    require(sha256_file(rows) == CONFIRMATION_ROWS_SHA256, "CONFIRMATION_ROWS_SHA")
    require(sha256_file(manifest_path) == CONFIRMATION_MANIFEST_SHA256, "CONFIRMATION_MANIFEST_SHA")
    facts = _read_jsonl(source)
    materialized = _read_jsonl(rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(len(facts) == PAIR_COUNT, "CONFIRMATION_FACT_COUNT")
    require(len(materialized) == ROWS, "CONFIRMATION_ROW_COUNT")
    require([str(row["pair_id"]) for row in facts] == list(PAIR_IDS), "CONFIRMATION_PAIR_ORDER")
    holdout_builder.prior.base.validate_materialized_rows(materialized, expected_pairs=PAIR_COUNT)
    require(manifest["schema_version"] == "GEN4_MAMBA1_FIVE_SCALE_LADDER_XG1_PROSPECTIVE_HOLDOUT_V1", "CONFIRMATION_SCHEMA")
    require(manifest["result"] == "PASS_MAMBA1_FIVE_SCALE_LADDER_XG1_STRUCTURAL", "CONFIRMATION_RESULT")
    require(manifest["role"] == "confirmation", "CONFIRMATION_ROLE")
    require(manifest["prospective_use"] == "core_confirmation_only_after_discovery_selection_freeze", "CONFIRMATION_USE")
    require(manifest["planned_endpoint"] == "D_CORE=Q_restored(k*)-Q_control(c*)", "CONFIRMATION_ENDPOINT")
    require(manifest["planned_primary_test"] == {"test": "one_sample_student_t", "alternative": "greater", "alpha": 0.05, "n": 300, "primary_p_value_count": 1}, "CONFIRMATION_PRIMARY_TEST")
    require(manifest["success_rule"] == "mean(D_CORE)>0 and one-sided primary p<0.05", "CONFIRMATION_SUCCESS_RULE")
    require(manifest["selection_allowed"] is False, "CONFIRMATION_SELECTION")
    require(manifest["discovery_raw_response_access_allowed"] is False, "DISCOVERY_RAW_RESPONSE_BOUNDARY")
    require(manifest["readout_response_access_allowed"] is False, "READOUT_RESPONSE_BOUNDARY")
    require(manifest["rescue_policy"] == "none", "CONFIRMATION_RESCUE")
    return facts, materialized


def load_frozen_geometry() -> dict[str, Any]:
    frozen = discovery.load_frozen_geometry()

    vectors: list[torch.Tensor] = []
    for plane in PLANE_ORDER:
        vectors.extend([
            frozen["planes"][plane]["plus"],
            frozen["planes"][plane]["minus"],
        ])

    matrix = torch.stack(vectors, dim=1)
    gram = matrix.T @ matrix
    residual = float(
        torch.max(
            torch.abs(
                gram
                - torch.eye(
                    10,
                    dtype=torch.float64,
                )
            )
        ).item()
    )
    require(
        residual <= PLANE_TOL,
        f"FULL_PLANE_GRAM:{residual}",
    )
    frozen["full_plane_gram_max_abs_residual"] = residual
    return frozen


def runtime_gate_single_visible_gpu(physical_device: int) -> torch.device:
    return discovery.runtime_gate_single_visible_gpu(physical_device)


def validate_runtime_geometry(
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> None:
    discovery.validate_runtime_geometry(runtime_ctx, frozen)


def build_input_state(
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    facts, rows = load_confirmation_population()
    tokenizer, _provenance = geom.load_tokenizer(snapshot)
    encoded = adapter.encode_gen4_rows(rows, tokenizer)

    require(
        tuple(encoded["input_ids"].shape) == (ROWS, 128),
        "ENCODED_INPUT_SHAPE",
    )
    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }
    require(len(facts_by_id) == PAIR_COUNT, "FACT_LOOKUP_COUNT")

    events: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        require(pair in facts_by_id, f"MISSING_FACT:{pair}")

        analyzed = tokenizer_gate.analyze_required_anchors_for_row(
            row,
            facts_by_id[pair],
            tokenizer,
        )
        for event in analyzed:
            key = (
                pair,
                cell,
                str(event["anchor_name"]),
            )
            require(key not in events, f"ANCHOR_DUPLICATE:{key}")
            require(bool(event["post4_eligible"]), f"ANCHOR_INELIGIBLE:{key}")
            events[key] = dict(event)

    required_cells = (
        TARGET_PLUS_CELL,
        TARGET_MINUS_CELL,
        REFERENCE_PLUS_CELL,
        REFERENCE_MINUS_CELL,
    )
    for pair in PAIR_IDS:
        for cell in required_cells:
            key = (pair, cell, ANCHOR_NAME)
            require(key in events, f"MISSING_REQUIRED_ANCHOR:{key}")

    return rows, encoded, events


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
    require(len(out) == ROWS, "ROW_INDEX_COUNT")
    return out


def input_row(
    encoded: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> torch.Tensor:
    key = (pair, cell)
    require(key in lookup, f"MISSING_INPUT:{key}")
    index = lookup[key]
    return (
        encoded["input_ids"][index:index + 1]
        .detach().cpu().contiguous()
    )


def anchors_for_pair(
    pair: str,
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, int]:
    cells = {
        "tp": TARGET_PLUS_CELL,
        "tm": TARGET_MINUS_CELL,
        "rp": REFERENCE_PLUS_CELL,
        "rm": REFERENCE_MINUS_CELL,
    }
    return {
        role: int(
            events[(pair, cell, ANCHOR_NAME)][
                "absolute_anchor_token_index"
            ]
        )
        for role, cell in cells.items()
    }


def plane_component(
    h: torch.Tensor,
    plane: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    require(plane in PLANE_ORDER, f"PLANE:{plane}")
    value = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(value.shape) == (DIM,), "H_SHAPE")
    require(bool(torch.isfinite(value).all().item()), "H_NONFINITE")

    plus = planes[plane]["plus"]
    minus = planes[plane]["minus"]
    a = float(torch.dot(value, plus).item())
    b = float(torch.dot(value, minus).item())
    component = (a * plus + b * minus).contiguous()

    return {
        "a": a,
        "b": b,
        "component": component,
        "component_l2": float(torch.linalg.vector_norm(component).item()),
    }


def condition_correction(
    h: torch.Tensor,
    *,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    require(condition in CONDITION_ORDER, f"CONDITION:{condition}")
    value = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(value.shape) == (DIM,), "CONDITION_H_SHAPE")
    selected = plane_component(value, SELECTED_PLANE, planes)
    selected_component = selected["component"]
    selected_post_max = None
    matched_norm_mismatch = None
    restoration_round_trip = None
    if condition == "dominant_restored":
        correction = (-selected_component + selected_component).contiguous()
        restoration_round_trip = float(torch.max(torch.abs(correction)).item())
        require(restoration_round_trip <= TOL, f"CORE_RESTORATION_ROUND_TRIP:{restoration_round_trip}")
    else:
        control_plus = planes[CONTROL_PLANE]["plus"]
        control_minus = planes[CONTROL_PLANE]["minus"]
        control_component = (float(selected["a"]) * control_plus + float(selected["b"]) * control_minus).contiguous()
        matched_norm_mismatch = abs(float(torch.linalg.vector_norm(selected_component).item()) - float(torch.linalg.vector_norm(control_component).item()))
        require(matched_norm_mismatch <= TOL, f"CORE_MATCHED_NORM:{matched_norm_mismatch}")
        correction = (-selected_component + control_component).contiguous()
    post = (value + correction).contiguous()
    selected_plus = planes[SELECTED_PLANE]["plus"]
    selected_minus = planes[SELECTED_PLANE]["minus"]
    native_selected = [float(torch.dot(value, selected_plus).item()), float(torch.dot(value, selected_minus).item())]
    post_selected = [float(torch.dot(post, selected_plus).item()), float(torch.dot(post, selected_minus).item())]
    if condition == "dominant_control":
        selected_post_max = max(abs(post_selected[0]), abs(post_selected[1]))
        require(selected_post_max <= TOL, f"CORE_SELECTED_NEUTRALIZATION:{selected_post_max}")
    require(bool(torch.isfinite(correction).all().item()), "CORRECTION_NONFINITE")
    return {
        "condition": condition,
        "correction": correction,
        "correction_l2": float(torch.linalg.vector_norm(correction).item()),
        "selected_native_coefficients": native_selected,
        "selected_post_coefficients": post_selected,
        "selected_post_max_abs_projection": selected_post_max,
        "residual_post_max_abs_projection": None,
        "selected_coefficient_drift_max_abs": None,
        "dominant_matched_norm_abs_mismatch": matched_norm_mismatch,
        "dominant_restoration_round_trip_max_abs": restoration_round_trip,
    }


def install_probe_hook(
    mixer: Any,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    direction: torch.Tensor,
    orientation: int,
    branch_sign: int,
    audit: dict[str, Any],
):
    require(orientation in (-1, 1), "ORIENTATION")
    require(branch_sign in (-1, 1), "BRANCH_SIGN")

    vector = direction.detach().cpu().to(torch.float64).contiguous()
    require(tuple(vector.shape) == (DIM,), "DIRECTION_SHAPE")
    require(
        abs(float(torch.linalg.vector_norm(vector).item()) - 1.0)
        <= PLANE_TOL,
        "DIRECTION_NORM",
    )

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * geom.INTERMEDIATE_SIZE,
            "INPROJ_SHAPE",
        )
        require(0 <= token_index < output.shape[1], "TOKEN_INDEX")

        before = output.detach().clone()
        mask = strong_mask.detach().cpu().bool().contiguous()
        require(
            mask.numel() == geom.INTERMEDIATE_SIZE
            and int(mask.sum().item()) == DIM,
            "STRONG_MASK",
        )
        mask_device = mask.to(before.device)

        h = (
            before[
                0,
                token_index,
                :geom.INTERMEDIATE_SIZE,
            ][mask_device]
            .detach().cpu().to(torch.float64).contiguous()
        )

        info = condition_correction(
            h,
            condition=condition,
            planes=planes,
        )

        probe = vector * (
            float(branch_sign)
            * float(orientation)
            * EPS
        )
        total = (info["correction"] + probe).contiguous()

        out = output.clone()
        intended = total.to(
            device=out.device,
            dtype=out.dtype,
        )
        out[
            0,
            token_index,
            :geom.INTERMEDIATE_SIZE,
        ][mask_device] += intended

        require(
            torch.equal(
                out[:, :, geom.INTERMEDIATE_SIZE:],
                before[:, :, geom.INTERMEDIATE_SIZE:],
            ),
            "GATE_CHANGED",
        )
        require(
            torch.equal(
                out[:, :, :geom.INTERMEDIATE_SIZE][:, :, ~mask_device],
                before[:, :, :geom.INTERMEDIATE_SIZE][:, :, ~mask_device],
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
            out[
                0,
                token_index,
                :geom.INTERMEDIATE_SIZE,
            ][mask_device]
            - before[
                0,
                token_index,
                :geom.INTERMEDIATE_SIZE,
            ][mask_device]
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
            residual <= transport_runtime.RUNTIME_CAST_TOL,
            f"APPLIED_RESIDUAL:{residual}",
        )

        audit.clear()
        audit.update({
            "condition": condition,
            "orientation": int(orientation),
            "branch_sign": int(branch_sign),
            "token_index": int(token_index),
            "condition_correction_l2": float(info["correction_l2"]),
            "probe_correction_l2": float(
                torch.linalg.vector_norm(probe).item()
            ),
            "selected_post_max_abs_projection":
                info["selected_post_max_abs_projection"],
            "residual_post_max_abs_projection":
                info["residual_post_max_abs_projection"],
            "selected_coefficient_drift_max_abs":
                info["selected_coefficient_drift_max_abs"],
            "dominant_matched_norm_abs_mismatch":
                info["dominant_matched_norm_abs_mismatch"],
            "dominant_restoration_round_trip_max_abs":
                info["dominant_restoration_round_trip_max_abs"],
            "applied_correction_max_abs_residual": residual,
        })
        return out

    return mixer.in_proj.register_forward_hook(hook)


class ForwardBudget:
    def __init__(self, expected: int) -> None:
        self.expected = int(expected)
        self.count = 0

    def consume(self) -> None:
        require(self.count < self.expected, "FORWARD_BUDGET_OVERFLOW")
        self.count += 1

    def assert_exact(self) -> None:
        require(
            self.count == self.expected,
            f"FORWARD_BUDGET:{self.count}:{self.expected}",
        )


def flatten_state(state: torch.Tensor) -> np.ndarray:
    out = (
        state.detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .numpy()
        .reshape(-1)
        .copy()
    )
    require(out.size == STATE_WIDTH, f"STATE_WIDTH:{out.size}")
    require(bool(np.isfinite(out).all()), "STATE_NONFINITE")
    return out


def capture_branch(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    kernels: Mapping[str, Any],
    input_ids: torch.Tensor,
    anchor: int,
    device: torch.device,
    budget: ForwardBudget,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    direction: torch.Tensor,
    orientation: int,
    branch_sign: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import transformers.models.mamba.modeling_mamba as mm

    mixer = runtime_ctx["intervention_mixer"]
    target_abs = int(anchor) + geom.TARGET_OFFSET
    require(tuple(input_ids.shape) == (1, 128), "INPUT_SHAPE")

    intervention_audit: dict[str, Any] = {}
    handle = install_probe_hook(
        mixer,
        token_index=target_abs,
        strong_mask=runtime_ctx["strong_mask"],
        condition=condition,
        planes=planes,
        direction=direction,
        orientation=orientation,
        branch_sign=branch_sign,
        audit=intervention_audit,
    )

    kernel_scan = kernels["selective_scan_fn"]
    kernel_update = kernels["selective_state_update"]

    active = {"value": False}
    captured: list[tuple[torch.Tensor, ...]] = []
    fast_calls = {"count": 0}

    def scan_wrapper(*args, **kwargs):
        if active["value"]:
            require(len(captured) == 0, "SCAN_CAPTURE_DUPLICATE")
            require(len(args) >= 8, "SCAN_ARG_COUNT")
            captured.append(
                tuple(
                    value.detach().clone()
                    for value in args[:8]
                )
            )
        return kernel_scan(*args, **kwargs)

    original_scan = mm.selective_scan_fn
    original_cuda = mixer.cuda_kernels_forward

    def cuda_wrapper(
        _self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        require(not active["value"], "CUDA_FAST_REENTRY")
        fast_calls["count"] += 1
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
    mixer.cuda_kernels_forward = types.MethodType(
        cuda_wrapper,
        mixer,
    )

    budget.consume()
    try:
        model.mamba.eval()
        with torch.inference_mode():
            _ = model.mamba(
                input_ids=input_ids.detach().to(device).contiguous()
            )
        torch.cuda.synchronize(device)
    finally:
        handle.remove()
        mm.selective_scan_fn = original_scan
        if "cuda_kernels_forward" in mixer.__dict__:
            del mixer.__dict__["cuda_kernels_forward"]

    require(bool(intervention_audit), "INTERVENTION_AUDIT_MISSING")
    require(fast_calls["count"] == 1, f"CUDA_FAST_PATH_CALLS:{fast_calls['count']}")
    require(len(captured) == 1, f"SCAN_CAPTURE_COUNT:{len(captured)}")

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
    require(prefix_end + 4 <= u.shape[-1], "POST4_RANGE")

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

    window = [flatten_state(state)]

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
        window.append(flatten_state(state))

    filler = window[0]
    states = [
        filler.copy()
        for _ in range(int(input_ids.shape[1]))
    ]
    for offset, vector in enumerate(window):
        states[int(anchor) + offset] = vector

    path_efficiency = measurement.post4_path_efficiency(
        states,
        int(anchor),
    )
    require(math.isfinite(path_efficiency), "PATH_EFFICIENCY_NONFINITE")

    return {
        "anchor": int(anchor),
        "target_abs": target_abs,
        "path_efficiency": float(path_efficiency),
        "fast_path_calls": 1,
    }, intervention_audit


def run_signed(
    *,
    pair: str,
    condition: str,
    direction: torch.Tensor,
    orientation: int,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    kernels: Mapping[str, Any],
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    encoded: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    device: torch.device,
    budget: ForwardBudget,
) -> dict[str, Any]:
    anchors = anchors_for_pair(pair, events)
    cells = {
        "tp": TARGET_PLUS_CELL,
        "tm": TARGET_MINUS_CELL,
    }

    efficiencies: dict[str, float] = {}
    audits: list[dict[str, Any]] = []

    for role, branch_sign in (("tp", 1), ("tm", -1)):
        branch, audit = capture_branch(
            model=model,
            runtime_ctx=runtime_ctx,
            kernels=kernels,
            input_ids=input_row(
                encoded,
                lookup,
                pair,
                cells[role],
            ),
            anchor=anchors[role],
            device=device,
            budget=budget,
            condition=condition,
            planes=planes,
            direction=direction,
            orientation=orientation,
            branch_sign=branch_sign,
        )
        efficiencies[role] = float(branch["path_efficiency"])
        audits.append(audit)

    value = efficiencies["tp"] - efficiencies["tm"]
    require(math.isfinite(value), "F_NONFINITE")

    optional_keys = (
        "selected_post_max_abs_projection",
        "residual_post_max_abs_projection",
        "selected_coefficient_drift_max_abs",
        "dominant_matched_norm_abs_mismatch",
        "dominant_restoration_round_trip_max_abs",
    )
    maxima: dict[str, float] = {}
    for key in optional_keys:
        maxima[key] = max(
            (
                abs(float(a[key]))
                for a in audits
                if a[key] is not None
            ),
            default=0.0,
        )

    return {
        "F": value,
        "plus_path_efficiency": efficiencies["tp"],
        "minus_path_efficiency": efficiencies["tm"],
        "max_applied_correction_residual": max(
            float(a["applied_correction_max_abs_residual"])
            for a in audits
        ),
        **maxima,
        "model_forward_count": FORWARDS_PER_SIGNED,
    }


def run_condition(
    *,
    pair: str,
    condition: str,
    bases: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    audit_maxima: dict[str, float] = {}

    for family in ("xg2", "xg4"):
        for index in range(K):
            direction = bases[family][:, index]
            positive = run_signed(
                pair=pair,
                condition=condition,
                direction=direction,
                orientation=1,
                **kwargs,
            )
            negative = run_signed(
                pair=pair,
                condition=condition,
                direction=direction,
                orientation=-1,
                **kwargs,
            )

            f_plus = float(positive["F"])
            f_minus = float(negative["F"])
            j_value = (f_plus - f_minus) / (2.0 * EPS)
            require(math.isfinite(j_value), "J_NONFINITE")

            probes.append({
                "direction_key": f"{family}_{index}",
                "basis_family": family,
                "basis_index": index,
                "F_plus": f_plus,
                "F_minus": f_minus,
                "J": j_value,
                "J_squared": j_value * j_value,
            })

            for source in (positive, negative):
                for key, value in source.items():
                    if key.startswith("max_"):
                        audit_maxima[key] = max(
                            audit_maxima.get(key, 0.0),
                            abs(float(value)),
                        )

    require(
        [probe["direction_key"] for probe in probes]
        == list(DIRECTION_ORDER),
        "DIRECTION_ORDER",
    )

    e_xg2 = math.fsum(
        float(probe["J_squared"])
        for probe in probes[:K]
    ) / K
    e_xg4 = math.fsum(
        float(probe["J_squared"])
        for probe in probes[K:]
    ) / K
    q_value = e_xg2 - e_xg4

    require(
        all(math.isfinite(v) for v in (e_xg2, e_xg4, q_value)),
        "Q_NONFINITE",
    )

    return {
        "condition": condition,
        "direction_order": list(DIRECTION_ORDER),
        "direction_probes": probes,
        "E_XG2": e_xg2,
        "E_XG4": e_xg4,
        "Q": q_value,
        "audit_maxima": dict(sorted(audit_maxima.items())),
        "model_forward_count": FORWARDS_PER_CONDITION,
    }


def run_pair(
    *, pair_index: int, pair: str, bases: Mapping[str, torch.Tensor], **kwargs: Any,
) -> dict[str, Any]:
    require(pair == PAIR_IDS[pair_index], f"PAIR:{pair_index}:{pair}")
    conditions = {name: run_condition(pair=pair, condition=name, bases=bases, **kwargs) for name in CONDITION_ORDER}
    q_restored = float(conditions["dominant_restored"]["Q"])
    q_control = float(conditions["dominant_control"]["Q"])
    d_core = q_restored - q_control
    require(all(math.isfinite(v) for v in (q_restored, q_control, d_core)), "PAIR_ENDPOINT_NONFINITE")
    return {
        "schema_version": ITEM_SCHEMA,
        "source_pair_id": pair,
        "pair_index": pair_index,
        "epsilon": EPS,
        "selected_dominant_candidate": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "condition_order": list(CONDITION_ORDER),
        "conditions": conditions,
        "Q_restored": q_restored,
        "Q_control": q_control,
        "D_CORE": d_core,
        "scientific_model_forward_count_this_run": FORWARDS_PER_PAIR,
        "confirmatory_inference_performed_in_item": False,
        "selection_reopened": False,
        "readout_response_accessed": False,
        "rescue_performed": False,
    }


def descriptive_stats(
    values: Sequence[float],
) -> dict[str, float]:
    from scipy import stats

    require(len(values) == PAIR_COUNT, "DESCRIPTIVE_N")
    array = np.asarray(values, dtype=np.float64)
    require(bool(np.isfinite(array).all()), "DESCRIPTIVE_NONFINITE")

    n = int(array.size)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    require(sd > 0.0 and math.isfinite(sd), "DESCRIPTIVE_SD")

    se = sd / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, df=n - 1))
    require(math.isfinite(critical), "CI_CRITICAL_NONFINITE")

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_t_low": mean - critical * se,
        "ci95_t_high": mean + critical * se,
        "cohen_dz": mean / sd,
        "fraction_positive": float(np.mean(array > 0.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def one_sample_greater(
    values: Sequence[float],
) -> dict[str, float]:
    from scipy import stats

    descriptive = descriptive_stats(values)
    n = int(descriptive["n"])
    mean = float(descriptive["mean"])
    sd = float(descriptive["sd"])

    t_stat = mean / (sd / math.sqrt(n))
    require(math.isfinite(t_stat), "T_STAT_NONFINITE")

    p_raw = float(stats.t.sf(t_stat, df=n - 1))
    require(
        math.isfinite(p_raw)
        and 0.0 <= p_raw <= 1.0,
        f"P_RAW:{p_raw}",
    )

    return {
        **descriptive,
        "df": n - 1,
        "null_mean": 0.0,
        "alternative": "greater",
        "t_statistic": t_stat,
        "p_raw": p_raw,
    }


def infer_confirmation(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(items) == PAIR_COUNT, "INFERENCE_ITEM_COUNT")
    d_core = [float(item["D_CORE"]) for item in items]
    core = one_sample_greater(d_core)
    supported = bool(float(core["mean"]) > 0.0 and float(core["p_raw"]) < ALPHA)
    return {
        "schema_version": INFERENCE_SCHEMA,
        "result": RESULT_PASS,
        "primary_test": {
            "endpoint": "D_CORE=Q_restored(P2)-Q_control(P5)",
            "null": "E[D_CORE] <= 0",
            "alternative": "E[D_CORE] > 0",
            "test": "one_sample_student_t",
            "tail": "greater",
            "alpha": ALPHA,
            "p_value_count": 1,
            **core,
        },
        "core_supported": supported,
        "scientific_conclusion": SUCCESS_LABEL if supported else FAILURE_LABEL,
        "selected_dominant_candidate": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "selection_reopened": False,
        "readout_response_accessed": False,
        "rescue_performed": False,
        "additional_p_values_executed": False,
        "layer_sweep_executed": False,
        "epsilon_sweep_executed": False,
        "token_sweep_executed": False,
    }


def _worker_paths(temp_dir: Path, shard_id: int) -> dict[str, Path]:
    return {
        "items": temp_dir / f"shard_{shard_id}_items.jsonl",
        "meta": temp_dir / f"shard_{shard_id}_meta.json",
        "error": temp_dir / f"shard_{shard_id}.error.txt",
    }


def worker_run(
    *,
    shard: Mapping[str, Any],
    expected_head: str,
    model_snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
) -> None:
    shard_id = int(shard["shard_id"])
    physical_device = int(shard["physical_device"])
    paths = _worker_paths(Path(temp_dir), shard_id)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)

        validate_protocol()
        authenticate_repo(expected_head)
        device = runtime_gate_single_visible_gpu(physical_device)

        snapshot = Path(model_snapshot)
        checkpoint = Path(compact_checkpoint)

        load_frozen_selection()
        frozen = load_frozen_geometry()
        rows, encoded, events = build_input_state(snapshot)
        lookup = row_index(rows)

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        kernel_compat.validate_transformers_kernel_bindings(kernels)

        runtime_ctx = geom.runtime_components(model)
        validate_runtime_geometry(runtime_ctx, frozen)

        budget = ForwardBudget(int(shard["forward_budget"]))
        items: list[dict[str, Any]] = []

        for global_index in range(
            int(shard["start_index"]),
            int(shard["end_index"]),
        ):
            pair = PAIR_IDS[global_index]
            items.append(
                run_pair(
                    pair_index=global_index,
                    pair=pair,
                    bases=frozen["bases"],
                    model=model,
                    runtime_ctx=runtime_ctx,
                    kernels=kernels,
                    planes=frozen["planes"],
                    encoded=encoded,
                    lookup=lookup,
                    events=events,
                    device=device,
                    budget=budget,
                )
            )

        budget.assert_exact()
        torch.cuda.synchronize(device)

        require(len(items) == int(shard["pair_count"]), "WORKER_ITEM_COUNT")
        require(
            items[0]["source_pair_id"] == shard["pair_first"]
            and items[-1]["source_pair_id"] == shard["pair_last"],
            "WORKER_PAIR_RANGE",
        )

        paths["items"].write_bytes(jsonl_bytes(items))
        meta = {
            "schema_version": "gen4-mamba790m-core-confirmation-worker-v1",
            "shard_id": shard_id,
            "physical_device": physical_device,
            "logical_device": 0,
            "cuda_visible_devices": str(physical_device),
            "device_name": torch.cuda.get_device_name(0),
            "pair_first": shard["pair_first"],
            "pair_last": shard["pair_last"],
            "pair_count": len(items),
            "scientific_model_forward_count":
                int(shard["forward_budget"]),
            "items_sha256": sha256_file(paths["items"]),
            "model_provenance": model_provenance,
            "geometry_summary_sha256":
                discovery.GEOMETRY_SUMMARY_SHA256,
            "discovery_selection_sha256":
                DISCOVERY_SELECTION_SHA256,
            "selected_dominant_candidate": SELECTED_PLANE,
            "response_blind_control_plane": CONTROL_PLANE,
        }
        paths["meta"].write_bytes(pretty_json_bytes(meta))

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def merge_worker_items(
    temp_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []

    for shard in SHARDS:
        shard_id = int(shard["shard_id"])
        paths = _worker_paths(temp_dir, shard_id)

        require(paths["meta"].is_file(), f"WORKER_META_MISSING:{shard_id}")
        require(paths["items"].is_file(), f"WORKER_ITEMS_MISSING:{shard_id}")

        meta = json.loads(
            paths["meta"].read_text(encoding="utf-8")
        )
        require(meta["shard_id"] == shard_id, "MERGE_SHARD_ID")
        require(
            meta["physical_device"] == shard["physical_device"],
            "MERGE_PHYSICAL_DEVICE",
        )
        require(meta["logical_device"] == 0, "MERGE_LOGICAL_DEVICE")
        require(
            meta["pair_count"] == shard["pair_count"],
            "MERGE_PAIR_COUNT",
        )
        require(
            meta["scientific_model_forward_count"]
            == shard["forward_budget"],
            "MERGE_FORWARD_COUNT",
        )
        require(
            meta["items_sha256"] == sha256_file(paths["items"]),
            "MERGE_ITEMS_SHA",
        )
        require(
            meta["selected_dominant_candidate"] == SELECTED_PLANE,
            "MERGE_SELECTED",
        )
        require(
            meta["response_blind_control_plane"] == CONTROL_PLANE,
            "MERGE_CONTROL",
        )

        shard_items = _read_jsonl(paths["items"])
        require(
            len(shard_items) == shard["pair_count"],
            "MERGE_SHARD_ITEM_COUNT",
        )
        items.extend(shard_items)
        metas.append(meta)

    require(len(items) == PAIR_COUNT, "MERGED_ITEM_COUNT")
    require(
        [str(item["source_pair_id"]) for item in items]
        == list(PAIR_IDS),
        "MERGED_PAIR_ORDER",
    )
    require(
        sum(
            int(item["scientific_model_forward_count_this_run"])
            for item in items
        )
        == TOTAL_FORWARD_BUDGET,
        "MERGED_FORWARD_SUM",
    )

    return items, metas


def write_outputs(
    *, output_dir: Path, expected_head: str, items: Sequence[Mapping[str, Any]], worker_meta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    import scipy
    selection = load_frozen_selection()
    frozen = load_frozen_geometry()
    inference = infer_confirmation(items)
    inference.update({
        "execution_head": expected_head,
        "geometry_freeze_commit": GEOMETRY_FREEZE_COMMIT,
        "confirmation_holdout_freeze_commit": CONFIRMATION_HOLDOUT_FREEZE_COMMIT,
        "discovery_freeze_commit": DISCOVERY_FREEZE_COMMIT,
        "discovery_selection_sha256": DISCOVERY_SELECTION_SHA256,
        "confirmation_population": {
            "pair_first": PAIR_IDS[0], "pair_last": PAIR_IDS[-1], "source_pair_count": PAIR_COUNT,
            "source_sha256": CONFIRMATION_SOURCE_SHA256, "rows_sha256": CONFIRMATION_ROWS_SHA256,
            "structural_manifest_sha256": CONFIRMATION_MANIFEST_SHA256,
        },
        "geometry_full_plane_gram_max_abs_residual": frozen["full_plane_gram_max_abs_residual"],
        "scipy_version": scipy.__version__,
        "discovery_selection_execution_head": selection["execution_head"],
    })
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / ITEM_FILE).write_bytes(jsonl_bytes(items))
    (output_dir / INFERENCE_FILE).write_bytes(pretty_json_bytes(inference))
    file_hashes = {ITEM_FILE: sha256_file(output_dir / ITEM_FILE), INFERENCE_FILE: sha256_file(output_dir / INFERENCE_FILE)}
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "geometry_freeze_commit": GEOMETRY_FREEZE_COMMIT,
        "confirmation_holdout_freeze_commit": CONFIRMATION_HOLDOUT_FREEZE_COMMIT,
        "discovery_freeze_commit": DISCOVERY_FREEZE_COMMIT,
        "discovery_selection_sha256": DISCOVERY_SELECTION_SHA256,
        "selected_dominant_candidate": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "confirmation_pair_first": PAIR_IDS[0],
        "confirmation_pair_last": PAIR_IDS[-1],
        "confirmation_pair_count": PAIR_COUNT,
        "scientific_model_forward_count": TOTAL_FORWARD_BUDGET,
        "worker_count": GPU_COUNT,
        "workers": list(worker_meta),
        "primary_p_value_count": 1,
        "test": "one_sample_student_t",
        "alternative": "greater",
        "alpha": ALPHA,
        "core_supported": bool(inference["core_supported"]),
        "scientific_conclusion": inference["scientific_conclusion"],
        "selection_reopened": False,
        "readout_response_accessed": False,
        "rescue_performed": False,
        "additional_p_values_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "output_file_sha256": dict(sorted(file_hashes.items())),
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))
    file_hashes[MANIFEST_FILE] = sha256_file(output_dir / MANIFEST_FILE)
    (output_dir / CHECKSUM_FILE).write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(file_hashes.items())), encoding="utf-8", newline="\n")
    return inference


def run_confirmation(
    *,
    expected_head: str,
    model_snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)
    load_frozen_selection()
    load_confirmation_population()
    geom.validate_snapshot(model_snapshot)

    require(
        compact_checkpoint.resolve()
        == (ROOT / geom.COMPACT_CHECKPOINT_REL).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        sha256_file(compact_checkpoint)
        == geom.COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "PHYSICAL_GPU_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba790m_core_confirmation_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for shard in SHARDS:
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "shard": shard,
                    "expected_head": expected_head,
                    "model_snapshot": str(model_snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                },
                name=(
                    "mamba790m-core-confirmation-"
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
                raise ConfirmationError(
                    "WORKER_FAILED:"
                    f"{shard['shard_id']}:\n{detail}"
                )

        items, worker_meta = merge_worker_items(temp_dir)
        return write_outputs(
            output_dir=output_dir,
            expected_head=expected_head,
            items=items,
            worker_meta=worker_meta,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Mamba-790M prospective core confirmation on XG1 7201..7500. Uses frozen discovery P2/P5 selection and executes exactly one one-sided one-sample Student t-test on D_CORE."))
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--compact-checkpoint", type=Path, default=ROOT / geom.COMPACT_CHECKPOINT_REL)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    inference = run_confirmation(expected_head=args.expected_head, model_snapshot=args.model_snapshot, compact_checkpoint=args.compact_checkpoint, output_dir=args.output_dir)
    core = inference["primary_test"]
    print("RESULT=" + str(inference["result"]))
    print("SELECTED_DOMINANT_CANDIDATE=" + SELECTED_PLANE)
    print("RESPONSE_BLIND_CONTROL_PLANE=" + CONTROL_PLANE)
    print("D_CORE_MEAN=" + format(float(core["mean"]), ".17g"))
    print("D_CORE_SD=" + format(float(core["sd"]), ".17g"))
    print("D_CORE_T=" + format(float(core["t_statistic"]), ".17g"))
    print("D_CORE_P=" + format(float(core["p_raw"]), ".17g"))
    print("D_CORE_COHEN_DZ=" + format(float(core["cohen_dz"]), ".17g"))
    print("D_CORE_FRACTION_POSITIVE=" + format(float(core["fraction_positive"]), ".17g"))
    print("PRIMARY_P_VALUE_COUNT=1")
    print("PRIMARY_ALTERNATIVE=greater")
    print("ALPHA=0.05")
    print("CORE_SUPPORTED=" + str(bool(inference["core_supported"])))
    print("SCIENTIFIC_CONCLUSION=" + str(inference["scientific_conclusion"]))
    print("SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=" + str(TOTAL_FORWARD_BUDGET))
    print("CONFIRMATION_RANGE=xg1_fact_7201..xg1_fact_7500")
    print("SELECTION_REOPENED=False")
    print("READOUT_RESPONSE_ACCESSED=False")
    print("RESCUE_PERFORMED=False")
    print("ADDITIONAL_P_VALUES_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


if __name__ == "__main__":
    main()
