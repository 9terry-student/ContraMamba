#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import tempfile
import traceback
import types
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import (
    build_reason_router_gen4_mamba14b_xg1_holdouts
    as holdout_builder,
)
from scripts import (
    reason_router_gen4_generator_family_prevalence_kernel_compat
    as kernel_compat,
)
from scripts import (
    reason_router_gen4_k_directional_alignment_transport_runtime
    as transport_runtime,
)
from scripts import (
    reason_router_gen4_mamba14b_discovery_fast_cuda
    as discovery,
)
from scripts import (
    reason_router_gen4_mamba14b_geometry_prepare_fast_cuda
    as geom,
)
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement
from scripts import (
    reason_router_gen4_six_cell_tier2_inference_adapter
    as adapter,
)
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

HOLDOUT_FREEZE_COMMIT = "e988cf3e20990a9397b8238df0a1edaf03674522"
GEOMETRY_FREEZE_COMMIT = "f97b597fb4da08a8d360ed07e48c6727e15ae0be"
DISCOVERY_FREEZE_COMMIT = "e7804bcab88a50dd57c2edd86af88daeb90a56d5"
PRIOR_CORE_CONFIRMATION_FREEZE_COMMIT = (
    "693c12fef2194cbc48d17ed8fc563cc5065dcae0"
)

DISCOVERY_SELECTION_PATH = Path(
    "reports/reason_router_gen4_mamba14b_discovery_runs/"
    "g4k-mamba14b-discovery-xg1-3901-4200-2gpu-73d8bf8/"
    "discovery_selection.json"
)
DISCOVERY_SELECTION_SHA256 = (
    "deeb9a2514503a317fd184c77c963d6fc296dc0cf75ce4d180fd46d319595c09"
)

DATA_ROOT = Path(
    "data/reason_router_gen4_mamba14b_xg1_residual_characterization_v1"
)
SOURCE_SHA256 = (
    "beeba91a246f5a13e2ba1843a87f49bd41efe9dc1c1fea3e9744dd42c892312d"
)
ROWS_SHA256 = (
    "8e4d330a9b75a4b3a5f16b53ab6f2f0b25ff70cc908eefadbfb1eb9133d8a002"
)
STRUCTURAL_MANIFEST_SHA256 = (
    "94fcd70d15dfc91b81651da7f2e7fa147a2f4adb230f5d2cad12c211293fc18d"
)

PAIR_FIRST = 4501
PAIR_LAST = 4800
PAIR_COUNT = 300
ROWS = 1800
PAIR_IDS = tuple(
    f"xg1_fact_{index:03d}"
    for index in range(PAIR_FIRST, PAIR_LAST + 1)
)

PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")
SELECTED_DOMINANT = "P5"
RESIDUAL_PLANES = tuple(
    plane for plane in PLANE_ORDER
    if plane != SELECTED_DOMINANT
)

CONDITION_ORDER = (
    "native",
    "p1_neutralized",
    "p2_neutralized",
    "p3_neutralized",
    "p4_neutralized",
    "residual_all_neutralized",
)

K = 5
EPS = 0.025
DIM = 829
STATE_WIDTH = geom.INTERMEDIATE_SIZE * geom.STATE_SIZE
DIRECTION_ORDER = tuple(
    [f"xg2_{index}" for index in range(K)]
    + [f"xg4_{index}" for index in range(K)]
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
        "pair_first": "xg1_fact_4501",
        "pair_last": "xg1_fact_4650",
        "pair_count": 150,
        "forward_budget": FORWARDS_PER_SHARD,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_4651",
        "pair_last": "xg1_fact_4800",
        "pair_count": 150,
        "forward_budget": FORWARDS_PER_SHARD,
    },
)

ITEM_FILE = "residual_characterization_items.jsonl"
SUMMARY_FILE = "residual_characterization_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

RESULT_PASS = "PASS_MAMBA14B_RESIDUAL_CHARACTERIZATION"
ITEM_SCHEMA = "gen4-mamba14b-residual-characterization-item-v1"
SUMMARY_SCHEMA = "gen4-mamba14b-residual-characterization-summary-v1"
MANIFEST_SCHEMA = "gen4-mamba14b-residual-characterization-manifest-v1"

TOL = 1.0e-12
PLANE_TOL = 1.0e-9


class ResidualDecompositionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ResidualDecompositionError(message)


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
        raise ResidualDecompositionError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(status == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (GEOMETRY_FREEZE_COMMIT, "GEOMETRY"),
        (HOLDOUT_FREEZE_COMMIT, "HOLDOUT"),
        (DISCOVERY_FREEZE_COMMIT, "DISCOVERY"),
        (PRIOR_CORE_CONFIRMATION_FREEZE_COMMIT, "PRIOR_CORE_CONFIRMATION"),
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
    require(DIM == 829, "DIM")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(SELECTED_DOMINANT == "P5", "SELECTED_DOMINANT")
    require(RESIDUAL_PLANES == ("P1", "P2", "P3", "P4"), "RESIDUAL_PLANES")
    require(
        CONDITION_ORDER
        == (
            "native",
            "p1_neutralized",
            "p2_neutralized",
            "p3_neutralized",
            "p4_neutralized",
            "residual_all_neutralized",
        ),
        "CONDITION_ORDER",
    )
    require(FORWARDS_PER_CONDITION == 40, "FORWARDS_PER_CONDITION")
    require(FORWARDS_PER_PAIR == 240, "FORWARDS_PER_PAIR")
    require(TOTAL_FORWARD_BUDGET == 72000, "TOTAL_FORWARD_BUDGET")
    require(FORWARDS_PER_SHARD == 36000, "FORWARDS_PER_SHARD")

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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}:{line_no}",
        )
        output.append(value)
    return output


def load_frozen_selection() -> dict[str, Any]:
    path = ROOT / DISCOVERY_SELECTION_PATH
    require(path.is_file(), "DISCOVERY_SELECTION_MISSING")
    require(
        sha256_file(path) == DISCOVERY_SELECTION_SHA256,
        "DISCOVERY_SELECTION_SHA",
    )
    selection = json.loads(path.read_text(encoding="utf-8-sig"))
    require(
        selection["result"] == "PASS_MAMBA14B_DISCOVERY_SELECTION_FREEZE",
        "DISCOVERY_SELECTION_RESULT",
    )
    require(
        selection["selected_dominant_candidate"] == SELECTED_DOMINANT,
        "DISCOVERY_SELECTED_DOMINANT",
    )
    require(selection["selection_unique"] is True, "DISCOVERY_SELECTION_UNIQUE")
    require(
        selection["positivity_gate_applied"] is False,
        "DISCOVERY_POSITIVITY_GATE",
    )
    require(
        selection["geometry_freeze_commit"] == GEOMETRY_FREEZE_COMMIT,
        "DISCOVERY_GEOMETRY_FREEZE",
    )
    return selection


def load_population() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = DATA_ROOT / holdout_builder.SOURCE_FILE
    rows = DATA_ROOT / holdout_builder.ROW_FILE
    manifest_path = DATA_ROOT / "structural_manifest.json"

    require(source.is_file(), "SOURCE_MISSING")
    require(rows.is_file(), "ROWS_MISSING")
    require(manifest_path.is_file(), "MANIFEST_MISSING")
    require(sha256_file(source) == SOURCE_SHA256, "SOURCE_SHA")
    require(sha256_file(rows) == ROWS_SHA256, "ROWS_SHA")
    require(
        sha256_file(manifest_path) == STRUCTURAL_MANIFEST_SHA256,
        "STRUCTURAL_MANIFEST_SHA",
    )

    facts = _read_jsonl(source)
    materialized = _read_jsonl(rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    require(len(facts) == PAIR_COUNT, "FACT_COUNT")
    require(len(materialized) == ROWS, "ROW_COUNT")
    require(
        [str(row["pair_id"]) for row in facts] == list(PAIR_IDS),
        "PAIR_ORDER",
    )
    holdout_builder.m370.prior.base.validate_materialized_rows(
        materialized,
        expected_pairs=PAIR_COUNT,
    )

    require(
        manifest["schema_version"]
        == "GEN4_MAMBA14B_XG1_PROSPECTIVE_HOLDOUT_V1",
        "HOLDOUT_SCHEMA",
    )
    require(
        manifest["result"]
        == "PASS_MAMBA14B_XG1_RESIDUAL_4501_4800_STRUCTURAL",
        "HOLDOUT_RESULT",
    )
    require(manifest["role"] == "residual_characterization", "HOLDOUT_ROLE")
    require(
        manifest["prospective_use"]
        == "descriptive_residual_characterization_only",
        "HOLDOUT_USE",
    )
    require(
        manifest["planned_symbolic_endpoints"] == {
            "S_k": "Q_native-Q_k-neutralized for k != k*",
            "S_RES": "Q_native-Q_all-residual-neutralized",
            "S_SUM": "sum_{k != k*} S_k",
            "I_RES": "S_RES-S_SUM",
            "C_k": "mean_branch[a_k^2+b_k^2]",
        },
        "HOLDOUT_ENDPOINTS",
    )
    require(manifest["formal_inference_allowed"] is False, "HOLDOUT_INFERENCE")
    require(manifest["p_value_count"] == 0, "HOLDOUT_P_VALUE_COUNT")
    require(manifest["selection_allowed"] is False, "HOLDOUT_SELECTION")
    require(
        manifest["core_confirmation_inference_access_allowed"] is False,
        "HOLDOUT_CORE_CONFIRMATION_ACCESS",
    )
    require(
        manifest["rescue_of_failed_core_test"] is False,
        "HOLDOUT_RESCUE",
    )
    require(
        manifest["dominant_plane_preselected"] is False
        and manifest["residual_plane_set_preselected"] is False
        and manifest["p3_preselected"] is False
        and manifest["p5_predesignated"] is False,
        "HOLDOUT_PRESELECTION_BOUNDARY",
    )
    require(
        manifest["layer_sweep_allowed"] is False
        and manifest["epsilon_sweep_allowed"] is False
        and manifest["token_sweep_allowed"] is False,
        "HOLDOUT_SWEEP_BOUNDARY",
    )
    return facts, materialized


def build_input_state(
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    facts, rows = load_population()
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
            require(
                bool(event["post4_eligible"]),
                f"ANCHOR_INELIGIBLE:{key}",
            )
            events[key] = dict(event)

    for pair in PAIR_IDS:
        for cell in (
            TARGET_PLUS_CELL,
            TARGET_MINUS_CELL,
            REFERENCE_PLUS_CELL,
            REFERENCE_MINUS_CELL,
        ):
            key = (pair, cell, ANCHOR_NAME)
            require(key in events, f"MISSING_REQUIRED_ANCHOR:{key}")

    return rows, encoded, events


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
        "component_l2": float(
            torch.linalg.vector_norm(component).item()
        ),
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

    residual_info = {
        plane: plane_component(value, plane, planes)
        for plane in RESIDUAL_PLANES
    }
    native_energy = {
        plane: (
            float(residual_info[plane]["a"]) ** 2
            + float(residual_info[plane]["b"]) ** 2
        )
        for plane in RESIDUAL_PLANES
    }

    if condition == "native":
        neutralized_planes: tuple[str, ...] = ()
    elif condition == "residual_all_neutralized":
        neutralized_planes = RESIDUAL_PLANES
    else:
        plane = condition[:2].upper()
        require(plane in RESIDUAL_PLANES, f"CONDITION_PLANE:{condition}")
        neutralized_planes = (plane,)

    component_sum = torch.zeros(DIM, dtype=torch.float64)
    for plane in neutralized_planes:
        component_sum += residual_info[plane]["component"]

    correction = (-component_sum).contiguous()
    post = (value + correction).contiguous()

    neutralized_max = max(
        (
            abs(float(torch.dot(post, planes[plane][sign]).item()))
            for plane in neutralized_planes
            for sign in ("plus", "minus")
        ),
        default=0.0,
    )
    require(
        neutralized_max <= TOL,
        f"NEUTRALIZATION_RESIDUAL:{condition}:{neutralized_max}",
    )

    selected_before = [
        float(torch.dot(value, planes[SELECTED_DOMINANT]["plus"]).item()),
        float(torch.dot(value, planes[SELECTED_DOMINANT]["minus"]).item()),
    ]
    selected_after = [
        float(torch.dot(post, planes[SELECTED_DOMINANT]["plus"]).item()),
        float(torch.dot(post, planes[SELECTED_DOMINANT]["minus"]).item()),
    ]
    selected_drift = max(
        abs(a - b)
        for a, b in zip(selected_before, selected_after, strict=True)
    )
    require(
        selected_drift <= TOL,
        f"SELECTED_DOMINANT_DRIFT:{condition}:{selected_drift}",
    )
    require(bool(torch.isfinite(correction).all().item()), "CORRECTION_NONFINITE")

    return {
        "condition": condition,
        "neutralized_planes": list(neutralized_planes),
        "native_residual_coefficient_energy": native_energy,
        "correction": correction,
        "correction_l2": float(torch.linalg.vector_norm(correction).item()),
        "neutralized_post_max_abs_projection": neutralized_max,
        "selected_dominant_coefficient_drift_max_abs": selected_drift,
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
            "neutralized_planes": info["neutralized_planes"],
            "orientation": int(orientation),
            "branch_sign": int(branch_sign),
            "token_index": int(token_index),
            "condition_correction_l2": float(info["correction_l2"]),
            "neutralized_post_max_abs_projection":
                float(info["neutralized_post_max_abs_projection"]),
            "selected_dominant_coefficient_drift_max_abs":
                float(info["selected_dominant_coefficient_drift_max_abs"]),
            "native_residual_coefficient_energy":
                dict(info["native_residual_coefficient_energy"]),
            "probe_correction_l2": float(
                torch.linalg.vector_norm(probe).item()
            ),
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
    require(
        fast_calls["count"] == 1,
        f"CUDA_FAST_PATH_CALLS:{fast_calls['count']}",
    )
    require(
        len(captured) == 1,
        f"SCAN_CAPTURE_COUNT:{len(captured)}",
    )

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


def mean_branch_coefficient_energy(
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    require(len(audits) == 2, "COEFFICIENT_BRANCH_COUNT")
    output: dict[str, float] = {}
    for plane in RESIDUAL_PLANES:
        values = [
            float(audit["native_residual_coefficient_energy"][plane])
            for audit in audits
        ]
        require(
            all(math.isfinite(value) and value >= 0.0 for value in values),
            f"COEFFICIENT_ENERGY:{plane}",
        )
        output[plane] = math.fsum(values) / 2.0
    return output


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
    anchors = discovery.anchors_for_pair(pair, events)
    cells = {"tp": TARGET_PLUS_CELL, "tm": TARGET_MINUS_CELL}
    efficiencies: dict[str, float] = {}
    audits: list[dict[str, Any]] = []

    for role, branch_sign in (("tp", 1), ("tm", -1)):
        branch, audit = capture_branch(
            model=model,
            runtime_ctx=runtime_ctx,
            kernels=kernels,
            input_ids=discovery.input_row(
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
    return {
        "F": value,
        "plus_path_efficiency": efficiencies["tp"],
        "minus_path_efficiency": efficiencies["tm"],
        "coefficient_energy_by_plane":
            mean_branch_coefficient_energy(audits),
        "max_applied_correction_residual": max(
            float(a["applied_correction_max_abs_residual"])
            for a in audits
        ),
        "max_neutralized_post_projection": max(
            float(a["neutralized_post_max_abs_projection"])
            for a in audits
        ),
        "max_selected_dominant_coefficient_drift": max(
            float(a["selected_dominant_coefficient_drift_max_abs"])
            for a in audits
        ),
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
    maxima = {
        "max_applied_correction_residual": 0.0,
        "max_neutralized_post_projection": 0.0,
        "max_selected_dominant_coefficient_drift": 0.0,
    }
    coefficient_reference: dict[str, float] | None = None

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

            for source in (positive, negative):
                observed = {
                    plane: float(source["coefficient_energy_by_plane"][plane])
                    for plane in RESIDUAL_PLANES
                }
                if coefficient_reference is None:
                    coefficient_reference = observed
                else:
                    for plane in RESIDUAL_PLANES:
                        ref = float(coefficient_reference[plane])
                        value = float(observed[plane])
                        tol = PLANE_TOL * max(1.0, abs(ref), abs(value))
                        require(
                            abs(value - ref) <= tol,
                            f"COEFFICIENT_REPEATABILITY:{plane}:{value}:{ref}",
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
                for key in maxima:
                    maxima[key] = max(maxima[key], abs(float(source[key])))

    require(coefficient_reference is not None, "COEFFICIENT_REFERENCE")
    require(
        [probe["direction_key"] for probe in probes]
        == list(DIRECTION_ORDER),
        "DIRECTION_ORDER",
    )
    e_xg2 = math.fsum(float(p["J_squared"]) for p in probes[:K]) / K
    e_xg4 = math.fsum(float(p["J_squared"]) for p in probes[K:]) / K
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
        "coefficient_energy_by_plane": dict(coefficient_reference),
        "audit_maxima": maxima,
        "scientific_model_forward_count": FORWARDS_PER_CONDITION,
    }


def run_pair(
    *,
    pair_index: int,
    pair: str,
    bases: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, Any]:
    require(pair == PAIR_IDS[pair_index], f"PAIR:{pair_index}:{pair}")
    conditions = {
        name: run_condition(
            pair=pair,
            condition=name,
            bases=bases,
            **kwargs,
        )
        for name in CONDITION_ORDER
    }
    q_native = float(conditions["native"]["Q"])
    effects = {
        plane: (
            q_native
            - float(conditions[f"{plane.lower()}_neutralized"]["Q"])
        )
        for plane in RESIDUAL_PLANES
    }
    energies = {
        plane: float(
            conditions["native"]["coefficient_energy_by_plane"][plane]
        )
        for plane in RESIDUAL_PLANES
    }
    q_all = float(conditions["residual_all_neutralized"]["Q"])
    s_res = q_native - q_all
    s_sum = math.fsum(float(effects[plane]) for plane in RESIDUAL_PLANES)
    i_res = s_res - s_sum

    require(
        all(
            math.isfinite(v)
            for v in (
                q_native,
                q_all,
                s_res,
                s_sum,
                i_res,
                *effects.values(),
                *energies.values(),
            )
        ),
        "PAIR_ENDPOINT_NONFINITE",
    )
    require(all(v >= 0.0 for v in energies.values()), "PAIR_C_NEGATIVE")

    item = {
        "schema_version": ITEM_SCHEMA,
        "source_pair_id": pair,
        "pair_index": pair_index,
        "epsilon": EPS,
        "selected_dominant_candidate_frozen": SELECTED_DOMINANT,
        "residual_plane_order": list(RESIDUAL_PLANES),
        "condition_order": list(CONDITION_ORDER),
        "conditions": conditions,
        "Q_native": q_native,
        "Q_residual_all_neutralized": q_all,
        "S_RES": float(s_res),
        "S_SUM": float(s_sum),
        "I_RES": float(i_res),
        "scientific_model_forward_count_this_run": FORWARDS_PER_PAIR,
        "formal_inference_performed": False,
        "p_value_count": 0,
        "selection_performed": False,
        "core_confirmation_inference_accessed": False,
        "rescue_performed": False,
    }
    for plane in RESIDUAL_PLANES:
        item[f"S_{plane}"] = float(effects[plane])
        item[f"C_{plane}"] = float(energies[plane])
    return item


def descriptive(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (PAIR_COUNT,), f"DESCRIPTIVE_SHAPE:{x.shape}")
    require(bool(np.isfinite(x).all()), "DESCRIPTIVE_NONFINITE")

    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "fraction_positive": float(np.mean(x > 0.0)),
        "fraction_negative": float(np.mean(x < 0.0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    require(
        a.shape == b.shape == (PAIR_COUNT,),
        "CORR_SHAPE",
    )
    require(
        bool(np.isfinite(a).all())
        and bool(np.isfinite(b).all()),
        "CORR_NONFINITE",
    )
    require(
        float(np.std(a, ddof=1)) > 0.0
        and float(np.std(b, ddof=1)) > 0.0,
        "CORR_ZERO_SD",
    )
    value = float(np.corrcoef(a, b)[0, 1])
    require(math.isfinite(value), "CORR_RESULT_NONFINITE")
    return value


def summarize_items(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(items) == PAIR_COUNT, "SUMMARY_ITEM_COUNT")
    keys = [
        *(f"S_{plane}" for plane in RESIDUAL_PLANES),
        "S_RES",
        "S_SUM",
        "I_RES",
        *(f"C_{plane}" for plane in RESIDUAL_PLANES),
        "Q_native",
        "Q_residual_all_neutralized",
    ]
    arrays = {
        key: np.asarray([float(item[key]) for item in items], dtype=np.float64)
        for key in keys
    }
    for key, value in arrays.items():
        require(
            value.shape == (PAIR_COUNT,)
            and bool(np.isfinite(value).all()),
            f"SUMMARY_ARRAY:{key}",
        )

    return {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "source_pair_count": PAIR_COUNT,
        "pair_id_first": PAIR_IDS[0],
        "pair_id_last": PAIR_IDS[-1],
        "selected_dominant_candidate_frozen": SELECTED_DOMINANT,
        "residual_planes": list(RESIDUAL_PLANES),
        "condition_order": list(CONDITION_ORDER),
        "endpoint_definitions": {
            "S_k": "Q_native-Q_k-neutralized for k != k*",
            "S_RES": "Q_native-Q_all-residual-neutralized",
            "S_SUM": "sum_{k != k*} S_k",
            "I_RES": "S_RES-S_SUM",
            "C_k": "mean_branch[a_k^2+b_k^2]",
        },
        "individual_effects": {
            plane: descriptive(arrays[f"S_{plane}"])
            for plane in RESIDUAL_PLANES
        },
        "aggregate_effect": descriptive(arrays["S_RES"]),
        "individual_sum": descriptive(arrays["S_SUM"]),
        "interaction_residual": descriptive(arrays["I_RES"]),
        "coefficient_energy": {
            plane: descriptive(arrays[f"C_{plane}"])
            for plane in RESIDUAL_PLANES
        },
        "q_native": descriptive(arrays["Q_native"]),
        "q_residual_all_neutralized": descriptive(
            arrays["Q_residual_all_neutralized"]
        ),
        "formal_inference_performed": False,
        "p_value_count": 0,
        "selection_performed": False,
        "core_confirmation_inference_accessed": False,
        "scientific_conclusion_established": False,
        "rescue_performed": False,
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

        device = discovery.runtime_gate_single_visible_gpu(
            physical_device
        )

        snapshot = Path(model_snapshot)
        checkpoint = Path(compact_checkpoint)

        load_frozen_selection()
        frozen = discovery.load_frozen_geometry()
        rows, encoded, events = build_input_state(snapshot)
        lookup = discovery.row_index(rows)

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        kernel_compat.validate_transformers_kernel_bindings(kernels)

        runtime_ctx = geom.runtime_components(model)
        discovery.validate_runtime_geometry(
            runtime_ctx,
            frozen,
        )

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

        require(
            len(items) == int(shard["pair_count"]),
            "WORKER_ITEM_COUNT",
        )
        require(
            items[0]["source_pair_id"] == shard["pair_first"]
            and items[-1]["source_pair_id"] == shard["pair_last"],
            "WORKER_PAIR_RANGE",
        )

        paths["items"].write_bytes(jsonl_bytes(items))
        meta = {
            "schema_version":
                "gen4-mamba14b-residual-characterization-worker-v1",
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
            "holdout_source_sha256": SOURCE_SHA256,
            "holdout_rows_sha256": ROWS_SHA256,
            "holdout_manifest_sha256":
                STRUCTURAL_MANIFEST_SHA256,
            "formal_inference_performed": False,
            "p_value_count": 0,
            "selection_performed": False,
            "core_confirmation_inference_accessed": False,
            "rescue_performed": False,
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

        require(
            paths["meta"].is_file(),
            f"WORKER_META_MISSING:{shard_id}",
        )
        require(
            paths["items"].is_file(),
            f"WORKER_ITEMS_MISSING:{shard_id}",
        )

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
            meta["formal_inference_performed"] is False
            and meta["p_value_count"] == 0
            and meta["selection_performed"] is False
            and meta["rescue_performed"] is False,
            "MERGE_SCIENCE_BOUNDARY",
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
    *,
    output_dir: Path,
    expected_head: str,
    items: Sequence[Mapping[str, Any]],
    worker_meta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    summary = summarize_items(items)
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / ITEM_FILE).write_bytes(jsonl_bytes(items))
    (output_dir / SUMMARY_FILE).write_bytes(pretty_json_bytes(summary))
    file_hashes = {
        ITEM_FILE: sha256_file(output_dir / ITEM_FILE),
        SUMMARY_FILE: sha256_file(output_dir / SUMMARY_FILE),
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "holdout_freeze_commit": HOLDOUT_FREEZE_COMMIT,
        "geometry_freeze_commit": GEOMETRY_FREEZE_COMMIT,
        "discovery_freeze_commit": DISCOVERY_FREEZE_COMMIT,
        "prior_core_confirmation_freeze_commit":
            PRIOR_CORE_CONFIRMATION_FREEZE_COMMIT,
        "discovery_selection_sha256": DISCOVERY_SELECTION_SHA256,
        "holdout_source_sha256": SOURCE_SHA256,
        "holdout_rows_sha256": ROWS_SHA256,
        "holdout_structural_manifest_sha256":
            STRUCTURAL_MANIFEST_SHA256,
        "selected_dominant_candidate_frozen": SELECTED_DOMINANT,
        "residual_planes": list(RESIDUAL_PLANES),
        "condition_order": list(CONDITION_ORDER),
        "pair_first": PAIR_IDS[0],
        "pair_last": PAIR_IDS[-1],
        "source_pair_count": PAIR_COUNT,
        "scientific_model_forward_count": TOTAL_FORWARD_BUDGET,
        "worker_count": GPU_COUNT,
        "workers": list(worker_meta),
        "formal_inference_performed": False,
        "p_value_count": 0,
        "selection_performed": False,
        "core_confirmation_inference_accessed": False,
        "scientific_conclusion_established": False,
        "rescue_performed": False,
        "training_executed": False,
        "backward_executed": False,
        "output_file_sha256": dict(sorted(file_hashes.items())),
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))
    file_hashes[MANIFEST_FILE] = sha256_file(output_dir / MANIFEST_FILE)
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(file_hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def run_decomposition(
    *,
    expected_head: str,
    model_snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)
    load_frozen_selection()
    load_population()
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
        prefix="gen4_mamba14b_residual_characterization_"
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
                    "mamba14b-residual-characterization-"
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
                raise ResidualDecompositionError(
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


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mamba-1.4B residual characterization on XG1 4501..4800. "
            "Descriptive only: S_k, S_RES, S_SUM, I_RES, C_k; "
            "zero p-values, no selection, no core-confirmation inference access."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=ROOT / geom.COMPACT_CHECKPOINT_REL,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    summary = run_decomposition(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT=" + str(summary["result"]))
    print("SELECTED_DOMINANT_CANDIDATE=P5")
    print("RESIDUAL_PLANES=P1,P2,P3,P4")
    for plane in RESIDUAL_PLANES:
        print(
            f"S_{plane}_MEAN="
            + format(float(summary["individual_effects"][plane]["mean"]), ".17g")
        )
        print(
            f"C_{plane}_MEAN="
            + format(float(summary["coefficient_energy"][plane]["mean"]), ".17g")
        )
    print(
        "S_RES_MEAN="
        + format(float(summary["aggregate_effect"]["mean"]), ".17g")
    )
    print(
        "S_SUM_MEAN="
        + format(float(summary["individual_sum"]["mean"]), ".17g")
    )
    print(
        "I_RES_MEAN="
        + format(float(summary["interaction_residual"]["mean"]), ".17g")
    )
    print(
        "I_RES_SD="
        + format(float(summary["interaction_residual"]["sd"]), ".17g")
    )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(TOTAL_FORWARD_BUDGET)
    )
    print("RESIDUAL_RANGE=xg1_fact_4501..xg1_fact_4800")
    print("FORMAL_INFERENCE_PERFORMED=False")
    print("P_VALUE_COUNT=0")
    print("SELECTION_PERFORMED=False")
    print("CORE_CONFIRMATION_INFERENCE_ACCESSED=False")
    print("SCIENTIFIC_CONCLUSION_ESTABLISHED=False")
    print("RESCUE_PERFORMED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


if __name__ == "__main__":
    main()
