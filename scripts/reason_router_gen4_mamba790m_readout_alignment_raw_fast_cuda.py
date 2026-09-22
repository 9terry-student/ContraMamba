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
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_mamba790m_confirmation_fast_cuda
    as confirmation,
)
from scripts import (
    reason_router_gen4_mamba790m_geometry_prepare_fast_cuda
    as geom,
)
from scripts import (
    reason_router_gen4_six_cell_tier2_inference_adapter
    as adapter,
)
from scripts import (
    reason_router_gen4_xg1_tokenizer_anchor_eligibility
    as tokenizer_gate,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"

READOUT_HOLDOUT_FREEZE_COMMIT = (
    "6c0989a45382db31c9052b89b088e999b3e7cf59"
)
CONFIRMATION_FREEZE_COMMIT = (
    "dfb0cdb5677f89fd85eaaef0300d2511da27f292"
)

CONFIRMATION_MANIFEST_PATH = Path(
    "reports/reason_router_gen4_mamba790m_confirmation_runs/"
    "g4k-mamba790m-confirmation-xg1-7201-7500-p2-p5-2gpu-d3117c1-fresh/"
    "artifact_manifest.json"
)
CONFIRMATION_MANIFEST_SHA256 = (
    "f00656a44c86f3b46b93b2c2f49d1cad7c3cb4844f18505266ee9e8559c826f5"
)

READOUT_ROOT = Path(
    "data/reason_router_gen4_mamba790m_xg1_readout_v1"
)
READOUT_SOURCE_SHA256 = (
    "8ab0e7ecab5ce7c4f197c566f02a59bd90ca64a8d6d9aabc9fcc4640e27bb4ee"
)
READOUT_ROWS_SHA256 = (
    "1c11d426b3602de84fe41ad6a389abb672d430d57e2acc222d493a11da14088c"
)
READOUT_MANIFEST_SHA256 = (
    "2795823deafd0d6338b076ffaf0c662a67bdc3b5958602d81f4babfccdac2fa1"
)

PAIR_FIRST = 7501
PAIR_LAST = 7800
PAIR_COUNT = 300
PAIR_IDS = tuple(
    f"xg1_fact_{index:03d}"
    for index in range(PAIR_FIRST, PAIR_LAST + 1)
)

ROW_COUNT = 1800
ROWS_PER_PAIR = 6

TARGET_CELLS = ("C0_SHAM", "C2_NAME")
LABEL_ID_BY_CELL = {
    "C0_SHAM": 2,
    "C2_NAME": 1,
}
ANCHOR_NAME = "A_IDENTITY"

SELECTED_PLANE = "P2"
CONTROL_PLANE = "P5"

DIM = 975
ARM = "G3-GROUP-D-HALF"
D_EDGE_OWNERSHIP_LAMBDA = 0.5
FORWARD_EQUIVALENT_SCALE = 2.0

GPU_COUNT = 2
PAIRS_PER_SHARD = 150
ROWS_PER_SHARD = PAIRS_PER_SHARD * len(TARGET_CELLS)

NATIVE_MODEL_FORWARD_COUNT = PAIR_COUNT * len(TARGET_CELLS)
LOCAL_BACKWARD_COUNT = PAIR_COUNT * len(TARGET_CELLS)

SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start_index": 0,
        "end_index": 150,
        "pair_first": "xg1_fact_7501",
        "pair_last": "xg1_fact_7650",
        "pair_count": 150,
        "native_forward_count": 300,
        "local_backward_count": 300,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_7651",
        "pair_last": "xg1_fact_7800",
        "pair_count": 150,
        "native_forward_count": 300,
        "local_backward_count": 300,
    },
)

ITEM_FILE = "readout_alignment_items.jsonl"
SUMMARY_FILE = "raw_readout_alignment_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-mamba790m-fresh-readout-item-v1"
SUMMARY_SCHEMA = "gen4-mamba790m-fresh-readout-raw-summary-v1"
MANIFEST_SCHEMA = "gen4-mamba790m-fresh-readout-raw-manifest-v1"

RESULT_PASS = "PASS_MAMBA790M_FRESH_DELTA_L_READOUT_RAW"


class ReadoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReadoutError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
        out.append(value)
    return out


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReadoutError(
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
    require(
        head == expected_head,
        f"HEAD_MISMATCH:{head}",
    )
    require(
        status == "",
        "WORKTREE_NOT_CLEAN",
    )

    for ancestor, label in (
        (
            READOUT_HOLDOUT_FREEZE_COMMIT,
            "READOUT_HOLDOUT",
        ),
        (
            confirmation.GEOMETRY_FREEZE_COMMIT,
            "GEOMETRY",
        ),
        (
            CONFIRMATION_FREEZE_COMMIT,
            "CONFIRMATION",
        ),
    ):
        rc = subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                ancestor,
                head,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            rc == 0,
            f"{label}_FREEZE_NOT_ANCESTOR",
        )


def validate_protocol() -> None:
    require(
        geom.HF_REPO == "state-spaces/mamba-790m-hf",
        "HF_REPO",
    )
    require(
        geom.HF_REVISION
        == "9822dd4b76af2bd9099b6ce2f19efd8329189a7e",
        "HF_REVISION",
    )
    require(
        geom.COMPACT_CHECKPOINT_SHA256
        == "af5582df61ed2dc2f7c77c0154c10e40b7aa6646a2141a6b3b30c52a45fc9072",
        "COMPACT_CHECKPOINT_SHA",
    )

    require(
        (
            geom.SOURCE_BLOCK,
            geom.TARGET_RESIDUAL_LAYER,
            geom.INTERVENTION_LAYER,
            geom.TARGET_OFFSET,
        )
        == (33, 34, 35, 2),
        "LAYER_MAPPING",
    )
    require(
        geom.INTERMEDIATE_SIZE == 3072,
        "INTERMEDIATE_SIZE",
    )

    require(
        PAIR_IDS
        == tuple(
            f"xg1_fact_{index:03d}"
            for index in range(7501, 7801)
        ),
        "PAIR_IDS",
    )
    require(
        PAIR_COUNT == 300,
        "PAIR_COUNT",
    )
    require(
        ROW_COUNT == 1800
        and ROWS_PER_PAIR == 6,
        "READOUT_ROW_COUNTS",
    )
    require(
        TARGET_CELLS == ("C0_SHAM", "C2_NAME"),
        "TARGET_CELLS",
    )

    require(
        SELECTED_PLANE == "P2"
        and CONTROL_PLANE == "P5",
        "PLANE_IDS",
    )
    require(
        confirmation.SELECTED_PLANE == SELECTED_PLANE,
        "CONFIRMATION_SELECTED_PLANE",
    )
    require(
        confirmation.CONTROL_PLANE == CONTROL_PLANE,
        "CONFIRMATION_CONTROL_PLANE",
    )

    require(
        DIM == confirmation.DIM == 975,
        "DIM",
    )
    require(
        ARM == geom.ARM == "G3-GROUP-D-HALF",
        "ARM",
    )

    edge_map = adapter.expected_edge_gradient_lambdas(ARM)
    for key in ("F_TO_D", "P_TO_D", "S_TO_D", "Q_TO_D"):
        require(
            edge_map[key] == D_EDGE_OWNERSHIP_LAMBDA,
            f"D_EDGE_OWNERSHIP:{key}",
        )

    require(
        D_EDGE_OWNERSHIP_LAMBDA == 0.5,
        "D_EDGE_LAMBDA",
    )
    require(
        FORWARD_EQUIVALENT_SCALE == 2.0,
        "FORWARD_EQUIVALENT_SCALE",
    )

    require(
        NATIVE_MODEL_FORWARD_COUNT == 600,
        "NATIVE_FORWARD_COUNT",
    )
    require(
        LOCAL_BACKWARD_COUNT == 600,
        "LOCAL_BACKWARD_COUNT",
    )

    covered: list[int] = []
    for expected_id, shard in enumerate(SHARDS):
        require(
            shard["shard_id"] == expected_id,
            "SHARD_ID",
        )
        require(
            shard["physical_device"] == expected_id,
            "SHARD_DEVICE",
        )
        require(
            shard["pair_count"] == PAIRS_PER_SHARD,
            "SHARD_PAIR_COUNT",
        )
        require(
            shard["native_forward_count"] == ROWS_PER_SHARD,
            "SHARD_FORWARD_COUNT",
        )
        require(
            shard["local_backward_count"] == ROWS_PER_SHARD,
            "SHARD_BACKWARD_COUNT",
        )
        covered.extend(
            range(
                int(shard["start_index"]),
                int(shard["end_index"]),
            )
        )

    require(
        covered == list(range(PAIR_COUNT)),
        "SHARD_COVERAGE",
    )


def load_frozen_confirmation_manifest() -> dict[str, Any]:
    path = ROOT / CONFIRMATION_MANIFEST_PATH

    require(
        path.is_file(),
        "CONFIRMATION_MANIFEST_MISSING",
    )
    require(
        sha256_file(path) == CONFIRMATION_MANIFEST_SHA256,
        "CONFIRMATION_MANIFEST_SHA",
    )

    manifest = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    require(
        manifest["result"]
        == "PASS_MAMBA790M_CORE_CONFIRMATION",
        "CONFIRMATION_RESULT",
    )
    require(
        manifest["core_supported"] is True,
        "CONFIRMATION_CORE_NOT_SUPPORTED",
    )
    require(
        manifest["selected_dominant_candidate"]
        == SELECTED_PLANE,
        "CONFIRMATION_SELECTED",
    )
    require(
        manifest["response_blind_control_plane"]
        == CONTROL_PLANE,
        "CONFIRMATION_CONTROL",
    )
    require(
        manifest["selection_reopened"] is False,
        "CONFIRMATION_SELECTION_REOPENED",
    )
    require(
        manifest["readout_response_accessed"] is False,
        "READOUT_ALREADY_ACCESSED",
    )
    require(
        manifest["rescue_performed"] is False,
        "CONFIRMATION_RESCUE",
    )
    require(
        manifest["additional_p_values_executed"] is False,
        "CONFIRMATION_EXTRA_PVALUES",
    )

    return manifest


def load_readout_population() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    builder = confirmation.holdout_builder

    source_path = ROOT / READOUT_ROOT / builder.SOURCE_FILE
    rows_path = ROOT / READOUT_ROOT / builder.ROW_FILE
    manifest_path = ROOT / READOUT_ROOT / "structural_manifest.json"

    require(
        source_path.is_file(),
        "READOUT_SOURCE_MISSING",
    )
    require(
        rows_path.is_file(),
        "READOUT_ROWS_MISSING",
    )
    require(
        manifest_path.is_file(),
        "READOUT_MANIFEST_MISSING",
    )

    require(
        sha256_file(source_path) == READOUT_SOURCE_SHA256,
        "READOUT_SOURCE_SHA",
    )
    require(
        sha256_file(rows_path) == READOUT_ROWS_SHA256,
        "READOUT_ROWS_SHA",
    )
    require(
        sha256_file(manifest_path) == READOUT_MANIFEST_SHA256,
        "READOUT_MANIFEST_SHA",
    )

    facts = read_jsonl(source_path)
    rows = read_jsonl(rows_path)
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )

    require(
        len(facts) == PAIR_COUNT,
        "READOUT_FACT_COUNT",
    )
    require(
        len(rows) == ROW_COUNT,
        "READOUT_ROW_COUNT",
    )
    require(
        [str(row["pair_id"]) for row in facts]
        == list(PAIR_IDS),
        "READOUT_PAIR_ORDER",
    )

    builder.prior.base.validate_materialized_rows(
        rows,
        expected_pairs=PAIR_COUNT,
    )

    require(
        manifest["schema_version"]
        == "GEN4_MAMBA1_FIVE_SCALE_LADDER_XG1_PROSPECTIVE_HOLDOUT_V1",
        "READOUT_SCHEMA",
    )
    require(
        manifest["result"]
        == "PASS_MAMBA1_FIVE_SCALE_LADDER_XG1_STRUCTURAL",
        "READOUT_RESULT",
    )
    require(
        manifest["role"] == "readout",
        "READOUT_ROLE",
    )
    require(
        manifest["scale"] == "mamba790m",
        "READOUT_SCALE",
    )
    require(
        manifest["pair_id_first"] == "xg1_fact_7501"
        and manifest["pair_id_last"] == "xg1_fact_7800",
        "READOUT_PAIR_RANGE",
    )
    require(
        manifest["source_pair_count"] == PAIR_COUNT,
        "READOUT_PAIR_COUNT_MANIFEST",
    )
    require(
        manifest["row_count"] == ROW_COUNT
        and manifest["rows_per_pair"] == ROWS_PER_PAIR,
        "READOUT_ROWS_MANIFEST",
    )
    require(
        manifest["source_file_sha256"]
        == READOUT_SOURCE_SHA256,
        "READOUT_SOURCE_MANIFEST_SHA",
    )
    require(
        manifest["row_file_sha256"]
        == READOUT_ROWS_SHA256,
        "READOUT_ROWS_MANIFEST_SHA",
    )

    require(
        manifest["planned_quantity"]
        == "Delta_L_owned_with_forward_equivalent_recorded_if_applicable",
        "READOUT_PLANNED_QUANTITY",
    )
    require(
        manifest["prospective_use"]
        == "fresh_readout_only_after_core_confirmation",
        "READOUT_PROSPECTIVE_USE",
    )

    require(
        manifest["selected_component_fixed_before_readout"] is True,
        "READOUT_SELECTED_NOT_FIXED",
    )
    require(
        manifest["response_blind_control_fixed_before_readout"] is True,
        "READOUT_CONTROL_NOT_FIXED",
    )
    require(
        manifest["selection_allowed"] is False,
        "READOUT_SELECTION_ALLOWED",
    )
    require(
        manifest["response_guided_selection_allowed"] is False,
        "READOUT_RESPONSE_GUIDED_SELECTION",
    )
    require(
        manifest["cohort_replacement_allowed"] is False,
        "READOUT_COHORT_REPLACEMENT",
    )
    require(
        manifest["row_filtering_allowed"] is False,
        "READOUT_ROW_FILTERING",
    )
    require(
        manifest["rescue_policy"] == "none",
        "READOUT_RESCUE_POLICY",
    )
    require(
        manifest["discovery_raw_response_access_allowed"] is False,
        "DISCOVERY_RAW_ACCESS_ALLOWED",
    )
    require(
        manifest["confirmation_raw_response_access_allowed"] is False,
        "CONFIRMATION_RAW_ACCESS_ALLOWED",
    )
    require(
        manifest["response_fields_present"] is False
        and manifest["endpoint_values_present"] is False,
        "READOUT_PREPOPULATED_RESPONSE",
    )

    return facts, rows, manifest


def build_input_state(
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    facts, rows, _manifest = load_readout_population()

    tokenizer, _provenance = geom.load_tokenizer(snapshot)
    encoded = adapter.encode_gen4_rows(
        rows,
        tokenizer,
    )

    require(
        tuple(encoded["input_ids"].shape)
        == (ROW_COUNT, 128),
        "ENCODED_INPUT_SHAPE",
    )
    require(
        list(encoded["source_pair_id"])
        == [
            str(row["source_pair_id"])
            for row in rows
        ],
        "ENCODED_PAIR_ORDER",
    )

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }
    require(
        len(facts_by_id) == PAIR_COUNT,
        "FACT_LOOKUP_COUNT",
    )

    events: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for row in rows:
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])

        require(
            pair in facts_by_id,
            f"MISSING_FACT:{pair}",
        )

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
            require(
                key not in events,
                f"ANCHOR_DUPLICATE:{key}",
            )
            require(
                bool(event["post4_eligible"]),
                f"ANCHOR_INELIGIBLE:{key}",
            )
            events[key] = dict(event)

    for pair in PAIR_IDS:
        for cell in TARGET_CELLS:
            key = (
                pair,
                cell,
                ANCHOR_NAME,
            )
            require(
                key in events,
                f"MISSING_READOUT_ANCHOR:{key}",
            )

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
        require(
            key not in out,
            f"DUPLICATE_ROW:{key}",
        )
        out[key] = index

    require(
        len(out) == ROW_COUNT,
        "ROW_INDEX_COUNT",
    )
    return out


def feature_batch(
    encoded: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    key = (pair, cell)
    require(
        key in lookup,
        f"MISSING_FEATURE_ROW:{key}",
    )
    index = int(lookup[key])

    output: dict[str, torch.Tensor] = {}
    for name in (
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
    ):
        value = (
            encoded[name][index:index + 1]
            .detach()
            .to(device)
            .contiguous()
        )
        output[name] = value

    require(
        tuple(output["input_ids"].shape) == (1, 128),
        "FEATURE_INPUT_SHAPE",
    )
    return output


def _active_margin(
    logits: torch.Tensor,
    label_id: int,
) -> tuple[torch.Tensor, int]:
    require(
        torch.is_tensor(logits)
        and tuple(logits.shape) == (1, 3),
        "LOGITS_SHAPE",
    )
    require(
        label_id in (0, 1, 2),
        "LABEL_ID",
    )

    wrong_ids = [
        index
        for index in range(3)
        if index != label_id
    ]
    wrong = logits[0, wrong_ids]

    require(
        float(wrong[0].detach().item())
        != float(wrong[1].detach().item()),
        "WRONG_CLASS_EXACT_TIE",
    )

    local = int(
        torch.argmax(wrong).item()
    )
    wrong_id = int(wrong_ids[local])

    margin = (
        logits[0, label_id]
        - logits[0, wrong_id]
    )

    require(
        bool(torch.isfinite(margin.detach()).item()),
        "MARGIN_NONFINITE",
    )
    return margin, wrong_id


def _cosine_from_dot(
    dot: float,
    grad_norm: float,
    component_norm: float,
) -> float | None:
    if grad_norm == 0.0 or component_norm == 0.0:
        return None

    value = dot / (
        grad_norm * component_norm
    )
    require(
        math.isfinite(value),
        "COSINE_NONFINITE",
    )
    require(
        -1.0 - 1e-9
        <= value
        <= 1.0 + 1e-9,
        "COSINE_RANGE",
    )

    return float(
        min(1.0, max(-1.0, value))
    )


def _install_local_leaf_hook(
    mixer: Any,
    *,
    token_index: int,
    intermediate_size: int,
    strong_mask: torch.Tensor,
    capture: dict[str, Any],
):
    mask_cpu = (
        strong_mask.detach()
        .cpu()
        .bool()
        .contiguous()
    )

    require(
        mask_cpu.numel() == intermediate_size,
        "MASK_WIDTH",
    )
    require(
        int(mask_cpu.sum().item()) == DIM,
        "MASK_STRONG_DIM",
    )

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1]
            == 2 * intermediate_size,
            "INPROJ_SHAPE",
        )
        require(
            0 <= token_index < output.shape[1],
            "TOKEN_INDEX",
        )

        # Upstream model parameters and input tensors do not require
        # gradients. At this exact native in-projection boundary we cut
        # all history and create the only differentiable leaf.
        before = output.detach()

        native_full = (
            before[
                0,
                token_index,
                :intermediate_size,
            ]
            .clone()
            .contiguous()
        )

        leaf = (
            native_full.clone()
            .requires_grad_(True)
        )

        gate = (
            before[
                0,
                token_index,
                intermediate_size:,
            ]
            .clone()
            .contiguous()
        )

        token = torch.cat(
            [leaf, gate],
            dim=0,
        ).view(1, 1, -1)

        out = torch.cat(
            [
                before[:, :token_index, :],
                token,
                before[:, token_index + 1:, :],
            ],
            dim=1,
        )

        require(
            tuple(out.shape)
            == tuple(output.shape),
            "HOOK_OUTPUT_SHAPE",
        )
        require(
            torch.equal(
                out.detach(),
                before,
            ),
            "HOOK_VALUE_DRIFT",
        )

        capture.clear()
        capture["leaf"] = leaf
        capture["native_full"] = (
            native_full.detach()
            .cpu()
            .contiguous()
        )
        capture["strong_mask"] = mask_cpu

        return out

    return mixer.in_proj.register_forward_hook(
        hook
    )


def _plane_component(
    native_strong: torch.Tensor,
    *,
    plane: str,
    planes: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
) -> dict[str, Any]:
    require(
        plane in confirmation.PLANE_ORDER,
        f"PLANE:{plane}",
    )

    value = (
        native_strong.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    require(
        tuple(value.shape) == (DIM,),
        "NATIVE_STRONG_SHAPE",
    )

    plus = (
        planes[plane]["plus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    minus = (
        planes[plane]["minus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    require(
        tuple(plus.shape) == (DIM,)
        and tuple(minus.shape) == (DIM,),
        "PLANE_SHAPE",
    )

    a = float(
        torch.dot(value, plus).item()
    )
    b = float(
        torch.dot(value, minus).item()
    )

    component = (
        a * plus
        + b * minus
    ).contiguous()

    require(
        bool(
            torch.isfinite(
                component
            ).all().item()
        ),
        "PLANE_COMPONENT_NONFINITE",
    )

    return {
        "a": a,
        "b": b,
        "component": component,
    }


def _plane_components(
    native_strong: torch.Tensor,
    *,
    planes: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
) -> dict[str, Any]:
    selected = _plane_component(
        native_strong,
        plane=SELECTED_PLANE,
        planes=planes,
    )

    a = float(selected["a"])
    b = float(selected["b"])

    control_plus = (
        planes[CONTROL_PLANE]["plus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    control_minus = (
        planes[CONTROL_PLANE]["minus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    control_component = (
        a * control_plus
        + b * control_minus
    ).contiguous()

    selected_component = (
        selected["component"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    selected_norm = float(
        torch.linalg.vector_norm(
            selected_component
        ).item()
    )
    control_norm = float(
        torch.linalg.vector_norm(
            control_component
        ).item()
    )

    mismatch = abs(
        selected_norm
        - control_norm
    )
    require(
        mismatch <= confirmation.TOL,
        f"MATCHED_COMPONENT_NORM:{mismatch}",
    )

    return {
        "a": a,
        "b": b,
        "selected_component": selected_component,
        "control_component": control_component,
    }


def _owned_to_forward_equivalent(
    value: float,
) -> float:
    require(
        math.isfinite(value),
        "OWNED_DELTA_NONFINITE",
    )
    output = (
        FORWARD_EQUIVALENT_SCALE
        * float(value)
    )
    require(
        math.isfinite(output),
        "FORWARD_DELTA_NONFINITE",
    )
    return output


def freeze_all_parameters_for_local_leaf(
    model: torch.nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)

    require(
        all(
            not parameter.requires_grad
            for parameter in model.parameters()
        ),
        "MODEL_PARAMETER_REQUIRES_GRAD",
    )
    require(
        all(
            parameter.grad is None
            for parameter in model.parameters()
        ),
        "MODEL_PARAMETER_GRAD_PREEXISTS",
    )


def run_native_gradient_row(
    *,
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    planes: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
    encoded: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], int],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    pair: str,
    cell: str,
    device: torch.device,
) -> dict[str, Any]:
    require(
        cell in TARGET_CELLS,
        f"TARGET_CELL:{cell}",
    )

    event_key = (
        pair,
        cell,
        ANCHOR_NAME,
    )
    require(
        event_key in events,
        f"ANCHOR_MISSING:{event_key}",
    )

    anchor_index = int(
        events[event_key][
            "absolute_anchor_token_index"
        ]
    )
    target_index = (
        anchor_index
        + geom.TARGET_OFFSET
    )

    strong_mask = (
        runtime_ctx["strong_mask"]
        .detach()
        .cpu()
        .bool()
        .contiguous()
    )
    require(
        strong_mask.numel()
        == geom.INTERMEDIATE_SIZE,
        "STRONG_MASK_WIDTH",
    )
    require(
        int(strong_mask.sum().item())
        == DIM,
        "STRONG_MASK_DIM",
    )

    capture: dict[str, Any] = {}

    handle = _install_local_leaf_hook(
        runtime_ctx["intervention_mixer"],
        token_index=target_index,
        intermediate_size=geom.INTERMEDIATE_SIZE,
        strong_mask=strong_mask,
        capture=capture,
    )

    try:
        output = adapter.historical_forward(
            model,
            feature_batch(
                encoded,
                lookup,
                pair,
                cell,
                device,
            ),
            arm=ARM,
        )
    finally:
        handle.remove()

    require(
        "leaf" in capture
        and "native_full" in capture,
        "HOOK_CAPTURE",
    )

    leaf = capture["leaf"]
    label_id = LABEL_ID_BY_CELL[cell]

    margin, active_wrong_id = _active_margin(
        output["logits"],
        label_id,
    )

    grad = torch.autograd.grad(
        margin,
        leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

    require(
        tuple(grad.shape)
        == (geom.INTERMEDIATE_SIZE,),
        "GRAD_SHAPE",
    )
    require(
        bool(torch.isfinite(grad).all().item()),
        "GRAD_NONFINITE",
    )

    require(
        all(
            parameter.grad is None
            for parameter in model.parameters()
        ),
        "PARAMETER_GRAD_CREATED",
    )

    native_full = capture["native_full"]
    native_strong = native_full[
        strong_mask
    ]

    grad_full = (
        grad.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    grad_strong = grad_full[
        strong_mask
    ]

    component = _plane_components(
        native_strong,
        planes=planes,
    )

    selected_component = component[
        "selected_component"
    ]
    control_component = component[
        "control_component"
    ]

    selected_plus = (
        planes[SELECTED_PLANE]["plus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    selected_minus = (
        planes[SELECTED_PLANE]["minus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    control_plus = (
        planes[CONTROL_PLANE]["plus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    control_minus = (
        planes[CONTROL_PLANE]["minus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    selected_coords = [
        float(
            torch.dot(
                grad_strong,
                selected_plus,
            ).item()
        ),
        float(
            torch.dot(
                grad_strong,
                selected_minus,
            ).item()
        ),
    ]
    control_coords = [
        float(
            torch.dot(
                grad_strong,
                control_plus,
            ).item()
        ),
        float(
            torch.dot(
                grad_strong,
                control_minus,
            ).item()
        ),
    ]

    grad_norm = float(
        torch.linalg.vector_norm(
            grad_full
        ).item()
    )
    selected_projection_norm = float(
        math.hypot(*selected_coords)
    )
    control_projection_norm = float(
        math.hypot(*control_coords)
    )
    selected_component_norm = float(
        torch.linalg.vector_norm(
            selected_component
        ).item()
    )
    control_component_norm = float(
        torch.linalg.vector_norm(
            control_component
        ).item()
    )

    l_selected_owned = float(
        torch.dot(
            grad_strong,
            selected_component,
        ).item()
    )
    l_control_owned = float(
        torch.dot(
            grad_strong,
            control_component,
        ).item()
    )
    delta_l_owned = (
        l_selected_owned
        - l_control_owned
    )
    delta_l_forward = (
        _owned_to_forward_equivalent(
            delta_l_owned
        )
    )

    numeric = (
        grad_norm,
        selected_projection_norm,
        control_projection_norm,
        selected_component_norm,
        control_component_norm,
        l_selected_owned,
        l_control_owned,
        delta_l_owned,
        delta_l_forward,
    )
    require(
        all(
            math.isfinite(value)
            for value in numeric
        ),
        "READOUT_NONFINITE",
    )

    require(
        delta_l_forward
        == 2.0 * delta_l_owned,
        "OWNED_FORWARD_RELATION",
    )

    return {
        "schema_version": ITEM_SCHEMA,
        "scale": "mamba790m",
        "source_pair_id": pair,
        "contrast_cell_id": cell,
        "correct_label_id": label_id,
        "active_wrong_class_id": active_wrong_id,
        "native_correct_class_margin":
            float(
                margin.detach()
                .cpu()
                .item()
            ),
        "model_repo": geom.HF_REPO,
        "model_revision": geom.HF_REVISION,
        "compact_checkpoint_sha256":
            geom.COMPACT_CHECKPOINT_SHA256,
        "gradient_ownership_arm": ARM,
        "d_edge_gradient_lambda":
            D_EDGE_OWNERSHIP_LAMBDA,
        "selected_plane": SELECTED_PLANE,
        "control_plane": CONTROL_PLANE,
        "source_block": geom.SOURCE_BLOCK,
        "target_residual_layer":
            geom.TARGET_RESIDUAL_LAYER,
        "intervention_layer":
            geom.INTERVENTION_LAYER,
        "target_offset": geom.TARGET_OFFSET,
        "anchor_name": ANCHOR_NAME,
        "absolute_anchor_token_index":
            anchor_index,
        "target_intervention_token_index":
            target_index,
        "native_selected_coordinates": [
            float(component["a"]),
            float(component["b"]),
        ],
        "gradient_full_l2": grad_norm,
        "selected_projection_l2":
            selected_projection_norm,
        "control_projection_l2":
            control_projection_norm,
        "selected_projection_fraction":
            selected_projection_norm
            / max(grad_norm, 1e-12),
        "control_projection_fraction":
            control_projection_norm
            / max(grad_norm, 1e-12),
        "selected_directional_coordinates":
            selected_coords,
        "control_directional_coordinates":
            control_coords,
        "selected_component_l2":
            selected_component_norm,
        "control_component_l2":
            control_component_norm,
        "L_selected_owned":
            l_selected_owned,
        "L_control_owned":
            l_control_owned,
        "Delta_L_owned":
            delta_l_owned,
        "Delta_L_forward_equivalent":
            delta_l_forward,
        "forward_equivalent_scale":
            FORWARD_EQUIVALENT_SCALE,
        "cosine_gradient_selected_component":
            _cosine_from_dot(
                l_selected_owned,
                grad_norm,
                selected_component_norm,
            ),
        "cosine_gradient_control_component":
            _cosine_from_dot(
                l_control_owned,
                grad_norm,
                control_component_norm,
            ),
        "native_model_forward_count": 1,
        "local_leaf_backward_count": 1,
        "parameter_gradient_count": 0,
        "intervention_condition_forward_count": 0,
        "inferential_test_performed": False,
        "p_value_count_executed": 0,
        "selection_reopened": False,
        "rescue_performed": False,
        "sweep_performed": False,
    }


def _worker_paths(
    temp_dir: Path,
    shard_id: int,
) -> dict[str, Path]:
    prefix = f"shard{shard_id}"
    return {
        "items":
            temp_dir / f"{prefix}_items.jsonl",
        "meta":
            temp_dir / f"{prefix}_meta.json",
        "error":
            temp_dir / f"{prefix}_error.txt",
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
    physical_device = int(
        shard["physical_device"]
    )
    paths = _worker_paths(
        Path(temp_dir),
        shard_id,
    )

    try:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = str(physical_device)

        validate_protocol()
        authenticate_repo(expected_head)

        device = (
            confirmation
            .runtime_gate_single_visible_gpu(
                physical_device
            )
        )

        snapshot = Path(model_snapshot)
        checkpoint = Path(
            compact_checkpoint
        )

        geom.validate_snapshot(snapshot)

        require(
            checkpoint.resolve()
            == (
                ROOT
                / geom.COMPACT_CHECKPOINT_REL
            ).resolve(),
            "COMPACT_CHECKPOINT_PATH",
        )
        require(
            sha256_file(checkpoint)
            == geom.COMPACT_CHECKPOINT_SHA256,
            "COMPACT_CHECKPOINT_SHA",
        )

        load_frozen_confirmation_manifest()
        confirmation.load_frozen_selection()
        frozen = (
            confirmation.load_frozen_geometry()
        )

        rows, encoded, events = (
            build_input_state(snapshot)
        )
        lookup = row_index(rows)

        model, kernels, model_provenance = (
            geom.reconstruct_model(
                snapshot=snapshot,
                compact_checkpoint=checkpoint,
                gpu_id=0,
            )
        )

        confirmation.kernel_compat.validate_transformers_kernel_bindings(
            kernels
        )

        runtime_ctx = geom.runtime_components(
            model
        )
        confirmation.validate_runtime_geometry(
            runtime_ctx,
            frozen,
        )

        freeze_all_parameters_for_local_leaf(
            model
        )
        model.eval()

        items: list[dict[str, Any]] = []

        for global_index in range(
            int(shard["start_index"]),
            int(shard["end_index"]),
        ):
            pair = PAIR_IDS[
                global_index
            ]

            for cell in TARGET_CELLS:
                items.append(
                    run_native_gradient_row(
                        model=model,
                        runtime_ctx=runtime_ctx,
                        planes=frozen[
                            "planes"
                        ],
                        encoded=encoded,
                        lookup=lookup,
                        events=events,
                        pair=pair,
                        cell=cell,
                        device=device,
                    )
                )

        torch.cuda.synchronize(device)

        require(
            len(items)
            == int(
                shard[
                    "native_forward_count"
                ]
            ),
            "WORKER_ITEM_COUNT",
        )

        require(
            sum(
                int(
                    item[
                        "native_model_forward_count"
                    ]
                )
                for item in items
            )
            == int(
                shard[
                    "native_forward_count"
                ]
            ),
            "WORKER_FORWARD_COUNT",
        )

        require(
            sum(
                int(
                    item[
                        "local_leaf_backward_count"
                    ]
                )
                for item in items
            )
            == int(
                shard[
                    "local_backward_count"
                ]
            ),
            "WORKER_BACKWARD_COUNT",
        )

        require(
            sum(
                int(
                    item[
                        "parameter_gradient_count"
                    ]
                )
                for item in items
            )
            == 0,
            "WORKER_PARAMETER_GRAD_COUNT",
        )

        require(
            items[0]["source_pair_id"]
            == shard["pair_first"]
            and items[-1]["source_pair_id"]
            == shard["pair_last"],
            "WORKER_PAIR_RANGE",
        )

        paths["items"].write_bytes(
            jsonl_bytes(items)
        )

        meta = {
            "schema_version":
                "gen4-mamba790m-fresh-readout-worker-v1",
            "shard_id": shard_id,
            "physical_device":
                physical_device,
            "logical_device": 0,
            "cuda_visible_devices":
                str(physical_device),
            "device_name":
                torch.cuda.get_device_name(0),
            "pair_first":
                shard["pair_first"],
            "pair_last":
                shard["pair_last"],
            "pair_count":
                shard["pair_count"],
            "item_count":
                len(items),
            "native_model_forward_count":
                shard[
                    "native_forward_count"
                ],
            "local_leaf_backward_count":
                shard[
                    "local_backward_count"
                ],
            "parameter_gradient_count": 0,
            "intervention_condition_forward_count": 0,
            "p_value_count_executed": 0,
            "selected_plane":
                SELECTED_PLANE,
            "control_plane":
                CONTROL_PLANE,
            "readout_source_sha256":
                READOUT_SOURCE_SHA256,
            "readout_rows_sha256":
                READOUT_ROWS_SHA256,
            "readout_manifest_sha256":
                READOUT_MANIFEST_SHA256,
            "confirmation_manifest_sha256":
                CONFIRMATION_MANIFEST_SHA256,
            "items_sha256":
                sha256_file(
                    paths["items"]
                ),
            "model_provenance":
                model_provenance,
            "selection_reopened": False,
            "rescue_performed": False,
            "sweep_performed": False,
        }

        paths["meta"].write_bytes(
            pretty_json_bytes(meta)
        )

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def merge_worker_items(
    temp_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    items: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []

    for shard in SHARDS:
        shard_id = int(
            shard["shard_id"]
        )
        paths = _worker_paths(
            temp_dir,
            shard_id,
        )

        require(
            paths["items"].is_file(),
            f"WORKER_ITEMS_MISSING:{shard_id}",
        )
        require(
            paths["meta"].is_file(),
            f"WORKER_META_MISSING:{shard_id}",
        )

        meta = json.loads(
            paths["meta"].read_text(
                encoding="utf-8"
            )
        )

        require(
            meta["shard_id"] == shard_id,
            "MERGE_SHARD_ID",
        )
        require(
            meta["physical_device"]
            == shard["physical_device"],
            "MERGE_PHYSICAL_DEVICE",
        )
        require(
            meta["item_count"]
            == shard[
                "native_forward_count"
            ],
            "MERGE_ITEM_COUNT",
        )
        require(
            meta["native_model_forward_count"]
            == shard[
                "native_forward_count"
            ],
            "MERGE_FORWARD_COUNT",
        )
        require(
            meta["local_leaf_backward_count"]
            == shard[
                "local_backward_count"
            ],
            "MERGE_BACKWARD_COUNT",
        )
        require(
            meta["parameter_gradient_count"]
            == 0,
            "MERGE_PARAMETER_GRAD",
        )
        require(
            meta["intervention_condition_forward_count"]
            == 0,
            "MERGE_INTERVENTION_FORWARD",
        )
        require(
            meta["p_value_count_executed"]
            == 0,
            "MERGE_PVALUE_COUNT",
        )
        require(
            meta["items_sha256"]
            == sha256_file(
                paths["items"]
            ),
            "MERGE_ITEMS_SHA",
        )

        shard_items = read_jsonl(
            paths["items"]
        )
        items.extend(shard_items)
        metas.append(meta)

    require(
        len(items)
        == NATIVE_MODEL_FORWARD_COUNT,
        "MERGED_ITEM_COUNT",
    )

    expected_order = [
        (pair, cell)
        for pair in PAIR_IDS
        for cell in TARGET_CELLS
    ]
    observed_order = [
        (
            str(item["source_pair_id"]),
            str(item["contrast_cell_id"]),
        )
        for item in items
    ]
    require(
        observed_order == expected_order,
        "MERGED_ITEM_ORDER",
    )

    require(
        sum(
            int(
                item[
                    "native_model_forward_count"
                ]
            )
            for item in items
        )
        == NATIVE_MODEL_FORWARD_COUNT,
        "MERGED_FORWARD_SUM",
    )
    require(
        sum(
            int(
                item[
                    "local_leaf_backward_count"
                ]
            )
            for item in items
        )
        == LOCAL_BACKWARD_COUNT,
        "MERGED_BACKWARD_SUM",
    )
    require(
        sum(
            int(
                item[
                    "parameter_gradient_count"
                ]
            )
            for item in items
        )
        == 0,
        "MERGED_PARAMETER_GRAD_SUM",
    )
    require(
        sum(
            int(
                item[
                    "intervention_condition_forward_count"
                ]
            )
            for item in items
        )
        == 0,
        "MERGED_INTERVENTION_FORWARD_SUM",
    )
    require(
        all(
            float(
                item[
                    "Delta_L_forward_equivalent"
                ]
            )
            == 2.0
            * float(
                item["Delta_L_owned"]
            )
            for item in items
        ),
        "MERGED_OWNERSHIP_RELATION",
    )

    return items, metas


def write_outputs(
    *,
    output_dir: Path,
    expected_head: str,
    items: Sequence[Mapping[str, Any]],
    worker_meta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(
        not output_dir.exists(),
        "OUTPUT_COLLISION",
    )
    require(
        len(items)
        == NATIVE_MODEL_FORWARD_COUNT,
        "OUTPUT_ITEM_COUNT",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (output_dir / ITEM_FILE).write_bytes(
        jsonl_bytes(items)
    )

    summary = {
        "schema_version":
            SUMMARY_SCHEMA,
        "result":
            RESULT_PASS,
        "execution_head":
            expected_head,
        "population":
            "xg1_fact_7501..xg1_fact_7800",
        "pair_count":
            PAIR_COUNT,
        "target_cells":
            list(TARGET_CELLS),
        "item_count":
            NATIVE_MODEL_FORWARD_COUNT,
        "scale":
            "mamba790m",
        "selected_plane":
            SELECTED_PLANE,
        "control_plane":
            CONTROL_PLANE,
        "planned_quantity":
            "Delta_L_owned_with_forward_equivalent_recorded_if_applicable",
        "gradient_ownership_arm":
            ARM,
        "d_edge_gradient_lambda":
            D_EDGE_OWNERSHIP_LAMBDA,
        "ownership_relation":
            "Delta_L_forward_equivalent=2*Delta_L_owned",
        "native_model_forward_count":
            NATIVE_MODEL_FORWARD_COUNT,
        "local_leaf_backward_count":
            LOCAL_BACKWARD_COUNT,
        "parameter_gradient_count":
            0,
        "intervention_condition_forward_count":
            0,
        "p_value_count_executed":
            0,
        "inferential_test_performed":
            False,
        "scientific_conclusion":
            None,
        "pair_level_sign_aggregation_performed":
            False,
        "selection_reopened":
            False,
        "rescue_performed":
            False,
        "sweep_performed":
            False,
        "discovery_raw_response_accessed":
            False,
        "confirmation_raw_response_accessed":
            False,
        "fresh_readout_measurement_executed":
            True,
        "readout_source_sha256":
            READOUT_SOURCE_SHA256,
        "readout_rows_sha256":
            READOUT_ROWS_SHA256,
        "readout_manifest_sha256":
            READOUT_MANIFEST_SHA256,
        "confirmation_manifest_sha256":
            CONFIRMATION_MANIFEST_SHA256,
        "model_repo":
            geom.HF_REPO,
        "model_revision":
            geom.HF_REVISION,
        "compact_checkpoint_sha256":
            geom.COMPACT_CHECKPOINT_SHA256,
    }

    (output_dir / SUMMARY_FILE).write_bytes(
        pretty_json_bytes(summary)
    )

    file_hashes = {
        ITEM_FILE:
            sha256_file(
                output_dir / ITEM_FILE
            ),
        SUMMARY_FILE:
            sha256_file(
                output_dir / SUMMARY_FILE
            ),
    }

    manifest = {
        "schema_version":
            MANIFEST_SCHEMA,
        "result":
            RESULT_PASS,
        "execution_head":
            expected_head,
        "readout_holdout_freeze_commit":
            READOUT_HOLDOUT_FREEZE_COMMIT,
        "geometry_freeze_commit":
            confirmation.GEOMETRY_FREEZE_COMMIT,
        "confirmation_freeze_commit":
            CONFIRMATION_FREEZE_COMMIT,
        "readout_source_sha256":
            READOUT_SOURCE_SHA256,
        "readout_rows_sha256":
            READOUT_ROWS_SHA256,
        "readout_manifest_sha256":
            READOUT_MANIFEST_SHA256,
        "confirmation_manifest_sha256":
            CONFIRMATION_MANIFEST_SHA256,
        "source_pair_count":
            PAIR_COUNT,
        "pair_first":
            PAIR_IDS[0],
        "pair_last":
            PAIR_IDS[-1],
        "target_cells":
            list(TARGET_CELLS),
        "item_count":
            NATIVE_MODEL_FORWARD_COUNT,
        "selected_plane":
            SELECTED_PLANE,
        "control_plane":
            CONTROL_PLANE,
        "native_model_forward_count":
            NATIVE_MODEL_FORWARD_COUNT,
        "local_leaf_backward_count":
            LOCAL_BACKWARD_COUNT,
        "parameter_gradient_count":
            0,
        "intervention_condition_forward_count":
            0,
        "p_value_count_executed":
            0,
        "inferential_test_performed":
            False,
        "scientific_conclusion":
            None,
        "selection_reopened":
            False,
        "rescue_performed":
            False,
        "sweep_performed":
            False,
        "discovery_raw_response_accessed":
            False,
        "confirmation_raw_response_accessed":
            False,
        "training_executed":
            False,
        "worker_count":
            GPU_COUNT,
        "workers":
            list(worker_meta),
        "output_file_sha256":
            dict(sorted(file_hashes.items())),
    }

    (output_dir / MANIFEST_FILE).write_bytes(
        pretty_json_bytes(manifest)
    )

    file_hashes[MANIFEST_FILE] = (
        sha256_file(
            output_dir / MANIFEST_FILE
        )
    )

    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest
            in sorted(file_hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )

    return summary


def run_raw(
    *,
    expected_head: str,
    model_snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)

    geom.validate_snapshot(
        model_snapshot
    )

    require(
        compact_checkpoint.resolve()
        == (
            ROOT
            / geom.COMPACT_CHECKPOINT_REL
        ).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        sha256_file(
            compact_checkpoint
        )
        == geom.COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )

    load_frozen_confirmation_manifest()
    confirmation.load_frozen_selection()
    load_readout_population()

    require(
        not output_dir.exists(),
        "OUTPUT_COLLISION",
    )

    require(
        torch.cuda.is_available(),
        "CUDA_UNAVAILABLE",
    )
    require(
        torch.cuda.device_count()
        >= GPU_COUNT,
        "PHYSICAL_GPU_COUNT",
    )

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba790m_readout_"
    ) as tmp:
        temp_dir = Path(tmp)

        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for shard in SHARDS:
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "shard": shard,
                    "expected_head":
                        expected_head,
                    "model_snapshot":
                        str(model_snapshot),
                    "compact_checkpoint":
                        str(compact_checkpoint),
                    "temp_dir":
                        str(temp_dir),
                },
                name=(
                    "mamba790m-readout-"
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
                    paths["error"].read_text(
                        encoding="utf-8"
                    )
                    if paths[
                        "error"
                    ].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )

                raise ReadoutError(
                    "WORKER_FAILED:"
                    f"{shard['shard_id']}:\n"
                    f"{detail}"
                )

        items, worker_meta = (
            merge_worker_items(
                temp_dir
            )
        )

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
            "Mamba-790M prospective fresh Delta_L readout on "
            "xg1_fact_7501..7800. Measures exactly 600 native "
            "forward/local-leaf-backward rows for frozen P2/P5. "
            "No intervention-condition forward and no statistical "
            "inference are executed."
        )
    )

    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=(
            ROOT
            / geom.COMPACT_CHECKPOINT_REL
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    summary = run_raw(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )

    print(
        "RESULT="
        + str(summary["result"])
    )
    print(
        "READOUT_RANGE="
        "xg1_fact_7501..xg1_fact_7800"
    )
    print(
        "SELECTED_PLANE=P2"
    )
    print(
        "CONTROL_PLANE=P5"
    )
    print(
        "NATIVE_MODEL_FORWARD_COUNT=600"
    )
    print(
        "LOCAL_LEAF_BACKWARD_COUNT=600"
    )
    print(
        "PARAMETER_GRADIENT_COUNT=0"
    )
    print(
        "INTERVENTION_CONDITION_FORWARD_COUNT=0"
    )
    print(
        "P_VALUE_COUNT_EXECUTED=0"
    )
    print(
        "PAIR_LEVEL_SIGN_AGGREGATION_PERFORMED=False"
    )
    print(
        "SELECTION_REOPENED=False"
    )
    print(
        "RESCUE_PERFORMED=False"
    )
    print(
        "SWEEP_PERFORMED=False"
    )
    print(
        "OWNERSHIP_RELATION="
        "Delta_L_forward_equivalent=2*Delta_L_owned"
    )


if __name__ == "__main__":
    main()