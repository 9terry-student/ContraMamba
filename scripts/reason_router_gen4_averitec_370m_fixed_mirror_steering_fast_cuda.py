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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as bridge,
)


EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "d4296763ac1110b9b8cd2f6c05c92b6bc56e6479"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_causal_atlas_guided_steering_design.md"
)
DESIGN_ARTIFACT_GIT_BLOB = "0d35f9aebb37e6203bea40304d3a55ab2361b684"

SOURCE_ELIGIBILITY_CORRECTION = Path(
    "reports/reason_router_gen4_causal_atlas_guided_steering_source_eligibility_correction.md"
)
SOURCE_ELIGIBILITY_CORRECTION_GIT_BLOB = (
    "3a55da349d7c9780f8e51de76e3fd9abd67531aa"
)

COHORT_ROOT = Path(
    "data/reason_router_gen4_averitec_train_fresh_steering_boundary_v1"
)
COHORT_FILE = COHORT_ROOT / "fresh_compatible_cohort.jsonl"
MANIFEST_FILE = COHORT_ROOT / "token_gate_manifest.json"
COHORT_SUMS_FILE = COHORT_ROOT / "SHA256SUMS.txt"

COHORT_GIT_BLOB = "f6c8edcfa37c10feb7dfa66ecd36648bd1cc2bf1"
MANIFEST_GIT_BLOB = "c31b20880eac0db2c3330a977fa7a5f66a788ddd"
COHORT_SUMS_GIT_BLOB = "6a480e21178de09a7e24415413626a6576479693"

COHORT_SHA256 = "ce271c1b57b33e12399bc180f06d417ceef6e4e92ad9e5383e2d8187829a2810"
MANIFEST_SHA256 = "7fef7f8c9544f46fbdeb1184bee313259bd52154181c346c4ae6d367d08da738"
COHORT_SUMS_SHA256 = "db799689d76f96b5bef74d5c509a77ea3777386cb23e9c5b8241b01c549a2f85"

BRIDGE_SCRIPT = Path(
    "scripts/reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda.py"
)
BRIDGE_SCRIPT_GIT_BLOB = "07a6ccd97e7f616de4400053fbd442408b158398"

CONFIRMATION_SCRIPT = Path(
    "scripts/reason_router_gen4_mamba370m_confirmation_fast_cuda.py"
)
CONFIRMATION_SCRIPT_GIT_BLOB = "ba9aaa6de002759f1f8b5f42c6cfb0cedbc67984"

GEOMETRY_SCRIPT = Path(
    "scripts/reason_router_gen4_mamba370m_geometry_prepare_fast_cuda.py"
)
GEOMETRY_SCRIPT_GIT_BLOB = "12cfc608ddca1f16892ebaed1e1f05f09b4c452c"

N = 2799
CONDITIONS = (
    "native",
    "p3_mirror_steer",
    "p5_matched_control",
)
SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"
INTERVENTION_LAYER = 35
ANCHOR_NAME = "A_CLAIM_EVIDENCE_BOUNDARY"
TARGET_OFFSET = 2
EXPECTED_FORWARD_BUDGET = N * len(CONDITIONS)

SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start": 0,
        "end": 1400,
        "example_count": 1400,
        "forward_budget": 4200,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start": 1400,
        "end": 2799,
        "example_count": 1399,
        "forward_budget": 4197,
    },
)
SHARD_COUNT = len(SHARDS)

ROW_FILE = "steering_raw_rows.jsonl"
SUMMARY_FILE = "execution_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ROW_SCHEMA = "gen4-averitec-370m-fixed-mirror-steering-raw-row-v1"
SUMMARY_SCHEMA = "gen4-averitec-370m-fixed-mirror-steering-raw-summary-v1"
SHARD_SCHEMA = "gen4-averitec-370m-fixed-mirror-steering-shard-v1"
RESULT_PASS = "PASS_AVERITEC_370M_FIXED_MIRROR_STEERING_RAW"

PRIMARY_INFERENCE_EXECUTED = False
P_VALUE_COUNT_ADDED = 0


class SteeringRawError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SteeringRawError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SteeringRawError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"JSONL_MISSING:{path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}:{line_no}",
        )
        rows.append(value)
    return rows


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR")

    frozen_blobs = {
        DESIGN_ARTIFACT.as_posix(): DESIGN_ARTIFACT_GIT_BLOB,
        SOURCE_ELIGIBILITY_CORRECTION.as_posix():
            SOURCE_ELIGIBILITY_CORRECTION_GIT_BLOB,
        COHORT_FILE.as_posix(): COHORT_GIT_BLOB,
        MANIFEST_FILE.as_posix(): MANIFEST_GIT_BLOB,
        COHORT_SUMS_FILE.as_posix(): COHORT_SUMS_GIT_BLOB,
        BRIDGE_SCRIPT.as_posix(): BRIDGE_SCRIPT_GIT_BLOB,
        CONFIRMATION_SCRIPT.as_posix(): CONFIRMATION_SCRIPT_GIT_BLOB,
        GEOMETRY_SCRIPT.as_posix(): GEOMETRY_SCRIPT_GIT_BLOB,
    }
    for path, expected_blob in frozen_blobs.items():
        require(
            git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(N == 2799, "N")
    require(
        CONDITIONS
        == (
            "native",
            "p3_mirror_steer",
            "p5_matched_control",
        ),
        "CONDITIONS",
    )
    require(SELECTED_PLANE == "P3", "SELECTED_PLANE")
    require(CONTROL_PLANE == "P5", "CONTROL_PLANE")
    require(INTERVENTION_LAYER == 35, "INTERVENTION_LAYER")
    require(ANCHOR_NAME == "A_CLAIM_EVIDENCE_BOUNDARY", "ANCHOR")
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(EXPECTED_FORWARD_BUDGET == 8397, "FORWARD_BUDGET")
    require(SHARD_COUNT == 2, "SHARD_COUNT")

    covered: list[int] = []
    forward_sum = 0
    for shard_id, shard in enumerate(SHARDS):
        require(int(shard["shard_id"]) == shard_id, "SHARD_ID")
        require(int(shard["physical_device"]) == shard_id, "SHARD_GPU")
        start = int(shard["start"])
        end = int(shard["end"])
        count = int(shard["example_count"])
        budget = int(shard["forward_budget"])
        require(end - start == count, f"SHARD_COUNT:{shard_id}")
        require(
            budget == count * len(CONDITIONS),
            f"SHARD_BUDGET:{shard_id}",
        )
        covered.extend(range(start, end))
        forward_sum += budget

    require(covered == list(range(N)), "SHARD_COVERAGE")
    require(forward_sum == EXPECTED_FORWARD_BUDGET, "SHARD_FORWARD_SUM")


def validate_cohort() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(
        sha256_file(ROOT / COHORT_FILE) == COHORT_SHA256,
        "COHORT_SHA256",
    )
    require(
        sha256_file(ROOT / MANIFEST_FILE) == MANIFEST_SHA256,
        "MANIFEST_SHA256",
    )
    require(
        sha256_file(ROOT / COHORT_SUMS_FILE) == COHORT_SUMS_SHA256,
        "COHORT_SUMS_SHA256",
    )

    sums: dict[str, str] = {}
    for line in (ROOT / COHORT_SUMS_FILE).read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        sums[name] = digest
    require(
        sums
        == {
            "fresh_compatible_cohort.jsonl": COHORT_SHA256,
            "token_gate_manifest.json": MANIFEST_SHA256,
        },
        "COHORT_SUMS",
    )

    manifest = json.loads(
        (ROOT / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    require(manifest["result"] == "PASS_2799_OF_2799", "GATE_RESULT")
    require(manifest["fresh_cohort_count"] == N, "GATE_N")
    require(manifest["token_gate"]["target_count"] == N, "GATE_TARGET")
    require(manifest["token_gate"]["pass_count"] == N, "GATE_PASS")
    require(manifest["token_gate"]["fail_count"] == 0, "GATE_FAIL")
    require(
        manifest["fresh_cohort_label_counts"]
        == {
            "Not Enough Evidence": 267,
            "Refuted": 1727,
            "Supported": 805,
        },
        "GATE_LABEL_COUNTS",
    )
    require(
        manifest["freshness"]["exclusion_counts"]
        == {
            "dev_claim_overlap": 6,
            "dev_url_overlap": 1,
            "empty_normalized_claim": 1,
            "incompatible_label": 195,
            "later_within_train_duplicate_claim": 66,
        },
        "GATE_EXCLUSIONS",
    )
    require(
        manifest["planned_steering_execution"]
        == {
            "conditions": list(CONDITIONS),
            "expected_full_model_forward_count": EXPECTED_FORWARD_BUDGET,
            "expected_row_count": N,
            "model_scale": "mamba370m",
            "scientific_execution_authorized_by_this_artifact": False,
        },
        "GATE_PLANNED_EXECUTION",
    )
    require(
        manifest["anchor_contract"]["anchor_name"] == ANCHOR_NAME
        and manifest["anchor_contract"]["target_offset"] == TARGET_OFFSET,
        "GATE_ANCHOR",
    )
    require(
        manifest["cohort_file_sha256"] == COHORT_SHA256,
        "GATE_COHORT_SHA",
    )
    require(manifest["model_forward_count"] == 0, "GATE_MODEL_FORWARD")
    require(
        manifest["scientific_inference_executed"] is False,
        "GATE_INFERENCE",
    )
    require(manifest["p_value_count_added"] == 0, "GATE_PVALUE")

    rows = read_jsonl(ROOT / COHORT_FILE)
    require(len(rows) == N, "COHORT_N")
    require(
        len({str(row["example_id"]) for row in rows}) == N,
        "COHORT_IDS",
    )
    require(
        [int(row["averitec_train_index"]) for row in rows]
        == sorted(int(row["averitec_train_index"]) for row in rows),
        "COHORT_ORDER",
    )
    indices = {int(row["averitec_train_index"]) for row in rows}
    require(438 in indices, "EMPTY_QUESTION_ROW_438_MISSING")
    require(1948 not in indices, "EMPTY_CLAIM_ROW_1948_PRESENT")

    counts = Counter(str(row["source_label"]) for row in rows)
    require(
        dict(counts)
        == {
            "Refuted": 1727,
            "Supported": 805,
            "Not Enough Evidence": 267,
        },
        f"COHORT_LABEL_COUNTS:{dict(counts)}",
    )

    for row in rows:
        require(row["token_gate_pass"] is True, "ROW_GATE")
        require(row["token_gate_reasons"] == [], "ROW_GATE_REASONS")
        require(bool(str(row["normalized_claim"])), "ROW_EMPTY_CLAIM")
        require(row["anchor_name"] == ANCHOR_NAME, "ROW_ANCHOR")
        require(int(row["target_offset"]) == TARGET_OFFSET, "ROW_OFFSET")
        require(
            int(row["target_intervention_token_index"])
            == int(row["absolute_anchor_token_index"]) + TARGET_OFFSET,
            "ROW_TARGET",
        )
        require(
            int(row["claim_consumed_token_count"]) >= 1,
            "ROW_CLAIM_TOKENS",
        )
        require(
            int(row["evidence_consumed_token_count"]) >= 2,
            "ROW_EVIDENCE_TOKENS",
        )
        require(int(row["correct_label_id"]) in (0, 1, 2), "ROW_LABEL_ID")
        require(
            str(row["correct_label"])
            in ("REFUTE", "NOT_ENTITLED", "SUPPORT"),
            "ROW_LABEL",
        )
    return rows, manifest


def model_rows(
    cohort: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in cohort:
        example_id = str(row["example_id"])
        out.append({
            "row_id": example_id,
            "source_pair_id": example_id,
            "contrast_cell_id": "AVERITEC_STEERING",
            "claim": str(row["claim"]),
            "evidence": str(row["evidence"]),
        })
    require(len(out) == N, "MODEL_ROWS_N")
    return out


def validate_encoding(
    *,
    cohort: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
) -> None:
    for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
        require(key in encoded, f"ENCODED_KEY:{key}")
        require(torch.is_tensor(encoded[key]), f"ENCODED_TENSOR:{key}")
        require(
            tuple(encoded[key].shape) == (N, 128),
            f"ENCODED_SHAPE:{key}:{tuple(encoded[key].shape)}",
        )

    for index, row in enumerate(cohort):
        attention = encoded["attention_mask"][index]
        claim = encoded["claim_mask"][index]
        evidence = encoded["evidence_mask"][index]

        attended = int(attention.sum().item())
        claim_count = int(claim.sum().item())
        evidence_count = int(evidence.sum().item())
        anchor = int(row["absolute_anchor_token_index"])
        target = int(row["target_intervention_token_index"])

        require(
            attended == int(row["serialized_attended_length"]),
            f"ATTENDED:{index}",
        )
        require(
            claim_count == int(row["claim_consumed_token_count"]),
            f"CLAIM_COUNT:{index}",
        )
        require(
            evidence_count == int(row["evidence_consumed_token_count"]),
            f"EVIDENCE_COUNT:{index}",
        )
        require(anchor == claim_count, f"ANCHOR_INDEX:{index}")
        require(target == anchor + TARGET_OFFSET, f"TARGET_INDEX:{index}")
        require(target < attended, f"TARGET_ATTENDED:{index}")
        require(bool(evidence[target].item()), f"TARGET_EVIDENCE:{index}")


def build_input(
    *,
    snapshot: Path,
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = bridge.scale_spec("mamba370m")
    spec["geom"].validate_snapshot(snapshot)
    tokenizer, tokenizer_provenance = spec["geom"].load_tokenizer(snapshot)
    rows = model_rows(cohort)
    encoded = spec["adapter"].encode_gen4_rows(rows, tokenizer)
    validate_encoding(cohort=cohort, encoded=encoded)
    return rows, encoded, tokenizer_provenance


def feature_batch(
    encoded: Mapping[str, Any],
    index: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
        tensor = encoded[key]
        require(torch.is_tensor(tensor), f"FEATURE:{key}")
        out[key] = (
            tensor[index:index + 1]
            .detach()
            .to(device)
            .contiguous()
        )
    return out


def correct_margin(logits: Sequence[float], label_id: int) -> float:
    require(
        len(logits) == 3 and label_id in (0, 1, 2),
        "MARGIN_INPUT",
    )
    correct = float(logits[label_id])
    wrong = max(float(logits[i]) for i in range(3) if i != label_id)
    value = correct - wrong
    require(math.isfinite(value), "MARGIN_NONFINITE")
    return value


def mirror_condition_correction(
    h: torch.Tensor,
    *,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    selected_plane: str,
    control_plane: str,
    dim: int,
    tol: float,
) -> dict[str, Any]:
    require(
        condition in ("p3_mirror_steer", "p5_matched_control"),
        f"CONDITION:{condition}",
    )
    require(selected_plane == SELECTED_PLANE, "SELECTED_PLANE_DRIFT")
    require(control_plane == CONTROL_PLANE, "CONTROL_PLANE_DRIFT")

    value = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(value.shape) == (dim,), "H_SHAPE")
    require(bool(torch.isfinite(value).all().item()), "H_NONFINITE")

    selected = bridge.plane_component(
        value,
        plane=selected_plane,
        planes=planes,
        dim=dim,
    )
    selected_component = selected["component"]

    control_plus = planes[control_plane]["plus"]
    control_minus = planes[control_plane]["minus"]
    require(
        tuple(control_plus.shape) == (dim,)
        and tuple(control_minus.shape) == (dim,),
        "CONTROL_PLANE_SHAPE",
    )
    control_component = (
        float(selected["a"]) * control_plus
        + float(selected["b"]) * control_minus
    ).contiguous()

    selected_l2 = float(torch.linalg.vector_norm(selected_component).item())
    control_l2 = float(torch.linalg.vector_norm(control_component).item())
    matched_norm_mismatch = abs(selected_l2 - control_l2)
    require(
        matched_norm_mismatch <= tol,
        f"MATCHED_CONTROL_NORM:{matched_norm_mismatch}",
    )

    delta = (selected_component - control_component).contiguous()
    require(bool(torch.isfinite(delta).all().item()), "DELTA_NONFINITE")

    historical = bridge.condition_correction(
        value,
        condition="dominant_control",
        planes=planes,
        selected_plane=selected_plane,
        control_plane=control_plane,
        dim=dim,
        tol=tol,
    )
    historical_control = historical["correction"].detach().cpu().to(
        torch.float64
    ).contiguous()

    historical_identity_residual = float(
        torch.max(torch.abs(historical_control + delta)).item()
    )
    require(
        historical_identity_residual <= tol,
        f"HISTORICAL_CONTROL_IDENTITY:{historical_identity_residual}",
    )

    if condition == "p3_mirror_steer":
        correction = delta
        sign = 1
    else:
        correction = (-delta).contiguous()
        sign = -1
        control_residual = float(
            torch.max(torch.abs(correction - historical_control)).item()
        )
        require(
            control_residual <= tol,
            f"CONTROL_REPRODUCTION:{control_residual}",
        )

    require(
        bool(torch.isfinite(correction).all().item()),
        "CORRECTION_NONFINITE",
    )

    post = (value + correction).contiguous()
    selected_post = [
        float(
            torch.dot(
                post,
                planes[selected_plane]["plus"],
            ).item()
        ),
        float(
            torch.dot(
                post,
                planes[selected_plane]["minus"],
            ).item()
        ),
    ]

    selected_post_max = max(abs(v) for v in selected_post)
    if condition == "p5_matched_control":
        require(
            selected_post_max <= tol,
            f"CONTROL_SELECTED_NEUTRALIZATION:{selected_post_max}",
        )

    return {
        "condition": condition,
        "mirror_sign": sign,
        "a": float(selected["a"]),
        "b": float(selected["b"]),
        "selected_component_l2": selected_l2,
        "matched_control_component_l2": control_l2,
        "matched_control_norm_abs_difference": matched_norm_mismatch,
        "delta_l2": float(torch.linalg.vector_norm(delta).item()),
        "correction_l2": float(torch.linalg.vector_norm(correction).item()),
        "historical_control_identity_max_abs_residual":
            historical_identity_residual,
        "selected_post_max_abs_projection": selected_post_max,
        "correction": correction,
    }


def steering_hook(
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

    info = mirror_condition_correction(
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
        "matched_control_component_l2":
            float(info["matched_control_component_l2"]),
        "matched_control_norm_abs_difference":
            float(info["matched_control_norm_abs_difference"]),
        "delta_l2": float(info["delta_l2"]),
        "mirror_sign": int(info["mirror_sign"]),
        "correction_l2": float(info["correction_l2"]),
        "historical_control_identity_max_abs_residual":
            float(info["historical_control_identity_max_abs_residual"]),
        "selected_post_max_abs_projection":
            float(info["selected_post_max_abs_projection"]),
        "applied_correction_max_abs_residual": residual,
        "free_steering_coefficient_present": False,
        "response_dependent_strength_present": False,
    })
    return out


def install_steering_hook(
    mixer: Any,
    **kwargs: Any,
):
    def hook(_module, _args, output):
        return steering_hook(output, **kwargs)

    return mixer.in_proj.register_forward_hook(hook)


def serialize_output(
    output: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    model_row: Mapping[str, Any],
    cohort_row: Mapping[str, Any],
    condition: str,
    intervention_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    serialized = bridge.serialize_output(
        output,
        row=model_row,
        scale="mamba370m",
        checkpoint_sha256=str(spec["checkpoint_sha256"]),
    )
    label_id = int(cohort_row["correct_label_id"])
    serialized.update({
        "schema_version": ROW_SCHEMA,
        "condition": condition,
        "averitec_train_index": int(cohort_row["averitec_train_index"]),
        "example_id": str(cohort_row["example_id"]),
        "source_label": str(cohort_row["source_label"]),
        "correct_label_id": label_id,
        "correct_label": str(cohort_row["correct_label"]),
        "is_correct": int(serialized["prediction_id"]) == label_id,
        "correct_class_logit_margin":
            correct_margin(serialized["final_logits"], label_id),
        "anchor_name": ANCHOR_NAME,
        "absolute_anchor_token_index":
            int(cohort_row["absolute_anchor_token_index"]),
        "target_intervention_token_index":
            int(cohort_row["target_intervention_token_index"]),
        "selected_plane": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "intervention_layer": INTERVENTION_LAYER,
        "intervention_audit":
            None if intervention_audit is None else dict(intervention_audit),
        "scientific_full_model_forward_count": 1,
    })
    return serialized


def run_condition(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
    encoded: Mapping[str, Any],
    model_row: Mapping[str, Any],
    cohort_row: Mapping[str, Any],
    row_index: int,
    condition: str,
    device: torch.device,
) -> dict[str, Any]:
    require(condition in CONDITIONS, f"RUN_CONDITION:{condition}")

    handle = None
    audit: dict[str, Any] | None = None

    if condition != "native":
        audit = {}
        handle = install_steering_hook(
            runtime_ctx["intervention_mixer"],
            token_index=int(cohort_row["target_intervention_token_index"]),
            strong_mask=runtime_ctx["strong_mask"],
            condition=condition,
            planes=frozen["planes"],
            selected_plane=SELECTED_PLANE,
            control_plane=CONTROL_PLANE,
            dim=int(spec["dim"]),
            intermediate_size=int(spec["geom"].INTERMEDIATE_SIZE),
            tol=float(spec["confirmation"].TOL),
            cast_tol=float(
                spec["confirmation"].transport_runtime.RUNTIME_CAST_TOL
            ),
            audit=audit,
        )

    try:
        with torch.inference_mode():
            output = spec["adapter"].historical_forward(
                model,
                feature_batch(encoded, row_index, device),
                arm=spec["geom"].ARM,
            )
    finally:
        if handle is not None:
            handle.remove()

    return serialize_output(
        output,
        spec=spec,
        model_row=model_row,
        cohort_row=cohort_row,
        condition=condition,
        intervention_audit=audit,
    )


def load_model_for_worker(
    *,
    snapshot: Path,
    physical_device: int,
) -> tuple[
    Mapping[str, Any],
    torch.nn.Module,
    torch.device,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)

    spec = bridge.scale_spec("mamba370m")
    require(str(spec["selected_plane"]) == SELECTED_PLANE, "SPEC_SELECTED")
    require(str(spec["control_plane"]) == CONTROL_PLANE, "SPEC_CONTROL")
    require(int(spec["geom"].INTERVENTION_LAYER) == INTERVENTION_LAYER, "SPEC_LAYER")
    require(int(spec["geom"].TARGET_OFFSET) == TARGET_OFFSET, "SPEC_OFFSET")

    device = spec["confirmation"].runtime_gate_single_visible_gpu(
        physical_device
    )
    require(str(device) == "cuda:0", f"LOGICAL_DEVICE:{device}")

    spec["geom"].validate_snapshot(snapshot)
    checkpoint = ROOT / spec["checkpoint_rel"]
    require(checkpoint.is_file(), "CHECKPOINT_MISSING")
    require(
        sha256_file(checkpoint) == str(spec["checkpoint_sha256"]),
        "CHECKPOINT_SHA256",
    )

    spec["confirmation"].load_frozen_selection()
    frozen = spec["confirmation"].load_frozen_geometry()

    model, kernels, model_provenance = spec["geom"].reconstruct_model(
        snapshot=snapshot,
        compact_checkpoint=checkpoint,
        gpu_id=0,
    )
    spec["confirmation"].kernel_compat.validate_transformers_kernel_bindings(
        kernels
    )
    runtime_ctx = spec["geom"].runtime_components(model)
    spec["confirmation"].validate_runtime_geometry(runtime_ctx, frozen)

    require(
        int(runtime_ctx["partition"]["strong_count"]) == int(spec["dim"]),
        "STRONG_DIM",
    )
    return (
        spec,
        model,
        device,
        runtime_ctx,
        frozen,
        model_provenance,
    )


def shard_worker(
    *,
    shard: Mapping[str, int],
    snapshot: str,
    temp_output: str,
) -> None:
    try:
        physical_device = int(shard["physical_device"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        snapshot_path = Path(snapshot)

        cohort, _manifest = validate_cohort()
        model_rows_value, encoded, tokenizer_provenance = build_input(
            snapshot=snapshot_path,
            cohort=cohort,
        )

        (
            spec,
            model,
            device,
            runtime_ctx,
            frozen,
            model_provenance,
        ) = load_model_for_worker(
            snapshot=snapshot_path,
            physical_device=physical_device,
        )

        rows: list[dict[str, Any]] = []
        for index in range(
            int(shard["start"]),
            int(shard["end"]),
        ):
            cohort_row = cohort[index]
            model_row = model_rows_value[index]
            for condition in CONDITIONS:
                rows.append(
                    run_condition(
                        spec=spec,
                        model=model,
                        runtime_ctx=runtime_ctx,
                        frozen=frozen,
                        encoded=encoded,
                        model_row=model_row,
                        cohort_row=cohort_row,
                        row_index=index,
                        condition=condition,
                        device=device,
                    )
                )

        torch.cuda.synchronize(device)
        require(
            len(rows) == int(shard["forward_budget"]),
            "SHARD_ROW_COUNT",
        )

        payload = {
            "schema_version": SHARD_SCHEMA,
            "shard": dict(shard),
            "checkpoint_sha256": str(spec["checkpoint_sha256"]),
            "hf_repo": str(spec["hf_repo"]),
            "hf_revision": str(spec["hf_revision"]),
            "geometry_freeze_commit": str(spec["geometry_freeze_commit"]),
            "discovery_freeze_commit": str(spec["discovery_freeze_commit"]),
            "discovery_selection_sha256":
                str(spec["discovery_selection_sha256"]),
            "model_provenance": dict(model_provenance),
            "tokenizer": dict(tokenizer_provenance),
            "rows": rows,
        }
        Path(temp_output).write_bytes(canonical_json_bytes(payload))

    except BaseException:
        traceback.print_exc()
        raise


def launch_shards(
    *,
    snapshot: Path,
    temp_dir: Path,
) -> list[Path]:
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    outputs: list[Path] = []

    for shard in SHARDS:
        out = temp_dir / f"shard{shard['shard_id']}.json"
        outputs.append(out)

        process = ctx.Process(
            target=shard_worker,
            kwargs={
                "shard": shard,
                "snapshot": str(snapshot),
                "temp_output": str(out),
            },
            name=f"averitec-steering-shard-{shard['shard_id']}",
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        require(
            process.exitcode == 0,
            f"SHARD_EXIT:{process.name}:{process.exitcode}",
        )

    require(
        all(path.is_file() for path in outputs),
        "SHARD_OUTPUT_MISSING",
    )
    return outputs


def merge_shards(
    *,
    shard_paths: Sequence[Path],
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(len(shard_paths) == SHARD_COUNT, "MERGE_SHARD_COUNT")

    rows: list[dict[str, Any]] = []
    shard_meta: list[dict[str, Any]] = []
    identity_ref: Mapping[str, Any] | None = None
    tokenizer_ref: Mapping[str, Any] | None = None

    for shard_id, path in enumerate(shard_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload["schema_version"] == SHARD_SCHEMA, "MERGE_SCHEMA")
        require(
            int(payload["shard"]["shard_id"]) == shard_id,
            "MERGE_SHARD_ID",
        )

        expected_shard = SHARDS[shard_id]
        require(payload["shard"] == expected_shard, "MERGE_SHARD_SPEC")

        identity = payload["model_provenance"]
        tokenizer = payload["tokenizer"]

        if identity_ref is None:
            identity_ref = identity
            tokenizer_ref = tokenizer
        else:
            for key in (
                "compact_checkpoint_sha256",
                "source_full_checkpoint_sha256",
                "mamba_state_canonical_sha256",
                "downstream_state_canonical_sha256",
                "full_state_canonical_sha256",
            ):
                require(
                    identity[key] == identity_ref[key],
                    f"MERGE_MODEL_IDENTITY:{key}",
                )
            require(
                tokenizer == tokenizer_ref,
                "MERGE_TOKENIZER_IDENTITY",
            )

        shard_rows = payload["rows"]
        require(
            len(shard_rows) == int(expected_shard["forward_budget"]),
            "MERGE_SHARD_ROWS",
        )
        rows.extend(shard_rows)

        shard_meta.append({
            "shard": payload["shard"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "hf_repo": payload["hf_repo"],
            "hf_revision": payload["hf_revision"],
            "geometry_freeze_commit": payload["geometry_freeze_commit"],
            "discovery_freeze_commit": payload["discovery_freeze_commit"],
            "discovery_selection_sha256":
                payload["discovery_selection_sha256"],
            "model_provenance": identity,
            "tokenizer": tokenizer,
            "raw_row_count": len(shard_rows),
        })

    require(len(rows) == EXPECTED_FORWARD_BUDGET, "MERGE_ROW_COUNT")

    expected_order = [
        (str(row["example_id"]), condition)
        for row in cohort
        for condition in CONDITIONS
    ]
    observed_order = [
        (str(row["example_id"]), str(row["condition"]))
        for row in rows
    ]
    require(observed_order == expected_order, "MERGE_ROW_ORDER")

    return rows, shard_meta


def validate_raw_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    cohort: Sequence[Mapping[str, Any]],
) -> None:
    require(len(rows) == EXPECTED_FORWARD_BUDGET, "RAW_ROW_COUNT")

    by_example: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        require(row["schema_version"] == ROW_SCHEMA, "RAW_SCHEMA")
        require(str(row["condition"]) in CONDITIONS, "RAW_CONDITION")
        require(row["scale"] == "mamba370m", "RAW_SCALE")
        require(
            row["selected_plane"] == SELECTED_PLANE,
            "RAW_SELECTED",
        )
        require(
            row["response_blind_control_plane"] == CONTROL_PLANE,
            "RAW_CONTROL",
        )
        require(
            int(row["intervention_layer"]) == INTERVENTION_LAYER,
            "RAW_LAYER",
        )
        require(row["anchor_name"] == ANCHOR_NAME, "RAW_ANCHOR")
        require(
            int(row["target_intervention_token_index"])
            == int(row["absolute_anchor_token_index"]) + TARGET_OFFSET,
            "RAW_TARGET",
        )
        require(
            int(row["scientific_full_model_forward_count"]) == 1,
            "RAW_FORWARD_COUNT",
        )
        require(int(row["prediction_id"]) in (0, 1, 2), "RAW_PREDICTION_ID")
        require(
            str(row["prediction"])
            in ("REFUTE", "NOT_ENTITLED", "SUPPORT"),
            "RAW_PREDICTION",
        )
        require(
            len(row["final_logits"]) == 3
            and all(math.isfinite(float(x)) for x in row["final_logits"]),
            "RAW_LOGITS",
        )
        require(
            math.isfinite(float(row["correct_class_logit_margin"])),
            "RAW_MARGIN",
        )

        condition = str(row["condition"])
        audit = row["intervention_audit"]
        if condition == "native":
            require(audit is None, "RAW_NATIVE_AUDIT")
        else:
            require(isinstance(audit, dict), "RAW_INTERVENTION_AUDIT")
            require(audit["condition"] == condition, "RAW_AUDIT_CONDITION")
            require(
                audit["selected_plane"] == SELECTED_PLANE,
                "RAW_AUDIT_SELECTED",
            )
            require(
                audit["control_plane"] == CONTROL_PLANE,
                "RAW_AUDIT_CONTROL",
            )
            require(
                int(audit["token_index"])
                == int(row["target_intervention_token_index"]),
                "RAW_AUDIT_TOKEN",
            )
            require(
                bool(audit["free_steering_coefficient_present"]) is False,
                "RAW_FREE_COEFFICIENT",
            )
            require(
                bool(audit["response_dependent_strength_present"]) is False,
                "RAW_RESPONSE_DEPENDENT_STRENGTH",
            )
            expected_sign = 1 if condition == "p3_mirror_steer" else -1
            require(
                int(audit["mirror_sign"]) == expected_sign,
                "RAW_MIRROR_SIGN",
            )

        by_example.setdefault(str(row["example_id"]), []).append(row)

    require(len(by_example) == N, "RAW_EXAMPLE_COUNT")

    cohort_by_id = {
        str(row["example_id"]): row
        for row in cohort
    }
    require(set(by_example) == set(cohort_by_id), "RAW_EXAMPLE_SET")

    for example_id, group in by_example.items():
        require(len(group) == len(CONDITIONS), "RAW_GROUP_SIZE")
        require(
            tuple(str(row["condition"]) for row in group) == CONDITIONS,
            "RAW_GROUP_CONDITION_ORDER",
        )
        source = cohort_by_id[example_id]
        for row in group:
            require(
                int(row["averitec_train_index"])
                == int(source["averitec_train_index"]),
                "RAW_SOURCE_INDEX",
            )
            require(
                int(row["correct_label_id"])
                == int(source["correct_label_id"]),
                "RAW_GOLD_ID",
            )
            require(
                str(row["correct_label"])
                == str(source["correct_label"]),
                "RAW_GOLD_LABEL",
            )


def build_summary(
    *,
    execution_head: str,
    shard_meta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(shard_meta) == SHARD_COUNT, "SUMMARY_SHARDS")

    return {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": execution_head,
        "experiment": "CAUSAL_ATLAS_GUIDED_STEERING",
        "model_scale": "mamba370m",
        "conditions": list(CONDITIONS),
        "fresh_cohort_count": N,
        "full_model_forward_budget": EXPECTED_FORWARD_BUDGET,
        "raw_row_count": EXPECTED_FORWARD_BUDGET,
        "selected_plane": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "intervention_layer": INTERVENTION_LAYER,
        "anchor_name": ANCHOR_NAME,
        "target_offset": TARGET_OFFSET,
        "cohort_sha256": COHORT_SHA256,
        "token_gate_manifest_sha256": MANIFEST_SHA256,
        "design_artifact_git_blob": DESIGN_ARTIFACT_GIT_BLOB,
        "source_eligibility_correction_git_blob":
            SOURCE_ELIGIBILITY_CORRECTION_GIT_BLOB,
        "shards": [dict(value) for value in shard_meta],
        "primary_inference_executed": False,
        "p_value_count_added": 0,
        "scientific_conclusion": None,
        "response_based_selection_performed": False,
        "rescue_performed": False,
        "magnitude_search_performed": False,
        "sign_search_performed": False,
        "layer_search_performed": False,
        "token_search_performed": False,
        "training_executed": False,
        "backward_executed": False,
    }


def write_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    rows_raw = jsonl_bytes(rows)
    summary_raw = pretty_json_bytes(summary)

    (output_dir / ROW_FILE).write_bytes(rows_raw)
    (output_dir / SUMMARY_FILE).write_bytes(summary_raw)

    hashes = {
        ROW_FILE: hashlib.sha256(rows_raw).hexdigest(),
        SUMMARY_FILE: hashlib.sha256(summary_raw).hexdigest(),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_raw(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    cohort, _manifest = validate_cohort()

    spec = bridge.scale_spec("mamba370m")
    spec["geom"].validate_snapshot(mamba370m_snapshot)

    checkpoint = ROOT / spec["checkpoint_rel"]
    require(checkpoint.is_file(), "CHECKPOINT_MISSING")
    require(
        sha256_file(checkpoint) == str(spec["checkpoint_sha256"]),
        "CHECKPOINT_SHA256",
    )

    require(not output_dir.exists(), "OUTPUT_ROOT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= SHARD_COUNT, "PHYSICAL_GPU_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="contramamba_averitec_steering_"
    ) as temp:
        temp_dir = Path(temp)
        shard_paths = launch_shards(
            snapshot=mamba370m_snapshot,
            temp_dir=temp_dir,
        )
        rows, shard_meta = merge_shards(
            shard_paths=shard_paths,
            cohort=cohort,
        )

    validate_raw_rows(rows=rows, cohort=cohort)

    summary = build_summary(
        execution_head=expected_head,
        shard_meta=shard_meta,
    )
    write_outputs(
        output_dir=output_dir,
        rows=rows,
        summary=summary,
    )

    print("RESULT=" + RESULT_PASS)
    print("MODEL_SCALE=mamba370m")
    print("FRESH_COHORT_COUNT=2799")
    print("CONDITIONS=native,p3_mirror_steer,p5_matched_control")
    print("SHARD0_EXAMPLES=1400")
    print("SHARD1_EXAMPLES=1399")
    print("SHARD0_FORWARDS=4200")
    print("SHARD1_FORWARDS=4197")
    print("FULL_MODEL_FORWARDS_TOTAL=8397")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")
    print("SCIENTIFIC_CONCLUSION=None")
    print("RESPONSE_BASED_SELECTION_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    print("MAGNITUDE_SEARCH_PERFORMED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen Mamba-370M AVeriTeC fixed mirror-steering "
            "three-condition raw response collection. No statistical inference."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--mamba370m-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    run_raw(
        expected_head=str(args.expected_head),
        mamba370m_snapshot=args.mamba370m_snapshot,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
