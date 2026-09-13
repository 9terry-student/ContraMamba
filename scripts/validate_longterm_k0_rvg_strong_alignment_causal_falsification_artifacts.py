#!/usr/bin/env python3
"""Independent validator for the frozen K0 strong-alignment causal-falsification run.

This validator intentionally does NOT import the causal runner. It authenticates
frozen repository inputs and recomputes public artifact invariants directly from
JSON/JSONL bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"
EXECUTION_COMMIT = "06585fae09db09b0bf222fcfe9494b9eeb674d70"
DESIGN_COMMIT = "edc951f8e845e5e41e4f17ad3678eed297c62970"
DESIGN_BLOB = "fcfa00d3de9e8b9c1cba200626fd7a6ee4b73220"
STOP_COMMIT = "f5cf85692bf8900f019a06979c54e637144b2d68"

RUNNER_REL = "scripts/longterm_k0_rvg_strong_alignment_causal_falsification.py"
TEST_REL = "tests/test_longterm_k0_rvg_strong_alignment_causal_falsification.py"

GEOMETRY_EVIDENCE_COMMIT = "431e8faa6e5c82a20d87f532b4ab960fcf641ec2"
GEOMETRY_IMPLEMENTATION_COMMIT = "0b1168182a265bc405652d2ce519f63310433c3e"
GEOMETRY_RUNNER_REL = "scripts/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_audit.py"
GEOMETRY_RUNNER_SHA256 = "0c487e3d29236e298f8c74cbdbcdfce3599af587dfd27a6e5fcf169df64eade5"
GEOMETRY_RUNNER_BLOB = "007d9ec21487ccfe7c56d6410d1dac6f832a9a22"
GEOMETRY_ITEM_REL = (
    "reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/"
    "layer20_y20_strong_interaction_geometry_item_metrics.jsonl"
)
GEOMETRY_ITEM_SHA256 = "e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe"
GEOMETRY_CHANNEL_REL = (
    "reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/"
    "layer20_y20_strong_interaction_geometry_channel_validation.jsonl"
)
GEOMETRY_SUMMARY_REL = (
    "reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/summary.json"
)
GEOMETRY_MANIFEST_REL = (
    "reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/execution_manifest.json"
)

CARRY_EVIDENCE_COMMIT = "21b36fa4579bee53644fa7f3da84cdc947ddef5b"
CARRY_IMPLEMENTATION_COMMIT = "378ac88cf3413dc46c9c7fa153ef60785fd370a4"
CARRY_ITEM_REL = (
    "reports/longterm_k0_rvg_layer22_carry_write_factorization_378ac88_v1/"
    "layer22_carry_write_factorization_metrics.jsonl"
)
CARRY_ITEM_SHA256 = "c9aa47f5cff0b8d4bd7ca81dfcef886e47fef02df9d6ca4e8e2dc2419367f2ad"
CARRY_SUMMARY_REL = (
    "reports/longterm_k0_rvg_layer22_carry_write_factorization_378ac88_v1/summary.json"
)

EXPECTED_HANDOFF_SHA256 = "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
EXPECTED_CHECKPOINT_SHA256 = "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
EXPECTED_ENCODER_CANONICAL = "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
EXPECTED_ENCODER_RAW = "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"
EXPECTED_MAMBA_SOURCE_SHA256 = "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"

OUTPUT_DIR_NAME = "longterm_k0_rvg_strong_alignment_causal_falsification_06585fa_v1"
ITEM_FILE = "strong_alignment_causal_falsification_item_metrics.jsonl"
SUMMARY_FILE = "summary.json"
MANIFEST_FILE = "execution_manifest.json"
EXPECTED_OUTPUT_SHA256 = {
    MANIFEST_FILE: "bf35236c29d6bf359cb6051a3843c7b94756cee03cb75149063bd1d9a05358ac",
    ITEM_FILE: "45660a82a177c38dd8724bfe0402acd8d5213d39432978e65d0d717d76558518",
    SUMMARY_FILE: "0663d354ba55912e22b6428600362837667cf8240445564c239c12a3b873d5c8",
}

ITEM_SCHEMA = "k0-rvg-strong-alignment-causal-falsification-item-v1"
SUMMARY_SCHEMA = "k0-rvg-strong-alignment-causal-falsification-summary-v1"
MANIFEST_SCHEMA = "k0-rvg-strong-alignment-causal-falsification-execution-manifest-v1"
PARENT_GEOMETRY_SCHEMA = "k0-rvg-layer20-y20-strong-interaction-geometry-item-v1"

EXPECTED_COUNT = 330
FULL_FORWARD_BUDGET = 2640
SOURCE_BLOCK = 20
TARGET_RESIDUAL_LAYER = 21
INTERVENTION_LAYER = 22
TARGET_K = 2

BRIDGE_TOL = 1e-12
VECTOR_TOL = 5e-12
CAST_TOL = 5e-6
MIDPOINT_TOL = 5e-6
RECOMPUTE_TOL = 1e-12

ITEM_FIELDS = frozenset({
    "schema_version", "local_template_index", "stable_item_id", "in_common_ddsssss_cohort",
    "source_block", "target_residual_layer", "intervention_layer", "relative_coordinate",
    "A_corr", "A_ctrl", "B_corr", "B_ctrl", "C_corr", "C_ctrl", "I_corr", "I_ctrl",
    "geometry_bridge_max_abs_residual", "native_bridge_max_abs_residual",
    "alignment_target_cosine", "alignment_realized_cosine",
    "alignment_A_preservation_abs_residual", "alignment_B_preservation_abs_residual",
    "alignment_midpoint_max_abs_residual", "alignment_runtime_correction_l2",
    "alignment_applied_correction_max_abs_residual",
    "magnitude_target_A", "magnitude_target_B", "magnitude_realized_A", "magnitude_realized_B",
    "magnitude_realized_cosine", "magnitude_cosine_preservation_abs_residual",
    "magnitude_midpoint_max_abs_residual", "magnitude_runtime_correction_l2",
    "magnitude_applied_correction_max_abs_residual",
    "baseline_corr_delta_s22_post_l2", "baseline_corr_delta_w_l2", "baseline_corr_delta_carry_l2",
    "baseline_ctrl_delta_s22_post_l2", "baseline_ctrl_delta_w_l2", "baseline_ctrl_delta_carry_l2",
    "alignment_corr_delta_s22_post_l2", "alignment_corr_delta_w_l2", "alignment_corr_delta_carry_l2",
    "magnitude_corr_delta_s22_post_l2", "magnitude_corr_delta_w_l2", "magnitude_corr_delta_carry_l2",
    "S_c0_minus_ctrl", "S_cA_minus_ctrl", "S_cM_minus_ctrl",
    "W_c0_minus_ctrl", "W_cA_minus_ctrl", "W_cM_minus_ctrl",
    "C_c0_minus_cA", "C_c0_minus_cM", "S_c0_minus_cA", "S_c0_minus_cM",
    "W_c0_minus_cA", "W_c0_minus_cM",
})

SUMMARY_FIELDS = frozenset({
    "schema_version", "population_size", "model_forward_count",
    "G_S0", "G_SA", "G_SM", "G_W0", "G_WA", "G_WM", "R_CA", "R_CM",
    "max_geometry_bridge_abs_residual", "max_native_bridge_abs_residual",
    "max_alignment_A_preservation_abs_residual", "max_alignment_B_preservation_abs_residual",
    "max_alignment_cosine_abs_residual", "max_alignment_midpoint_abs_residual",
    "max_alignment_applied_correction_abs_residual", "max_magnitude_A_abs_residual",
    "max_magnitude_B_abs_residual", "max_magnitude_cosine_preservation_abs_residual",
    "max_magnitude_midpoint_abs_residual", "max_magnitude_applied_correction_abs_residual",
    "alignment_runtime_correction_l2", "magnitude_runtime_correction_l2",
    "state_alignment_reduction", "state_magnitude_reduction",
    "write_alignment_reduction", "write_magnitude_reduction",
    "itemwise_counts", "forbidden_action_flags", "R_SA", "R_SM", "R_WA", "R_WM", "F_SA",
    "alignment_sign_strata", "all_mandatory_manipulation_checks_pass", "outcome",
})

MANIFEST_FIELDS = frozenset({
    "schema_version", "runtime_branch", "runtime_git_head", "design_freeze_commit", "design_blob",
    "implementation_commit", "runner_rel", "runner_sha256", "runner_blob", "stopping_boundary_freeze",
    "geometry_evidence_freeze", "geometry_implementation_commit", "geometry_runner_sha256",
    "geometry_runner_blob", "geometry_item_sha256", "geometry_item_blob", "geometry_channel_sha256",
    "geometry_channel_blob", "geometry_summary_sha256", "geometry_summary_blob",
    "geometry_manifest_sha256", "geometry_manifest_blob", "carry_evidence_freeze",
    "carry_implementation_commit", "carry_item_sha256", "carry_item_blob", "carry_summary_sha256",
    "carry_summary_blob", "handoff_zip_sha256", "checkpoint_sha256", "encoder_canonical_digest",
    "encoder_raw_concat_digest", "mamba_source_sha256", "model_forward_count", "training_executed",
    "task_heads_executed", "logits_read", "tokenizer_invoked", "causal_intervention_executed",
    "raw_vectors_persisted", "k1_executed", "outputs",
})

NEGATIVE_FLAGS = (
    "training_executed", "task_heads_executed", "logits_read", "tokenizer_invoked",
    "raw_vectors_persisted", "k1_executed",
)


class ValidationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValidationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError("GIT_FAILURE:" + " ".join(args)) from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError("GIT_SHOW_FAILURE:" + spec) from exc


def is_ancestor(root: Path, ancestor: str, head: str) -> bool:
    return subprocess.call(
        ["git", "merge-base", "--is-ancestor", ancestor, head], cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) == 0


def json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise ValidationError(label + "_JSON_PARSE_FAILURE") from exc
    require(isinstance(obj, dict), label + "_NOT_OBJECT")
    return obj


def jsonl_objects(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(raw.splitlines(), start=1):
        require(bool(line), f"{label}_BLANK_LINE:{n}")
        try:
            obj = json.loads(line)
        except Exception as exc:
            raise ValidationError(f"{label}_JSON_PARSE_FAILURE:{n}") from exc
        require(isinstance(obj, dict), f"{label}_ROW_NOT_OBJECT:{n}")
        rows.append(obj)
    return rows


def finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), label + "_NOT_NUMBER")
    out = float(value)
    require(math.isfinite(out), label + "_NONFINITE")
    return out


def close(a: Any, b: Any, label: str, tol: float = RECOMPUTE_TOL) -> None:
    av = finite_number(a, label + ":A")
    bv = finite_number(b, label + ":B")
    require(abs(av - bv) <= tol, f"{label}_MISMATCH:{av}:{bv}:{abs(av-bv)}")


def reject_private_payload(obj: Any, path: str = "root") -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            lower = str(key).lower()
            field_path = path + "." + str(key)
            if lower in {"logits_read", "raw_vectors_persisted"}:
                require(value is False, "NEGATIVE_PUBLIC_FLAG_NOT_FALSE:" + field_path)
            else:
                for forbidden in ("raw_vector", "activation_vector", "logits", "checkpoint_bytes", "token_text"):
                    require(forbidden not in lower, "FORBIDDEN_PUBLIC_FIELD:" + field_path)
            reject_private_payload(value, field_path)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            reject_private_payload(value, f"{path}[{i}]")


def aggregate(values: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    require(bool(vals) and all(math.isfinite(v) for v in vals), "BAD_AGGREGATE_INPUT")
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    vals = [float(r[field]) for r in rows]
    return {
        "decrease": sum(v > 0.0 for v in vals),
        "increase": sum(v < 0.0 for v in vals),
        "equal": sum(v == 0.0 for v in vals),
    }


def classify(sa: float, wa: float, sm: float, wm: float, manipulation_ok: bool) -> str:
    if not manipulation_ok:
        return "Invalid"
    if sa > 0.0 and wa > 0.0:
        return "A" if sa > sm and wa > wm else "B"
    if (sa > 0.0) != (wa > 0.0):
        return "C"
    return "D"


def compare_nested(expected: Any, actual: Any, path: str = "root", tol: float = RECOMPUTE_TOL) -> None:
    if isinstance(expected, Mapping):
        require(isinstance(actual, Mapping), path + "_TYPE_MISMATCH")
        require(set(actual) == set(expected), path + "_KEY_SET_MISMATCH")
        for k in expected:
            compare_nested(expected[k], actual[k], path + "." + str(k), tol)
        return
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + "_LIST_MISMATCH")
        for i, (e, a) in enumerate(zip(expected, actual)):
            compare_nested(e, a, f"{path}[{i}]", tol)
        return
    if isinstance(expected, float):
        close(expected, actual, path, tol)
        return
    require(actual == expected, f"{path}_MISMATCH:{actual!r}:{expected!r}")


def frozen_file(root: Path, commit: str, rel: str, expected_sha: str | None, label: str) -> tuple[bytes, str]:
    raw = git_bytes(root, f"{commit}:{rel}")
    if expected_sha is not None:
        require(sha256_bytes(raw) == expected_sha, label + "_SHA256_MISMATCH")
    current = root / rel
    require(current.is_file(), label + "_MISSING")
    require(current.read_bytes() == raw, label + "_WORKTREE_DRIFT")
    blob = git(root, "rev-parse", f"{commit}:{rel}")
    return raw, blob


def validate_repo(root: Path) -> dict[str, str]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "BRANCH_MISMATCH")
    require(is_ancestor(root, EXECUTION_COMMIT, head), "EXECUTION_COMMIT_NOT_ANCESTOR")
    require(git(root, "rev-parse", f"{DESIGN_COMMIT}:reports/longterm_k0_rvg_strong_alignment_causal_falsification_static_design_candidate.md") == DESIGN_BLOB, "DESIGN_BLOB_MISMATCH")

    runner_exec = git_bytes(root, f"{EXECUTION_COMMIT}:{RUNNER_REL}")
    test_exec = git_bytes(root, f"{EXECUTION_COMMIT}:{TEST_REL}")
    require(git(root, "rev-parse", f"HEAD:{RUNNER_REL}") == git(root, "rev-parse", f"{EXECUTION_COMMIT}:{RUNNER_REL}"), "RUNNER_DRIFT_AFTER_EXECUTION")
    require(git(root, "rev-parse", f"HEAD:{TEST_REL}") == git(root, "rev-parse", f"{EXECUTION_COMMIT}:{TEST_REL}"), "TEST_DRIFT_AFTER_EXECUTION")
    return {
        "branch": branch,
        "head": head,
        "runner_sha256": sha256_bytes(runner_exec),
        "runner_blob": git(root, "rev-parse", f"{EXECUTION_COMMIT}:{RUNNER_REL}"),
        "test_sha256": sha256_bytes(test_exec),
    }


def validate_parent_inputs(root: Path):
    geometry_raw, geometry_blob = frozen_file(
        root, GEOMETRY_EVIDENCE_COMMIT, GEOMETRY_ITEM_REL, GEOMETRY_ITEM_SHA256, "GEOMETRY_ITEM"
    )
    carry_raw, carry_blob = frozen_file(
        root, CARRY_EVIDENCE_COMMIT, CARRY_ITEM_REL, CARRY_ITEM_SHA256, "CARRY_ITEM"
    )
    geometry_rows = jsonl_objects(geometry_raw, "GEOMETRY_ITEM")
    carry_rows = jsonl_objects(carry_raw, "CARRY_ITEM")
    geo = {}
    for row in geometry_rows:
        require(row.get("schema_version") == PARENT_GEOMETRY_SCHEMA, "GEOMETRY_SCHEMA_MISMATCH")
        idx = int(row["local_template_index"])
        require(idx not in geo, "GEOMETRY_DUPLICATE_INDEX")
        geo[idx] = row
    carry = {}
    for row in carry_rows:
        if int(row.get("relative_coordinate", -999)) != TARGET_K or not bool(row.get("in_common_ddsssss_cohort")):
            continue
        key = (int(row["local_template_index"]), str(row["role"]))
        require(key not in carry, "CARRY_DUPLICATE_COMMON_K2")
        carry[key] = row
    common = sorted(i for i in geo if (i, "corr") in carry and (i, "ctrl") in carry)
    require(len(common) == EXPECTED_COUNT, "PARENT_COMMON_330_MISMATCH")
    return geo, carry, common, geometry_blob, carry_blob


def validate_item_rows(rows, geo, carry, common):
    require(len(rows) == EXPECTED_COUNT, "ITEM_COUNT_MISMATCH")
    indices = [int(r.get("local_template_index", -1)) for r in rows]
    require(indices == common, "ITEM_INDEX_ORDER_OR_SET_MISMATCH")
    require(len(set(indices)) == EXPECTED_COUNT, "ITEM_INDEX_DUPLICATE")
    stable_ids = set()

    for row in rows:
        require(set(row) == ITEM_FIELDS, f"ITEM_FIELD_SET_MISMATCH:{row.get('local_template_index')}")
        require(row["schema_version"] == ITEM_SCHEMA, "ITEM_SCHEMA_MISMATCH")
        require(row["in_common_ddsssss_cohort"] is True, "ITEM_COMMON_FLAG_NOT_TRUE")
        require(int(row["source_block"]) == SOURCE_BLOCK, "ITEM_SOURCE_BLOCK_MISMATCH")
        require(int(row["target_residual_layer"]) == TARGET_RESIDUAL_LAYER, "ITEM_TARGET_LAYER_MISMATCH")
        require(int(row["intervention_layer"]) == INTERVENTION_LAYER, "ITEM_INTERVENTION_LAYER_MISMATCH")
        require(int(row["relative_coordinate"]) == TARGET_K, "ITEM_RELATIVE_COORDINATE_MISMATCH")
        reject_private_payload(row)
        idx = int(row["local_template_index"])
        parent = geo[idx]
        require(row["stable_item_id"] == parent["stable_item_id"], f"STABLE_ID_PARENT_MISMATCH:{idx}")
        require(row["stable_item_id"] not in stable_ids, "STABLE_ID_DUPLICATE")
        stable_ids.add(row["stable_item_id"])

        for field in ("A_corr", "A_ctrl", "B_corr", "B_ctrl", "C_corr", "C_ctrl", "I_corr", "I_ctrl"):
            close(row[field], parent[field], f"GEOMETRY_PARENT_BRIDGE:{idx}:{field}", BRIDGE_TOL)

        for role in ("corr", "ctrl"):
            p = carry[idx, role]
            mapping = {
                f"baseline_{role}_delta_s22_post_l2": "delta_s22_post_l2",
                f"baseline_{role}_delta_w_l2": "delta_w_l2",
                f"baseline_{role}_delta_carry_l2": "delta_carry_l2",
            }
            for public_field, parent_field in mapping.items():
                require(parent_field in p, f"CARRY_PARENT_FIELD_MISSING:{idx}:{role}:{parent_field}")
                close(row[public_field], p[parent_field], f"CARRY_PARENT_BRIDGE:{idx}:{role}:{parent_field}", BRIDGE_TOL)

        derived = {
            "S_c0_minus_ctrl": row["baseline_corr_delta_s22_post_l2"] - row["baseline_ctrl_delta_s22_post_l2"],
            "S_cA_minus_ctrl": row["alignment_corr_delta_s22_post_l2"] - row["baseline_ctrl_delta_s22_post_l2"],
            "S_cM_minus_ctrl": row["magnitude_corr_delta_s22_post_l2"] - row["baseline_ctrl_delta_s22_post_l2"],
            "W_c0_minus_ctrl": row["baseline_corr_delta_w_l2"] - row["baseline_ctrl_delta_w_l2"],
            "W_cA_minus_ctrl": row["alignment_corr_delta_w_l2"] - row["baseline_ctrl_delta_w_l2"],
            "W_cM_minus_ctrl": row["magnitude_corr_delta_w_l2"] - row["baseline_ctrl_delta_w_l2"],
            "C_c0_minus_cA": row["baseline_corr_delta_carry_l2"] - row["alignment_corr_delta_carry_l2"],
            "C_c0_minus_cM": row["baseline_corr_delta_carry_l2"] - row["magnitude_corr_delta_carry_l2"],
            "S_c0_minus_cA": row["baseline_corr_delta_s22_post_l2"] - row["alignment_corr_delta_s22_post_l2"],
            "S_c0_minus_cM": row["baseline_corr_delta_s22_post_l2"] - row["magnitude_corr_delta_s22_post_l2"],
            "W_c0_minus_cA": row["baseline_corr_delta_w_l2"] - row["alignment_corr_delta_w_l2"],
            "W_c0_minus_cM": row["baseline_corr_delta_w_l2"] - row["magnitude_corr_delta_w_l2"],
        }
        for field, expected in derived.items():
            close(row[field], expected, f"ITEM_DERIVED:{idx}:{field}")

        require(abs(float(row["alignment_realized_cosine"]) - float(row["alignment_target_cosine"])) <= VECTOR_TOL, f"ALIGN_COSINE_GATE:{idx}")
        require(float(row["alignment_A_preservation_abs_residual"]) <= VECTOR_TOL, f"ALIGN_A_GATE:{idx}")
        require(float(row["alignment_B_preservation_abs_residual"]) <= VECTOR_TOL, f"ALIGN_B_GATE:{idx}")
        require(float(row["alignment_midpoint_max_abs_residual"]) <= MIDPOINT_TOL, f"ALIGN_MIDPOINT_GATE:{idx}")
        require(float(row["alignment_applied_correction_max_abs_residual"]) <= CAST_TOL, f"ALIGN_CAST_GATE:{idx}")
        require(abs(float(row["magnitude_realized_A"]) - float(row["magnitude_target_A"])) <= VECTOR_TOL, f"MAG_A_GATE:{idx}")
        require(abs(float(row["magnitude_realized_B"]) - float(row["magnitude_target_B"])) <= VECTOR_TOL, f"MAG_B_GATE:{idx}")
        require(float(row["magnitude_cosine_preservation_abs_residual"]) <= VECTOR_TOL, f"MAG_COSINE_GATE:{idx}")
        require(float(row["magnitude_midpoint_max_abs_residual"]) <= MIDPOINT_TOL, f"MAG_MIDPOINT_GATE:{idx}")
        require(float(row["magnitude_applied_correction_max_abs_residual"]) <= CAST_TOL, f"MAG_CAST_GATE:{idx}")
        require(float(row["geometry_bridge_max_abs_residual"]) <= BRIDGE_TOL, f"GEOMETRY_BRIDGE_GATE:{idx}")
        require(float(row["native_bridge_max_abs_residual"]) <= BRIDGE_TOL, f"NATIVE_BRIDGE_GATE:{idx}")


def recompute_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mean = lambda field: float(statistics.fmean(float(r[field]) for r in rows))
    out: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "population_size": EXPECTED_COUNT,
        "model_forward_count": FULL_FORWARD_BUDGET,
        "G_S0": mean("S_c0_minus_ctrl"),
        "G_SA": mean("S_cA_minus_ctrl"),
        "G_SM": mean("S_cM_minus_ctrl"),
        "G_W0": mean("W_c0_minus_ctrl"),
        "G_WA": mean("W_cA_minus_ctrl"),
        "G_WM": mean("W_cM_minus_ctrl"),
        "R_CA": mean("C_c0_minus_cA"),
        "R_CM": mean("C_c0_minus_cM"),
        "max_geometry_bridge_abs_residual": max(float(r["geometry_bridge_max_abs_residual"]) for r in rows),
        "max_native_bridge_abs_residual": max(float(r["native_bridge_max_abs_residual"]) for r in rows),
        "max_alignment_A_preservation_abs_residual": max(float(r["alignment_A_preservation_abs_residual"]) for r in rows),
        "max_alignment_B_preservation_abs_residual": max(float(r["alignment_B_preservation_abs_residual"]) for r in rows),
        "max_alignment_cosine_abs_residual": max(abs(float(r["alignment_realized_cosine"]) - float(r["alignment_target_cosine"])) for r in rows),
        "max_alignment_midpoint_abs_residual": max(float(r["alignment_midpoint_max_abs_residual"]) for r in rows),
        "max_alignment_applied_correction_abs_residual": max(float(r["alignment_applied_correction_max_abs_residual"]) for r in rows),
        "max_magnitude_A_abs_residual": max(abs(float(r["magnitude_realized_A"]) - float(r["magnitude_target_A"])) for r in rows),
        "max_magnitude_B_abs_residual": max(abs(float(r["magnitude_realized_B"]) - float(r["magnitude_target_B"])) for r in rows),
        "max_magnitude_cosine_preservation_abs_residual": max(float(r["magnitude_cosine_preservation_abs_residual"]) for r in rows),
        "max_magnitude_midpoint_abs_residual": max(float(r["magnitude_midpoint_max_abs_residual"]) for r in rows),
        "max_magnitude_applied_correction_abs_residual": max(float(r["magnitude_applied_correction_max_abs_residual"]) for r in rows),
        "alignment_runtime_correction_l2": aggregate([r["alignment_runtime_correction_l2"] for r in rows]),
        "magnitude_runtime_correction_l2": aggregate([r["magnitude_runtime_correction_l2"] for r in rows]),
        "state_alignment_reduction": aggregate([r["S_c0_minus_cA"] for r in rows]),
        "state_magnitude_reduction": aggregate([r["S_c0_minus_cM"] for r in rows]),
        "write_alignment_reduction": aggregate([r["W_c0_minus_cA"] for r in rows]),
        "write_magnitude_reduction": aggregate([r["W_c0_minus_cM"] for r in rows]),
        "itemwise_counts": {
            "state_alignment": counts(rows, "S_c0_minus_cA"),
            "state_magnitude": counts(rows, "S_c0_minus_cM"),
            "write_alignment": counts(rows, "W_c0_minus_cA"),
            "write_magnitude": counts(rows, "W_c0_minus_cM"),
        },
        "forbidden_action_flags": {
            "training_executed": False,
            "task_heads_executed": False,
            "logits_read": False,
            "tokenizer_invoked": False,
            "raw_vectors_persisted": False,
            "k1_executed": False,
        },
    }
    out["R_SA"] = out["G_S0"] - out["G_SA"]
    out["R_SM"] = out["G_S0"] - out["G_SM"]
    out["R_WA"] = out["G_W0"] - out["G_WA"]
    out["R_WM"] = out["G_W0"] - out["G_WM"]
    out["F_SA"] = out["R_SA"] / out["G_S0"] if out["G_S0"] > 0.0 else None

    gt = [r for r in rows if float(r["C_corr"]) > float(r["C_ctrl"])]
    lt = [r for r in rows if float(r["C_corr"]) < float(r["C_ctrl"])]
    eq = [r for r in rows if float(r["C_corr"]) == float(r["C_ctrl"])]
    out["alignment_sign_strata"] = {
        "corr_gt_ctrl": {"count": len(gt), "state": counts(gt, "S_c0_minus_cA"), "write": counts(gt, "W_c0_minus_cA")},
        "corr_lt_ctrl": {"count": len(lt), "state": counts(lt, "S_c0_minus_cA"), "write": counts(lt, "W_c0_minus_cA")},
        "equal": {"count": len(eq), "state": counts(eq, "S_c0_minus_cA"), "write": counts(eq, "W_c0_minus_cA")},
    }
    manipulation_ok = (
        out["max_geometry_bridge_abs_residual"] <= BRIDGE_TOL
        and out["max_native_bridge_abs_residual"] <= BRIDGE_TOL
        and out["max_alignment_A_preservation_abs_residual"] <= VECTOR_TOL
        and out["max_alignment_B_preservation_abs_residual"] <= VECTOR_TOL
        and out["max_alignment_cosine_abs_residual"] <= VECTOR_TOL
        and out["max_alignment_midpoint_abs_residual"] <= MIDPOINT_TOL
        and out["max_alignment_applied_correction_abs_residual"] <= CAST_TOL
        and out["max_magnitude_A_abs_residual"] <= VECTOR_TOL
        and out["max_magnitude_B_abs_residual"] <= VECTOR_TOL
        and out["max_magnitude_cosine_preservation_abs_residual"] <= VECTOR_TOL
        and out["max_magnitude_midpoint_abs_residual"] <= MIDPOINT_TOL
        and out["max_magnitude_applied_correction_abs_residual"] <= CAST_TOL
    )
    out["all_mandatory_manipulation_checks_pass"] = bool(manipulation_ok)
    out["outcome"] = classify(out["R_SA"], out["R_WA"], out["R_SM"], out["R_WM"], manipulation_ok)
    return out


def validate_manifest(root: Path, manifest: Mapping[str, Any], repo: Mapping[str, str], output_hashes: Mapping[str, str], geometry_blob: str, carry_blob: str) -> None:
    require(set(manifest) == MANIFEST_FIELDS, "MANIFEST_FIELD_SET_MISMATCH")
    require(manifest["schema_version"] == MANIFEST_SCHEMA, "MANIFEST_SCHEMA_MISMATCH")
    reject_private_payload(manifest)
    exact = {
        "runtime_branch": EXPECTED_BRANCH,
        "runtime_git_head": EXECUTION_COMMIT,
        "design_freeze_commit": DESIGN_COMMIT,
        "design_blob": DESIGN_BLOB,
        "implementation_commit": EXECUTION_COMMIT,
        "runner_rel": RUNNER_REL,
        "runner_sha256": repo["runner_sha256"],
        "runner_blob": repo["runner_blob"],
        "stopping_boundary_freeze": STOP_COMMIT,
        "geometry_evidence_freeze": GEOMETRY_EVIDENCE_COMMIT,
        "geometry_implementation_commit": GEOMETRY_IMPLEMENTATION_COMMIT,
        "geometry_runner_sha256": GEOMETRY_RUNNER_SHA256,
        "geometry_runner_blob": GEOMETRY_RUNNER_BLOB,
        "geometry_item_sha256": GEOMETRY_ITEM_SHA256,
        "geometry_item_blob": geometry_blob,
        "carry_evidence_freeze": CARRY_EVIDENCE_COMMIT,
        "carry_implementation_commit": CARRY_IMPLEMENTATION_COMMIT,
        "carry_item_sha256": CARRY_ITEM_SHA256,
        "carry_item_blob": carry_blob,
        "handoff_zip_sha256": EXPECTED_HANDOFF_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "encoder_canonical_digest": EXPECTED_ENCODER_CANONICAL,
        "encoder_raw_concat_digest": EXPECTED_ENCODER_RAW,
        "mamba_source_sha256": EXPECTED_MAMBA_SOURCE_SHA256,
        "model_forward_count": FULL_FORWARD_BUDGET,
        "causal_intervention_executed": True,
    }
    for field, expected in exact.items():
        require(manifest[field] == expected, f"MANIFEST_{field}_MISMATCH")
    for flag in NEGATIVE_FLAGS:
        require(manifest[flag] is False, "MANIFEST_NEGATIVE_FLAG_NOT_FALSE:" + flag)
    require(set(manifest["outputs"]) == {ITEM_FILE, SUMMARY_FILE}, "MANIFEST_OUTPUT_KEY_SET_MISMATCH")
    require(manifest["outputs"][ITEM_FILE] == output_hashes[ITEM_FILE], "MANIFEST_ITEM_HASH_MISMATCH")
    require(manifest["outputs"][SUMMARY_FILE] == output_hashes[SUMMARY_FILE], "MANIFEST_SUMMARY_HASH_MISMATCH")

    # Authenticate the remaining parent identities dynamically against frozen commits.
    parent_specs = (
        ("geometry_channel_sha256", "geometry_channel_blob", GEOMETRY_EVIDENCE_COMMIT, GEOMETRY_CHANNEL_REL),
        ("geometry_summary_sha256", "geometry_summary_blob", GEOMETRY_EVIDENCE_COMMIT, GEOMETRY_SUMMARY_REL),
        ("geometry_manifest_sha256", "geometry_manifest_blob", GEOMETRY_EVIDENCE_COMMIT, GEOMETRY_MANIFEST_REL),
        ("carry_summary_sha256", "carry_summary_blob", CARRY_EVIDENCE_COMMIT, CARRY_SUMMARY_REL),
    )
    for sha_field, blob_field, commit, rel in parent_specs:
        raw = git_bytes(root, f"{commit}:{rel}")
        require(manifest[sha_field] == sha256_bytes(raw), "MANIFEST_PARENT_SHA_MISMATCH:" + sha_field)
        require(manifest[blob_field] == git(root, "rev-parse", f"{commit}:{rel}"), "MANIFEST_PARENT_BLOB_MISMATCH:" + blob_field)


def validate(output_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    repo = validate_repo(root)
    require(output_dir.resolve().parent == (root / "reports").resolve(), "OUTPUT_DIR_NOT_UNDER_REPORTS")
    require(output_dir.name == OUTPUT_DIR_NAME, "OUTPUT_DIR_NAME_MISMATCH")
    require(output_dir.is_dir(), "OUTPUT_DIR_MISSING")
    require(not Path(str(output_dir) + ".partial").exists(), "PARTIAL_OUTPUT_PRESENT")
    files = {p.name for p in output_dir.iterdir() if p.is_file()}
    require(files == {ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE}, "OUTPUT_FILE_SET_MISMATCH")
    require(not any(p.is_dir() for p in output_dir.iterdir()), "UNEXPECTED_OUTPUT_SUBDIRECTORY")

    output_hashes = {name: sha256_file(output_dir / name) for name in files}
    require(output_hashes == EXPECTED_OUTPUT_SHA256, "OUTPUT_SHA256_IDENTITY_MISMATCH")

    rows = jsonl_objects((output_dir / ITEM_FILE).read_bytes(), "OUTPUT_ITEMS")
    summary = json_object((output_dir / SUMMARY_FILE).read_bytes(), "OUTPUT_SUMMARY")
    manifest = json_object((output_dir / MANIFEST_FILE).read_bytes(), "OUTPUT_MANIFEST")
    reject_private_payload(rows)
    reject_private_payload(summary)
    reject_private_payload(manifest)
    require(set(summary) == SUMMARY_FIELDS, "SUMMARY_FIELD_SET_MISMATCH")
    require(summary["schema_version"] == SUMMARY_SCHEMA, "SUMMARY_SCHEMA_MISMATCH")

    geo, carry, common, geometry_blob, carry_blob = validate_parent_inputs(root)
    validate_item_rows(rows, geo, carry, common)
    expected_summary = recompute_summary(rows)
    compare_nested(expected_summary, summary, "summary", RECOMPUTE_TOL)
    validate_manifest(root, manifest, repo, output_hashes, geometry_blob, carry_blob)

    return {
        "validator_mode": "independent_no_runner_import",
        "validated_execution_commit": EXECUTION_COMMIT,
        "validator_runtime_head": repo["head"],
        "item_count": len(rows),
        "model_forward_count": manifest["model_forward_count"],
        "output_sha256": output_hashes,
        "all_mandatory_manipulation_checks_pass": summary["all_mandatory_manipulation_checks_pass"],
        "outcome_recomputed": expected_summary["outcome"],
        "R_SA_recomputed": expected_summary["R_SA"],
        "R_WA_recomputed": expected_summary["R_WA"],
        "R_SM_recomputed": expected_summary["R_SM"],
        "R_WM_recomputed": expected_summary["R_WM"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.output_dir)
    print("PASS_INDEPENDENT_STRONG_ALIGNMENT_CAUSAL_ARTIFACT_VALIDATION")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print("BLOCKED:", exc)
        raise SystemExit(2)
