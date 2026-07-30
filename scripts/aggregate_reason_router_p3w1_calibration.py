"""Aggregate P3-W1 reason-router calibration unit artifacts.

Pure JSON validator/aggregator: imports no model code and no torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

UNIT_SCHEMA = "reason_router_p3w1_calibration_unit_v1"
AGGREGATE_SCHEMA = "reason_router_p3w1_calibration_aggregate_v1"
EXPECTED_SEEDS = [180, 181, 182]
UNIT_DECISION = "P3W1_CALIBRATION_UNIT_PASS"
AGGREGATE_DECISION = "P3W1_CALIBRATION_AGGREGATE_PASS_PENDING_REVIEW"
FULL_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
TOL = 1e-9
CALIBRATION_FORWARD_BATCH_SIZE = 8
LOGICAL_UNITS_PER_SEED = 1
LOGICAL_UNIT_SCOPE = "COMPLETE_AUTHORITATIVE_TRAIN_SPLIT"
EXPECTED_DEV_RATIO = 0.2
EXPECTED_SPLIT_SEED = 174
EXPECTED_CONFIGURATION = {
    "architecture": "v6b_minimal",
    "backbone": "mamba",
    "model_name": "state-spaces/mamba-130m-hf",
    "max_length": 128,
    "device": "cuda",
    "flag_source": "controlled_heuristic",
    "freeze_encoder": True,
    "reason_router_epsilon": 1e-8,
    "train_batch_size": None,
    "balanced_sampler": False,
    "weighted_label_loss": False,
    "class_weighting": "none",
}
UNIT_REQUIRED_FIELDS = {
    "schema_version", "status", "seed", "unit_index", "unit_scope",
    "ordered_train_row_count", "ordered_train_row_identity_hash",
    "model_mode", "measurement_arm", "measurement_gradient_ownership",
    "reason_loss_weight_placeholder", "architecture", "backbone", "model_name",
    "max_length", "device", "flag_source", "freeze_encoder",
    "reason_router_epsilon", "train_batch_size", "balanced_sampler",
    "weighted_label_loss", "class_weighting", "calibration_forward_batch_size",
    "logical_units_per_seed", "logical_unit_scope",
    "fresh_initialization", "checkpoint_loaded", "gradient_tracking_enabled",
    "before_backward", "before_optimizer_step", "before_scheduler_step",
    "parameter_update_count", "optimizer_step_executed", "scheduler_step_executed",
    "dev_forward_executed", "calibration_data_scope", "train_reason_supervision_built",
    "dev_reason_supervision_built", "dev_inputs_accessed_for_calibration",
    "dev_labels_used_for_calibration", "dev_counts_used_for_gate",
    "dev_metrics_used_for_calibration", "a0_reference_predictions_required",
    "a0_reference_predictions_accessed", "a0_predictions_used_for_calibration",
    "a0_logits_used_for_calibration", "a0_metrics_used_for_calibration",
    "a0_checkpoint_used_for_calibration", "external_eval_executed", "normal_training_report_written",
    "causal_checkpoint_written", "final_loss_mean", "final_applicable_count",
    "final_loss_sum_reconstructed", "final_loss_finite", "reason_loss_mean",
    "reason_eligible_count", "reason_loss_sum_reconstructed", "reason_loss_finite",
    "dataset_path", "dataset_sha256", "sidecar_path", "sidecar_semantic_sha256",
    "expected_sidecar_semantic_sha256", "sidecar_semantic_sha256_verified",
    "split_seed", "dev_ratio", "execution_commit", "declared_execution_commit",
    "execution_commit_verified", "decision",
}


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"unit artifact must be a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"P3W1_AGGREGATE_GATE: {message}")


def _finite_positive(value: Any, name: str) -> float:
    _require(type(value) in {int, float}, f"{name} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and result > 0.0, f"{name} must be finite and > 0")
    return result


def _finite_dev_ratio(value: Any, name: str) -> float:
    _require(type(value) in {int, float}, f"{name} must be numeric")
    result = float(value)
    _require(
        math.isfinite(result)
        and math.isclose(result, EXPECTED_DEV_RATIO, rel_tol=0.0, abs_tol=1e-12),
        f"{name} must be exactly 0.2",
    )
    return result


def _positive_int(value: Any, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be an integer > 0")
    return value


def _exact_int(value: Any, name: str) -> int:
    _require(type(value) is int, f"{name} must be an exact integer")
    return value


def _non_empty_str(value: Any, name: str) -> str:
    _require(type(value) is str and value != "", f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    value = _non_empty_str(value, name)
    _require(SHA256_RE.fullmatch(value) is not None, f"{name} must be a 64-hex SHA256")
    return value


def _full_sha(value: Any, name: str) -> str:
    value = _non_empty_str(value, name)
    _require(FULL_SHA_RE.fullmatch(value) is not None, f"{name} must be a 40-hex commit")
    return value


def _validate_expected_identity(
    *,
    expected_execution_commit: str,
    expected_dataset_sha256: str,
    expected_sidecar_semantic_sha256: str,
    expected_ordered_train_row_count: int,
    expected_ordered_train_row_identity_hash: str,
    expected_dev_ratio: float,
    expected_split_seed: int,
) -> None:
    _full_sha(expected_execution_commit, "expected execution commit")
    _sha256(expected_dataset_sha256, "expected dataset sha256")
    _sha256(expected_sidecar_semantic_sha256, "expected sidecar sha256")
    _positive_int(expected_ordered_train_row_count, "expected ordered train row count")
    _sha256(expected_ordered_train_row_identity_hash, "expected ordered train identity hash")
    _finite_dev_ratio(expected_dev_ratio, "expected dev ratio")
    _require(type(expected_split_seed) is int, "expected split seed must be an exact integer")
    _require(expected_split_seed == EXPECTED_SPLIT_SEED, "expected split seed must be exactly 174")


def validate_unit_artifact(
    unit: dict[str, Any],
    *,
    expected_execution_commit: str,
    expected_dataset_sha256: str,
    expected_sidecar_semantic_sha256: str,
    expected_split_seed: int,
    expected_ordered_train_row_count: int,
    expected_ordered_train_row_identity_hash: str,
    expected_dev_ratio: float,
) -> dict[str, Any]:
    missing = sorted(UNIT_REQUIRED_FIELDS - unit.keys())
    _require(not missing, f"missing unit fields: {missing}")
    for key in (
        "schema_version", "status", "unit_scope", "logical_unit_scope", "model_mode",
        "measurement_arm", "measurement_gradient_ownership", "ordered_train_row_identity_hash",
        "architecture", "backbone", "model_name", "device", "flag_source", "class_weighting",
        "calibration_data_scope", "dataset_path", "dataset_sha256", "sidecar_path", "sidecar_semantic_sha256",
        "expected_sidecar_semantic_sha256", "execution_commit", "declared_execution_commit",
        "decision",
    ):
        _non_empty_str(unit.get(key), key)
    _require(unit.get("schema_version") == UNIT_SCHEMA, "unit schema mismatch")
    _require(unit.get("status") == "PASS", "unit status must be PASS")
    _require(unit.get("decision") == UNIT_DECISION, "unit decision mismatch")
    _require(unit.get("calibration_data_scope") == "TRAIN_ONLY", "calibration_data_scope mismatch")
    _require(unit.get("train_reason_supervision_built") is True, "train_reason_supervision_built mismatch")
    for key in (
        "dev_reason_supervision_built",
        "dev_inputs_accessed_for_calibration",
        "dev_labels_used_for_calibration",
        "dev_counts_used_for_gate",
        "dev_metrics_used_for_calibration",
        "a0_reference_predictions_required",
        "a0_reference_predictions_accessed",
        "a0_predictions_used_for_calibration",
        "a0_logits_used_for_calibration",
        "a0_metrics_used_for_calibration",
        "a0_checkpoint_used_for_calibration",
    ):
        _require(unit.get(key) is False, f"{key} mismatch")
    for key, expected in EXPECTED_CONFIGURATION.items():
        if key in {"freeze_encoder", "balanced_sampler", "weighted_label_loss"}:
            _require(unit.get(key) is expected, f"{key} mismatch")
        elif key == "train_batch_size":
            _require(unit.get(key) is None, "train_batch_size mismatch")
        elif key == "max_length":
            _require(type(unit.get(key)) is int and unit.get(key) == expected, "max_length mismatch")
        elif key == "reason_router_epsilon":
            _require(
                type(unit.get(key)) in {int, float}
                and math.isfinite(float(unit.get(key)))
                and math.isclose(float(unit.get(key)), expected, rel_tol=0.0, abs_tol=0.0),
                "reason_router_epsilon mismatch",
            )
        else:
            _require(unit.get(key) == expected, f"{key} mismatch")
    seed = _exact_int(unit.get("seed"), "seed")
    split_seed = _exact_int(unit.get("split_seed"), "split_seed")
    _require(split_seed == expected_split_seed, "split seed mismatch")
    _require(unit.get("execution_commit") == expected_execution_commit, "execution commit mismatch")
    _require(unit.get("declared_execution_commit") == expected_execution_commit, "declared execution commit mismatch")
    _require(unit.get("execution_commit_verified") is True, "execution_commit_verified mismatch")
    _require(unit.get("dataset_sha256") == expected_dataset_sha256, "dataset sha256 mismatch")
    _require(unit.get("sidecar_semantic_sha256") == expected_sidecar_semantic_sha256, "sidecar semantic sha256 mismatch")
    _require(unit.get("expected_sidecar_semantic_sha256") == expected_sidecar_semantic_sha256, "expected sidecar semantic sha256 mismatch")
    _require(unit.get("sidecar_semantic_sha256_verified") is True, "sidecar_semantic_sha256_verified mismatch")
    _full_sha(unit.get("execution_commit"), "execution_commit")
    _full_sha(unit.get("declared_execution_commit"), "declared_execution_commit")
    _sha256(unit.get("dataset_sha256"), "dataset_sha256")
    _sha256(unit.get("sidecar_semantic_sha256"), "sidecar_semantic_sha256")
    _sha256(unit.get("expected_sidecar_semantic_sha256"), "expected_sidecar_semantic_sha256")
    _require(
        type(unit.get("reason_loss_weight_placeholder")) in {int, float}
        and unit.get("reason_loss_weight_placeholder") == 0.0,
        "reason_loss_weight_placeholder mismatch",
    )
    exact_values = {
        "unit_index": 0,
        "unit_scope": LOGICAL_UNIT_SCOPE,
        "logical_unit_scope": LOGICAL_UNIT_SCOPE,
        "model_mode": "train",
        "measurement_arm": "conditional_first_blocker",
        "measurement_gradient_ownership": "explicit_local",
        "calibration_forward_batch_size": CALIBRATION_FORWARD_BATCH_SIZE,
        "logical_units_per_seed": LOGICAL_UNITS_PER_SEED,
        "parameter_update_count": 0,
    }
    for key, expected in exact_values.items():
        _require(unit.get(key) == expected, f"{key} mismatch")
    for key in ("unit_index", "calibration_forward_batch_size", "logical_units_per_seed", "parameter_update_count"):
        _exact_int(unit.get(key), key)
    for key, expected in {
        "fresh_initialization": True,
        "checkpoint_loaded": False,
        "gradient_tracking_enabled": False,
        "before_backward": True,
        "before_optimizer_step": True,
        "before_scheduler_step": True,
        "optimizer_step_executed": False,
        "scheduler_step_executed": False,
        "dev_forward_executed": False,
        "external_eval_executed": False,
        "normal_training_report_written": False,
        "causal_checkpoint_written": False,
        "final_loss_finite": True,
        "reason_loss_finite": True,
    }.items():
        _require(unit.get(key) is expected, f"{key} mismatch")
    row_hash = _sha256(unit.get("ordered_train_row_identity_hash"), "ordered_train_row_identity_hash")
    row_count = _positive_int(unit.get("ordered_train_row_count"), "ordered_train_row_count")
    _require(row_count == expected_ordered_train_row_count, "ordered train row count does not match expected authority")
    _require(row_hash == expected_ordered_train_row_identity_hash, "ordered train identity hash does not match expected authority")
    dev_ratio = _finite_dev_ratio(unit.get("dev_ratio"), "dev_ratio")
    _require(
        math.isclose(dev_ratio, float(expected_dev_ratio), rel_tol=0.0, abs_tol=1e-12),
        "unit dev ratio does not match expected authority",
    )
    final_count = _positive_int(unit.get("final_applicable_count"), "final_applicable_count")
    reason_count = _positive_int(unit.get("reason_eligible_count"), "reason_eligible_count")
    final_mean = _finite_positive(unit.get("final_loss_mean"), "final_loss_mean")
    reason_mean = _finite_positive(unit.get("reason_loss_mean"), "reason_loss_mean")
    final_sum = _finite_positive(unit.get("final_loss_sum_reconstructed"), "final_loss_sum_reconstructed")
    reason_sum = _finite_positive(unit.get("reason_loss_sum_reconstructed"), "reason_loss_sum_reconstructed")
    _require(math.isclose(final_sum, final_mean * final_count, rel_tol=TOL, abs_tol=TOL), "final reconstructed sum mismatch")
    _require(math.isclose(reason_sum, reason_mean * reason_count, rel_tol=TOL, abs_tol=TOL), "reason reconstructed sum mismatch")
    return {
        "seed": seed,
        "row_count": row_count,
        "row_hash": row_hash,
        "final_count": final_count,
        "reason_count": reason_count,
        "final_sum": final_sum,
        "reason_sum": reason_sum,
        "split_seed": split_seed,
    }


def build_aggregate(
    *,
    unit_paths: list[Path],
    output_json: Path,
    expected_execution_commit: str,
    expected_dataset_sha256: str,
    expected_sidecar_semantic_sha256: str,
    expected_split_seed: int,
    expected_ordered_train_row_count: int,
    expected_ordered_train_row_identity_hash: str,
    expected_dev_ratio: float,
) -> dict[str, Any]:
    _validate_expected_identity(
        expected_execution_commit=expected_execution_commit,
        expected_dataset_sha256=expected_dataset_sha256,
        expected_sidecar_semantic_sha256=expected_sidecar_semantic_sha256,
        expected_ordered_train_row_count=expected_ordered_train_row_count,
        expected_ordered_train_row_identity_hash=expected_ordered_train_row_identity_hash,
        expected_dev_ratio=expected_dev_ratio,
        expected_split_seed=expected_split_seed,
    )
    _require(len(unit_paths) == 3, "exactly three --unit-json paths are required")
    units = [_load_json(path) for path in unit_paths]
    summaries = [
        validate_unit_artifact(
            unit,
            expected_execution_commit=expected_execution_commit,
            expected_dataset_sha256=expected_dataset_sha256,
            expected_sidecar_semantic_sha256=expected_sidecar_semantic_sha256,
            expected_split_seed=expected_split_seed,
            expected_ordered_train_row_count=expected_ordered_train_row_count,
            expected_ordered_train_row_identity_hash=expected_ordered_train_row_identity_hash,
            expected_dev_ratio=expected_dev_ratio,
        )
        for unit in units
    ]
    seeds = [summary["seed"] for summary in summaries]
    _require(sorted(seeds) == EXPECTED_SEEDS, "seeds must be exactly [180, 181, 182]")
    _require(len(set(seeds)) == 3, "duplicate calibration seed")
    row_counts = {summary["row_count"] for summary in summaries}
    row_hashes = {summary["row_hash"] for summary in summaries}
    split_seeds = {summary["split_seed"] for summary in summaries}
    _require(len(row_counts) == 1, "ordered train row count mismatch")
    _require(len(row_hashes) == 1, "ordered train identity hash mismatch")
    _require(len(split_seeds) == 1, "split seed mismatch")
    total_final_count = sum(summary["final_count"] for summary in summaries)
    total_reason_count = sum(summary["reason_count"] for summary in summaries)
    total_final_loss_sum = sum(summary["final_sum"] for summary in summaries)
    total_reason_loss_sum = sum(summary["reason_sum"] for summary in summaries)
    mu_final = total_final_loss_sum / total_final_count
    mu_reason = total_reason_loss_sum / total_reason_count
    resolved_reason_loss_weight = mu_final / mu_reason
    _require(math.isfinite(resolved_reason_loss_weight) and resolved_reason_loss_weight > 0.0, "resolved weight must be finite and > 0")
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS",
        "calibration_seeds": EXPECTED_SEEDS,
        "seed_unit_paths": [str(path) for path in unit_paths],
        "seed_unit_artifact_sha256": {str(path): _file_sha256(path) for path in unit_paths},
        "execution_commit": expected_execution_commit,
        "dataset_sha256": expected_dataset_sha256,
        "sidecar_semantic_sha256": expected_sidecar_semantic_sha256,
        "split_seed": next(iter(split_seeds)),
        "expected_split_seed": EXPECTED_SPLIT_SEED,
        "split_seed_verified": True,
        **EXPECTED_CONFIGURATION,
        "ordered_train_row_count": next(iter(row_counts)),
        "ordered_train_row_identity_hash": next(iter(row_hashes)),
        "expected_ordered_train_row_count": expected_ordered_train_row_count,
        "expected_ordered_train_row_identity_hash": expected_ordered_train_row_identity_hash,
        "expected_dev_ratio": float(expected_dev_ratio),
        "calibration_forward_batch_size": CALIBRATION_FORWARD_BATCH_SIZE,
        "logical_units_per_seed": LOGICAL_UNITS_PER_SEED,
        "logical_unit_scope": LOGICAL_UNIT_SCOPE,
        "calibration_data_scope": "TRAIN_ONLY",
        "all_train_reason_supervision_built": True,
        "all_dev_reason_supervision_absent": True,
        "all_dev_inputs_unaccessed": True,
        "all_dev_labels_unused": True,
        "all_dev_counts_unused": True,
        "all_dev_metrics_unused": True,
        "all_a0_reference_predictions_unrequired": True,
        "all_a0_reference_predictions_unaccessed": True,
        "all_a0_predictions_unused": True,
        "all_a0_logits_unused": True,
        "all_a0_metrics_unused": True,
        "all_a0_checkpoints_unused": True,
        "total_final_count": total_final_count,
        "total_reason_count": total_reason_count,
        "total_final_loss_sum": total_final_loss_sum,
        "total_reason_loss_sum": total_reason_loss_sum,
        "mu_final": mu_final,
        "mu_reason": mu_reason,
        "resolved_reason_loss_weight": resolved_reason_loss_weight,
        "all_three_seeds_present": True,
        "all_unit_gates_pass": True,
        "all_sidecar_hashes_verified": True,
        "all_execution_commits_verified": True,
        "nonfinite_count": 0,
        "A1_A3_common_weight": True,
        "A1_A3_released": False,
        "decision": AGGREGATE_DECISION,
    }


def write_json_atomic_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"P3W1_AGGREGATE_WRITE_REFUSED: destination exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with temporary_path.open("xb") as handle:
            handle.write(data)
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"P3W1_AGGREGATE_WRITE_REFUSED: destination exists: {path}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-json", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-execution-commit", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-sidecar-semantic-sha256", required=True)
    parser.add_argument("--expected-split-seed", type=int, required=True)
    parser.add_argument("--expected-ordered-train-row-count", type=int, required=True)
    parser.add_argument("--expected-ordered-train-row-identity-hash", required=True)
    parser.add_argument("--expected-dev-ratio", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    aggregate = build_aggregate(
        unit_paths=args.unit_json,
        output_json=args.output_json,
        expected_execution_commit=args.expected_execution_commit,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_sidecar_semantic_sha256=args.expected_sidecar_semantic_sha256,
        expected_split_seed=args.expected_split_seed,
        expected_ordered_train_row_count=args.expected_ordered_train_row_count,
        expected_ordered_train_row_identity_hash=args.expected_ordered_train_row_identity_hash,
        expected_dev_ratio=args.expected_dev_ratio,
    )
    write_json_atomic_no_overwrite(args.output_json, aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())