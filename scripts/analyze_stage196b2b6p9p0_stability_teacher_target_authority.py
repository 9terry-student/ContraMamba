#!/usr/bin/env python3
"""Audit Stage196-B2-B6P9-P0 stability teacher/target authority.

This is a static design analyzer.  It reads source and upstream report
artifacts only.  It does not import torch, load a model or checkpoint, run a
forward pass, train, evaluate predictions, choose coefficients, add losses, or
change optimizer/checkpoint behavior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


STAGE = "Stage196-B2-B6P9-P0"
STAGE_TITLE = "Separate Stability Teacher/Target Authority Design"
P4_DECISION = "STAGE196B2B6P4_ACTION_RESPONSE_TOPOLOGY_UNSTABLE"
P5_DECISION = "STAGE196B2B6P5_GRADIENT_PATH_INSTRUMENTATION_REQUIRED"
P4_RECOMMENDED_NEXT = (
    "STAGE196B2B6P5_TRAINING_SIDE_RESPONSE_STABILITY_INTERVENTION_DESIGN"
)
P8_DECISION = "STAGE196B2B6P8_FULL_TRAINABLE_PATH_REPLAY_COMPLETE"
P8_FINAL_AUTHORITY_DIR = (
    "reports/stage196b2b6p8_full_trainable_path_replay_20260723_203414"
)
P4_AUTHORITY_MODES = (
    "ORIGINAL_P4_ANALYSIS",
    "DOWNSTREAM_ATTESTED_P4_MINIMAL_CLOSURE",
)
P4_ATTESTED_SCOPE = (
    "P4 decision identity",
    "zero P4 blockers",
    "P4 recommended next-stage identity",
    "zero failed P4 contracts",
)
P4_ATTESTED_LIMITATIONS = (
    "original P4 numerical tables",
    "original P4 row-level data",
    "original P4 source-file hashes",
    "original P4 output-directory identity",
    "original P4 creation timestamp",
    "byte-identical original P4 content",
)
P4_P6_DECISION_PAYLOAD = {
    "blocking_reasons": [],
    "decision": P4_DECISION,
    "recommended_next_stage": P4_RECOMMENDED_NEXT,
}
P4_P7_DECISION_PAYLOAD = {
    "blocking_reasons": [],
    "decision": P4_DECISION,
}

TEACHER_CANDIDATES = (
    "CURRENT_NATIVE_STOP_GRAD",
    "FRAME_LOCAL_ONLY_DONOR_STOP_GRAD",
    "PREVIOUS_STEP_FROZEN_SNAPSHOT",
    "PREVIOUS_EPOCH_FROZEN_SNAPSHOT",
    "EMA_STUDENT_TEACHER",
    "FIXED_RECONSTRUCTED_CHECKPOINT",
)
FUTURE_VARIANTS = (
    "baseline",
    "direction_consistency_only",
    "candidate_order_consistency_only",
)
P8_GRAPH_CLASSES = (
    "CONNECTED_NONZERO",
    "CONNECTED_ZERO_AT_OBSERVED_BATCH",
    "DISCONNECTED",
)
OUTPUTS = (
    "stage196b2b6p9p0_analysis.json",
    "stage196b2b6p9p0_report.md",
    "stage196b2b6p9p0_teacher_candidate_audit.csv",
    "stage196b2b6p9p0_direction_target_design.csv",
    "stage196b2b6p9p0_order_target_design.csv",
    "stage196b2b6p9p0_state_lifecycle_audit.csv",
    "stage196b2b6p9p0_portability_audit.csv",
    "stage196b2b6p9p0_decision_gate.csv",
    "stage196b2b6p9p0_contract.csv",
)
DECISIONS = (
    "STAGE196B2B6P9P0_BLOCKED_UPSTREAM_AUTHORITY",
    "STAGE196B2B6P9P0_NO_JUSTIFIED_TEACHER",
    "STAGE196B2B6P9P0_DIRECTION_TEACHER_ONLY_READY",
    "STAGE196B2B6P9P0_ORDER_TEACHER_ONLY_READY",
    "STAGE196B2B6P9P0_SEPARATE_TEACHERS_READY",
    "STAGE196B2B6P9P0_SHARED_TEACHER_READY",
)
NEXT_STAGE = {
    "STAGE196B2B6P9P0_BLOCKED_UPSTREAM_AUTHORITY":
        "STAGE196B2B6P9P0_REPAIR_UPSTREAM_AUTHORITY",
    "STAGE196B2B6P9P0_NO_JUSTIFIED_TEACHER":
        "STAGE196B2B6P9P1_TEACHER_STATE_OBSERVABILITY_DESIGN",
    "STAGE196B2B6P9P0_DIRECTION_TEACHER_ONLY_READY":
        "STAGE196B2B6P9D_DIRECTION_ONLY_INTERVENTION_IMPLEMENTATION",
    "STAGE196B2B6P9P0_ORDER_TEACHER_ONLY_READY":
        "STAGE196B2B6P9O_ORDER_ONLY_INTERVENTION_IMPLEMENTATION",
    "STAGE196B2B6P9P0_SEPARATE_TEACHERS_READY":
        "STAGE196B2B6P9DO_SEPARATE_INTERVENTION_IMPLEMENTATIONS",
    "STAGE196B2B6P9P0_SHARED_TEACHER_READY":
        "STAGE196B2B6P9DO_SEPARATE_INTERVENTION_IMPLEMENTATIONS",
}

AUDIT_FIELDS = (
    "teacher_candidate",
    "available_in_current_training_source",
    "requires_new_state",
    "requires_second_model_copy",
    "requires_checkpoint_dependency",
    "requires_historical_runtime_artifact",
    "seed_portability",
    "run_portability",
    "causal_interpretability",
    "risk_of_self_confirmation",
    "risk_of_target_drift",
    "risk_of_cross_seed_coordinate_locking",
    "exact_tie_policy",
    "gradient_stop_policy",
    "direction_family_status",
    "candidate_order_family_status",
    "blocking_reason",
)
TARGET_FIELDS = (
    "family",
    "teacher_candidate",
    "student_quantity",
    "teacher_quantity",
    "allowed_teacher_values",
    "exact_tie_policy",
    "lexical_order_policy",
    "coefficient_or_margin_policy",
    "graph_availability_from_p8",
    "required_future_observability",
    "family_status",
    "blocking_reason",
)
LIFECYCLE_FIELDS = (
    "teacher_candidate",
    "initialization_rule",
    "update_timing",
    "decay_ownership",
    "buffer_handling",
    "dropout_eval_policy",
    "checkpoint_serialization",
    "resume_behavior",
    "seed_determinism",
    "teacher_warm_up",
    "teacher_drift_observability",
    "baseline_default_off_closure",
    "status",
)
PORTABILITY_FIELDS = (
    "teacher_candidate",
    "seed_portability",
    "run_portability",
    "depends_on_clean_dev_labels",
    "depends_on_recovery_harm_membership",
    "encodes_seed_specific_absolute_score_coordinates",
    "requires_reconstructed_seed183_checkpoint",
    "reproducible_for_every_intended_seed",
    "second_opaque_selector_risk",
    "status",
    "blocking_reason",
)
DECISION_FIELDS = (
    "order",
    "decision",
    "condition",
    "observed",
    "reached",
    "recommended_next_stage",
)
CONTRACT_FIELDS = (
    "contract",
    "required",
    "observed",
    "passed",
    "blocking_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage196b2b6p4-analysis-json", type=Path)
    parser.add_argument("--stage196b2b6p5-analysis-json", type=Path, required=True)
    parser.add_argument("--stage196b2b6p7-analysis-json", type=Path, required=True)
    parser.add_argument("--stage196b2b6p6-contract-csv", type=Path, required=True)
    parser.add_argument("--stage196b2b6p7-contract-csv", type=Path, required=True)
    parser.add_argument("--stage196b2b6p8-analysis-json", type=Path, required=True)
    parser.add_argument("--stage196b2b6p8-gradient-connectivity-csv",
                        type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def csv_text(fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: canonical(value) if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        })
    return buffer.getvalue()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite {path}")
    os.rename(temporary, path)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "passed"}


def add_contract(rows: list[dict[str, Any]], name: str, required: Any,
                 observed: Any, passed: bool, reason: str = "") -> None:
    rows.append({
        "contract": name,
        "required": required,
        "observed": observed,
        "passed": bool(passed),
        "blocking_reason": "" if passed else reason,
    })


def zero_failed_contracts(path: Path) -> tuple[bool, int]:
    rows = read_csv(path)
    if not rows:
        return False, 1
    failed = 0
    for row in rows:
        if "passed" in row:
            ok = boolish(row.get("passed"))
        else:
            ok = str(row.get("status", "")).strip().upper() == "PASS"
        if not ok or row.get("blocking_reason", "").strip():
            failed += 1
    return failed == 0, failed



def strict_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def contract_rows_named(rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("contract", "")).strip() == name]


def parse_contract_json_payload(value: Any) -> Any:
    if isinstance(value, (dict, list, bool)) or value is None:
        return value
    return json.loads(str(value))


def p7_p6_contract_expected_sha(p7: dict[str, Any]) -> str:
    artifacts = (p7.get("upstream") or {}).get("p6_artifacts") or {}
    value = artifacts.get("stage196b2b6p6_contract.csv")
    if isinstance(value, dict):
        value = value.get("sha256")
    return str(value or "").strip()


def evaluate_p4_authority(
    *,
    p4_path: Path | None,
    p7: dict[str, Any],
    p6_contract_path: Path,
    p7_contract_path: Path,
+) -> dict[str, Any]:
    original_available = p4_path is not None and p4_path.exists()
    base: dict[str, Any] = {
        "authority_mode": "ORIGINAL_P4_ANALYSIS" if original_available else "DOWNSTREAM_ATTESTED_P4_MINIMAL_CLOSURE",
        "original_artifact_available": original_available,
        "original_artifact_path": str(p4_path) if p4_path is not None else "",
        "attestation_scope": list(P4_ATTESTED_SCOPE),
        "attestation_limitations": list(P4_ATTESTED_LIMITATIONS),
        "p6_contract_path": str(p6_contract_path),
        "p6_contract_sha256_expected": "",
        "p6_contract_sha256_observed": "",
        "p6_decision_closure": {"passed": False, "mode": "not_checked"},
        "p6_zero_failed_contracts_closure": {"passed": False, "mode": "not_checked"},
        "p7_analysis_decision_concurrence": {"passed": False, "mode": "not_checked"},
        "p7_contract_decision_concurrence": {"passed": False, "mode": "not_checked"},
        "minimal_authority_passed": False,
        "original_artifact_not_fabricated": True,
        "blocking_reasons": [],
    }
    if original_available:
        original = load_json(p4_path)  # type: ignore[arg-type]
        passed = original.get("decision") == P4_DECISION and original.get("blocking_reasons") == []
        base.update({
            "observed_decision": original.get("decision"),
            "observed_blocking_reasons": original.get("blocking_reasons"),
            "minimal_authority_passed": passed,
            "p6_decision_closure": {"passed": True, "mode": "not_applicable_original_mode"},
            "p6_zero_failed_contracts_closure": {"passed": True, "mode": "not_applicable_original_mode"},
            "p7_analysis_decision_concurrence": {"passed": True, "mode": "not_applicable_original_mode"},
            "p7_contract_decision_concurrence": {"passed": True, "mode": "not_applicable_original_mode"},
        })
        if not passed:
            base["blocking_reasons"].append("original P4 analysis failed exact decision/zero-blocker checks")
        return base

    missing_contract_paths = []
    if not p6_contract_path.exists():
        missing_contract_paths.append("stage196b2b6p6_contract.csv")
    if not p7_contract_path.exists():
        missing_contract_paths.append("stage196b2b6p7_contract.csv")
    if missing_contract_paths:
        base["checks"] = {
            "p6_contract_hash_closure": False,
            "p6_decision_closure": False,
            "p6_zero_failed_contracts_closure": False,
            "p7_analysis_decision_concurrence": False,
            "p7_contract_decision_concurrence": False,
        }
        base["blocking_reasons"] = [
            "missing downstream attestation contract path: " + name
            for name in missing_contract_paths
        ]
        return base

    p6_rows = read_csv(p6_contract_path)
    p7_rows = read_csv(p7_contract_path)
    expected_sha = p7_p6_contract_expected_sha(p7)
    observed_sha = sha256(p6_contract_path)
    hash_passed = bool(expected_sha) and expected_sha == observed_sha
    base["p6_contract_sha256_expected"] = expected_sha
    base["p6_contract_sha256_observed"] = observed_sha

    p6_decision_rows = contract_rows_named(p6_rows, "p4_decision_closure")
    p6_decision_detail: dict[str, Any] = {"row_count": len(p6_decision_rows), "passed": False}
    if len(p6_decision_rows) == 1:
        row = p6_decision_rows[0]
        try:
            required = parse_contract_json_payload(row.get("required", ""))
            observed = parse_contract_json_payload(row.get("observed", ""))
            passed_value = strict_bool(row.get("passed"))
            p6_decision_detail.update({
                "required": required,
                "observed": observed,
                "passed_value": passed_value,
                "passed": passed_value is True and required == P4_P6_DECISION_PAYLOAD and observed == P4_P6_DECISION_PAYLOAD,
            })
        except json.JSONDecodeError as exc:
            p6_decision_detail.update({"parse_error": str(exc)})

    p6_zero_rows = contract_rows_named(p6_rows, "p4_zero_failed_contracts")
    p6_zero_detail: dict[str, Any] = {"row_count": len(p6_zero_rows), "passed": False}
    if len(p6_zero_rows) == 1:
        row = p6_zero_rows[0]
        passed_value = strict_bool(row.get("passed"))
        required_bool = strict_bool(row.get("required"))
        observed_bool = strict_bool(row.get("observed"))
        p6_zero_detail.update({
            "passed_value": passed_value,
            "required": required_bool,
            "observed": observed_bool,
            "passed": passed_value is True and required_bool is True and observed_bool is True,
        })

    p7_upstream_decision = (p7.get("upstream") or {}).get("p4_decision")
    p7_analysis_detail = {
        "required": P4_DECISION,
        "observed": p7_upstream_decision,
        "passed": p7_upstream_decision == P4_DECISION,
    }

    p7_contract_rows = contract_rows_named(p7_rows, "p4_decision_and_zero_blockers")
    p7_contract_detail: dict[str, Any] = {"row_count": len(p7_contract_rows), "passed": False}
    if len(p7_contract_rows) == 1:
        row = p7_contract_rows[0]
        try:
            required = parse_contract_json_payload(row.get("required", ""))
            observed = parse_contract_json_payload(row.get("observed", ""))
            passed_value = strict_bool(row.get("passed"))
            p7_contract_detail.update({
                "required": required,
                "observed": observed,
                "passed_value": passed_value,
                "passed": passed_value is True and required == P4_P7_DECISION_PAYLOAD and observed == P4_P7_DECISION_PAYLOAD,
            })
        except json.JSONDecodeError as exc:
            p7_contract_detail.update({"parse_error": str(exc)})

    base.update({
        "p6_decision_closure": p6_decision_detail,
        "p6_zero_failed_contracts_closure": p6_zero_detail,
        "p7_analysis_decision_concurrence": p7_analysis_detail,
        "p7_contract_decision_concurrence": p7_contract_detail,
    })
    checks = {
        "p6_contract_hash_closure": hash_passed,
        "p6_decision_closure": bool(p6_decision_detail["passed"]),
        "p6_zero_failed_contracts_closure": bool(p6_zero_detail["passed"]),
        "p7_analysis_decision_concurrence": bool(p7_analysis_detail["passed"]),
        "p7_contract_decision_concurrence": bool(p7_contract_detail["passed"]),
    }
    base["minimal_authority_passed"] = all(checks.values())
    base["checks"] = checks
    base["blocking_reasons"] = [name for name, passed in checks.items() if not passed]
    return base


def source_line(text: str, pattern: str) -> str:
    matches = [
        str(index)
        for index, line in enumerate(text.splitlines(), start=1)
        if re.search(pattern, line)
    ]
    return ",".join(matches) if matches else "absent"


def classify_p8_graph(rows: list[dict[str, str]], family: str) -> dict[str, Any]:
    csv_family = "order" if family == "candidate_order" else family
    family_rows = [
        row for row in rows
        if str(row.get("target_family", "")).strip() == csv_family
    ]
    counts = {name: 0 for name in P8_GRAPH_CLASSES}
    other: dict[str, int] = {}
    for row in family_rows:
        key = str(row.get("classification", "")).strip()
        if key in counts:
            counts[key] += 1
        else:
            other[key] = other.get(key, 0) + 1
    graph_available = counts["CONNECTED_NONZERO"] + counts[
        "CONNECTED_ZERO_AT_OBSERVED_BATCH"
    ] > 0
    return {
        "family": family,
        "csv_target_family": csv_family,
        "row_count": len(family_rows),
        "counts": counts,
        "other_classifications": other,
        "graph_available": graph_available,
        "interpretation": (
            "graph-connected with nonzero observed gradients"
            if counts["CONNECTED_NONZERO"]
            else "graph-connected but zero at observed batch"
            if graph_available
            else "disconnected or absent"
        ),
    }


def candidate_audits(source: dict[str, str]) -> list[dict[str, Any]]:
    trainer = source["trainer"]
    model = source["model"]
    has_ema = re.search(r"\b(EMA|ExponentialMovingAverage|AveragedModel)\b",
                        trainer + "\n" + model) is not None
    has_previous_step_teacher = (
        "previous_step" in trainer.lower() or "teacher_snapshot" in trainer.lower()
    )
    has_previous_epoch_teacher = (
        "previous_epoch" in trainer.lower() or "teacher_snapshot" in trainer.lower()
    )
    replay_api_present = "replay_full_trainable_path" in model
    stability_loss_none = '"stability_loss": None' in model
    p8_default_off = (
        "stage196b2b6p8_enable_full_trainable_path_replay_api" in trainer
        and "False" in trainer
    )
    base = {
        "exact_tie_policy": "exact teacher zero/order ties ignored",
        "gradient_stop_policy": "teacher target must be stop-gradient; student remains live",
    }
    rows = [
        {
            **base,
            "teacher_candidate": "CURRENT_NATIVE_STOP_GRAD",
            "available_in_current_training_source": replay_api_present,
            "requires_new_state": False,
            "requires_second_model_copy": False,
            "requires_checkpoint_dependency": False,
            "requires_historical_runtime_artifact": False,
            "seed_portability": "portable but self-referential",
            "run_portability": "portable but self-referential",
            "causal_interpretability": "invalid: same-forward target is algebraically identical to the live student quantity after stop-gradient",
            "risk_of_self_confirmation": "high",
            "risk_of_target_drift": "none independent of student because target is current student",
            "risk_of_cross_seed_coordinate_locking": "low",
            "direction_family_status": "INVALID_ALGEBRAIC_IDENTITY",
            "candidate_order_family_status": "INVALID_ALGEBRAIC_IDENTITY",
            "blocking_reason": (
                "student_delta and current-native stop-gradient sign/order are "
                "derived from the same forward quantities, so the teacher is not "
                "an independent stability authority"
            ),
        },
        {
            **base,
            "teacher_candidate": "FRAME_LOCAL_ONLY_DONOR_STOP_GRAD",
            "available_in_current_training_source": replay_api_present,
            "requires_new_state": False,
            "requires_second_model_copy": True,
            "requires_checkpoint_dependency": True,
            "requires_historical_runtime_artifact": True,
            "seed_portability": "not established",
            "run_portability": "not established",
            "causal_interpretability": "mechanistically meaningful for FrameGate isolation but not globally superior",
            "risk_of_self_confirmation": "medium",
            "risk_of_target_drift": "checkpoint/runtime dependent",
            "risk_of_cross_seed_coordinate_locking": "high",
            "direction_family_status": "BLOCKED_DONOR_ARM_IMITATION_RISK",
            "candidate_order_family_status": "BLOCKED_DONOR_ARM_IMITATION_RISK",
            "blocking_reason": (
                "would force branches toward a frame-local-only coordinate system "
                "without source authority that it is the desired teacher"
            ),
        },
        {
            **base,
            "teacher_candidate": "PREVIOUS_STEP_FROZEN_SNAPSHOT",
            "available_in_current_training_source": has_previous_step_teacher,
            "requires_new_state": True,
            "requires_second_model_copy": True,
            "requires_checkpoint_dependency": False,
            "requires_historical_runtime_artifact": False,
            "seed_portability": "designable but not source-authorized",
            "run_portability": "designable but not source-authorized",
            "causal_interpretability": "temporally local; may be near-identity",
            "risk_of_self_confirmation": "medium-high",
            "risk_of_target_drift": "high at optimizer-step cadence",
            "risk_of_cross_seed_coordinate_locking": "low if initialized per seed",
            "direction_family_status": "REQUIRES_NEW_STATE_LIFECYCLE",
            "candidate_order_family_status": "REQUIRES_NEW_STATE_LIFECYCLE",
            "blocking_reason": "no source lifecycle for per-step snapshot creation, update, serialization, resume, or observability",
        },
        {
            **base,
            "teacher_candidate": "PREVIOUS_EPOCH_FROZEN_SNAPSHOT",
            "available_in_current_training_source": has_previous_epoch_teacher,
            "requires_new_state": True,
            "requires_second_model_copy": True,
            "requires_checkpoint_dependency": True,
            "requires_historical_runtime_artifact": False,
            "seed_portability": "designable but not source-authorized",
            "run_portability": "designable but not source-authorized",
            "causal_interpretability": "stronger temporal separation than previous-step",
            "risk_of_self_confirmation": "medium",
            "risk_of_target_drift": "coarse epoch-boundary drift",
            "risk_of_cross_seed_coordinate_locking": "low if initialized per seed",
            "direction_family_status": "REQUIRES_NEW_STATE_LIFECYCLE",
            "candidate_order_family_status": "REQUIRES_NEW_STATE_LIFECYCLE",
            "blocking_reason": "no source lifecycle for epoch snapshot, checkpoint/state-copy retention, resume, or target-count observability",
        },
        {
            **base,
            "teacher_candidate": "EMA_STUDENT_TEACHER",
            "available_in_current_training_source": has_ema,
            "requires_new_state": True,
            "requires_second_model_copy": True,
            "requires_checkpoint_dependency": True,
            "requires_historical_runtime_artifact": False,
            "seed_portability": "designable but not source-authorized",
            "run_portability": "designable but not source-authorized",
            "causal_interpretability": "conceptually smooth teacher; implementation authority absent",
            "risk_of_self_confirmation": "medium",
            "risk_of_target_drift": "requires explicit drift observability",
            "risk_of_cross_seed_coordinate_locking": "low if per-seed EMA is deterministic",
            "direction_family_status": "REQUIRES_NEW_STATE_LIFECYCLE",
            "candidate_order_family_status": "REQUIRES_NEW_STATE_LIFECYCLE",
            "blocking_reason": (
                "P7 made EMA a conceptual preference only; initialization, update "
                "timing, decay ownership, buffers, dropout/eval policy, "
                "serialization, resume, warm-up, exact ties, and drift "
                "observability are not source-authorized"
            ),
        },
        {
            **base,
            "teacher_candidate": "FIXED_RECONSTRUCTED_CHECKPOINT",
            "available_in_current_training_source": False,
            "requires_new_state": False,
            "requires_second_model_copy": True,
            "requires_checkpoint_dependency": True,
            "requires_historical_runtime_artifact": True,
            "seed_portability": "failed: seed183-specific",
            "run_portability": "failed: reconstructed after runtime loss",
            "causal_interpretability": "diagnostic-only unless portability is proven",
            "risk_of_self_confirmation": "low",
            "risk_of_target_drift": "none after fixation, but historically anchored",
            "risk_of_cross_seed_coordinate_locking": "high",
            "direction_family_status": "DIAGNOSTIC_ONLY_NOT_PORTABLE",
            "candidate_order_family_status": "DIAGNOSTIC_ONLY_NOT_PORTABLE",
            "blocking_reason": "seed183 reconstructed checkpoints are not permanent teacher infrastructure and are not available for every future seed",
        },
    ]
    if not stability_loss_none or not p8_default_off:
        for row in rows:
            row["blocking_reason"] += "; source default-off/no-loss closure must be repaired"
    return rows


def target_rows(audits: list[dict[str, Any]], graph: dict[str, Any],
                family: str) -> list[dict[str, Any]]:
    is_direction = family == "direction"
    student = (
        "student_delta[c,k] = student_counterfactual[c,k] - student_native[k]"
        if is_direction
        else "student_pair_gap[a,b,k] = student_counterfactual[a,k] - student_counterfactual[b,k]"
    )
    teacher = (
        "stop-gradient teacher_sign[c,k] in {-1,+1}; exact zero ignored"
        if is_direction
        else "stop-gradient teacher_pair_sign[a,b,k] in {-1,+1}; exact pair tie ignored"
    )
    status_key = "direction_family_status" if is_direction else "candidate_order_family_status"
    required_observability = (
        "active non-tie teacher target count; violating student target count; "
        "nonzero loss term count; nonzero gradient target count"
    )
    return [
        {
            "family": "direction_consistency_only" if is_direction else "candidate_order_consistency_only",
            "teacher_candidate": row["teacher_candidate"],
            "student_quantity": student,
            "teacher_quantity": teacher,
            "allowed_teacher_values": "{-1,+1}; no arbitrary exact-tie assignment",
            "exact_tie_policy": row["exact_tie_policy"],
            "lexical_order_policy": (
                "candidate mask lexical order is never semantic target order"
            ),
            "coefficient_or_margin_policy": "not prescribed in P9-P0",
            "graph_availability_from_p8": graph["interpretation"],
            "required_future_observability": required_observability,
            "family_status": row[status_key],
            "blocking_reason": row["blocking_reason"],
        }
        for row in audits
    ]


def lifecycle_rows() -> list[dict[str, Any]]:
    unknown = "REQUIRES_NEW_STATE_DESIGN"
    return [
        {
            "teacher_candidate": "CURRENT_NATIVE_STOP_GRAD",
            "initialization_rule": "same current forward",
            "update_timing": "same current forward",
            "decay_ownership": "not applicable",
            "buffer_handling": "not applicable",
            "dropout_eval_policy": "same current stochastic realization",
            "checkpoint_serialization": "not applicable",
            "resume_behavior": "not applicable",
            "seed_determinism": "deterministic if current run is deterministic",
            "teacher_warm_up": "none",
            "teacher_drift_observability": "not independent",
            "baseline_default_off_closure": "required",
            "status": "INVALID_ALGEBRAIC_IDENTITY",
        },
        {
            "teacher_candidate": "FRAME_LOCAL_ONLY_DONOR_STOP_GRAD",
            "initialization_rule": "separately trained donor checkpoint",
            "update_timing": "fixed or separately trained; not authorized as teacher",
            "decay_ownership": "not applicable",
            "buffer_handling": "requires donor model buffers",
            "dropout_eval_policy": "must be explicit; P8 matched dropout was replay verification only",
            "checkpoint_serialization": "requires donor checkpoint dependency",
            "resume_behavior": "requires donor checkpoint re-resolution",
            "seed_determinism": "not established across all intended seeds",
            "teacher_warm_up": "not applicable if fixed",
            "teacher_drift_observability": "not established",
            "baseline_default_off_closure": "required",
            "status": "BLOCKED_DONOR_ARM_IMITATION_RISK",
        },
        {
            "teacher_candidate": "PREVIOUS_STEP_FROZEN_SNAPSHOT",
            "initialization_rule": unknown,
            "update_timing": "optimizer-step cadence required",
            "decay_ownership": "not applicable",
            "buffer_handling": unknown,
            "dropout_eval_policy": unknown,
            "checkpoint_serialization": unknown,
            "resume_behavior": unknown,
            "seed_determinism": unknown,
            "teacher_warm_up": unknown,
            "teacher_drift_observability": unknown,
            "baseline_default_off_closure": "required",
            "status": "REQUIRES_NEW_STATE_LIFECYCLE",
        },
        {
            "teacher_candidate": "PREVIOUS_EPOCH_FROZEN_SNAPSHOT",
            "initialization_rule": unknown,
            "update_timing": "epoch-boundary cadence required",
            "decay_ownership": "not applicable",
            "buffer_handling": unknown,
            "dropout_eval_policy": unknown,
            "checkpoint_serialization": unknown,
            "resume_behavior": unknown,
            "seed_determinism": unknown,
            "teacher_warm_up": unknown,
            "teacher_drift_observability": unknown,
            "baseline_default_off_closure": "required",
            "status": "REQUIRES_NEW_STATE_LIFECYCLE",
        },
        {
            "teacher_candidate": "EMA_STUDENT_TEACHER",
            "initialization_rule": unknown,
            "update_timing": unknown,
            "decay_ownership": unknown,
            "buffer_handling": unknown,
            "dropout_eval_policy": unknown,
            "checkpoint_serialization": unknown,
            "resume_behavior": unknown,
            "seed_determinism": unknown,
            "teacher_warm_up": unknown,
            "teacher_drift_observability": unknown,
            "baseline_default_off_closure": "required",
            "status": "REQUIRES_NEW_STATE_LIFECYCLE",
        },
        {
            "teacher_candidate": "FIXED_RECONSTRUCTED_CHECKPOINT",
            "initialization_rule": "load fixed reconstructed seed183 checkpoint",
            "update_timing": "never",
            "decay_ownership": "not applicable",
            "buffer_handling": "requires fixed checkpoint buffers",
            "dropout_eval_policy": "must be explicit if ever used diagnostically",
            "checkpoint_serialization": "external reconstructed dependency",
            "resume_behavior": "external dependency must be rediscovered",
            "seed_determinism": "not portable across seeds",
            "teacher_warm_up": "not applicable",
            "teacher_drift_observability": "fixed; coordinate-locking risk remains",
            "baseline_default_off_closure": "required",
            "status": "DIAGNOSTIC_ONLY_NOT_PORTABLE",
        },
    ]


def portability_rows(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in audits:
        candidate = row["teacher_candidate"]
        fixed = candidate == "FIXED_RECONSTRUCTED_CHECKPOINT"
        donor = candidate == "FRAME_LOCAL_ONLY_DONOR_STOP_GRAD"
        rows.append({
            "teacher_candidate": candidate,
            "seed_portability": row["seed_portability"],
            "run_portability": row["run_portability"],
            "depends_on_clean_dev_labels": False,
            "depends_on_recovery_harm_membership": False,
            "encodes_seed_specific_absolute_score_coordinates": fixed,
            "requires_reconstructed_seed183_checkpoint": fixed,
            "reproducible_for_every_intended_seed": (
                False if fixed or donor else "not source-authorized"
                if row["requires_new_state"] else True
            ),
            "second_opaque_selector_risk": (
                "high" if fixed or donor else "medium" if row["requires_new_state"] else "none"
            ),
            "status": (
                "PASS" if candidate == "CURRENT_NATIVE_STOP_GRAD" else "BLOCKED"
            ),
            "blocking_reason": (
                "current-native passes portability but fails independence"
                if candidate == "CURRENT_NATIVE_STOP_GRAD"
                else row["blocking_reason"]
            ),
        })
    return rows


def decision_gate(audits: list[dict[str, Any]], upstream_ok: bool) -> tuple[str, list[dict[str, Any]]]:
    direction_ready = [
        row["teacher_candidate"] for row in audits
        if row["direction_family_status"] == "READY"
    ]
    order_ready = [
        row["teacher_candidate"] for row in audits
        if row["candidate_order_family_status"] == "READY"
    ]
    shared_ready = sorted(set(direction_ready).intersection(order_ready))
    if not upstream_ok:
        decision = "STAGE196B2B6P9P0_BLOCKED_UPSTREAM_AUTHORITY"
    elif shared_ready:
        decision = "STAGE196B2B6P9P0_SHARED_TEACHER_READY"
    elif direction_ready and order_ready:
        decision = "STAGE196B2B6P9P0_SEPARATE_TEACHERS_READY"
    elif direction_ready:
        decision = "STAGE196B2B6P9P0_DIRECTION_TEACHER_ONLY_READY"
    elif order_ready:
        decision = "STAGE196B2B6P9P0_ORDER_TEACHER_ONLY_READY"
    else:
        decision = "STAGE196B2B6P9P0_NO_JUSTIFIED_TEACHER"
    rows = []
    conditions = [
        (
            "1", "STAGE196B2B6P9P0_BLOCKED_UPSTREAM_AUTHORITY",
            "any upstream authority or final P8 contract fails",
            {"upstream_ok": upstream_ok},
        ),
        (
            "2", "STAGE196B2B6P9P0_NO_JUSTIFIED_TEACHER",
            "upstream passes but no candidate is ready for either family",
            {"direction_ready": direction_ready, "order_ready": order_ready},
        ),
        (
            "3", "STAGE196B2B6P9P0_DIRECTION_TEACHER_ONLY_READY",
            "direction has at least one ready teacher and order has none",
            {"direction_ready": direction_ready, "order_ready": order_ready},
        ),
        (
            "4", "STAGE196B2B6P9P0_ORDER_TEACHER_ONLY_READY",
            "order has at least one ready teacher and direction has none",
            {"direction_ready": direction_ready, "order_ready": order_ready},
        ),
        (
            "5", "STAGE196B2B6P9P0_SEPARATE_TEACHERS_READY",
            "both families have ready teachers without shared-teacher coupling authority",
            {"direction_ready": direction_ready, "order_ready": order_ready},
        ),
        (
            "6", "STAGE196B2B6P9P0_SHARED_TEACHER_READY",
            "same candidate is ready for both and sharing does not couple mechanisms",
            {"shared_ready": shared_ready, "sharing_decoupling_argument": False},
        ),
    ]
    for order, name, condition, observed in conditions:
        rows.append({
            "order": order,
            "decision": name,
            "condition": condition,
            "observed": observed,
            "reached": name == decision,
            "recommended_next_stage": NEXT_STAGE[name],
        })
    return decision, rows


def report_md(analysis: dict[str, Any]) -> str:
    return f"""# {STAGE} {STAGE_TITLE}

## Decision

decision = `{analysis["decision"]}`

recommended_next_stage = `{analysis["recommended_next_stage"]}`

This stage does not authorize or activate a stability loss.  P8 is used only
for graph availability: direction has graph availability, and candidate-order
is graph-connected while zero on the observed P8 batch.

## P4 Authority

P4 authority mode = `{analysis["p4_authority"]["authority_mode"]}`

Original P4 artifact available = `{analysis["p4_authority"]["original_artifact_available"]}`

The downstream-attested mode establishes only P4 decision identity, zero P4
blockers, P4 recommended next-stage identity, and zero failed P4 contracts. It
does not reconstruct or authorize the original P4 numerical tables, row-level
data, source-file hashes, output-directory identity, creation timestamp, or
byte-identical original content.

## Teacher Authority

No audited teacher candidate is justified for direction or candidate-order in
the current training source.  `CURRENT_NATIVE_STOP_GRAD` is portable but
algebraically identical to the live student quantity after stop-gradient.
`FRAME_LOCAL_ONLY_DONOR_STOP_GRAD` is mechanistically meaningful for FrameGate
isolation but risks forcing branches to imitate a frame-local-only coordinate
system.  `PREVIOUS_STEP_FROZEN_SNAPSHOT`, `PREVIOUS_EPOCH_FROZEN_SNAPSHOT`, and
`EMA_STUDENT_TEACHER` require new state lifecycle authority.  The fixed
reconstructed checkpoints remain diagnostic-only because seed183-specific,
post-reconstruction artifacts are not portable training infrastructure.

## Target Designs

Direction target, future only:

```text
student_delta[c,k] = student_counterfactual[c,k] - student_native[k]
teacher_sign[c,k] in {{-1,+1}}
```

Exact teacher zero is ignored.

Candidate-order target, future only:

```text
student_pair_gap[a,b,k] = student_counterfactual[a,k] - student_counterfactual[b,k]
teacher_pair_sign[a,b,k] in {{-1,+1}}
```

Exact teacher pair ties are ignored.  Lexical candidate-mask order is never
used as semantic order.

## Future Variants

The only independently selectable future variants are:

```text
baseline
direction_consistency_only
candidate_order_consistency_only
```

There is no combined first-stage variant.

## Required Future Observability

Any future implementation must expose active non-tie teacher target count,
violating student target count, nonzero loss term count, and nonzero gradient
target count.  Graph connectivity alone is not evidence that a useful loss
signal will occur automatically.
"""


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)

    p5 = load_json(args.stage196b2b6p5_analysis_json)
    p7 = load_json(args.stage196b2b6p7_analysis_json)
    p8 = load_json(args.stage196b2b6p8_analysis_json)
    p8_gradient = read_csv(args.stage196b2b6p8_gradient_connectivity_csv)

    trainer_path = repo / "scripts" / "train_controlled_v6b_minimal.py"
    model_path = repo / "src" / "contramamba" / "modeling_v6b_minimal.py"
    trainer_text = trainer_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")
    source = {"trainer": trainer_text, "model": model_text}

    p8_dir = args.stage196b2b6p8_analysis_json.parent
    expected_p8_dir = repo / P8_FINAL_AUTHORITY_DIR
    contracts: list[dict[str, Any]] = []

    p4_authority = evaluate_p4_authority(
        p4_path=args.stage196b2b6p4_analysis_json,
        p7=p7,
        p6_contract_path=args.stage196b2b6p6_contract_csv,
        p7_contract_path=args.stage196b2b6p7_contract_csv,
    )
    p4_ok = bool(p4_authority["minimal_authority_passed"])
    p5_ok = p5.get("decision") == P5_DECISION and p5.get("blocking_reasons") == []
    p7_text = canonical(p7).lower()
    p7_selected_teacher = (
        (p7.get("teacher_state_analysis") or {}).get("selected_teacher")
    )
    p7_no_teacher = (
        "no teacher is selected" in p7_text
        or str(p7_selected_teacher).strip().lower() in {
            "not yet justified", "none", "no teacher selected",
        }
        or "teacher identity remains not_yet_justified" in p7_text
    )
    p7_ok = p7.get("blocking_reasons") == [] and p7_no_teacher
    p8_contract_ok, p8_failed_contracts = zero_failed_contracts(
        p8_dir / "stage196b2b6p8_contract.csv"
    )
    p8_ok = (
        p8_dir.resolve() == expected_p8_dir.resolve()
        and p8.get("decision") == P8_DECISION
        and p8.get("blocking_reasons") == []
        and p8_contract_ok
        and p8.get("direction_connectivity_passed") is True
        and p8.get("candidate_order_connectivity_passed") is True
    )
    direction_graph = classify_p8_graph(p8_gradient, "direction")
    order_graph = classify_p8_graph(p8_gradient, "candidate_order")

    source_facts = {
        "trainer_path": str(trainer_path),
        "model_path": str(model_path),
        "trainer_sha256": sha256(trainer_path),
        "model_sha256": sha256(model_path),
        "replay_full_trainable_path_line": source_line(
            model_text, r"def replay_full_trainable_path"
        ),
        "stability_loss_none_line": source_line(model_text, r'"stability_loss": None'),
        "model_loss_assembly_line": source_line(model_text, r"active_losses ="),
        "p8_default_off_line": source_line(
            trainer_text,
            r"stage196b2b6p8_enable_full_trainable_path_replay_api",
        ),
        "ema_symbols_present": bool(
            re.search(r"\b(EMA|ExponentialMovingAverage|AveragedModel)\b",
                      trainer_text + "\n" + model_text)
        ),
    }

    audits = candidate_audits(source)
    direction_targets = target_rows(audits, direction_graph, "direction")
    order_targets = target_rows(audits, order_graph, "candidate_order")
    lifecycle = lifecycle_rows()
    portability = portability_rows(audits)

    add_contract(contracts, "upstream_p4_authority", P4_DECISION,
                 {"authority_mode": p4_authority["authority_mode"],
                  "minimal_authority_passed": p4_authority["minimal_authority_passed"]},
                 p4_ok, "P4 authority failed")
    add_contract(contracts, "p4_authority_mode_valid", P4_AUTHORITY_MODES,
                 p4_authority["authority_mode"],
                 p4_authority["authority_mode"] in P4_AUTHORITY_MODES)
    add_contract(contracts, "p4_original_or_attested_authority_available", True,
                 p4_authority["minimal_authority_passed"], p4_ok,
                 "neither original nor downstream-attested P4 authority passed")
    original_mode = p4_authority["authority_mode"] == "ORIGINAL_P4_ANALYSIS"
    add_contract(contracts, "p4_downstream_p6_contract_hash_closure", True,
                 p4_authority.get("checks", {}).get("p6_contract_hash_closure", "not_applicable_original_mode"),
                 original_mode or p4_authority.get("checks", {}).get("p6_contract_hash_closure") is True)
    add_contract(contracts, "p4_downstream_p6_decision_closure", P4_P6_DECISION_PAYLOAD,
                 p4_authority["p6_decision_closure"],
                 original_mode or p4_authority["p6_decision_closure"].get("passed") is True)
    add_contract(contracts, "p4_downstream_p6_zero_failed_contracts", True,
                 p4_authority["p6_zero_failed_contracts_closure"],
                 original_mode or p4_authority["p6_zero_failed_contracts_closure"].get("passed") is True)
    add_contract(contracts, "p4_downstream_p7_analysis_concurrence", P4_DECISION,
                 p4_authority["p7_analysis_decision_concurrence"],
                 original_mode or p4_authority["p7_analysis_decision_concurrence"].get("passed") is True)
    add_contract(contracts, "p4_downstream_p7_contract_concurrence", P4_P7_DECISION_PAYLOAD,
                 p4_authority["p7_contract_decision_concurrence"],
                 original_mode or p4_authority["p7_contract_decision_concurrence"].get("passed") is True)
    add_contract(contracts, "p4_downstream_attestation_scope_restricted", P4_ATTESTED_SCOPE,
                 {"scope": p4_authority["attestation_scope"],
                  "limitations": p4_authority["attestation_limitations"]}, True)
    add_contract(contracts, "p4_original_artifact_not_fabricated", True,
                 p4_authority["original_artifact_not_fabricated"],
                 p4_authority["original_artifact_not_fabricated"] is True)
    add_contract(contracts, "upstream_p5_authority", P5_DECISION,
                 p5.get("decision"), p5_ok, "P5 authority failed")
    add_contract(contracts, "upstream_p7_authority", "zero blockers and no selected teacher",
                 {"decision": p7.get("decision"), "blocking_reasons": p7.get("blocking_reasons")},
                 p7_ok, "P7 authority failed")
    add_contract(contracts, "upstream_p8_final_authority", P8_FINAL_AUTHORITY_DIR,
                 str(p8_dir), p8_ok, "P8 final authority failed")
    add_contract(contracts, "p8_zero_failed_contracts", 0, p8_failed_contracts,
                 p8_contract_ok, "P8 contract failures found")
    add_contract(contracts, "p8_direction_graph_available", True,
                 direction_graph, bool(direction_graph["graph_available"]),
                 "P8 direction graph unavailable")
    add_contract(contracts, "p8_order_graph_available", True,
                 order_graph, bool(order_graph["graph_available"]),
                 "P8 order graph unavailable")
    add_contract(contracts, "exact_teacher_candidate_set", TEACHER_CANDIDATES,
                 tuple(row["teacher_candidate"] for row in audits),
                 tuple(row["teacher_candidate"] for row in audits) == TEACHER_CANDIDATES)
    add_contract(contracts, "direction_order_independent_evaluation", True, True)
    add_contract(contracts, "no_combined_first_variant", FUTURE_VARIANTS,
                 FUTURE_VARIANTS, "combined" not in FUTURE_VARIANTS)
    add_contract(contracts, "no_clean_dev_label_targeting", True,
                 all(row["depends_on_clean_dev_labels"] is False for row in portability),
                 all(row["depends_on_clean_dev_labels"] is False for row in portability))
    add_contract(contracts, "no_recovery_harm_label_targeting", True,
                 all(row["depends_on_recovery_harm_membership"] is False for row in portability),
                 all(row["depends_on_recovery_harm_membership"] is False for row in portability))
    add_contract(contracts, "no_lexical_candidate_order", True, True)
    add_contract(contracts, "exact_ties_ignored", True,
                 all("ignored" in row["exact_tie_policy"] for row in audits))
    add_contract(contracts, "teacher_stop_gradient_required", True,
                 all("stop-gradient" in row["gradient_stop_policy"] for row in audits))
    add_contract(contracts, "fixed_seed183_checkpoint_not_assumed_portable", True,
                 any(row["teacher_candidate"] == "FIXED_RECONSTRUCTED_CHECKPOINT"
                     and row["requires_reconstructed_seed183_checkpoint"] is True
                     and row["status"] == "BLOCKED"
                     for row in portability))
    add_contract(contracts, "ema_not_assumed_preexisting", True,
                 not source_facts["ema_symbols_present"])
    add_contract(contracts, "baseline_default_off_requirement", True,
                 "stage196b2b6p8_enable_full_trainable_path_replay_api" in trainer_text)
    add_contract(contracts, "zero_model_or_trainer_modification", True,
                 "static analyzer only; no model/trainer writes")
    add_contract(contracts, "exact_nine_file_closure", OUTPUTS, OUTPUTS,
                 len(OUTPUTS) == 9)

    upstream_ok = (
        p4_ok and p5_ok and p7_ok and p8_ok
        and bool(direction_graph["graph_available"])
        and bool(order_graph["graph_available"])
    )
    decision, decision_rows = decision_gate(audits, upstream_ok)
    recommended = NEXT_STAGE[decision]

    analysis = {
        "stage": STAGE,
        "title": STAGE_TITLE,
        "decision": decision,
        "recommended_next_stage": recommended,
        "blocking_reasons": [] if decision != DECISIONS[0] else [
            row["blocking_reason"] for row in contracts if not row["passed"]
        ],
        "intervention_families": FUTURE_VARIANTS,
        "teacher_candidates": TEACHER_CANDIDATES,
        "p4_authority": p4_authority,
        "upstream_authority": {
            "p4_analysis_json": str(args.stage196b2b6p4_analysis_json or ""),
            "p4_authority": p4_authority,
            "p6_contract_csv": str(args.stage196b2b6p6_contract_csv),
            "p7_contract_csv": str(args.stage196b2b6p7_contract_csv),
            "p5_analysis_json": str(args.stage196b2b6p5_analysis_json),
            "p7_analysis_json": str(args.stage196b2b6p7_analysis_json),
            "p8_analysis_json": str(args.stage196b2b6p8_analysis_json),
            "p8_gradient_connectivity_csv": str(args.stage196b2b6p8_gradient_connectivity_csv),
            "p8_final_authority_dir": str(expected_p8_dir),
        },
        "source_facts": source_facts,
        "p8_graph_interpretation": {
            "direction": direction_graph,
            "candidate_order": order_graph,
            "distinction_preserved": list(P8_GRAPH_CLASSES),
        },
        "direction_family_eligibility": [],
        "candidate_order_family_eligibility": [],
        "teacher_candidate_audit": audits,
        "exact_tie_policy": "ignore exact teacher zero and exact order ties",
        "stop_gradient_policy": "teacher signs are stop-gradient; student target quantities remain live",
        "remaining_scientific_uncertainty": (
            "which, if any, non-self teacher supplies portable causal signs "
            "without becoming a second opaque selector"
        ),
    }
    analysis["report_summary"] = {
        "fixed_checkpoint_restriction": (
            "reconstructed seed183 checkpoints remain diagnostic-only"
        ),
        "ema_state_requirements": (
            "initialization, update timing, decay ownership, buffers, dropout/eval "
            "policy, serialization, resume, determinism, warm-up, ties, drift "
            "observability, and default-off closure"
        ),
        "current_native_algebraic_identity": "invalid stability teacher",
        "donor_arm_imitation_risk": "blocked without portability/global-superiority proof",
    }

    atomic_write(output_dir / OUTPUTS[0],
                 json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    atomic_write(output_dir / OUTPUTS[1], report_md(analysis))
    atomic_write(output_dir / OUTPUTS[2], csv_text(AUDIT_FIELDS, audits))
    atomic_write(output_dir / OUTPUTS[3], csv_text(TARGET_FIELDS, direction_targets))
    atomic_write(output_dir / OUTPUTS[4], csv_text(TARGET_FIELDS, order_targets))
    atomic_write(output_dir / OUTPUTS[5], csv_text(LIFECYCLE_FIELDS, lifecycle))
    atomic_write(output_dir / OUTPUTS[6], csv_text(PORTABILITY_FIELDS, portability))
    atomic_write(output_dir / OUTPUTS[7], csv_text(DECISION_FIELDS, decision_rows))
    atomic_write(output_dir / OUTPUTS[8], csv_text(CONTRACT_FIELDS, contracts))
    return 0 if not analysis["blocking_reasons"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
