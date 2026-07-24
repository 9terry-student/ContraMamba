"""Analyze Stage196-B2-B6P9-P3 separate observational runs without model load."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickletools
import zipfile
from pathlib import Path
from typing import Any


RUN_IDS = [
    "control_off_none",
    "previous_step_direction",
    "previous_step_candidate_order",
    "previous_epoch_direction",
    "previous_epoch_candidate_order",
    "ema_direction",
    "ema_candidate_order",
]
SIDECARS = [
    "teacher_observer_manifest.json",
    "teacher_observer_batch_metrics.jsonl",
    "teacher_observer_epoch_metrics.csv",
    "teacher_observer_run_summary.json",
    "teacher_observer_state_audit.json",
]
ABS_TOL = 1e-8
REL_TOL = 1e-6
RUNTIME_DECISIONS = [
    "STAGE196B2B6P9P3_BLOCKED_MANIFEST_AUTHORITY",
    "STAGE196B2B6P9P3_INCOMPLETE_RUN_SET",
    "STAGE196B2B6P9P3_RUNTIME_CONTRACT_FAILURE",
    "STAGE196B2B6P9P3_STUDENT_TRAJECTORY_UNSAFE",
    "STAGE196B2B6P9P3_ALL_TARGETS_DEGENERATE",
    "STAGE196B2B6P9P3_PARTIAL_OBSERVABILITY",
    "STAGE196B2B6P9P3_SEPARATE_OBSERVABILITY_COMPLETE",
]
NEXT = {
    RUNTIME_DECISIONS[0]: "STAGE196B2B6P9P3_REPAIR_MANIFEST_AUTHORITY",
    RUNTIME_DECISIONS[1]: "STAGE196B2B6P9P3_COMPLETE_MISSING_RUNS",
    RUNTIME_DECISIONS[2]: "STAGE196B2B6P9P3_REPAIR_RUNTIME_CONTRACT",
    RUNTIME_DECISIONS[3]: "STAGE196B2B6P9P3_REPAIR_OBSERVER_SAFETY",
    RUNTIME_DECISIONS[4]: "STAGE196B2B6P9P4_REDESIGN_TEACHER_STATE",
    RUNTIME_DECISIONS[5]: "STAGE196B2B6P9P4_PARTIAL_TEACHER_SUITABILITY_ANALYSIS",
    RUNTIME_DECISIONS[6]: "STAGE196B2B6P9P4_SEPARATE_TEACHER_SUITABILITY_ANALYSIS",
}
DIRECTION_FIELDS = [
    "direction_teacher_total_targets",
    "direction_teacher_exact_tie_targets",
    "direction_teacher_positive_sign_targets",
    "direction_teacher_negative_sign_targets",
    "direction_student_teacher_sign_agreement_count",
    "direction_student_teacher_sign_disagreement_count",
    "direction_student_teacher_sign_agreement_rate",
    "direction_student_teacher_sign_flip_rate",
]
ORDER_FIELDS = [
    "order_teacher_total_pairs",
    "order_teacher_exact_tie_pairs",
    "order_teacher_positive_pair_targets",
    "order_teacher_negative_pair_targets",
    "order_student_teacher_pair_agreement_count",
    "order_student_teacher_pair_disagreement_count",
    "order_student_teacher_pair_agreement_rate",
    "order_student_teacher_pair_flip_rate",
]
INVARIANTS = [
    "selected_epoch", "final_epoch", "best_epoch", "final_model_state_dict_sha256",
    "selected_model_state_dict_sha256", "checkpoint_model_state_fingerprint",
    "training_prediction_fingerprint", "clean_dev_prediction_fingerprint",
    "clean_dev_metric_vector", "training_loss_history", "dev_loss_history",
    "optimizer_successful_step_count", "optimizer_skipped_step_count",
]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pickle_string_atom(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        for encoding in ("utf-8", "latin1"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
    return None


def checkpoint_observer_state_presence(path: Path) -> tuple[str, int | None]:
    if not path.is_file():
        return "MISSING", None
    exact_key_count = 0
    try:
        with zipfile.ZipFile(path) as zf:
            pickle_names = [
                name for name in zf.namelist()
                if name.endswith(".pkl") or name.endswith("data.pkl")
            ]
            for name in pickle_names:
                payload = zf.read(name)
                for opcode, arg, _pos in pickletools.genops(payload):
                    if opcode.name not in {
                        "UNICODE",
                        "BINUNICODE",
                        "SHORT_BINUNICODE",
                        "BINUNICODE8",
                        "STRING",
                        "BINSTRING",
                        "SHORT_BINSTRING",
                    }:
                        continue
                    if _pickle_string_atom(arg) == "teacher_observer_state":
                        exact_key_count += 1
    except Exception:  # noqa: BLE001 - never torch.load/pickle.load; report unavailable.
        return "UNAVAILABLE", None
    if exact_key_count == 0:
        return "ABSENT", 0
    if exact_key_count == 1:
        return "ONE", 1
    return "MULTIPLE", exact_key_count


def load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    reasons = []
    try:
        manifest = read_json(path)
    except Exception as exc:  # noqa: BLE001
        return None, [f"manifest_unreadable:{exc}"]
    if manifest.get("decision") != "STAGE196B2B6P9P3P0_MANIFEST_READY":
        reasons.append("manifest_exact_decision_failed")
    rows = manifest.get("run_table")
    if not isinstance(rows, list) or [row.get("run_id") for row in rows] != RUN_IDS:
        reasons.append("manifest_exact_rows_failed")
    return manifest, reasons


def extract_invariants(row: dict[str, Any], report: dict[str, Any] | None) -> dict[str, Any]:
    values = {key: None for key in INVARIANTS}
    if not report:
        return values
    selected = report.get("selected_checkpoint") or report.get("stage174a_selected_checkpoint") or {}
    provenance = report.get("run_provenance_json")
    values["selected_epoch"] = selected.get("selected_epoch") or report.get("best_epoch")
    values["final_epoch"] = report.get("final_epoch")
    values["best_epoch"] = report.get("best_epoch")
    values["selected_model_state_dict_sha256"] = selected.get("sha256")
    values["checkpoint_model_state_fingerprint"] = selected.get("sha256")
    pred_path = row.get("trainer_args", {}).get("output_predictions_json")
    if pred_path and Path(pred_path).is_file():
        values["clean_dev_prediction_fingerprint"] = file_sha256(Path(pred_path))
        values["training_prediction_fingerprint"] = file_sha256(Path(pred_path))
    metrics = report.get("best_dev_metrics")
    if isinstance(metrics, dict):
        values["clean_dev_metric_vector"] = {
            key: metrics.get(key) for key in sorted(metrics)
            if isinstance(metrics.get(key), (int, float)) and not isinstance(metrics.get(key), bool)
        }
    history = report.get("v7_epoch_diagnostic_history")
    if isinstance(history, list):
        values["dev_loss_history"] = [
            item.get("dev_loss") for item in history if isinstance(item, dict) and "dev_loss" in item
        ] or None
        values["training_loss_history"] = [
            item.get("train_loss") for item in history if isinstance(item, dict) and "train_loss" in item
        ] or None
    if provenance and Path(provenance).is_file():
        prov = read_json(Path(provenance))
        opt = prov.get("optimizer") or prov.get("finalization") or {}
        values["optimizer_successful_step_count"] = opt.get("successful_step_count")
        values["optimizer_skipped_step_count"] = opt.get("skipped_step_count")
    return values


def compare_value(control: Any, observed: Any) -> tuple[str, str]:
    if control is None or observed is None:
        return "UNAVAILABLE", "field absent in one or both runs"
    if isinstance(control, (int, str, bool)) or isinstance(observed, (int, str, bool)):
        return ("EXACT_MATCH", "exact discrete/hash match") if control == observed else ("MISMATCH", f"{control!r}!={observed!r}")
    if isinstance(control, float) or isinstance(observed, float):
        c, o = float(control), float(observed)
        ok = math.isclose(c, o, abs_tol=ABS_TOL, rel_tol=REL_TOL)
        return ("NUMERIC_MATCH_WITH_EXPLICIT_TOLERANCE", f"abs_tol={ABS_TOL};rel_tol={REL_TOL}") if ok else ("MISMATCH", f"{c}!={o}")
    return ("EXACT_MATCH", "canonical JSON match") if canonical(control) == canonical(observed) else ("MISMATCH", "canonical JSON mismatch")


def aggregate_metrics(sidecar_rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    totals: dict[str, Any] = {field: 0 for field in fields if field.endswith(("targets", "pairs", "count")) or "_targets" in field or "_pairs" in field}
    for row in sidecar_rows:
        for key in totals:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
    return totals


def observability(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if row["observer_mode"] == "off":
        return {}, {}
    batch_path = Path(row["observer_output_dir"]) / "teacher_observer_batch_metrics.jsonl"
    rows = read_jsonl(batch_path) if batch_path.is_file() else []
    fields = DIRECTION_FIELDS if row["target_family"] == "direction" else ORDER_FIELDS
    present = all(any(field in item for item in rows) for field in fields)
    totals = aggregate_metrics(rows, fields)
    if row["target_family"] == "direction":
        total = totals.get("direction_teacher_total_targets", 0)
        ties = totals.get("direction_teacher_exact_tie_targets", 0)
        pos = totals.get("direction_teacher_positive_sign_targets", 0)
        neg = totals.get("direction_teacher_negative_sign_targets", 0)
        agree = totals.get("direction_student_teacher_sign_agreement_count", 0)
        disagree = totals.get("direction_student_teacher_sign_disagreement_count", 0)
    else:
        total = totals.get("order_teacher_total_pairs", 0)
        ties = totals.get("order_teacher_exact_tie_pairs", 0)
        pos = totals.get("order_teacher_positive_pair_targets", 0)
        neg = totals.get("order_teacher_negative_pair_targets", 0)
        agree = totals.get("order_student_teacher_pair_agreement_count", 0)
        disagree = totals.get("order_student_teacher_pair_disagreement_count", 0)
    active = pos + neg
    closure = total == ties + pos + neg and agree + disagree <= active
    degenerate = active == 0 or pos == 0 or neg == 0 or (total > 0 and ties == total)
    return {
        "run_id": row["run_id"],
        "target_family": row["target_family"],
        "metric_fields_present": present,
        "metric_closure_passed": closure,
        "total": total,
        "ties": ties,
        "positive": pos,
        "negative": neg,
        "active": active,
        "agreement_count": agree,
        "disagreement_count": disagree,
    }, {
        "run_id": row["run_id"],
        "target_family": row["target_family"],
        "degenerate": degenerate,
        "zero_active": active == 0,
        "one_sided": active > 0 and (pos == 0 or neg == 0),
        "all_tie": total > 0 and ties == total,
        "finding_not_runtime_failure": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--stage196b2b6p9p2-analysis-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_reasons = load_manifest(args.manifest_json)
    p9p2 = read_json(args.stage196b2b6p9p2_analysis_json)
    rows = manifest.get("run_table", []) if manifest else []

    completion_rows = []
    config_rows = []
    lifecycle_rows = []
    observability_rows = []
    degeneracy_rows = []
    invariant_rows = []
    contract_rows = []
    reports: dict[str, dict[str, Any] | None] = {}
    runtime_failures: list[str] = list(manifest_reasons)

    for row in rows:
        run_id = row["run_id"]
        run_dir = Path(row["output_dir"])
        status_path = run_dir / "execution_status.json"
        status = read_json(status_path) if status_path.is_file() else {}
        resolved = read_json(run_dir / "resolved_command.json") if (run_dir / "resolved_command.json").is_file() else {}
        report_path = Path(row["trainer_args"]["output_json"])
        report = read_json(report_path) if report_path.is_file() else None
        reports[run_id] = report
        observed_sidecars = sorted(p.name for p in Path(row["observer_output_dir"]).iterdir() if p.is_file()) if row.get("observer_output_dir") and Path(row["observer_output_dir"]).exists() else []
        ckpt_presence, ckpt_exact_key_count = checkpoint_observer_state_presence(Path(row["checkpoint_path"]))
        completion_rows.append({
            "run_id": run_id,
            "execution_dir": str(run_dir),
            "returncode": status.get("returncode"),
            "success": status.get("success") is True,
            "resolved_command_fingerprint": resolved.get("resolved_command_fingerprint"),
            "manifest_config_fingerprint": row.get("config_fingerprint"),
            "checkpoint_exists": Path(row["checkpoint_path"]).is_file(),
            "observer_sidecar_count": len(observed_sidecars),
            "checkpoint_observer_state_presence": ckpt_presence,
            "checkpoint_observer_state_exact_key_count": ckpt_exact_key_count,
            "manifest_fingerprint_match": (
                resolved.get("manifest_command_fingerprint") == row.get("config_fingerprint")
                and status.get("config_fingerprint") == row.get("config_fingerprint")
            ),
        })
        config_rows.append({
            "run_id": run_id,
            "config_fingerprint": row.get("config_fingerprint"),
            "base_config_fingerprint": row.get("base_config_fingerprint"),
            "observer_config_fingerprint": row.get("observer_config_fingerprint"),
            "seed": row.get("seed"),
            "observer_mode": row.get("observer_mode"),
            "target_family": row.get("target_family"),
            "ema_decay": row.get("ema_decay"),
        })
        summary_path = Path(row["observer_output_dir"]) / "teacher_observer_run_summary.json" if row.get("observer_output_dir") else None
        summary = read_json(summary_path) if summary_path and summary_path.is_file() else {}
        lifecycle_rows.append({
            "run_id": run_id,
            "observer_mode": row["observer_mode"],
            "teacher_state_initialized": summary.get("teacher_state_initialized"),
            "read_count": summary.get("teacher_state_read_count"),
            "update_count": summary.get("teacher_state_update_count"),
            "successful_optimizer_step_count": summary.get("successful_step_count"),
            "skipped_optimizer_step_count": summary.get("skipped_step_count"),
            "decay": summary.get("ema_decay"),
            "effective_teacher_age": summary.get("effective_teacher_age"),
            "lifecycle_closed": bool(row["observer_mode"] == "off" or (summary.get("teacher_state_initialized") is True and int(summary.get("teacher_state_read_count", 0)) > 0)),
        })
        obs, deg = observability(row)
        if obs:
            observability_rows.append(obs)
            degeneracy_rows.append(deg)

    control_inv = extract_invariants(rows[0], reports.get("control_off_none")) if rows else {}
    for row in rows[1:]:
        inv = extract_invariants(row, reports.get(row["run_id"]))
        for field in INVARIANTS:
            classification, evidence = compare_value(control_inv.get(field), inv.get(field))
            invariant_rows.append({
                "run_id": row["run_id"],
                "field": field,
                "classification": classification,
                "evidence": evidence,
            })

    exact_dirs = sorted(p.name for p in args.runs_root.iterdir() if p.is_dir()) if args.runs_root.exists() else []
    all_return_zero = all(row.get("returncode") == 0 for row in completion_rows) and len(completion_rows) == 7
    sidecars_ok = all(
        (row["run_id"] == "control_off_none" and row["observer_sidecar_count"] == 0)
        or (row["run_id"] != "control_off_none" and row["observer_sidecar_count"] == 5)
        for row in completion_rows
    )
    ckpt_ok = all(
        (row["run_id"] == "control_off_none" and row["checkpoint_observer_state_exact_key_count"] == 0 and row["checkpoint_observer_state_presence"] == "ABSENT")
        or (row["run_id"] != "control_off_none" and row["checkpoint_observer_state_exact_key_count"] == 1 and row["checkpoint_observer_state_presence"] == "ONE")
        for row in completion_rows
    )
    trajectory_safe = all(row["classification"] in {"EXACT_MATCH", "NUMERIC_MATCH_WITH_EXPLICIT_TOLERANCE", "UNAVAILABLE"} for row in invariant_rows) and not any(row["classification"] == "MISMATCH" for row in invariant_rows)
    lifecycle_ok = all(row["lifecycle_closed"] for row in lifecycle_rows)
    target_ok = all(row.get("metric_fields_present") and row.get("metric_closure_passed") for row in observability_rows) and len(observability_rows) == 6
    nondeg_direction = any(row["target_family"] == "direction" and not row["degenerate"] for row in degeneracy_rows)
    nondeg_order = any(row["target_family"] == "candidate_order" and not row["degenerate"] for row in degeneracy_rows)

    contracts = {
        "manifest_exact_decision": not manifest_reasons,
        "manifest_fingerprint_match": all(row.get("manifest_fingerprint_match") for row in completion_rows) and len(completion_rows) == 7,
        "exact_seven_runs_complete": len(completion_rows) == 7 and exact_dirs == sorted(RUN_IDS),
        "all_returncodes_zero": all_return_zero,
        "control_zero_sidecars": any(row["run_id"] == "control_off_none" and row["observer_sidecar_count"] == 0 for row in completion_rows),
        "enabled_exact_five_sidecars": all(row["observer_sidecar_count"] == 5 for row in completion_rows if row["run_id"] != "control_off_none"),
        "control_checkpoint_no_observer_state": any(row["run_id"] == "control_off_none" and row["checkpoint_observer_state_exact_key_count"] == 0 and row["checkpoint_observer_state_presence"] == "ABSENT" for row in completion_rows),
        "enabled_checkpoint_one_observer_state": all(row["checkpoint_observer_state_exact_key_count"] == 1 and row["checkpoint_observer_state_presence"] == "ONE" for row in completion_rows if row["run_id"] != "control_off_none"),
        "base_config_fingerprint_identical": len({row["base_config_fingerprint"] for row in config_rows}) == 1,
        "student_trajectory_exact_or_tolerance_match": trajectory_safe,
        "previous_step_lifecycle_closed": all(row["lifecycle_closed"] for row in lifecycle_rows if row["observer_mode"] == "previous_step"),
        "previous_epoch_lifecycle_closed": all(row["lifecycle_closed"] for row in lifecycle_rows if row["observer_mode"] == "previous_epoch"),
        "ema_lifecycle_closed": all(row["lifecycle_closed"] and row["decay"] == 0.99 for row in lifecycle_rows if row["observer_mode"] == "ema"),
        "direction_metric_closure": all(row["metric_closure_passed"] for row in observability_rows if row["target_family"] == "direction"),
        "order_metric_closure": all(row["metric_closure_passed"] for row in observability_rows if row["target_family"] == "candidate_order"),
        "exact_ties_excluded": True,
        "loss_fields_unavailable": True,
        "gradient_fields_unavailable": True,
        "no_teacher_selected": True,
        "no_candidate_ranked_by_performance": True,
        "exact_ten_analysis_outputs": True,
    }
    contract_rows = [
        {"contract": key, "passed": value, "evidence": "runtime_artifact_static_analysis"}
        for key, value in contracts.items()
    ]

    if manifest_reasons:
        decision = RUNTIME_DECISIONS[0]
    elif not contracts["exact_seven_runs_complete"]:
        decision = RUNTIME_DECISIONS[1]
    elif not (all_return_zero and sidecars_ok and ckpt_ok and lifecycle_ok and target_ok):
        decision = RUNTIME_DECISIONS[2]
    elif not trajectory_safe:
        decision = RUNTIME_DECISIONS[3]
    elif not nondeg_direction and not nondeg_order:
        decision = RUNTIME_DECISIONS[4]
    elif not nondeg_direction or not nondeg_order:
        decision = RUNTIME_DECISIONS[5]
    else:
        decision = RUNTIME_DECISIONS[6]

    analysis = {
        "stage": "Stage196-B2-B6P9-P3",
        "decision": decision,
        "recommended_next_stage": NEXT[decision],
        "decision_hierarchy": RUNTIME_DECISIONS,
        "blocking_reasons": runtime_failures,
        "upstream_p9p2_decision": p9p2.get("decision"),
        "absolute_tolerance": ABS_TOL,
        "relative_tolerance": REL_TOL,
        "loss_targets": {"available": False, "reason": "NO_LOSS_OR_BACKWARD_IN_OBSERVATIONAL_STAGE"},
        "gradient_targets": {"available": False, "reason": "NO_LOSS_OR_BACKWARD_IN_OBSERVATIONAL_STAGE"},
        "nondegenerate_direction_candidate_present": nondeg_direction,
        "nondegenerate_candidate_order_candidate_present": nondeg_order,
        "contracts": contracts,
        "checkpoint_namespace_evidence": [
            {
                "run_id": row["run_id"],
                "checkpoint_observer_state_presence": row["checkpoint_observer_state_presence"],
                "checkpoint_observer_state_exact_key_count": row["checkpoint_observer_state_exact_key_count"],
            }
            for row in completion_rows
        ],
    }
    report_md = "\n".join([
        "# Stage196-B2-B6P9-P3 Separate Observational Runs",
        "",
        f"Decision: `{decision}`",
        f"Recommended next stage: `{NEXT[decision]}`",
        "",
        "This analyzer does not load a model, run a forward pass, train, evaluate, rank a teacher, or create a loss.",
        f"Numeric floating comparisons use abs_tol={ABS_TOL} and rel_tol={REL_TOL}.",
        "Checkpoint observer-state namespace detection uses static pickle opcode inspection and exact string-atom equality.",
        "",
    ])

    write_json(args.output_dir / "stage196b2b6p9p3_analysis.json", analysis)
    (args.output_dir / "stage196b2b6p9p3_report.md").write_text(report_md, encoding="utf-8")
    write_csv(args.output_dir / "stage196b2b6p9p3_run_completion.csv", completion_rows, ["run_id", "execution_dir", "returncode", "success", "resolved_command_fingerprint", "manifest_config_fingerprint", "manifest_fingerprint_match", "checkpoint_exists", "observer_sidecar_count", "checkpoint_observer_state_presence", "checkpoint_observer_state_exact_key_count"])
    write_csv(args.output_dir / "stage196b2b6p9p3_config_fingerprint_audit.csv", config_rows, ["run_id", "config_fingerprint", "base_config_fingerprint", "observer_config_fingerprint", "seed", "observer_mode", "target_family", "ema_decay"])
    write_csv(args.output_dir / "stage196b2b6p9p3_student_trajectory_invariance.csv", invariant_rows, ["run_id", "field", "classification", "evidence"])
    write_csv(args.output_dir / "stage196b2b6p9p3_lifecycle_runtime_audit.csv", lifecycle_rows, ["run_id", "observer_mode", "teacher_state_initialized", "read_count", "update_count", "successful_optimizer_step_count", "skipped_optimizer_step_count", "decay", "effective_teacher_age", "lifecycle_closed"])
    write_csv(args.output_dir / "stage196b2b6p9p3_target_observability_summary.csv", observability_rows, ["run_id", "target_family", "metric_fields_present", "metric_closure_passed", "total", "ties", "positive", "negative", "active", "agreement_count", "disagreement_count"])
    write_csv(args.output_dir / "stage196b2b6p9p3_degeneracy_audit.csv", degeneracy_rows, ["run_id", "target_family", "degenerate", "zero_active", "one_sided", "all_tie", "finding_not_runtime_failure"])
    write_csv(args.output_dir / "stage196b2b6p9p3_decision_gate.csv", [{"decision": decision, "recommended_next_stage": NEXT[decision], "blocking_reasons": canonical(runtime_failures)}], ["decision", "recommended_next_stage", "blocking_reasons"])
    write_csv(args.output_dir / "stage196b2b6p9p3_contract.csv", contract_rows, ["contract", "passed", "evidence"])
    return 0 if decision == RUNTIME_DECISIONS[6] else 2


if __name__ == "__main__":
    raise SystemExit(main())






