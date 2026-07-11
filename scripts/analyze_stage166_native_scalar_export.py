"""Stage167-A: analyze Stage166 native scalar export readiness.

This is an offline artifact analyzer. It reads existing Stage166 reports and
prediction artifacts, validates architecture-native scalar coverage, and audits
prediction artifact parsing. It does not train, tune thresholds, select
checkpoints, mutate predictions, or use external labels for model decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage167_scalar_schema import (
    ALL_KNOWN_SCALAR_FIELDS,
    SCALAR_SCHEMA_NAME,
    optional_scalars_for_architecture,
    required_scalars_for_architecture,
    unsupported_scalars_for_architecture,
)

PREDICTION_LIST_KEYS = ("predictions", "rows", "records", "examples", "items")
PREDICTION_FIELD_CANDIDATES = (
    "prediction",
    "pred_label",
    "final_prediction",
    "final_pred",
    "pred",
    "label_pred",
    "composed_prediction",
    "base_prediction",
)
SAFETY_POLICY = {
    "analysis_only": True,
    "training_run": False,
    "checkpoint_selection_modified": False,
    "threshold_tuning": False,
    "external_labels_used_for_tuning": False,
    "missing_scalars_synthesized": False,
    "model_decision_behavior_changed": False,
    "training_losses_changed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage167-A offline native scalar export analyzer for Stage166 artifacts."
    )
    parser.add_argument("--stage166-report", required=True, type=Path)
    parser.add_argument("--clean-predictions", required=True, type=Path)
    parser.add_argument("--external-predictions", required=True, type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")


def preview_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    text = json.dumps(value, sort_keys=True, default=json_safe)
    if len(text) > 240:
        return text[:237] + "..."
    return value


def split_dict_records(values: list[Any]) -> tuple[list[dict[str, Any]], int, list[Any]]:
    records: list[dict[str, Any]] = []
    skipped = 0
    samples: list[Any] = []
    for value in values:
        if isinstance(value, dict):
            records.append(value)
        else:
            skipped += 1
            if len(samples) < 10:
                samples.append(preview_value(value))
    return records, skipped, samples


def parse_prediction_artifact(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    audit: dict[str, Any] = {
        "path": str(path),
        "parser_mode": None,
        "whole_file_json_valid": False,
        "whole_file_json_type": None,
        "recognized_container_key": None,
        "total_decoded_values": 0,
        "dictionary_prediction_records": 0,
        "skipped_non_object_values": 0,
        "malformed_lines": 0,
        "sample_skipped_values": [],
    }

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    else:
        audit["whole_file_json_valid"] = True
        audit["whole_file_json_type"] = type(payload).__name__
        values: list[Any] = []
        if isinstance(payload, list):
            values = payload
            audit["parser_mode"] = "json_array"
        elif isinstance(payload, dict):
            for key in PREDICTION_LIST_KEYS:
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    values = candidate
                    audit["parser_mode"] = f"json_object_{key}"
                    audit["recognized_container_key"] = key
                    break
            if not values:
                values = [payload]
                audit["parser_mode"] = "json_object_single_record"
        else:
            values = [payload]
            audit["parser_mode"] = "json_scalar"
        records, skipped, samples = split_dict_records(values)
        audit["total_decoded_values"] = len(values)
        audit["dictionary_prediction_records"] = len(records)
        audit["skipped_non_object_values"] = skipped
        audit["sample_skipped_values"] = samples
        if records:
            return records, audit

    records = []
    skipped = 0
    malformed = 0
    samples: list[Any] = []
    decoded = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            malformed += 1
            continue
        decoded += 1
        if isinstance(value, dict):
            records.append(value)
        else:
            skipped += 1
            if len(samples) < 10:
                samples.append({"line": line_no, "value": preview_value(value)})
    audit.update(
        {
            "parser_mode": "jsonl" if not audit["whole_file_json_valid"] else "jsonl_fallback_after_whole_file_no_records",
            "total_decoded_values": decoded,
            "dictionary_prediction_records": len(records),
            "skipped_non_object_values": skipped,
            "malformed_lines": malformed,
            "sample_skipped_values": samples,
        }
    )
    return records, audit


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(path), "exists": True, "json_parse_error": str(exc)}
    if isinstance(value, dict):
        value.setdefault("path", str(path))
        value.setdefault("exists", True)
        return value
    return {"path": str(path), "exists": True, "json_type": type(value).__name__}


def prediction_for_row(row: dict[str, Any]) -> str | None:
    for field in PREDICTION_FIELD_CANDIDATES:
        value = row.get(field)
        if value is not None:
            return str(value)
    return None


def prediction_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(prediction_for_row(row) for row in rows)
    counts.pop(None, None)
    return dict(sorted((str(k), v) for k, v in counts.items()))


def is_noncollapsed(counts: dict[str, int]) -> bool:
    positive_labels = [label for label, count in counts.items() if count > 0]
    return len(positive_labels) >= 2


def scalar_coverage(rows: list[dict[str, Any]], architecture: str) -> dict[str, Any]:
    required = required_scalars_for_architecture(architecture)
    optional = optional_scalars_for_architecture(architecture)
    unsupported = unsupported_scalars_for_architecture(architecture)
    counts = {
        field: sum(1 for row in rows if row.get(field) is not None)
        for field in ALL_KNOWN_SCALAR_FIELDS
    }
    required_counts = {field: counts[field] for field in required}
    n_rows = len(rows)
    native_pass = bool(n_rows) and all(count == n_rows for count in required_counts.values())
    all_requested_pass = bool(n_rows) and all(counts[field] == n_rows for field in ALL_KNOWN_SCALAR_FIELDS)
    return {
        "scalar_schema": SCALAR_SCHEMA_NAME,
        "architecture": architecture,
        "n_rows": n_rows,
        "required_scalar_fields": list(required),
        "optional_scalar_fields": list(optional),
        "unsupported_scalar_fields": list(unsupported),
        "scalar_field_coverage_counts": counts,
        "required_scalar_counts": required_counts,
        "native_required_scalar_pass": native_pass,
        "all_requested_scalar_pass": all_requested_pass,
    }


def write_scalar_coverage_csv(path: Path, coverage: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    required = set(coverage["required_scalar_fields"])
    optional = set(coverage["optional_scalar_fields"])
    unsupported = set(coverage["unsupported_scalar_fields"])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["field", "count", "n_rows", "role", "complete"],
        )
        writer.writeheader()
        for field in ALL_KNOWN_SCALAR_FIELDS:
            if field in required:
                role = "required_native"
            elif field in unsupported:
                role = "unsupported_for_architecture"
            elif field in optional:
                role = "optional"
            else:
                role = "known"
            count = coverage["scalar_field_coverage_counts"][field]
            writer.writerow(
                {
                    "field": field,
                    "count": count,
                    "n_rows": coverage["n_rows"],
                    "role": role,
                    "complete": count == coverage["n_rows"],
                }
            )


def decide(native_scalar_pass: bool, external_noncollapsed: bool, architecture: str) -> str:
    arch_prefix = architecture.upper().replace("_MINIMAL", "")
    if not native_scalar_pass:
        return f"STAGE167A_{arch_prefix}_NATIVE_SCALAR_EXPORT_INCOMPLETE"
    if not external_noncollapsed:
        return f"STAGE167A_{arch_prefix}_NATIVE_SCALAR_EXPORT_COLLAPSED"
    return f"STAGE167A_{arch_prefix}_NATIVE_SCALAR_EXPORT_READY_FOR_COMMON_SCALAR_ANALYSIS"


def write_md_report(path: Path, report: dict[str, Any]) -> None:
    coverage = report["scalar_coverage"]
    lines = [
        "# Stage167-A Native Scalar Export Analysis",
        "",
        f"Decision: `{report['decision']}`",
        f"Architecture: `{coverage['architecture']}`",
        f"External rows parsed: {report['external_predictions']['n_rows']}",
        f"External prediction counts: `{report['external_predictions']['prediction_counts']}`",
        f"External non-collapsed: {report['external_predictions']['noncollapsed']}",
        f"Native required scalar pass: {coverage['native_required_scalar_pass']}",
        "",
        "## Scalar Schema",
        "",
        f"Required native fields: `{coverage['required_scalar_fields']}`",
        f"Unsupported fields for this architecture: `{coverage['unsupported_scalar_fields']}`",
        f"All requested scalar pass: {coverage['all_requested_scalar_pass']}",
        "",
        "## Parser Audit",
        "",
        f"Clean parser mode: `{report['clean_prediction_parser_audit']['parser_mode']}`",
        f"Clean prediction records: {report['clean_prediction_parser_audit']['dictionary_prediction_records']}",
        f"Clean skipped non-object values: {report['clean_prediction_parser_audit']['skipped_non_object_values']}",
        f"Clean malformed lines: {report['clean_prediction_parser_audit']['malformed_lines']}",
        "",
        "## Safety",
        "",
        "No training, checkpoint selection, threshold tuning, model decision behavior, training losses, or source predictions are modified.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage166_report = load_json_object(args.stage166_report)
    clean_rows, clean_audit = parse_prediction_artifact(args.clean_predictions)
    external_rows, external_audit = parse_prediction_artifact(args.external_predictions)

    external_counts = prediction_counts(external_rows)
    external_noncollapsed = is_noncollapsed(external_counts)
    coverage = scalar_coverage(external_rows, args.architecture)
    decision = decide(coverage["native_required_scalar_pass"], external_noncollapsed, args.architecture)

    report = {
        "stage": "Stage167-A",
        "decision": decision,
        "inputs": {
            "stage166_report": str(args.stage166_report),
            "clean_predictions": str(args.clean_predictions),
            "external_predictions": str(args.external_predictions),
        },
        "stage166_report": stage166_report,
        "scalar_coverage": coverage,
        "external_predictions": {
            "n_rows": len(external_rows),
            "prediction_counts": external_counts,
            "noncollapsed": external_noncollapsed,
            "parser_audit": external_audit,
        },
        "clean_predictions": {
            "n_rows": len(clean_rows),
            "prediction_counts": prediction_counts(clean_rows),
        },
        "clean_prediction_parser_audit": clean_audit,
        "safety_policy": SAFETY_POLICY,
    }

    write_json(args.output_dir / "stage167a_native_scalar_export_analysis_report.json", report)
    write_md_report(args.output_dir / "stage167a_native_scalar_export_analysis_report.md", report)
    write_scalar_coverage_csv(args.output_dir / "stage167a_scalar_coverage.csv", coverage)
    write_json(args.output_dir / "stage167a_clean_prediction_parser_audit.json", clean_audit)
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())