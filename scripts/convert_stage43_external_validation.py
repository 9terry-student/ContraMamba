"""Stage43-B0 external validation intake converter (scaffold).

Converts a user-supplied naturalistic claim/evidence/label dataset into the
ContraMamba Stage43 external-validation JSONL schema documented in
docs/stage43_external_validation_schema.md.

This script is preparation-only. It does not download any dataset, does not
fabricate rows, does not infer labels from model predictions, and does not
train, evaluate, or run any model. It only reads a user-supplied input file,
performs field extraction and label normalization, and writes accepted rows
(converted schema), rejected rows (with rejection reasons), and a report.

Converted output produced by this script must not be used for training,
calibration, threshold selection, checkpoint selection, or loss design --
see the leakage policy in docs/stage43_external_validation_schema.md and
reports/stage43b0_external_intake_plan.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SUPPORT_VALUES = {"supports", "support", "entailment", "true", "1"}
REFUTE_VALUES = {"refutes", "refute", "contradiction", "false", "-1"}
NOT_ENTITLED_VALUES = {
    "not enough info",
    "nei",
    "neutral",
    "unknown",
    "not_entitled",
    "0",
}

SAMPLE_LIMIT = 5


def normalize_label(raw_label: Any, allow_neutral_as_not_entitled: bool, strict_labels: bool) -> str | None:
    if raw_label is None:
        return None
    norm = str(raw_label).strip().lower()
    if not norm:
        return None

    if norm in SUPPORT_VALUES:
        return "SUPPORT"
    if norm in REFUTE_VALUES:
        return "REFUTE"

    if norm == "neutral":
        if allow_neutral_as_not_entitled and not strict_labels:
            return "NOT_ENTITLED"
        if strict_labels:
            return None

    if norm in NOT_ENTITLED_VALUES:
        return "NOT_ENTITLED"

    return None


def detect_format(input_path: Path, format_arg: str) -> str:
    if format_arg != "auto":
        return format_arg
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    raise ValueError(f"Cannot auto-detect format for input file: {input_path}")


def read_rows(input_path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "jsonl":
        rows: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    if fmt == "json":
        with input_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        if isinstance(data, dict):
            for key in ("predictions", "records", "items", "data", "rows"):
                value = data.get(key)
                if isinstance(value, list):
                    return [r for r in value if isinstance(r, dict)]
        raise ValueError(f"JSON input {input_path} is not a row-oriented list/dict of records.")

    if fmt in ("csv", "tsv"):
        delimiter = "," if fmt == "csv" else "\t"
        with input_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            return [dict(row) for row in reader]

    raise ValueError(f"Unsupported format: {fmt}")


def build_row_key(claim: str, evidence: str, label: str) -> tuple[str, str, str]:
    return (claim.strip(), evidence.strip(), label)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)

    field_mapping = {
        "claim_field": args.claim_field,
        "evidence_field": args.evidence_field,
        "label_field": args.label_field,
        "id_field": args.id_field,
    }

    try:
        fmt = detect_format(input_path, args.format)
        raw_rows = read_rows(input_path, fmt)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "decision": "STAGE43B0_EXTERNAL_INTAKE_FAILED",
            "input_file": str(input_path),
            "output_jsonl": str(args.output_jsonl),
            "rejected_jsonl": str(args.rejected_jsonl),
            "source_dataset": args.source_dataset,
            "total_rows": 0,
            "accepted_rows": 0,
            "rejected_rows": 0,
            "accepted_label_counts": {},
            "source_label_counts": {},
            "rejection_reason_counts": {},
            "field_mapping": field_mapping,
            "sample_accepted_rows": [],
            "sample_rejected_rows": [],
            "recommendation": f"Input file could not be read: {exc}",
            "leakage_policy": LEAKAGE_POLICY,
        }

    if args.max_rows is not None:
        raw_rows = raw_rows[: args.max_rows]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_label_counts: dict[str, int] = {}
    source_label_counts: dict[str, int] = {}
    rejection_reason_counts: dict[str, int] = {}
    seen_keys: set[tuple[str, str, str]] = set()

    for idx, record in enumerate(raw_rows):
        raw_claim = record.get(args.claim_field)
        raw_evidence = record.get(args.evidence_field)
        raw_label = record.get(args.label_field)

        claim = str(raw_claim).strip() if raw_claim is not None else ""
        evidence = str(raw_evidence).strip() if raw_evidence is not None else ""

        source_label_str = str(raw_label) if raw_label is not None else ""
        if source_label_str:
            source_label_counts[source_label_str] = source_label_counts.get(source_label_str, 0) + 1

        reason = None
        if not claim or len(claim) < args.min_text_chars:
            reason = "missing_or_too_short_claim"
        elif not evidence or len(evidence) < args.min_text_chars:
            reason = "missing_or_too_short_evidence"
        elif raw_label is None or not str(raw_label).strip():
            reason = "missing_label"
        else:
            mapped_label = normalize_label(raw_label, args.allow_neutral_as_not_entitled, args.strict_labels)
            if mapped_label is None:
                reason = "ambiguous_or_unmapped_label"

        if reason is None:
            mapped_label = normalize_label(raw_label, args.allow_neutral_as_not_entitled, args.strict_labels)
            if args.dedupe:
                key = build_row_key(claim, evidence, mapped_label)
                if key in seen_keys:
                    reason = "duplicate_claim_evidence_label"
                else:
                    seen_keys.add(key)

        if reason is not None:
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
            rejected.append(
                {
                    "row_index": idx,
                    "rejection_reason": reason,
                    "raw_record": record,
                }
            )
            continue

        row_id = None
        original_id = None
        if args.id_field and record.get(args.id_field) is not None:
            original_id = str(record.get(args.id_field))
            row_id = original_id
        if row_id is None:
            row_id = f"{args.source_dataset}_{idx}"

        out_row: dict[str, Any] = {
            "id": row_id,
            "claim": claim,
            "evidence": evidence,
            "label": mapped_label,
            "source_dataset": args.source_dataset,
            "source_label": source_label_str,
            "stage43_split": "external_validation",
            "metadata": {
                "row_index": idx,
                "input_file": str(input_path),
            },
        }
        if original_id is not None:
            out_row["original_id"] = original_id

        accepted.append(out_row)
        accepted_label_counts[mapped_label] = accepted_label_counts.get(mapped_label, 0) + 1

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for row in accepted:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    args.rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.rejected_jsonl.open("w", encoding="utf-8") as fh:
        for row in rejected:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    total_rows = len(raw_rows)
    accepted_rows = len(accepted)
    rejected_rows = len(rejected)
    distinct_labels = len(accepted_label_counts)

    if accepted_rows == 0:
        decision = "STAGE43B0_EXTERNAL_INTAKE_NO_VALID_ROWS"
        recommendation = (
            "No rows could be converted to the ContraMamba external validation schema. "
            "Check --claim-field/--evidence-field/--label-field against the input file's "
            "actual column/key names, and review rejected_jsonl for rejection reasons."
        )
    elif distinct_labels < 2:
        decision = "STAGE43B0_EXTERNAL_INTAKE_LABEL_IMBALANCED"
        recommendation = (
            "Rows were converted successfully but only one label class is represented "
            "in the accepted output. Stage43-B evaluation requires at least two label "
            "classes for a meaningful external validation signal; supply a source file "
            "(or subset) with more than one label before proceeding to Stage43-B."
        )
    else:
        decision = "STAGE43B0_EXTERNAL_INTAKE_READY"
        recommendation = (
            "Converted output is ready for Stage43-B external evaluation (eval-only). "
            "Do not use this output for training, calibration, threshold selection, "
            "checkpoint selection, or loss design."
        )

    return {
        "decision": decision,
        "input_file": str(input_path),
        "output_jsonl": str(args.output_jsonl),
        "rejected_jsonl": str(args.rejected_jsonl),
        "source_dataset": args.source_dataset,
        "total_rows": total_rows,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "accepted_label_counts": accepted_label_counts,
        "source_label_counts": source_label_counts,
        "rejection_reason_counts": rejection_reason_counts,
        "field_mapping": field_mapping,
        "sample_accepted_rows": accepted[:SAMPLE_LIMIT],
        "sample_rejected_rows": rejected[:SAMPLE_LIMIT],
        "recommendation": recommendation,
        "leakage_policy": LEAKAGE_POLICY,
    }


LEAKAGE_POLICY = (
    "Converted output produced by this script is external-evaluation-only. It must not "
    "be used for training, calibration, threshold selection, checkpoint selection, or "
    "loss design. See docs/stage43_external_validation_schema.md and "
    "reports/stage43b0_external_intake_plan.md."
)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Stage43-B0 External Validation Conversion Report")
    lines.append("")
    lines.append(
        "Preparation only. This report describes a single run of "
        "scripts/convert_stage43_external_validation.py. No model training, evaluation, "
        "or Kaggle/local model execution was performed to produce it."
    )
    lines.append("")

    lines.append("## 1. Decision")
    lines.append("")
    lines.append(f"**Decision:** `{report['decision']}`")
    lines.append("")

    lines.append("## 2. Run Summary")
    lines.append("")
    lines.append(f"- Input file: `{report['input_file']}`")
    lines.append(f"- Output JSONL: `{report['output_jsonl']}`")
    lines.append(f"- Rejected JSONL: `{report['rejected_jsonl']}`")
    lines.append(f"- Source dataset: `{report['source_dataset']}`")
    lines.append(f"- Total rows read: {report['total_rows']}")
    lines.append(f"- Accepted rows: {report['accepted_rows']}")
    lines.append(f"- Rejected rows: {report['rejected_rows']}")
    lines.append("")

    lines.append("## 3. Field Mapping")
    lines.append("")
    lines.append(f"`{json.dumps(report['field_mapping'])}`")
    lines.append("")

    lines.append("## 4. Accepted Label Counts")
    lines.append("")
    if report["accepted_label_counts"]:
        for label, count in report["accepted_label_counts"].items():
            lines.append(f"- `{label}`: {count}")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## 5. Source Label Counts")
    lines.append("")
    if report["source_label_counts"]:
        for label, count in report["source_label_counts"].items():
            lines.append(f"- `{label}`: {count}")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## 6. Rejection Reason Counts")
    lines.append("")
    if report["rejection_reason_counts"]:
        for reason, count in report["rejection_reason_counts"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## 7. Sample Accepted Rows")
    lines.append("")
    if report["sample_accepted_rows"]:
        lines.append("```json")
        lines.append(json.dumps(report["sample_accepted_rows"], indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## 8. Sample Rejected Rows")
    lines.append("")
    if report["sample_rejected_rows"]:
        lines.append("```json")
        lines.append(json.dumps(report["sample_rejected_rows"], indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## 9. Leakage Policy")
    lines.append("")
    lines.append(report["leakage_policy"])
    lines.append("")

    lines.append("## 10. Recommendation")
    lines.append("")
    lines.append(report["recommendation"])
    lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--rejected-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--source-dataset", type=str, required=True)
    parser.add_argument("--claim-field", type=str, required=True)
    parser.add_argument("--evidence-field", type=str, required=True)
    parser.add_argument("--label-field", type=str, required=True)
    parser.add_argument("--id-field", type=str, default=None)
    parser.add_argument(
        "--format",
        type=str,
        choices=["auto", "jsonl", "json", "csv", "tsv"],
        default="auto",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--min-text-chars", type=int, default=3)
    parser.add_argument(
        "--allow-neutral-as-not-entitled",
        type=lambda v: str(v).strip().lower() not in {"0", "false", "no"},
        default=True,
    )
    parser.add_argument("--strict-labels", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    report = convert(args)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    markdown = render_markdown(report)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    with args.report_md.open("w", encoding="utf-8") as fh:
        fh.write(markdown)

    print(f"Decision: {report['decision']}")
    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
