"""Stage43-B1 HuggingFace external validation acquisition (scaffold).

Pulls a user-specified HuggingFace dataset/config/split via
``datasets.load_dataset``, inspects its fields/labels, and converts rows
into the ContraMamba Stage43 canonical external-validation JSONL schema
documented in docs/stage43_external_validation_schema.md.

This script is preparation-only. It does not train, calibrate, or evaluate
any model. It does not fabricate rows, does not infer labels from model
predictions, and does not treat synthetic Stage34/35 probes as naturalistic
external evidence. It is not executed as part of writing this file -- it
is only run later, explicitly, by a human operator who supplies
--hf-dataset/--split (and optionally --hf-config, field names, etc).

Supported dataset loading pattern:

    from datasets import load_dataset

    if hf_config:
        dataset = load_dataset(hf_dataset, hf_config, split=split, ...)
    else:
        dataset = load_dataset(hf_dataset, split=split, ...)

Converted output produced by this script must not be used for training,
calibration, threshold selection, checkpoint selection, or loss design --
see the leakage policy in docs/stage43_external_validation_schema.md and
reports/stage43b1_hf_acquisition_plan.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SAMPLE_LIMIT = 5

DEFAULT_SUPPORT_VALUES = {"supports", "support", "entailment", "true", "1"}
DEFAULT_REFUTE_VALUES = {"refutes", "refute", "contradiction", "false", "-1"}
DEFAULT_NOT_ENTITLED_VALUES = {
    "not enough info",
    "nei",
    "neutral",
    "unknown",
    "not_entitled",
    "not_entailment",
    "0",
}

CLAIM_FIELD_ALIASES = ["claim", "hypothesis", "statement", "query", "sentence2", "premise2"]
EVIDENCE_FIELD_ALIASES = ["evidence", "premise", "context", "passage", "text", "sentence1"]
LABEL_FIELD_ALIASES = ["label", "gold", "gold_label", "answer", "verdict", "relation"]

LEAKAGE_POLICY = (
    "Converted output produced by this script is external-evaluation-only. It must not "
    "be used for training, calibration, threshold selection, checkpoint selection, or "
    "loss design. See docs/stage43_external_validation_schema.md and "
    "reports/stage43b1_hf_acquisition_plan.md."
)


def load_label_map(label_map_json: str | None) -> dict[str, str] | None:
    if not label_map_json:
        return None
    candidate_path = Path(label_map_json)
    if candidate_path.exists() and candidate_path.is_file():
        with candidate_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raw = json.loads(label_map_json)
    if not isinstance(raw, dict):
        raise ValueError("--label-map-json must decode to a JSON object")
    return {str(k).strip().lower(): str(v).strip().upper() for k, v in raw.items()}


def normalize_label(
    raw_label: Any,
    allow_neutral_as_not_entitled: bool,
    strict_labels: bool,
    manual_label_map: dict[str, str] | None,
) -> tuple[str | None, bool]:
    """Return (mapped_label_or_None, used_not_entailment_risk)."""
    if raw_label is None:
        return None, False
    norm = str(raw_label).strip().lower()
    if not norm:
        return None, False

    if manual_label_map is not None:
        mapped = manual_label_map.get(norm)
        if mapped in {"SUPPORT", "REFUTE", "NOT_ENTITLED"}:
            return mapped, False
        return None, False

    used_not_entailment_risk = False

    if norm in DEFAULT_SUPPORT_VALUES:
        return "SUPPORT", False
    if norm in DEFAULT_REFUTE_VALUES:
        return "REFUTE", False

    if norm == "not_entailment":
        used_not_entailment_risk = True
        return "NOT_ENTITLED", used_not_entailment_risk

    if norm == "neutral":
        if strict_labels:
            return None, False
        if allow_neutral_as_not_entitled:
            return "NOT_ENTITLED", False
        return None, False

    if norm in DEFAULT_NOT_ENTITLED_VALUES:
        return "NOT_ENTITLED", False

    return None, False


def detect_field(candidates: list[str], available_fields: list[str]) -> str | None:
    lowered = {f.lower(): f for f in available_fields}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def resolve_class_label(dataset: Any, field_name: str, raw_value: Any) -> Any:
    """Best-effort resolution of an integer ClassLabel to its string name."""
    try:
        features = getattr(dataset, "features", None)
        if features is None or field_name not in features:
            return raw_value
        feature = features[field_name]
        names = getattr(feature, "names", None)
        if names is None:
            return raw_value
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, int) and 0 <= raw_value < len(names):
            return names[raw_value]
    except Exception:
        return raw_value
    return raw_value


def build_row_key(claim: str, evidence: str, label: str) -> tuple[str, str, str]:
    return (claim.strip(), evidence.strip(), label)


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    source_dataset = args.source_dataset
    if source_dataset is None:
        source_dataset = args.hf_dataset
        if args.hf_config:
            source_dataset = f"{args.hf_dataset}/{args.hf_config}"

    base_report: dict[str, Any] = {
        "hf_dataset": args.hf_dataset,
        "hf_config": args.hf_config,
        "split": args.split,
        "source_dataset": source_dataset,
        "output_jsonl": str(args.output_jsonl),
        "rejected_jsonl": str(args.rejected_jsonl),
        "total_rows_seen": 0,
        "accepted_rows": 0,
        "rejected_rows": 0,
        "accepted_label_counts": {},
        "source_label_counts": {},
        "rejection_reason_counts": {},
        "field_mapping": {
            "claim_field": args.claim_field,
            "evidence_field": args.evidence_field,
            "label_field": args.label_field,
            "id_field": args.id_field,
        },
        "label_mapping": {},
        "sample_accepted_rows": [],
        "sample_rejected_rows": [],
        "risks": [],
        "recommendation": "",
        "leakage_policy": LEAKAGE_POLICY,
    }

    try:
        from datasets import load_dataset
    except ImportError as exc:
        base_report["decision"] = "STAGE43B1_HF_EXTERNAL_ACQUISITION_FAILED"
        base_report["recommendation"] = (
            f"The 'datasets' package is not importable in this environment: {exc}. "
            "Install the HuggingFace 'datasets' package before running this script."
        )
        return base_report

    try:
        load_kwargs: dict[str, Any] = {"split": args.split}
        if args.trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        if args.hf_config:
            dataset = load_dataset(args.hf_dataset, args.hf_config, **load_kwargs)
        else:
            dataset = load_dataset(args.hf_dataset, **load_kwargs)
    except Exception as exc:
        base_report["decision"] = "STAGE43B1_HF_EXTERNAL_ACQUISITION_FAILED"
        base_report["recommendation"] = (
            f"load_dataset({args.hf_dataset!r}, config={args.hf_config!r}, "
            f"split={args.split!r}) raised: {exc}"
        )
        return base_report

    try:
        available_fields = list(dataset.column_names)
    except Exception:
        available_fields = list(dataset.features.keys()) if hasattr(dataset, "features") else []

    claim_field = args.claim_field
    evidence_field = args.evidence_field
    label_field = args.label_field
    id_field = args.id_field

    risks: list[str] = []
    field_detection_notes: dict[str, Any] = {}

    if args.auto_detect_fields:
        if claim_field is None:
            claim_field = detect_field(CLAIM_FIELD_ALIASES, available_fields)
            field_detection_notes["claim_field_auto_detected"] = claim_field
        if evidence_field is None:
            evidence_field = detect_field(EVIDENCE_FIELD_ALIASES, available_fields)
            field_detection_notes["evidence_field_auto_detected"] = evidence_field
        if label_field is None:
            label_field = detect_field(LABEL_FIELD_ALIASES, available_fields)
            field_detection_notes["label_field_auto_detected"] = label_field

    base_report["field_mapping"] = {
        "claim_field": claim_field,
        "evidence_field": evidence_field,
        "label_field": label_field,
        "id_field": id_field,
        "available_fields": available_fields,
        "detection_notes": field_detection_notes,
    }

    if not claim_field or not evidence_field or not label_field:
        base_report["decision"] = "STAGE43B1_HF_EXTERNAL_ACQUISITION_NO_VALID_ROWS"
        base_report["recommendation"] = (
            "Could not resolve a confident claim/evidence/label field mapping. "
            f"Available fields: {available_fields}. Supply --claim-field/--evidence-field/"
            "--label-field explicitly, or pass --auto-detect-fields with a dataset whose "
            "field names match the known aliases."
        )
        base_report["risks"] = risks
        return base_report

    try:
        manual_label_map = load_label_map(args.label_map_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        base_report["decision"] = "STAGE43B1_HF_EXTERNAL_ACQUISITION_FAILED"
        base_report["recommendation"] = f"Could not parse --label-map-json: {exc}"
        return base_report

    if manual_label_map is not None:
        base_report["label_mapping"] = {"mode": "manual", "map": manual_label_map}
    else:
        base_report["label_mapping"] = {
            "mode": "default",
            "SUPPORT": sorted(DEFAULT_SUPPORT_VALUES),
            "REFUTE": sorted(DEFAULT_REFUTE_VALUES),
            "NOT_ENTITLED": sorted(DEFAULT_NOT_ENTITLED_VALUES),
            "not_entailment_policy": (
                "not_entailment maps to NOT_ENTITLED, not REFUTE, because RTE-style "
                "not_entailment does not distinguish contradiction from neutral."
            ),
        }

    try:
        indices = list(range(len(dataset)))
    except Exception:
        indices = list(range(sum(1 for _ in dataset)))

    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(indices)

    if args.max_rows is not None:
        indices = indices[: args.max_rows]

    if args.sample_size is not None:
        indices = indices[: args.sample_size]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_label_counts: dict[str, int] = {}
    source_label_counts: dict[str, int] = {}
    rejection_reason_counts: dict[str, int] = {}
    seen_keys: set[tuple[str, str, str]] = set()
    used_not_entailment_risk = False

    for idx in indices:
        record = dataset[idx]

        raw_claim = record.get(claim_field)
        raw_evidence = record.get(evidence_field)
        raw_label = record.get(label_field)

        resolved_label = resolve_class_label(dataset, label_field, raw_label)

        claim = str(raw_claim).strip() if raw_claim is not None else ""
        evidence = str(raw_evidence).strip() if raw_evidence is not None else ""

        source_label_str = str(resolved_label) if resolved_label is not None else ""
        if source_label_str:
            source_label_counts[source_label_str] = source_label_counts.get(source_label_str, 0) + 1

        reason = None
        mapped_label: str | None = None

        if not claim or len(claim) < args.min_text_chars:
            reason = "missing_or_too_short_claim"
        elif not evidence or len(evidence) < args.min_text_chars:
            reason = "missing_or_too_short_evidence"
        elif resolved_label is None or not str(resolved_label).strip():
            reason = "missing_label"
        else:
            mapped_label, row_risk = normalize_label(
                resolved_label, args.allow_neutral_as_not_entitled, args.strict_labels, manual_label_map
            )
            if mapped_label is None:
                reason = "ambiguous_or_unmapped_label"
            elif row_risk:
                used_not_entailment_risk = True

        if reason is None and args.dedupe:
            key = build_row_key(claim, evidence, mapped_label)
            if key in seen_keys:
                reason = "duplicate_claim_evidence_label"
            else:
                seen_keys.add(key)

        if reason is not None:
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
            compact_record = {k: record.get(k) for k in (claim_field, evidence_field, label_field, id_field) if k}
            rejected.append(
                {
                    "row_index": idx,
                    "rejection_reason": reason,
                    "source_label": source_label_str,
                    "raw_record": compact_record,
                }
            )
            continue

        original_id = None
        if id_field and record.get(id_field) is not None:
            original_id = str(record.get(id_field))
        row_id = original_id if original_id is not None else f"{source_dataset}_{idx}"

        out_row: dict[str, Any] = {
            "id": row_id,
            "claim": claim,
            "evidence": evidence,
            "label": mapped_label,
            "source_dataset": source_dataset,
            "source_label": source_label_str,
            "stage43_split": "external_validation",
            "metadata": {
                "row_index": idx,
                "hf_dataset": args.hf_dataset,
                "hf_config": args.hf_config,
                "hf_split": args.split,
            },
        }
        if original_id is not None:
            out_row["original_id"] = original_id
        if args.hf_config:
            out_row["source_config"] = args.hf_config
        out_row["source_split"] = args.split

        accepted.append(out_row)
        accepted_label_counts[mapped_label] = accepted_label_counts.get(mapped_label, 0) + 1

    if used_not_entailment_risk:
        risks.append(
            "Source dataset uses RTE-style 'not_entailment' labels, which were mapped to "
            "NOT_ENTITLED rather than REFUTE because RTE does not separate REFUTE from "
            "NOT_ENTITLED. Supply --label-map-json to override this if a different mapping "
            "is intended."
        )
    if args.hf_dataset.lower() in {"rte", "glue"} or (args.hf_config or "").lower() == "rte":
        risks.append(
            "RTE/GLUE-style sources are a weaker NLI transfer probe, not a dedicated "
            "fact-verification dataset; treat results as a secondary signal only."
        )
    if any(name in args.hf_dataset.lower() for name in ["mnli", "snli"]):
        risks.append(
            "MNLI/SNLI is broad NLI, not fact-verification-specific; claim/evidence "
            "semantics differ from VitaminC/FEVER-style verification."
        )

    total_rows_seen = len(indices)
    accepted_rows = len(accepted)
    rejected_rows = len(rejected)
    distinct_labels = len(accepted_label_counts)

    if not args.dry_run:
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

    if accepted_rows == 0:
        decision = "STAGE43B1_HF_EXTERNAL_ACQUISITION_NO_VALID_ROWS"
        recommendation = (
            "No rows could be converted to the ContraMamba external validation schema. "
            "Check --claim-field/--evidence-field/--label-field against the dataset's "
            f"actual columns ({available_fields}) and review rejected_jsonl for reasons."
        )
    elif distinct_labels < 2:
        decision = "STAGE43B1_HF_EXTERNAL_ACQUISITION_LABEL_IMBALANCED"
        recommendation = (
            "Rows were converted successfully but only one label class is represented "
            "in the accepted output. Stage43-B evaluation requires at least two label "
            "classes; consider a larger --max-rows/--sample-size or a different split."
        )
    else:
        decision = "STAGE43B1_HF_EXTERNAL_ACQUISITION_READY"
        if args.dry_run:
            recommendation = (
                "Dry run: field/label mapping looks valid and accepted rows would be "
                "written on a real run, but no output JSONL was written because "
                "--dry-run was set."
            )
        else:
            recommendation = (
                "Converted output is ready for Stage43-B2 external evaluation (eval-only). "
                "Do not use this output for training, calibration, threshold selection, "
                "checkpoint selection, or loss design."
            )

    base_report.update(
        {
            "decision": decision,
            "total_rows_seen": total_rows_seen,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "accepted_label_counts": accepted_label_counts,
            "source_label_counts": source_label_counts,
            "rejection_reason_counts": rejection_reason_counts,
            "sample_accepted_rows": accepted[:SAMPLE_LIMIT],
            "sample_rejected_rows": rejected[:SAMPLE_LIMIT],
            "risks": risks,
            "recommendation": recommendation,
        }
    )
    return base_report


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Stage43-B1 HuggingFace External Validation Acquisition Report")
    lines.append("")
    lines.append(
        "Preparation only. This report describes a single run of "
        "scripts/acquire_stage43_hf_external_validation.py using "
        "datasets.load_dataset. No model training or evaluation was performed."
    )
    lines.append("")

    lines.append("## 1. Decision")
    lines.append("")
    lines.append(f"**Decision:** `{report['decision']}`")
    lines.append("")

    lines.append("## 2. Dataset Source / Provenance")
    lines.append("")
    lines.append(f"- HF dataset: `{report['hf_dataset']}`")
    lines.append(f"- HF config: `{report['hf_config']}`")
    lines.append(f"- Split: `{report['split']}`")
    lines.append(f"- Source dataset label: `{report['source_dataset']}`")
    lines.append(f"- Output JSONL: `{report['output_jsonl']}`")
    lines.append(f"- Rejected JSONL: `{report['rejected_jsonl']}`")
    lines.append(f"- Total rows seen: {report['total_rows_seen']}")
    lines.append("")

    lines.append("## 3. Field Mapping")
    lines.append("")
    lines.append(f"```json\n{json.dumps(report['field_mapping'], indent=2)}\n```")
    lines.append("")

    lines.append("## 4. Label Mapping")
    lines.append("")
    lines.append(f"```json\n{json.dumps(report['label_mapping'], indent=2)}\n```")
    lines.append("")

    lines.append("## 5. Accepted/Rejected Summary")
    lines.append("")
    lines.append(f"- Accepted rows: {report['accepted_rows']}")
    lines.append(f"- Rejected rows: {report['rejected_rows']}")
    lines.append("")
    lines.append("### Rejection reason counts")
    lines.append("")
    if report["rejection_reason_counts"]:
        for reason, count in report["rejection_reason_counts"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("None.")
    lines.append("")

    lines.append("## 6. Label Distribution")
    lines.append("")
    lines.append("### Accepted label counts")
    lines.append("")
    if report["accepted_label_counts"]:
        for label, count in report["accepted_label_counts"].items():
            lines.append(f"- `{label}`: {count}")
    else:
        lines.append("None.")
    lines.append("")
    lines.append("### Source label counts")
    lines.append("")
    if report["source_label_counts"]:
        for label, count in report["source_label_counts"].items():
            lines.append(f"- `{label}`: {count}")
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

    lines.append("## 9. Risks")
    lines.append("")
    if report["risks"]:
        for risk in report["risks"]:
            lines.append(f"- {risk}")
    else:
        lines.append("None identified.")
    lines.append("")

    lines.append("## 10. Recommendation")
    lines.append("")
    lines.append(report["recommendation"])
    lines.append("")

    lines.append("## 11. Leakage Policy")
    lines.append("")
    lines.append(report["leakage_policy"])
    lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset", type=str, required=True)
    parser.add_argument("--hf-config", type=str, default=None)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--rejected-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)

    parser.add_argument("--claim-field", type=str, default=None)
    parser.add_argument("--evidence-field", type=str, default=None)
    parser.add_argument("--label-field", type=str, default=None)
    parser.add_argument("--id-field", type=str, default=None)
    parser.add_argument("--source-dataset", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--min-text-chars", type=int, default=3)
    parser.add_argument(
        "--allow-neutral-as-not-entitled",
        type=lambda v: str(v).strip().lower() not in {"0", "false", "no"},
        default=True,
    )
    parser.add_argument("--strict-labels", action="store_true")
    parser.add_argument("--label-map-json", type=str, default=None)
    parser.add_argument("--auto-detect-fields", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    report = acquire(args)

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
