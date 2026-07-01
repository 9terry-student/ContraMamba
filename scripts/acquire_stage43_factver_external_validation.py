"""Stage43-B1 targeted fact-verification HuggingFace acquisition.

Acquires Stage43 external-validation JSONL rows from the two targeted
fact-verification sources confirmed by prior EpistemicBERT work:

- ``tals/vitaminc``
- ``climate_fever``

This script is acquisition/preparation only. It does not run ContraMamba,
train models, evaluate models, infer labels from predictions, run Kaggle
commands, or create synthetic examples. Acquired rows are external-evaluation
only and must not be used for training, calibration, threshold selection,
checkpoint selection, or loss design.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Any

SAMPLE_LIMIT = 5
VALID_LABELS = {"SUPPORT", "REFUTE", "NOT_ENTITLED"}

DATASET_BY_PRESET = {
    "vitaminc": "tals/vitaminc",
    "climate_fever": "climate_fever",
}

VITAMINC_NUMERIC_FALLBACK = {
    0: "SUPPORT",
    1: "REFUTE",
    2: "NOT_ENTITLED",
}

CLIMATE_FEVER_NUMERIC_MAP = {
    0: "SUPPORT",
    1: "REFUTE",
    2: "NOT_ENTITLED",
}

LABEL_STRING_MAP = {
    "supports": "SUPPORT",
    "support": "SUPPORT",
    "refutes": "REFUTE",
    "refute": "REFUTE",
    "not enough info": "NOT_ENTITLED",
    "not_enough_info": "NOT_ENTITLED",
    "nei": "NOT_ENTITLED",
}

LEAKAGE_POLICY = (
    "Stage43-B1 acquired data is external-evaluation-only. It must not be used "
    "for training, calibration, threshold selection, checkpoint selection, loss "
    "design, or any other model-selection feedback loop."
)


def normalize_label_text(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def compact_value(value: Any, max_chars: int = 500) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"
    if isinstance(value, dict):
        return {str(k): compact_value(v, max_chars=max_chars) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [compact_value(v, max_chars=max_chars) for v in list(value)[:20]]
    return value


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return json_safe(value)

def is_numeric_only(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    if not text:
        return False
    return text.replace(".", "", 1).replace("-", "", 1).isdigit()


def get_classlabel_names(dataset: Any, label_field: str) -> list[str] | None:
    try:
        features = getattr(dataset, "features", None)
        if features is None or label_field not in features:
            return None
        names = getattr(features[label_field], "names", None)
        return list(names) if names else None
    except Exception:
        return None


def resolve_fields(preset: str, dataset: Any) -> dict[str, Any]:
    columns = list(getattr(dataset, "column_names", []) or [])
    if not columns and hasattr(dataset, "features"):
        columns = list(dataset.features.keys())

    if preset == "vitaminc":
        mapping = {
            "claim_field": "claim",
            "evidence_field": "evidence",
            "label_field": "label",
            "id_field": "id" if "id" in columns else None,
            "available_fields": columns,
        }
    else:
        evidence_field = "evidences" if "evidences" in columns else "evidence"
        label_field = "claim_label" if "claim_label" in columns else "label"
        mapping = {
            "claim_field": "claim",
            "evidence_field": evidence_field,
            "label_field": label_field,
            "id_field": "id" if "id" in columns else None,
            "available_fields": columns,
        }

    missing = [
        field
        for field in ("claim_field", "evidence_field", "label_field")
        if mapping[field] not in columns
    ]
    if missing:
        raise ValueError(f"Missing expected {preset} fields {missing}; available fields: {columns}")
    return mapping


def flatten_evidence(value: Any, mode: str, join_sep: str) -> tuple[str, dict[str, Any]]:
    notes: dict[str, Any] = {
        "evidence_was_structured": False,
        "evidence_flatten_strategy": "string",
    }

    def flatten_inner(item: Any) -> list[str]:
        if item is None:
            return []
        if isinstance(item, str):
            text = item.strip()
            return [text] if text else []
        if isinstance(item, (list, tuple)):
            notes["evidence_was_structured"] = True
            notes["evidence_flatten_strategy"] = (
                "list_first_non_empty" if mode == "first" else "list_join_all"
            )
            segments: list[str] = []
            for child in item:
                child_segments = flatten_inner(child)
                if mode == "first" and child_segments:
                    return [child_segments[0]]
                segments.extend(child_segments)
            return segments
        if isinstance(item, dict):
            notes["evidence_was_structured"] = True
            for key in ("evidence", "text", "sentence", "content", "passage", "context"):
                if key in item:
                    notes["evidence_flatten_strategy"] = f"dict_preferred_{key}"
                    return flatten_inner(item[key])
            notes["evidence_flatten_strategy"] = "dict_recursive_string_values"
            segments = []
            for child in item.values():
                if isinstance(child, (str, list, tuple, dict)):
                    child_segments = flatten_inner(child)
                    if mode == "first" and child_segments:
                        return [child_segments[0]]
                    segments.extend(child_segments)
            return segments
        if is_numeric_only(item):
            return []
        text = str(item).strip()
        if not text:
            return []
        notes["evidence_flatten_strategy"] = "scalar_string_cast"
        return [text]

    segments = flatten_inner(value)
    if mode == "first":
        flattened = segments[0].strip() if segments else ""
    else:
        flattened = join_sep.join(segment for segment in segments if segment).strip()
    return flattened, notes


def source_label_from_classlabel(
    raw_label: Any, classlabel_names: list[str] | None
) -> tuple[str | None, bool]:
    if classlabel_names is not None and isinstance(raw_label, Integral):
        label_index = int(raw_label)
        if 0 <= label_index < len(classlabel_names):
            return classlabel_names[label_index], True
    if raw_label is None:
        return None, False
    return str(raw_label), False


def map_vitaminc_label(
    raw_label: Any, source_label_name: str | None, used_classlabel_name: bool
) -> tuple[str | None, str | None]:
    if source_label_name:
        mapped = LABEL_STRING_MAP.get(normalize_label_text(source_label_name))
        if mapped:
            return mapped, None

    if isinstance(raw_label, Integral):
        mapped = VITAMINC_NUMERIC_FALLBACK.get(int(raw_label))
        if mapped:
            reason = None if used_classlabel_name else "vitaminc_numeric_fallback_used"
            return mapped, reason

    return None, None


def map_climate_fever_label(raw_label: Any, source_label_name: str | None) -> tuple[str | None, str | None]:
    if source_label_name and normalize_label_text(source_label_name) == "disputed":
        return None, "disputed_label_excluded"
    if isinstance(raw_label, Integral) and int(raw_label) == 3:
        return None, "disputed_label_excluded"

    if source_label_name:
        mapped = LABEL_STRING_MAP.get(normalize_label_text(source_label_name))
        if mapped:
            return mapped, None

    if isinstance(raw_label, Integral):
        mapped = CLIMATE_FEVER_NUMERIC_MAP.get(int(raw_label))
        if mapped:
            return mapped, None

    return None, None


def map_label(
    preset: str,
    raw_label: Any,
    source_label_name: str | None,
    used_classlabel_name: bool,
) -> tuple[str | None, str | None, str | None]:
    if raw_label is None or (isinstance(raw_label, str) and not raw_label.strip()):
        return None, "missing_label", None

    if preset == "vitaminc":
        mapped, note = map_vitaminc_label(raw_label, source_label_name, used_classlabel_name)
        if mapped:
            return mapped, None, note
        return None, "ambiguous_or_unmapped_label", note

    mapped, rejection = map_climate_fever_label(raw_label, source_label_name)
    if mapped:
        return mapped, None, None
    return None, rejection or "ambiguous_or_unmapped_label", None


def choose_indices(dataset: Any, args: argparse.Namespace) -> list[int]:
    total = len(dataset)
    limit = total if args.max_rows is None else min(total, args.max_rows)
    indices = list(range(limit))
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(indices)
    if args.sample_size is not None:
        indices = indices[: args.sample_size]
    return indices


def make_rejected(
    row_index: int,
    reason: str,
    raw_label: Any,
    record: dict[str, Any],
    claim: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "rejection_reason": reason,
        "source_label_raw": json_safe(raw_label),
        "original_row": to_jsonable(compact_value(record)),
        "detected_claim_snippet": claim[:200],
        "detected_evidence_snippet": evidence[:200],
    }


def classify_decision(label_counts: Counter[str]) -> tuple[str, str]:
    accepted_rows = sum(label_counts.values())
    distinct = len(label_counts)
    if accepted_rows == 0:
        return (
            "STAGE43B1_FACTVER_ACQUISITION_NO_VALID_ROWS",
            "No valid rows were acquired. Review field mapping, rejected rows, and label policy.",
        )
    if VALID_LABELS.issubset(set(label_counts)):
        return (
            "STAGE43B1_FACTVER_ACQUISITION_READY",
            "Accepted rows cover SUPPORT, REFUTE, and NOT_ENTITLED. The file is ready for future Stage43-B2 eval-only use.",
        )
    if distinct >= 2:
        return (
            "STAGE43B1_FACTVER_ACQUISITION_PARTIAL_READY",
            "Accepted rows cover at least two labels. Review whether the missing label is expected before Stage43-B2.",
        )
    return (
        "STAGE43B1_FACTVER_ACQUISITION_LABEL_IMBALANCED",
        "Accepted rows contain only one label. Use a broader split/subset before Stage43-B2 if possible.",
    )


def label_mapping_description(preset: str) -> dict[str, Any]:
    if preset == "vitaminc":
        return {
            "SUPPORT": ["SUPPORTS"],
            "REFUTE": ["REFUTES"],
            "NOT_ENTITLED": ["NOT ENOUGH INFO", "NOT_ENOUGH_INFO", "NEI"],
        }
    return {
        "SUPPORT": ["SUPPORTS", 0],
        "REFUTE": ["REFUTES", 1],
        "NOT_ENTITLED": ["NOT_ENOUGH_INFO", "NOT ENOUGH INFO", "NEI", 2],
        "excluded": {"DISPUTED": "disputed_label_excluded", 3: "disputed_label_excluded"},
    }


def numeric_label_policy(preset: str) -> dict[str, Any]:
    if preset == "vitaminc":
        return {
            "classlabel_first": True,
            "fallback": VITAMINC_NUMERIC_FALLBACK,
            "fallback_rationale": "EpistemicBERT-confirmed VitaminC internal mapping.",
            "fallback_recorded_in_report": True,
        }
    return {
        "classlabel_first": True,
        "mapping": CLIMATE_FEVER_NUMERIC_MAP,
        "excluded": {3: "disputed_label_excluded"},
        "disputed_policy": "DISPUTED is excluded, never mapped to NOT_ENTITLED.",
    }


def evidence_policy(args: argparse.Namespace, preset: str) -> dict[str, Any]:
    return {
        "preset": preset,
        "mode": args.evidence_mode,
        "join_separator": args.evidence_join_sep,
        "max_evidence_chars": args.max_evidence_chars,
        "dict_key_preference": ["evidence", "text", "sentence", "content", "passage", "context"],
        "empty_after_flattening": "reject",
    }


def base_report(args: argparse.Namespace) -> dict[str, Any]:
    hf_dataset = DATASET_BY_PRESET[args.preset]
    source_dataset = args.source_dataset or hf_dataset
    return {
        "decision": "STAGE43B1_FACTVER_ACQUISITION_FAILED",
        "preset": args.preset,
        "hf_dataset": hf_dataset,
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
        "field_mapping": {},
        "label_mapping": label_mapping_description(args.preset),
        "numeric_label_policy": numeric_label_policy(args.preset),
        "evidence_policy": evidence_policy(args, args.preset),
        "sample_accepted_rows": [],
        "sample_rejected_rows": [],
        "risks": [],
        "recommendation": "",
        "leakage_policy": LEAKAGE_POLICY,
    }


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    report = base_report(args)
    hf_dataset = report["hf_dataset"]
    source_dataset = report["source_dataset"]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        report["recommendation"] = f"Install HuggingFace datasets before running acquisition: {exc}"
        report["rejection_reason_counts"] = {"load_or_parse_failure": 1}
        return report

    try:
        dataset = load_dataset(
            hf_dataset,
            split=args.split,
            trust_remote_code=args.trust_remote_code,
        )
        field_mapping = resolve_fields(args.preset, dataset)
    except Exception as exc:
        report["recommendation"] = (
            f"Dataset loading/parsing failed for {hf_dataset!r} split {args.split!r}: {exc}"
        )
        report["rejection_reason_counts"] = {"load_or_parse_failure": 1}
        report["risks"] = ["No rows acquired because HuggingFace loading or field resolution failed."]
        return report

    report["field_mapping"] = field_mapping
    label_field = field_mapping["label_field"]
    classlabel_names = get_classlabel_names(dataset, label_field)
    indices = choose_indices(dataset, args)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_label_counts: Counter[str] = Counter()
    source_label_counts: Counter[str] = Counter()
    rejection_reason_counts: Counter[str] = Counter()
    risks: set[str] = set()
    seen: set[tuple[str, str, str]] = set()

    for row_index in indices:
        raw_record = dataset[row_index]
        record = dict(raw_record) if isinstance(raw_record, dict) else {"value": raw_record}

        raw_claim = record.get(field_mapping["claim_field"])
        raw_evidence = record.get(field_mapping["evidence_field"])
        raw_label = record.get(label_field)

        claim = str(raw_claim).strip() if raw_claim is not None else ""
        evidence, evidence_notes = flatten_evidence(
            raw_evidence,
            mode=args.evidence_mode,
            join_sep=args.evidence_join_sep,
        )
        if args.max_evidence_chars is not None and len(evidence) > args.max_evidence_chars:
            evidence = evidence[: args.max_evidence_chars].strip()
            evidence_notes["evidence_truncated_to_chars"] = args.max_evidence_chars

        source_label_name, used_classlabel_name = source_label_from_classlabel(raw_label, classlabel_names)
        source_label_raw = json_safe(raw_label)
        if source_label_name:
            source_label_counts[source_label_name] += 1

        mapped_label, label_rejection, numeric_note = map_label(
            args.preset,
            raw_label,
            source_label_name,
            used_classlabel_name,
        )
        if numeric_note:
            risks.add(numeric_note)

        reason = None
        if not claim or len(claim) < args.min_text_chars:
            reason = "missing_or_too_short_claim"
        elif not evidence or len(evidence) < args.min_text_chars:
            reason = "missing_or_too_short_evidence"
        elif label_rejection is not None:
            reason = label_rejection

        if reason is None and args.dedupe:
            key = (claim, evidence, str(mapped_label))
            if key in seen:
                reason = "duplicate_claim_evidence_label"
            else:
                seen.add(key)

        if reason is not None:
            rejection_reason_counts[reason] += 1
            rejected.append(make_rejected(row_index, reason, raw_label, record, claim, evidence))
            continue

        original_id = None
        id_field = field_mapping.get("id_field")
        if id_field and record.get(id_field) is not None:
            original_id = str(record.get(id_field))
        row_id = original_id or f"{source_dataset}_{args.split}_{row_index}"

        metadata = {
            "row_index": row_index,
            "hf_dataset": hf_dataset,
            "hf_config": None,
            "source_split": args.split,
            "preset": args.preset,
            "source_label_raw": source_label_raw,
            "source_label_name": source_label_name,
            "dropped_disputed": False,
            **evidence_notes,
        }
        if original_id is not None:
            metadata["original_id"] = original_id
        if used_classlabel_name:
            metadata["source_label_resolved_from_classlabel"] = True
        if numeric_note:
            metadata["numeric_label_policy_note"] = numeric_note

        out_row = {
            "id": row_id,
            "claim": claim,
            "evidence": evidence,
            "label": mapped_label,
            "source_dataset": source_dataset,
            "source_label": source_label_name or str(source_label_raw),
            "stage43_split": "external_validation",
            "metadata": metadata,
        }
        accepted.append(out_row)
        accepted_label_counts[str(mapped_label)] += 1

    if not args.dry_run:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as fh:
            for row in accepted:
                fh.write(json.dumps(to_jsonable(row), ensure_ascii=False))
                fh.write("\n")

        args.rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.rejected_jsonl.open("w", encoding="utf-8") as fh:
            for row in rejected:
                fh.write(json.dumps(to_jsonable(row), ensure_ascii=False))
                fh.write("\n")

    decision, recommendation = classify_decision(accepted_label_counts)
    report.update(
        {
            "decision": decision,
            "total_rows_seen": len(indices),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "accepted_label_counts": dict(accepted_label_counts),
            "source_label_counts": dict(source_label_counts),
            "rejection_reason_counts": dict(rejection_reason_counts),
            "sample_accepted_rows": accepted[:SAMPLE_LIMIT],
            "sample_rejected_rows": rejected[:SAMPLE_LIMIT],
            "risks": sorted(risks)
            + [
                "HuggingFace source schema or label metadata may change over time.",
                "Acquired rows must remain external-evaluation-only under the leakage policy.",
            ],
            "recommendation": recommendation,
        }
    )
    if args.preset == "climate_fever":
        report["risks"].append("Climate-FEVER DISPUTED rows are excluded for clean 3-way evaluation.")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage43-B1 Targeted Fact-Verification Acquisition Report",
        "",
        "Preparation/acquisition only. This report is produced by "
        "`scripts/acquire_stage43_factver_external_validation.py` and does not "
        "include model training, model evaluation, external probe evaluation, "
        "Kaggle commands, smoke tests, or local model execution.",
        "",
        "## 1. Overall decision",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "## 2. Dataset source/provenance",
        "",
        f"- Preset: `{report['preset']}`",
        f"- HuggingFace dataset: `{report['hf_dataset']}`",
        f"- Split: `{report['split']}`",
        f"- Source dataset: `{report['source_dataset']}`",
        f"- Output JSONL: `{report['output_jsonl']}`",
        f"- Rejected JSONL: `{report['rejected_jsonl']}`",
        "",
        "## 3. Preset",
        "",
        f"`{report['preset']}`",
        "",
        "## 4. Field mapping",
        "",
        "```json",
        json.dumps(report["field_mapping"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## 5. Label mapping",
        "",
        "```json",
        json.dumps(report["label_mapping"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## 6. Evidence handling",
        "",
        "```json",
        json.dumps(report["evidence_policy"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## 7. Accepted/rejected summary",
        "",
        f"- Total rows seen: {report['total_rows_seen']}",
        f"- Accepted rows: {report['accepted_rows']}",
        f"- Rejected rows: {report['rejected_rows']}",
        "",
        "## 8. Label distribution",
        "",
        "```json",
        json.dumps(report["accepted_label_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## 9. Sample accepted rows",
        "",
    ]
    if report["sample_accepted_rows"]:
        lines.extend(["```json", json.dumps(report["sample_accepted_rows"], indent=2, ensure_ascii=False), "```"])
    else:
        lines.append("None.")
    lines.extend(["", "## 10. Sample rejected rows", ""])
    if report["sample_rejected_rows"]:
        lines.extend(["```json", json.dumps(report["sample_rejected_rows"], indent=2, ensure_ascii=False), "```"])
    else:
        lines.append("None.")
    lines.extend(["", "## 11. Risks", ""])
    if report["risks"]:
        lines.extend(f"- {risk}" for risk in report["risks"])
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## 12. Recommendation",
            "",
            report["recommendation"],
            "",
            "## 13. Leakage policy",
            "",
            report["leakage_policy"],
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["vitaminc", "climate_fever"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--rejected-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--min-text-chars", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=None)
    parser.add_argument("--evidence-join-sep", default=" ")
    parser.add_argument("--evidence-mode", choices=["first", "join_all"], default="first")
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = acquire(args)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(report), fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    with args.report_md.open("w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))

    print(f"Decision: {report['decision']}")
    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")
    if not args.dry_run:
        print(f"Wrote {args.output_jsonl}")
        print(f"Wrote {args.rejected_jsonl}")
    return 0 if report["decision"] != "STAGE43B1_FACTVER_ACQUISITION_FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
