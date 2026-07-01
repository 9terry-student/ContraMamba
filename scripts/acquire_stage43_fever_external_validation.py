"""Stage43-B1 FEVER/VitaminC-first HuggingFace acquisition.

Loads a user-specified FEVER/VitaminC-style HuggingFace dataset with
``datasets.load_dataset`` and converts accepted rows into the ContraMamba
Stage43 external-validation JSONL schema.

This script is acquisition-only. It does not train, evaluate, run local
models, infer labels from model predictions, or select checkpoints. Acquired
rows are external-evaluation-only and must not be used for training,
calibration, threshold selection, checkpoint selection, or loss design.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

SAMPLE_LIMIT = 5

CLAIM_ALIASES = ("claim", "statement", "hypothesis", "query")
EVIDENCE_ALIASES = (
    "evidence",
    "evidence_text",
    "context",
    "passage",
    "text",
    "retrieved_evidence",
    "gold_evidence",
)
LABEL_ALIASES = ("label", "gold", "gold_label", "verdict", "answer")

DEFAULT_LABEL_MAP: dict[str, str] = {
    "supports": "SUPPORT",
    "support": "SUPPORT",
    "supported": "SUPPORT",
    "entailment": "SUPPORT",
    "true": "SUPPORT",
    "1": "SUPPORT",
    "refutes": "REFUTE",
    "refute": "REFUTE",
    "refuted": "REFUTE",
    "contradiction": "REFUTE",
    "false": "REFUTE",
    "-1": "REFUTE",
    "not enough info": "NOT_ENTITLED",
    "nei": "NOT_ENTITLED",
    "not_enough_info": "NOT_ENTITLED",
    "not enough information": "NOT_ENTITLED",
    "unknown": "NOT_ENTITLED",
    "not_entitled": "NOT_ENTITLED",
    "0": "NOT_ENTITLED",
}

CLASSLABEL_NUMERIC_NAMES = {"int", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}

LEAKAGE_POLICY = (
    "Stage43-B1 acquired data is external-evaluation-only. It must not be used "
    "for training, calibration, threshold selection, checkpoint selection, loss "
    "design, or any other model-selection feedback loop."
)


def normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_") if value is not None else ""


def compact_value(value: Any, max_chars: int = 500) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"
    if isinstance(value, dict):
        return {str(k): compact_value(v, max_chars=max_chars) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [compact_value(v, max_chars=max_chars) for v in list(value)[:20]]
    return value


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def load_label_map(label_map_json: str | None) -> dict[str, str]:
    if not label_map_json:
        return DEFAULT_LABEL_MAP.copy()

    candidate_path = Path(label_map_json)
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8") as fh:
            raw_map = json.load(fh)
    else:
        raw_map = json.loads(label_map_json)

    if not isinstance(raw_map, dict):
        raise ValueError("--label-map-json must be a JSON object or a path to one.")

    parsed: dict[str, str] = {}
    canonical_labels = {"SUPPORT", "REFUTE", "NOT_ENTITLED"}
    for key, value in raw_map.items():
        if str(key).upper() in canonical_labels and isinstance(value, list):
            for alias in value:
                parsed[normalize_key(alias)] = str(key).upper()
        elif str(value).upper() in canonical_labels:
            parsed[normalize_key(key)] = str(value).upper()
        else:
            raise ValueError(
                "--label-map-json values must be canonical labels or canonical-label lists."
            )
    return parsed


def get_column_names(dataset: Any) -> list[str]:
    names = getattr(dataset, "column_names", None)
    if names:
        return list(names)
    features = getattr(dataset, "features", None)
    if features:
        return list(features.keys())
    first = dataset[0] if len(dataset) else {}
    return list(first.keys()) if isinstance(first, dict) else []


def infer_field(column_names: list[str], explicit: str | None, aliases: tuple[str, ...], field_name: str) -> str:
    if explicit:
        if explicit not in column_names:
            raise ValueError(f"Requested {field_name} field '{explicit}' is not in dataset columns.")
        return explicit
    for alias in aliases:
        if alias in column_names:
            return alias
    raise ValueError(f"Could not infer {field_name} field from columns: {column_names}")


def build_field_mapping(args: argparse.Namespace, dataset: Any) -> dict[str, str | None]:
    column_names = get_column_names(dataset)
    if args.preset in {"fever_claim_evidence", "vitaminc_claim_evidence"}:
        mapping = {
            "claim_field": args.claim_field or "claim",
            "evidence_field": args.evidence_field or "evidence",
            "label_field": args.label_field or "label",
            "id_field": args.id_field,
        }
    elif args.preset == "manual":
        if not (args.claim_field and args.evidence_field and args.label_field):
            raise ValueError("manual preset requires --claim-field, --evidence-field, and --label-field.")
        mapping = {
            "claim_field": args.claim_field,
            "evidence_field": args.evidence_field,
            "label_field": args.label_field,
            "id_field": args.id_field,
        }
    else:
        mapping = {
            "claim_field": infer_field(column_names, args.claim_field, CLAIM_ALIASES, "claim"),
            "evidence_field": infer_field(column_names, args.evidence_field, EVIDENCE_ALIASES, "evidence"),
            "label_field": infer_field(column_names, args.label_field, LABEL_ALIASES, "label"),
            "id_field": args.id_field,
        }

    for label, field in mapping.items():
        if field is not None and field not in column_names:
            raise ValueError(f"{label} '{field}' is not in dataset columns: {column_names}")
    return mapping


def flatten_evidence(value: Any, join_sep: str) -> tuple[str, dict[str, Any]]:
    notes = {"evidence_was_structured": False, "evidence_flatten_strategy": "string"}

    def flatten_inner(item: Any) -> list[str]:
        if item is None:
            return []
        if isinstance(item, str):
            text = item.strip()
            return [text] if text else []
        if isinstance(item, (list, tuple)):
            notes["evidence_was_structured"] = True
            notes["evidence_flatten_strategy"] = "recursive_list_join"
            segments: list[str] = []
            for child in item:
                segments.extend(flatten_inner(child))
            return segments
        if isinstance(item, dict):
            notes["evidence_was_structured"] = True
            preferred = ("evidence", "text", "sentence", "content", "passage", "context")
            for key in preferred:
                if key in item:
                    notes["evidence_flatten_strategy"] = f"dict_preferred_{key}"
                    return flatten_inner(item[key])
            notes["evidence_flatten_strategy"] = "dict_recursive_string_values"
            segments = []
            for child in item.values():
                if isinstance(child, (str, list, tuple, dict)):
                    segments.extend(flatten_inner(child))
            return segments
        text = str(item).strip()
        if not text or text.replace(".", "", 1).replace("-", "", 1).isdigit():
            return []
        notes["evidence_flatten_strategy"] = "scalar_string_cast"
        return [text]

    flattened = join_sep.join(segment for segment in flatten_inner(value) if segment).strip()
    return flattened, notes


def get_classlabel_names(dataset: Any, label_field: str) -> list[str] | None:
    features = getattr(dataset, "features", None)
    if not features or label_field not in features:
        return None
    feature = features[label_field]
    names = getattr(feature, "names", None)
    return list(names) if names else None


def resolve_source_label(raw_label: Any, classlabel_names: list[str] | None) -> tuple[str | None, Any | None, bool]:
    original_numeric_label = None
    used_classlabel = False
    if raw_label is None:
        return None, original_numeric_label, used_classlabel
    if classlabel_names is not None and isinstance(raw_label, int):
        original_numeric_label = raw_label
        if 0 <= raw_label < len(classlabel_names):
            return classlabel_names[raw_label], original_numeric_label, True
    return str(raw_label), original_numeric_label, used_classlabel


def is_numeric_label_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    return text.lstrip("-").isdigit()


def map_label(
    source_label: str | None,
    raw_label: Any,
    label_map: dict[str, str],
    strict_labels: bool,
    allow_raw_numeric_mapping: bool,
) -> str | None:
    if source_label is None or not str(source_label).strip():
        return None
    if is_numeric_label_value(raw_label) and not allow_raw_numeric_mapping:
        return None
    mapped = label_map.get(normalize_key(source_label))
    if mapped:
        return mapped
    if strict_labels:
        return None
    if not isinstance(raw_label, int):
        return label_map.get(str(source_label).strip().lower())
    return None


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
    record: dict[str, Any],
    source_label: str | None,
    claim: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "rejection_reason": reason,
        "source_label": source_label or "",
        "original_row": compact_value(record),
        "detected_claim_snippet": claim[:200],
        "detected_evidence_snippet": evidence[:200],
    }


def source_dataset_name(args: argparse.Namespace) -> str:
    return args.source_dataset or (
        f"{args.hf_dataset}/{args.hf_config}" if args.hf_config else args.hf_dataset
    )


def classify_decision(accepted_label_counts: Counter[str]) -> tuple[str, str]:
    accepted_rows = sum(accepted_label_counts.values())
    distinct = len(accepted_label_counts)
    required = {"SUPPORT", "REFUTE", "NOT_ENTITLED"}
    if accepted_rows == 0:
        return (
            "STAGE43B1_FEVER_EXTERNAL_ACQUISITION_NO_VALID_ROWS",
            "No rows were accepted. Review field mapping, label mapping, and rejected rows before Stage43-B2.",
        )
    if required.issubset(set(accepted_label_counts)):
        return (
            "STAGE43B1_FEVER_EXTERNAL_ACQUISITION_READY",
            "Acquisition produced all three canonical labels and is ready for future Stage43-B2 eval-only use.",
        )
    if distinct >= 2:
        return (
            "STAGE43B1_FEVER_EXTERNAL_ACQUISITION_PARTIAL_READY",
            "Acquisition produced at least two labels. Review whether the missing label is expected before Stage43-B2.",
        )
    return (
        "STAGE43B1_FEVER_EXTERNAL_ACQUISITION_LABEL_IMBALANCED",
        "Acquisition produced only one label. A broader source split/subset is recommended before Stage43-B2.",
    )


def build_failure_report(args: argparse.Namespace, exc: Exception) -> dict[str, Any]:
    return {
        "decision": "STAGE43B1_FEVER_EXTERNAL_ACQUISITION_FAILED",
        "hf_dataset": args.hf_dataset,
        "hf_config": args.hf_config,
        "split": args.split,
        "preset": args.preset,
        "source_dataset": source_dataset_name(args),
        "output_jsonl": str(args.output_jsonl),
        "rejected_jsonl": str(args.rejected_jsonl),
        "total_rows_seen": 0,
        "accepted_rows": 0,
        "rejected_rows": 0,
        "accepted_label_counts": {},
        "source_label_counts": {},
        "rejection_reason_counts": {"load_or_parse_failure": 1},
        "field_mapping": {},
        "label_mapping": {},
        "used_classlabel_names": None,
        "sample_accepted_rows": [],
        "sample_rejected_rows": [{"rejection_reason": "load_or_parse_failure", "error": str(exc)}],
        "risks": ["Dataset loading/parsing failed; no external validation rows were acquired."],
        "recommendation": f"Fix the HuggingFace dataset/config/split or field arguments, then rerun acquisition. Error: {exc}",
        "leakage_policy": LEAKAGE_POLICY,
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from datasets import load_dataset

        if args.hf_config:
            dataset = load_dataset(
                args.hf_dataset,
                args.hf_config,
                split=args.split,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            dataset = load_dataset(
                args.hf_dataset,
                split=args.split,
                trust_remote_code=args.trust_remote_code,
            )

        field_mapping = build_field_mapping(args, dataset)
        label_field = str(field_mapping["label_field"])
        classlabel_names = get_classlabel_names(dataset, label_field)
        label_map = load_label_map(args.label_map_json)
        indices = choose_indices(dataset, args)

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        accepted_label_counts: Counter[str] = Counter()
        source_label_counts: Counter[str] = Counter()
        rejection_reason_counts: Counter[str] = Counter()
        seen_keys: set[tuple[str, str, str]] = set()
        source_name = source_dataset_name(args)

        for row_index in indices:
            raw_record = dataset[row_index]
            record = dict(raw_record) if isinstance(raw_record, dict) else {"value": raw_record}
            raw_claim = record.get(field_mapping["claim_field"])
            raw_evidence = record.get(field_mapping["evidence_field"])
            raw_label = record.get(label_field)

            claim = str(raw_claim).strip() if raw_claim is not None else ""
            evidence, evidence_notes = flatten_evidence(raw_evidence, args.evidence_join_sep)
            if args.max_evidence_chars is not None and len(evidence) > args.max_evidence_chars:
                evidence = evidence[: args.max_evidence_chars].strip()
                evidence_notes["evidence_truncated_to_chars"] = args.max_evidence_chars

            source_label, original_numeric_label, used_classlabel = resolve_source_label(
                raw_label, classlabel_names
            )
            if source_label:
                source_label_counts[source_label] += 1

            reason = None
            mapped_label = None
            if not claim or len(claim) < args.min_text_chars:
                reason = "missing_or_too_short_claim"
            elif not evidence or len(evidence) < args.min_text_chars:
                reason = "missing_or_too_short_evidence"
            elif raw_label is None or not str(raw_label).strip():
                reason = "missing_label"
            else:
                allow_raw_numeric_mapping = args.label_map_json is not None or used_classlabel
                mapped_label = map_label(
                    source_label,
                    raw_label,
                    label_map,
                    args.strict_labels,
                    allow_raw_numeric_mapping,
                )
                if mapped_label is None:
                    reason = "ambiguous_or_unmapped_label"

            if reason is None and args.dedupe:
                key = (claim, evidence, str(mapped_label))
                if key in seen_keys:
                    reason = "duplicate_claim_evidence_label"
                else:
                    seen_keys.add(key)

            if reason is not None:
                rejection_reason_counts[reason] += 1
                rejected.append(make_rejected(row_index, reason, record, source_label, claim, evidence))
                continue

            original_id = None
            if args.id_field and record.get(args.id_field) is not None:
                original_id = str(record.get(args.id_field))
            row_id = original_id or f"{source_name}_{args.split}_{row_index}"

            metadata = {
                "row_index": row_index,
                "hf_dataset": args.hf_dataset,
                "hf_config": args.hf_config,
                "source_config": args.hf_config,
                "source_split": args.split,
                "preset": args.preset,
                **evidence_notes,
            }
            if original_id is not None:
                metadata["original_id"] = original_id
            if original_numeric_label is not None:
                metadata["original_numeric_label"] = original_numeric_label
            if used_classlabel:
                metadata["source_label_resolved_from_classlabel"] = True

            out_row = {
                "id": row_id,
                "claim": claim,
                "evidence": evidence,
                "label": mapped_label,
                "source_dataset": source_name,
                "source_label": source_label or "",
                "stage43_split": "external_validation",
                "metadata": metadata,
            }
            accepted.append(out_row)
            accepted_label_counts[str(mapped_label)] += 1

        if not args.dry_run:
            args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.output_jsonl.open("w", encoding="utf-8") as fh:
                for row in accepted:
                    fh.write(json.dumps(jsonable(row), ensure_ascii=False))
                    fh.write("\n")

            args.rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.rejected_jsonl.open("w", encoding="utf-8") as fh:
                for row in rejected:
                    fh.write(json.dumps(jsonable(row), ensure_ascii=False))
                    fh.write("\n")

        decision, recommendation = classify_decision(accepted_label_counts)
        risks = [
            "HuggingFace source availability, schema, or label names may change over time.",
            "Structured FEVER-family evidence is flattened conservatively; review samples before evaluation.",
            "Acquired rows must remain external-evaluation-only under the leakage policy.",
        ]
        if "NOT_ENTITLED" not in accepted_label_counts:
            risks.append("Accepted rows do not include NOT_ENTITLED; insufficient-evidence behavior may be untested.")

        return {
            "decision": decision,
            "hf_dataset": args.hf_dataset,
            "hf_config": args.hf_config,
            "split": args.split,
            "preset": args.preset,
            "source_dataset": source_name,
            "output_jsonl": str(args.output_jsonl),
            "rejected_jsonl": str(args.rejected_jsonl),
            "dry_run": args.dry_run,
            "total_rows_seen": len(indices),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "accepted_label_counts": dict(accepted_label_counts),
            "source_label_counts": dict(source_label_counts),
            "rejection_reason_counts": dict(rejection_reason_counts),
            "field_mapping": field_mapping,
            "label_mapping": label_map,
            "used_classlabel_names": classlabel_names,
            "sample_accepted_rows": accepted[:SAMPLE_LIMIT],
            "sample_rejected_rows": rejected[:SAMPLE_LIMIT],
            "risks": risks,
            "recommendation": recommendation,
            "leakage_policy": LEAKAGE_POLICY,
        }
    except Exception as exc:
        return build_failure_report(args, exc)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Stage43-B1 FEVER/VitaminC HuggingFace Acquisition Report")
    lines.append("")
    lines.append(
        "Preparation/acquisition only. This report describes a run of "
        "`scripts/acquire_stage43_fever_external_validation.py`; it does not "
        "include model training, model evaluation, external probe evaluation, "
        "Kaggle commands, smoke tests, or local model execution."
    )
    lines.append("")
    lines.append("## 1. Overall decision")
    lines.append("")
    lines.append(f"**Decision:** `{report['decision']}`")
    lines.append("")
    lines.append("## 2. Dataset source/provenance")
    lines.append("")
    lines.append(f"- HuggingFace dataset: `{report['hf_dataset']}`")
    lines.append(f"- HuggingFace config: `{report.get('hf_config')}`")
    lines.append(f"- Split: `{report['split']}`")
    lines.append(f"- Source dataset name: `{report['source_dataset']}`")
    lines.append(f"- Output JSONL: `{report['output_jsonl']}`")
    lines.append(f"- Rejected JSONL: `{report['rejected_jsonl']}`")
    lines.append("")
    lines.append("## 3. Preset and field mapping")
    lines.append("")
    lines.append(f"- Preset: `{report['preset']}`")
    lines.append("```json")
    lines.append(json.dumps(report["field_mapping"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## 4. Evidence flattening behavior")
    lines.append("")
    lines.append(
        "String evidence is preserved. Lists/tuples are recursively flattened and "
        "joined. Dictionaries prefer `evidence`, `text`, `sentence`, `content`, "
        "`passage`, then `context`; otherwise string-like nested values are "
        "flattened. Empty or numeric-only scalar evidence is rejected."
    )
    lines.append("")
    lines.append("## 5. Label mapping")
    lines.append("")
    lines.append(f"- ClassLabel names used: `{report.get('used_classlabel_names')}`")
    lines.append("```json")
    lines.append(json.dumps(report["label_mapping"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## 6. Accepted/rejected summary")
    lines.append("")
    lines.append(f"- Total rows seen: {report['total_rows_seen']}")
    lines.append(f"- Accepted rows: {report['accepted_rows']}")
    lines.append(f"- Rejected rows: {report['rejected_rows']}")
    lines.append("")
    lines.append("## 7. Label distribution")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["accepted_label_counts"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## 8. Sample accepted rows")
    lines.append("")
    if report["sample_accepted_rows"]:
        lines.append("```json")
        lines.append(json.dumps(report["sample_accepted_rows"], indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("None.")
    lines.append("")
    lines.append("## 9. Sample rejected rows")
    lines.append("")
    if report["sample_rejected_rows"]:
        lines.append("```json")
        lines.append(json.dumps(report["sample_rejected_rows"], indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("None.")
    lines.append("")
    lines.append("## 10. Risks")
    lines.append("")
    for risk in report["risks"]:
        lines.append(f"- {risk}")
    lines.append("")
    lines.append("## 11. Recommendation")
    lines.append("")
    lines.append(report["recommendation"])
    lines.append("")
    lines.append("## 12. Leakage policy")
    lines.append("")
    lines.append(report["leakage_policy"])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--rejected-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument(
        "--preset",
        choices=["fever_claim_evidence", "vitaminc_claim_evidence", "auto_fever", "manual"],
        default="auto_fever",
    )
    parser.add_argument("--claim-field", default=None)
    parser.add_argument("--evidence-field", default=None)
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--id-field", default=None)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--min-text-chars", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=None)
    parser.add_argument("--evidence-join-sep", default=" ")
    parser.add_argument("--strict-labels", action="store_true")
    parser.add_argument("--label-map-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
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
    if not args.dry_run:
        print(f"Wrote {args.output_jsonl}")
        print(f"Wrote {args.rejected_jsonl}")
    return 0 if report["decision"] != "STAGE43B1_FEVER_EXTERNAL_ACQUISITION_FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

