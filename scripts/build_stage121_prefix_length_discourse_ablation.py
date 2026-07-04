"""Build Stage121-A prefix length/discourse ablation diagnostics."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "controlled_v5_v3_without_time_swap.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "stage121_prefix_length_discourse_ablation.jsonl"
SOURCE_DATASET = "controlled_v5_v3_without_time_swap"
DECISION = "STAGE121A_PREFIX_LENGTH_DISCOURSE_ABLATION_GENERATED"
DEFAULT_TARGET_LABELS = ("SUPPORT", "REFUTE", "NOT_ENTITLED")
LABEL_FIELDS = ("gold_label", "label", "final_label", "target", "target_label")
NUMERIC_LABEL_TO_TEXT = {0: "REFUTE", 1: "NOT_ENTITLED", 2: "SUPPORT", "0": "REFUTE", "1": "NOT_ENTITLED", "2": "SUPPORT"}
LABEL_ALIASES = {
    "SUPPORT": "SUPPORT", "SUPPORTS": "SUPPORT", "REFUTE": "REFUTE", "REFUTES": "REFUTE",
    "NOT_ENTITLED": "NOT_ENTITLED", "NEI": "NOT_ENTITLED", "NOT_ENOUGH_INFO": "NOT_ENTITLED",
    "NOT ENOUGH INFO": "NOT_ENTITLED",
}
FAMILY_PREFIXES: dict[str, tuple[str, str]] = {
    "original_control": ("", "none"),
    "short_neutral_fragment": ("Additional context:", "short_fragment"),
    "short_neutral_sentence": ("Context is provided.", "short_sentence"),
    "length_matched_nonsense": ("alpha beta gamma delta epsilon zeta eta theta", "length_only"),
    "length_matched_neutral_tokens": ("context note information detail passage text content statement", "weak_neutral_tokens"),
    "long_neutral_no_order": ("This text contains background information and relevant evidence.", "long_neutral_sentence"),
    "long_discourse_no_temporal": ("The passage contains context and the relevant statement.", "long_discourse_sentence"),
    "old_before_prefix": ("The following passage provides broader context before the relevant statement.", "temporal_discourse"),
    "explicit_before_temporal": ("Before the relevant statement, the passage provides broader context.", "temporal_discourse"),
}
DEFAULT_FAMILIES = (
    "original_control", "short_neutral_fragment", "short_neutral_sentence", "length_matched_nonsense",
    "length_matched_neutral_tokens", "long_neutral_no_order", "long_discourse_no_temporal", "old_before_prefix",
)
TEMPORAL_TRIGGERS = {"before", "after", "earlier", "later", "previous", "following", "prior", "subsequent", "during", "while", "when"}
DISCOURSE_MARKERS = {"context", "passage", "statement", "relevant", "evidence", "background", "information", "contains", "provides"}
SURFACE_KEYS = (
    "claim_token_len", "evidence_token_len", "prefix_token_len", "length_ratio_evidence_to_claim",
    "token_jaccard", "claim_token_coverage_in_evidence", "evidence_token_coverage_in_claim",
    "cap_entity_jaccard", "claim_cap_entity_coverage_in_evidence", "contains_before", "contains_after",
    "contains_following", "contains_temporal_trigger", "contains_discourse_marker", "is_sentence_prefix",
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CAP_CHUNK_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def first_present(row: dict[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        if field in row and row[field] not in (None, ""):
            return row[field]
    return None


def normalize_label(value: Any) -> str | None:
    if value in NUMERIC_LABEL_TO_TEXT:
        return NUMERIC_LABEL_TO_TEXT[value]
    if isinstance(value, float) and value.is_integer():
        return NUMERIC_LABEL_TO_TEXT.get(str(int(value)))
    return LABEL_ALIASES.get(str(value).strip().upper().replace("-", "_"))


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def cap_entities(text: str) -> set[str]:
    chunks: set[str] = set()
    for match in CAP_CHUNK_RE.finditer(text):
        chunk = " ".join(match.group(0).split())
        if len(chunk) > 1:
            chunks.add(chunk.lower())
    return chunks


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def surface_features(claim: str, evidence: str, prefix_text: str) -> dict[str, float | int]:
    claim_tokens = tokens(claim)
    evidence_tokens = tokens(evidence)
    claim_set = set(claim_tokens)
    evidence_set = set(evidence_tokens)
    claim_entities = cap_entities(claim)
    evidence_entities = cap_entities(evidence)
    token_union = claim_set | evidence_set
    entity_union = claim_entities | evidence_entities
    prefix_stripped = prefix_text.strip()
    return {
        "claim_token_len": len(claim_tokens),
        "evidence_token_len": len(evidence_tokens),
        "prefix_token_len": len(tokens(prefix_text)),
        "length_ratio_evidence_to_claim": safe_ratio(len(evidence_tokens), len(claim_tokens)),
        "token_jaccard": safe_ratio(len(claim_set & evidence_set), len(token_union)),
        "claim_token_coverage_in_evidence": safe_ratio(len(claim_set & evidence_set), len(claim_set)),
        "evidence_token_coverage_in_claim": safe_ratio(len(claim_set & evidence_set), len(evidence_set)),
        "cap_entity_jaccard": safe_ratio(len(claim_entities & evidence_entities), len(entity_union)),
        "claim_cap_entity_coverage_in_evidence": safe_ratio(len(claim_entities & evidence_entities), len(claim_entities)),
        "contains_before": int("before" in evidence_set),
        "contains_after": int("after" in evidence_set),
        "contains_following": int("following" in evidence_set),
        "contains_temporal_trigger": int(bool(evidence_set & TEMPORAL_TRIGGERS)),
        "contains_discourse_marker": int(bool(evidence_set & DISCOURSE_MARKERS)),
        "is_sentence_prefix": int(bool(prefix_stripped) and prefix_stripped[-1:] in {".", "!", "?"}),
    }


def feature_deltas(before: dict[str, float | int], after: dict[str, float | int]) -> dict[str, float | int]:
    return {key: after[key] - before[key] for key in SURFACE_KEYS}


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], Counter[str], int]:
    rows: list[dict[str, Any]] = []
    warnings: Counter[str] = Counter()
    n_input_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            n_input_rows += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                warnings["malformed_json"] += 1
                continue
            if not isinstance(row, dict):
                warnings["non_object_json"] += 1
                continue
            if not str(row.get("claim", "")).strip() or not str(row.get("evidence", "")).strip():
                warnings["missing_claim_or_evidence"] += 1
                continue
            label = normalize_label(first_present(row, LABEL_FIELDS))
            if label is None:
                warnings["missing_or_unknown_label"] += 1
                continue
            row["_stage121_gold_label"] = label
            row["_stage121_row_index"] = len(rows)
            rows.append(row)
    return rows, warnings, n_input_rows


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or f"row_{row['_stage121_row_index']:06d}")


def select_sources(rows: Sequence[dict[str, Any]], *, target_labels: set[str], max_rows: int | None, seed: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["_stage121_gold_label"] in target_labels]
    if max_rows is None or len(eligible) <= max_rows:
        return sorted(eligible, key=lambda row: row["_stage121_row_index"])
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_label[row["_stage121_gold_label"]].append(row)
    allocations: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []
    total = len(eligible)
    for label in sorted(by_label):
        exact = max_rows * len(by_label[label]) / total
        allocations[label] = int(exact)
        fractional.append((exact - int(exact), label))
    remaining = max_rows - sum(allocations.values())
    for _, label in sorted(fractional, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocations[label] += 1
    selected: list[dict[str, Any]] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        selected.extend(group[: allocations[label]])
    return sorted(selected, key=lambda row: row["_stage121_row_index"])


def make_evidence(prefix_text: str, original_evidence: str) -> str:
    return original_evidence if not prefix_text else f"{prefix_text} {original_evidence}"


def make_output_row(source: dict[str, Any], family: str) -> dict[str, Any]:
    prefix_text, prefix_category = FAMILY_PREFIXES[family]
    claim = str(source["claim"])
    original_evidence = str(source["evidence"])
    evidence = make_evidence(prefix_text, original_evidence)
    before = surface_features(claim, original_evidence, "")
    after = surface_features(claim, evidence, prefix_text)
    delta = feature_deltas(before, after)
    metadata = dict(source.get("metadata")) if isinstance(source.get("metadata"), dict) else {}
    metadata["stage121"] = {
        "family": family, "source_id": row_id(source), "prefix_text": prefix_text,
        "prefix_category": prefix_category, "label_preserved": True, "uses_external_data": False,
        "source_dataset": SOURCE_DATASET, "surface_before": before, "surface_after": after, "surface_delta": delta,
    }
    out: dict[str, Any] = dict(source)
    out.pop("_stage121_gold_label", None)
    out.pop("_stage121_row_index", None)
    out.update({
        "id": f"stage121a_{family}__{row_id(source)}",
        "source_id": row_id(source),
        "claim": claim,
        "evidence": evidence,
        "gold_label": source["_stage121_gold_label"],
        "stage121_family": family,
        "stage121_is_prefix_length_discourse_ablation": True,
        "stage121_label_preserved": True,
        "stage121_uses_external_data": False,
        "stage121_source_dataset": SOURCE_DATASET,
        "stage121_prefix_text": prefix_text,
        "stage121_prefix_category": prefix_category,
        "stage121_original_claim": claim,
        "stage121_original_evidence": original_evidence,
        "stage121_surface_before": before,
        "stage121_surface_after": after,
        "stage121_surface_delta": delta,
        "metadata": metadata,
    })
    return out


def build_rows(rows: Sequence[dict[str, Any]], *, target_labels: set[str], families: Sequence[str], max_rows: int | None, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = select_sources(rows, target_labels=target_labels, max_rows=max_rows, seed=seed)
    output = [make_output_row(source, family) for source in sources for family in families]
    return output, sources


def median(values: Sequence[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def aggregate_surface(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[f"{record['stage121_family']}::{record['gold_label']}"].append(record)
    summary: dict[str, Any] = {}
    for key in sorted(grouped):
        family, label = key.split("::", 1)
        group = grouped[key]
        summary[key] = {
            "stage121_family": family,
            "gold_label": label,
            "count": len(group),
            "before_median": {feature: median([row["stage121_surface_before"][feature] for row in group]) for feature in SURFACE_KEYS},
            "after_median": {feature: median([row["stage121_surface_after"][feature] for row in group]) for feature in SURFACE_KEYS},
            "delta_median": {feature: median([row["stage121_surface_delta"][feature] for row in group]) for feature in SURFACE_KEYS},
        }
    return summary


def trigger_counts_by_family(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys = ("contains_before", "contains_after", "contains_following", "contains_temporal_trigger", "contains_discourse_marker", "is_sentence_prefix")
    for record in records:
        grouped[record["stage121_family"]].append(record)
    return {family: {key: int(sum(row["stage121_surface_after"][key] for row in rows)) for key in keys} for family, rows in sorted(grouped.items())}


def build_summary(*, input_path: Path, output_path: Path, n_input_rows: int, valid_rows: Sequence[dict[str, Any]], selected_sources: Sequence[dict[str, Any]], output_rows: Sequence[dict[str, Any]], warning_counts: Counter[str], families: Sequence[str], target_labels: Sequence[str], seed: int) -> dict[str, Any]:
    return {
        "stage": "Stage121-A",
        "decision": DECISION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "seed": seed,
        "target_labels": list(target_labels),
        "families": list(families),
        "n_input_rows": int(n_input_rows),
        "n_valid_source_rows": int(len(valid_rows)),
        "n_selected_source_rows": int(len(selected_sources)),
        "n_output_rows": int(len(output_rows)),
        "warning_counts": dict(sorted(warning_counts.items())),
        "label_counts_source": dict(sorted(Counter(row["_stage121_gold_label"] for row in selected_sources).items())),
        "label_counts_output": dict(sorted(Counter(row["gold_label"] for row in output_rows).items())),
        "family_counts": dict(sorted(Counter(row["stage121_family"] for row in output_rows).items())),
        "prefix_category_counts": dict(sorted(Counter(row["stage121_prefix_category"] for row in output_rows).items())),
        "surface_medians_by_family_and_label": aggregate_surface(output_rows),
        "trigger_counts_by_family": trigger_counts_by_family(output_rows),
        "no_external_data": True,
        "label_preserved": True,
        "source_auxiliary_fields_preserved": True,
        "next_stage": "Stage121-B evaluate prefix length/discourse ablation with Stage118 generic diagnostic path",
    }


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(counter: dict[str, int], key_name: str) -> list[str]:
    lines = [f"| {key_name} | count |", "| --- | ---: |"]
    for key, count in counter.items():
        lines.append(f"| {key} | {count} |")
    return lines


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage121-A Prefix Length/Discourse Ablation", "", "## Purpose", "",
        "Generate label-preserved diagnostic rows that isolate whether Stage120 prefix failures come from prefix length, sentence/discourse wrapper form, or temporal/discourse lexical cues.",
        "", "## Hypothesis", "",
        "If collapse tracks only token count, length-matched nonsense and weak neutral tokens should behave like long discourse prefixes. If collapse tracks discourse or temporal wording, the long discourse and before-prefix families should be worse than length-only controls.",
        "", "## Prefix Families", "",
    ]
    for family in summary["families"]:
        prefix, category = FAMILY_PREFIXES[family]
        lines.append(f"- `{family}` ({category}): `{prefix}`")
    lines.extend([
        "", "## Safety Constraints", "", "- Claims are never rewritten.", "- Gold labels are normalized and preserved.",
        "- No external data is read or used.", "- Generated rows are diagnostic-only.", "", "## Summary Tables", "",
        f"- Decision: `{summary['decision']}`", f"- Input rows: {summary['n_input_rows']}",
        f"- Valid source rows: {summary['n_valid_source_rows']}", f"- Selected source rows: {summary['n_selected_source_rows']}",
        f"- Output rows: {summary['n_output_rows']}", "", "### Source Label Counts", "",
    ])
    lines.extend(md_table(summary["label_counts_source"], "label"))
    lines.extend(["", "### Output Label Counts", ""])
    lines.extend(md_table(summary["label_counts_output"], "label"))
    lines.extend(["", "### Family Counts", ""])
    lines.extend(md_table(summary["family_counts"], "family"))
    lines.extend(["", "### Prefix Category Counts", ""])
    lines.extend(md_table(summary["prefix_category_counts"], "prefix_category"))
    lines.extend(["", "### Trigger Counts By Family", ""])
    trigger_keys = ("contains_before", "contains_after", "contains_following", "contains_temporal_trigger", "contains_discourse_marker", "is_sentence_prefix")
    lines.append("| family | " + " | ".join(trigger_keys) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(trigger_keys)) + " |")
    for family, counts in summary["trigger_counts_by_family"].items():
        lines.append("| " + family + " | " + " | ".join(str(counts[key]) for key in trigger_keys) + " |")
    lines.extend(["", "## Next-Stage Recommendation", "", summary["next_stage"] + ".", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_args(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    families = parse_csv(args.families)
    unknown_families = sorted(set(families) - set(FAMILY_PREFIXES))
    if unknown_families:
        raise ValueError(f"unknown families: {unknown_families}; allowed: {sorted(FAMILY_PREFIXES)}")
    target_labels = tuple(normalize_label(label) or label for label in parse_csv(args.target_labels))
    unknown_labels = sorted(set(target_labels) - set(DEFAULT_TARGET_LABELS))
    if unknown_labels:
        raise ValueError(f"unknown target labels: {unknown_labels}; allowed: {list(DEFAULT_TARGET_LABELS)}")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive when provided")
    return families, target_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage121-A prefix length/discourse ablation JSONL.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=121001)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--target-labels", default=",".join(DEFAULT_TARGET_LABELS))
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        families, target_labels = validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not args.input_jsonl.exists():
        print(f"ERROR: input JSONL not found: {args.input_jsonl}", file=sys.stderr)
        return 1
    valid_rows, warning_counts, n_input_rows = load_jsonl(args.input_jsonl)
    if not valid_rows:
        print("ERROR: no valid input rows remain after JSONL parsing and label normalization", file=sys.stderr)
        return 1
    output_rows, selected_sources = build_rows(valid_rows, target_labels=set(target_labels), families=families, max_rows=args.max_rows, seed=args.seed)
    if not output_rows:
        print("ERROR: no Stage121 diagnostic rows were generated", file=sys.stderr)
        return 1
    write_jsonl(args.output_jsonl, output_rows)
    summary = build_summary(input_path=args.input_jsonl, output_path=args.output_jsonl, n_input_rows=n_input_rows, valid_rows=valid_rows, selected_sources=selected_sources, output_rows=output_rows, warning_counts=warning_counts, families=families, target_labels=target_labels, seed=args.seed)
    if args.summary_json is not None:
        write_summary_json(args.summary_json, summary)
    if args.summary_md is not None:
        write_summary_md(args.summary_md, summary)
    print(DECISION)
    print(f"input_rows\t{n_input_rows}")
    print(f"valid_source_rows\t{len(valid_rows)}")
    print(f"selected_source_rows\t{len(selected_sources)}")
    print(f"output_rows\t{len(output_rows)}")
    print("label_counts_source")
    for label, count in summary["label_counts_source"].items():
        print(f"{label}\t{count}")
    print("family_counts")
    for family, count in summary["family_counts"].items():
        print(f"{family}\t{count}")
    if warning_counts:
        print("warning_counts")
        for key, count in sorted(warning_counts.items()):
            print(f"{key}\t{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
