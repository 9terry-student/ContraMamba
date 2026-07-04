"""Build Stage122-A prefix/core localization diagnostics.

This generator expands Stage121 prefix ablation rows into paired variants that
separate prefix-only triggers from failures to localize the evidence core. It is
diagnostic-only: labels are preserved, no model code is imported, and no
external data is read.
"""

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
DEFAULT_OUTPUT = ROOT / "data" / "stage122_prefix_core_localization_diagnostic.jsonl"
SOURCE_DATASET = "stage121_prefix_length_discourse_ablation"
DECISION = "STAGE122A_PREFIX_CORE_LOCALIZATION_DIAGNOSTIC_GENERATED"

DEFAULT_TARGET_LABELS = ("SUPPORT", "REFUTE", "NOT_ENTITLED")
DEFAULT_SOURCE_FAMILIES = (
    "original_control",
    "short_neutral_fragment",
    "short_neutral_sentence",
    "length_matched_nonsense",
    "length_matched_neutral_tokens",
    "long_neutral_no_order",
    "long_discourse_no_temporal",
    "old_before_prefix",
    "explicit_before_temporal",
)
DEFAULT_VARIANTS = (
    "original_control",
    "prefix_plus_core",
    "prefix_only",
    "core_only",
    "core_plus_suffix",
    "core_with_neutral_marker",
)

LABEL_FIELDS = ("gold_label", "label", "final_label", "target", "target_label")
NUMERIC_LABEL_TO_TEXT = {
    0: "REFUTE",
    1: "NOT_ENTITLED",
    2: "SUPPORT",
    "0": "REFUTE",
    "1": "NOT_ENTITLED",
    "2": "SUPPORT",
}
LABEL_ALIASES = {
    "SUPPORT": "SUPPORT",
    "SUPPORTS": "SUPPORT",
    "REFUTE": "REFUTE",
    "REFUTES": "REFUTE",
    "NOT_ENTITLED": "NOT_ENTITLED",
    "NOT_ENOUGH_INFO": "NOT_ENTITLED",
    "NOT ENOUGH INFO": "NOT_ENTITLED",
    "NEI": "NOT_ENTITLED",
}

TEMPORAL_TRIGGERS = {
    "before",
    "after",
    "earlier",
    "later",
    "previous",
    "following",
    "prior",
    "subsequent",
    "during",
    "while",
    "when",
}
DISCOURSE_MARKERS = {
    "context",
    "passage",
    "statement",
    "relevant",
    "evidence",
    "background",
    "information",
    "contains",
    "provides",
}
EVIDENCE_MARKERS = {"evidence", "claim", "statement", "support", "refute"}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CAP_CHUNK_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")

SURFACE_KEYS = (
    "claim_token_len",
    "evidence_token_len",
    "prefix_token_len",
    "core_token_len",
    "length_ratio_evidence_to_claim",
    "token_jaccard",
    "claim_token_coverage_in_evidence",
    "evidence_token_coverage_in_claim",
    "cap_entity_jaccard",
    "claim_cap_entity_coverage_in_evidence",
    "prefix_position",
    "contains_before",
    "contains_after",
    "contains_temporal_trigger",
    "contains_discourse_marker",
    "contains_evidence_marker",
)

VARIANT_DESCRIPTIONS = {
    "original_control": "Core relation sanity baseline using the original Stage121 evidence core.",
    "prefix_plus_core": "Reconstructs prefixed evidence to reproduce prefix contamination.",
    "prefix_only": "Uses the prefix alone to test whether it is a learned polarity trigger.",
    "core_only": "Localization control verifying that evidence core alone remains label-correct.",
    "core_plus_suffix": "Moves the prefix after the core to test front-position sensitivity.",
    "core_with_neutral_marker": "Adds a minimal evidence marker before the core.",
    "prefix_separator_core": "Adds an explicit Evidence marker between prefix and core.",
}


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
    text = str(value).strip().upper().replace("-", "_")
    return LABEL_ALIASES.get(text)


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def cap_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for match in CAP_CHUNK_RE.finditer(text):
        entity = " ".join(match.group(0).split())
        if len(entity) > 1:
            entities.add(entity.lower())
    return entities


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def surface_features(
    claim: str,
    evidence: str,
    prefix_text: str,
    core_evidence: str,
    prefix_position: str,
) -> dict[str, float | int | str]:
    claim_tokens = tokens(claim)
    evidence_tokens = tokens(evidence)
    prefix_tokens = tokens(prefix_text)
    core_tokens = tokens(core_evidence)
    claim_set = set(claim_tokens)
    evidence_set = set(evidence_tokens)
    claim_entities = cap_entities(claim)
    evidence_entities = cap_entities(evidence)
    token_union = claim_set | evidence_set
    entity_union = claim_entities | evidence_entities
    return {
        "claim_token_len": len(claim_tokens),
        "evidence_token_len": len(evidence_tokens),
        "prefix_token_len": len(prefix_tokens),
        "core_token_len": len(core_tokens),
        "length_ratio_evidence_to_claim": safe_ratio(len(evidence_tokens), len(claim_tokens)),
        "token_jaccard": safe_ratio(len(claim_set & evidence_set), len(token_union)),
        "claim_token_coverage_in_evidence": safe_ratio(len(claim_set & evidence_set), len(claim_set)),
        "evidence_token_coverage_in_claim": safe_ratio(len(claim_set & evidence_set), len(evidence_set)),
        "cap_entity_jaccard": safe_ratio(len(claim_entities & evidence_entities), len(entity_union)),
        "claim_cap_entity_coverage_in_evidence": safe_ratio(len(claim_entities & evidence_entities), len(claim_entities)),
        "prefix_position": prefix_position,
        "contains_before": int("before" in evidence_set),
        "contains_after": int("after" in evidence_set),
        "contains_temporal_trigger": int(bool(evidence_set & TEMPORAL_TRIGGERS)),
        "contains_discourse_marker": int(bool(evidence_set & DISCOURSE_MARKERS)),
        "contains_evidence_marker": int(bool(evidence_set & EVIDENCE_MARKERS)),
    }


def feature_deltas(
    before: dict[str, float | int | str],
    after: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    delta: dict[str, float | int | str] = {}
    for key in SURFACE_KEYS:
        before_value = before[key]
        after_value = after[key]
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            delta[key] = after_value - before_value
        elif before_value == after_value:
            delta[key] = str(after_value)
        else:
            delta[key] = f"{before_value}->{after_value}"
    return delta


def clean_join(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or f"row_{row['_stage122_row_index']:06d}")


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
            claim = str(row.get("claim", "")).strip()
            evidence = str(row.get("evidence", "")).strip()
            core_evidence = str(row.get("stage121_original_evidence", "")).strip()
            family = str(row.get("stage121_family", "")).strip()
            if not claim:
                warnings["missing_claim"] += 1
                continue
            if not evidence:
                warnings["missing_evidence"] += 1
                continue
            if not core_evidence:
                warnings["missing_stage121_original_evidence"] += 1
                continue
            if not family:
                warnings["missing_stage121_family"] += 1
                continue
            if "stage121_prefix_text" not in row:
                warnings["missing_stage121_prefix_text"] += 1
                continue
            if not str(row.get("stage121_original_claim", "")).strip():
                warnings["missing_stage121_original_claim"] += 1
                continue
            label = normalize_label(first_present(row, LABEL_FIELDS))
            if label is None:
                warnings["missing_or_unknown_label"] += 1
                continue
            row["_stage122_gold_label"] = label
            row["_stage122_row_index"] = len(rows)
            rows.append(row)
    return rows, warnings, n_input_rows


def select_sources(
    rows: Sequence[dict[str, Any]],
    *,
    target_labels: set[str],
    source_families: set[str],
    max_rows: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["_stage122_gold_label"] in target_labels
        and str(row.get("stage121_family", "")).strip() in source_families
    ]
    if max_rows is None or len(eligible) <= max_rows:
        return sorted(eligible, key=lambda row: row["_stage122_row_index"])

    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_stratum[(row["_stage122_gold_label"], str(row["stage121_family"]))].append(row)

    allocations: dict[tuple[str, str], int] = {}
    fractional: list[tuple[float, tuple[str, str]]] = []
    total = len(eligible)
    for stratum in sorted(by_stratum):
        exact = max_rows * len(by_stratum[stratum]) / total
        allocations[stratum] = int(exact)
        fractional.append((exact - int(exact), stratum))

    remaining = max_rows - sum(allocations.values())
    for _, stratum in sorted(fractional, key=lambda item: (-item[0], item[1]))[:remaining]:
        allocations[stratum] += 1

    selected: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        group = list(by_stratum[stratum])
        rng.shuffle(group)
        selected.extend(group[: allocations[stratum]])
    return sorted(selected, key=lambda row: row["_stage122_row_index"])


def variant_evidence(variant: str, prefix_text: str, core_evidence: str) -> tuple[str, str]:
    if variant in {"original_control", "core_only"}:
        return core_evidence, "none"
    if variant == "prefix_plus_core":
        return clean_join(prefix_text, core_evidence), "front" if prefix_text.strip() else "none"
    if variant == "prefix_only":
        return prefix_text.strip(), "only"
    if variant == "core_plus_suffix":
        return clean_join(core_evidence, prefix_text), "suffix" if prefix_text.strip() else "none"
    if variant == "core_with_neutral_marker":
        return clean_join("Evidence:", core_evidence), "marker_front"
    if variant == "prefix_separator_core":
        return clean_join(prefix_text, "Evidence:", core_evidence), "separated_front" if prefix_text.strip() else "marker_front"
    raise ValueError(f"unknown variant: {variant}")


def make_output_row(source: dict[str, Any], variant: str) -> dict[str, Any] | None:
    prefix_text = str(source.get("stage121_prefix_text", ""))
    if variant == "prefix_only" and not prefix_text.strip():
        return None

    source_id = row_id(source)
    claim = str(source.get("claim", ""))
    original_claim = str(source.get("stage121_original_claim", claim))
    core_evidence = str(source.get("stage121_original_evidence", ""))
    evidence, prefix_position = variant_evidence(variant, prefix_text, core_evidence)
    before = surface_features(claim, core_evidence, "", core_evidence, "none")
    after = surface_features(claim, evidence, prefix_text, core_evidence, prefix_position)
    delta = feature_deltas(before, after)

    metadata = dict(source.get("metadata")) if isinstance(source.get("metadata"), dict) else {}
    metadata["stage122"] = {
        "stage": "Stage122-A",
        "variant": variant,
        "source_id": source_id,
        "source_stage121_family": source.get("stage121_family"),
        "source_stage121_prefix_category": source.get("stage121_prefix_category"),
        "prefix_text": prefix_text,
        "label_preserved": True,
        "uses_external_data": False,
        "source_dataset": SOURCE_DATASET,
        "surface_before": before,
        "surface_after": after,
        "surface_delta": delta,
    }

    output = dict(source)
    output.pop("_stage122_gold_label", None)
    output.pop("_stage122_row_index", None)
    output.update(
        {
            "id": f"stage122a_{variant}__{source_id}",
            "source_id": source_id,
            "claim": claim,
            "evidence": evidence,
            "gold_label": source["_stage122_gold_label"],
            "stage122_family": "prefix_core_localization",
            "stage122_variant": variant,
            "stage122_source_stage121_family": source.get("stage121_family"),
            "stage122_source_stage121_prefix_category": source.get("stage121_prefix_category"),
            "stage122_prefix_text": prefix_text,
            "stage122_original_claim": original_claim,
            "stage122_original_evidence": core_evidence,
            "stage122_evidence_core": core_evidence,
            "stage122_variant_description": VARIANT_DESCRIPTIONS[variant],
            "stage122_is_prefix_core_localization_diagnostic": True,
            "stage122_label_preserved": True,
            "stage122_uses_external_data": False,
            "stage122_source_dataset": SOURCE_DATASET,
            "stage122_surface_before": before,
            "stage122_surface_after": after,
            "stage122_surface_delta": delta,
            "metadata": metadata,
        }
    )
    return output


def build_rows(
    rows: Sequence[dict[str, Any]],
    *,
    target_labels: set[str],
    source_families: set[str],
    variants: Sequence[str],
    max_source_rows: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    selected_sources = select_sources(
        rows,
        target_labels=target_labels,
        source_families=source_families,
        max_rows=max_source_rows,
        seed=seed,
    )
    skipped: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for source in selected_sources:
        for variant in variants:
            output = make_output_row(source, variant)
            if output is None:
                skipped[f"{variant}_empty_prefix"] += 1
                continue
            output_rows.append(output)
    return output_rows, selected_sources, skipped


def median(values: Sequence[float | int]) -> float | None:
    return float(statistics.median(values)) if values else None


def aggregate_surface(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = "::".join(
            [
                str(record["stage122_source_stage121_family"]),
                str(record["stage122_variant"]),
                str(record["gold_label"]),
            ]
        )
        grouped[key].append(record)

    numeric_keys = [
        key
        for key in SURFACE_KEYS
        if key != "prefix_position"
    ]
    summary: dict[str, Any] = {}
    for key in sorted(grouped):
        source_family, variant, label = key.split("::", 2)
        group = grouped[key]
        summary[key] = {
            "stage121_family": source_family,
            "stage122_variant": variant,
            "gold_label": label,
            "count": len(group),
            "before_median": {
                feature: median([row["stage122_surface_before"][feature] for row in group])
                for feature in numeric_keys
            },
            "after_median": {
                feature: median([row["stage122_surface_after"][feature] for row in group])
                for feature in numeric_keys
            },
            "delta_median": {
                feature: median([row["stage122_surface_delta"][feature] for row in group])
                for feature in numeric_keys
            },
            "prefix_position_counts": dict(
                sorted(Counter(str(row["stage122_surface_after"]["prefix_position"]) for row in group).items())
            ),
        }
    return summary


def trigger_counts(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "contains_before",
        "contains_after",
        "contains_temporal_trigger",
        "contains_discourse_marker",
        "contains_evidence_marker",
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["stage122_source_stage121_family"]), str(record["stage122_variant"]))].append(record)
    return {
        f"{family}::{variant}": {
            "stage121_family": family,
            "stage122_variant": variant,
            **{key: int(sum(row["stage122_surface_after"][key] for row in rows)) for key in keys},
        }
        for (family, variant), rows in sorted(grouped.items())
    }


def nested_counter(records: Sequence[dict[str, Any]], *fields: str) -> dict[str, int]:
    return dict(sorted(Counter("::".join(str(row.get(field)) for field in fields) for row in records).items()))


def build_summary(
    *,
    input_path: Path,
    output_path: Path,
    n_input_rows: int,
    valid_rows: Sequence[dict[str, Any]],
    selected_sources: Sequence[dict[str, Any]],
    output_rows: Sequence[dict[str, Any]],
    skipped_counts: Counter[str],
    warning_counts: Counter[str],
    target_labels: Sequence[str],
    source_families: Sequence[str],
    variants: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    skipped_all = Counter(warning_counts)
    skipped_all.update(skipped_counts)
    return {
        "stage": "Stage122-A",
        "decision": DECISION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "seed": seed,
        "target_labels": list(target_labels),
        "source_families": list(source_families),
        "variants": list(variants),
        "n_input_rows": int(n_input_rows),
        "n_valid_source_rows": int(len(valid_rows)),
        "n_selected_source_rows": int(len(selected_sources)),
        "n_output_rows": int(len(output_rows)),
        "skipped_counts": dict(sorted(skipped_all.items())),
        "label_counts_source": dict(sorted(Counter(row["_stage122_gold_label"] for row in selected_sources).items())),
        "label_counts_output": dict(sorted(Counter(row["gold_label"] for row in output_rows).items())),
        "source_stage121_family_counts": dict(sorted(Counter(row["stage121_family"] for row in selected_sources).items())),
        "variant_counts": dict(sorted(Counter(row["stage122_variant"] for row in output_rows).items())),
        "source_family_x_variant_counts": nested_counter(output_rows, "stage122_source_stage121_family", "stage122_variant"),
        "surface_medians_by_source_family_variant_and_gold_label": aggregate_surface(output_rows),
        "trigger_counts_by_source_family_and_variant": trigger_counts(output_rows),
        "auxiliary_fields_preserved": True,
        "no_external_data": True,
        "label_preserved": True,
        "next_stage": "Stage122-B evaluate prefix/core localization diagnostic with Stage118 generic diagnostic path",
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
        "# Stage122-A Prefix/Core Localization Diagnostic",
        "",
        "## Purpose",
        "",
        "Generate paired, label-preserved variants from Stage121 rows to test whether prefix-contaminated evidence causes polarity triggers, core suppression, position sensitivity, or weak evidence-core localization.",
        "",
        "## Hypotheses",
        "",
        "- Prefix-only polarity trigger: the prefix alone can elicit REFUTE, especially for SUPPORT rows.",
        "- Core-evidence suppression: the core remains valid, but prefix+core hides or deweights it.",
        "- Prefix position/order sensitivity: moving the prefix to a suffix recovers behavior.",
        "- Discourse-wrapper semantics: discourse and temporal prefix wording drives inversion beyond length.",
        "- Evidence-core localization: explicit Evidence markers can help the controller find the core.",
        "",
        "## Variant Definitions",
        "",
    ]
    for variant in summary["variants"]:
        lines.append(f"- `{variant}`: {VARIANT_DESCRIPTIONS[variant]}")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Decision: `{summary['decision']}`",
            f"- Input rows: {summary['n_input_rows']}",
            f"- Valid source rows: {summary['n_valid_source_rows']}",
            f"- Selected source rows: {summary['n_selected_source_rows']}",
            f"- Output rows: {summary['n_output_rows']}",
            "",
            "### Source Label Counts",
            "",
        ]
    )
    lines.extend(md_table(summary["label_counts_source"], "label"))
    lines.extend(["", "### Output Label Counts", ""])
    lines.extend(md_table(summary["label_counts_output"], "label"))
    lines.extend(["", "### Source Stage121 Family Counts", ""])
    lines.extend(md_table(summary["source_stage121_family_counts"], "stage121_family"))
    lines.extend(["", "### Variant Counts", ""])
    lines.extend(md_table(summary["variant_counts"], "variant"))
    lines.extend(
        [
            "",
            "## Diagnostic Interpretation Guide",
            "",
            "- `prefix_only` predicts REFUTE for SUPPORT families: prefix text itself is a learned negative polarity trigger.",
            "- `core_only` recovers SUPPORT while `prefix_plus_core` fails: core relation is intact but prefix suppresses/localizes incorrectly.",
            "- `core_plus_suffix` recovers relative to `prefix_plus_core`: leading prefix position is the main contaminant.",
            "- `prefix_separator_core` recovers: explicit evidence-core marker helps localization.",
            "- `length_matched_nonsense` `prefix_plus_core` still NE, not REFUTE: pure length closes gates rather than flipping polarity.",
            "- Discourse prefixes produce REFUTE in `prefix_plus_core`: discourse semantics drive polarity inversion.",
            "",
            "## Next-Stage Recommendation",
            "",
            summary["next_stage"] + ".",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_args(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    target_labels = tuple(normalize_label(label) or label for label in parse_csv(args.target_labels))
    unknown_labels = sorted(set(target_labels) - set(DEFAULT_TARGET_LABELS))
    if unknown_labels:
        raise ValueError(f"unknown target labels: {unknown_labels}; allowed: {list(DEFAULT_TARGET_LABELS)}")

    source_families = parse_csv(args.source_families)
    if not source_families:
        raise ValueError("--source-families must contain at least one family")

    variants = parse_csv(args.variants)
    unknown_variants = sorted(set(variants) - set(VARIANT_DESCRIPTIONS))
    if unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}; allowed: {sorted(VARIANT_DESCRIPTIONS)}")
    if not variants:
        raise ValueError("--variants must contain at least one variant")

    if args.max_source_rows is not None and args.max_source_rows < 1:
        raise ValueError("--max-source-rows must be positive when provided")
    return target_labels, source_families, variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage122-A prefix/core localization diagnostic JSONL.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=122001)
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--target-labels", default=",".join(DEFAULT_TARGET_LABELS))
    parser.add_argument("--source-families", default=",".join(DEFAULT_SOURCE_FAMILIES))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        target_labels, source_families, variants = validate_args(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.input_jsonl.exists():
        print(f"ERROR: input JSONL not found: {args.input_jsonl}", file=sys.stderr)
        return 1

    valid_rows, warning_counts, n_input_rows = load_jsonl(args.input_jsonl)
    if not valid_rows:
        print("ERROR: no valid Stage121 source rows remain after parsing and validation", file=sys.stderr)
        return 1

    output_rows, selected_sources, skipped_counts = build_rows(
        valid_rows,
        target_labels=set(target_labels),
        source_families=set(source_families),
        variants=variants,
        max_source_rows=args.max_source_rows,
        seed=args.seed,
    )
    if not output_rows:
        print("ERROR: no Stage122 diagnostic rows were generated", file=sys.stderr)
        return 1

    write_jsonl(args.output_jsonl, output_rows)
    summary = build_summary(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        n_input_rows=n_input_rows,
        valid_rows=valid_rows,
        selected_sources=selected_sources,
        output_rows=output_rows,
        skipped_counts=skipped_counts,
        warning_counts=warning_counts,
        target_labels=target_labels,
        source_families=source_families,
        variants=variants,
        seed=args.seed,
    )
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
    print("variant_counts")
    for variant, count in summary["variant_counts"].items():
        print(f"{variant}\t{count}")
    if summary["skipped_counts"]:
        print("skipped_counts")
        for key, count in summary["skipped_counts"].items():
            print(f"{key}\t{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

