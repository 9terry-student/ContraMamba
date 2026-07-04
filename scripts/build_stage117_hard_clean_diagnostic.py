"""Build the Stage117-B hard-clean diagnostic scaffold.

This script creates diagnostic-only variants from the internal controlled clean
dataset. It never reads external datasets and never rewrites claims, labels,
entities, numbers, or dates. The generated rows preserve the original gold
label while stressing surface overlap, evidence length, and distractor
robustness.
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
DEFAULT_INPUT = ROOT / "data" / "controlled_v5_v3_without_time_swap.jsonl"
DEFAULT_FAMILIES = ("distractor_append", "evidence_prefix", "distractor_prefix")
DEFAULT_TARGET_LABELS = ("SUPPORT", "REFUTE", "NOT_ENTITLED")
SOURCE_DATASET = "controlled_v5_v3_without_time_swap"
DECISION_READY = "STAGE117B_HARD_CLEAN_DIAGNOSTIC_SCAFFOLD_READY"

NUMERIC_LABEL_TO_TEXT = {
    0: "REFUTE",
    1: "NOT_ENTITLED",
    2: "SUPPORT",
    "0": "REFUTE",
    "1": "NOT_ENTITLED",
    "2": "SUPPORT",
}
LABEL_ALIASES = {
    "REFUTES": "REFUTE",
    "REFUTED": "REFUTE",
    "REFUTE": "REFUTE",
    "CONTRADICT": "REFUTE",
    "CONTRADICTION": "REFUTE",
    "NOT_ENOUGH_INFO": "NOT_ENTITLED",
    "NOT ENOUGH INFO": "NOT_ENTITLED",
    "NEI": "NOT_ENTITLED",
    "NONE": "NOT_ENTITLED",
    "NOT_ENTITLED": "NOT_ENTITLED",
    "SUPPORTS": "SUPPORT",
    "SUPPORTED": "SUPPORT",
    "SUPPORT": "SUPPORT",
    "ENTAILMENT": "SUPPORT",
}
LABEL_FIELDS = ("gold_label", "label", "normalized_label", "final_label", "claim_label", "final_label_id")
SURFACE_KEYS = (
    "claim_token_len",
    "evidence_token_len",
    "length_ratio_evidence_to_claim",
    "token_jaccard",
    "claim_token_coverage_in_evidence",
    "evidence_token_coverage_in_claim",
    "claim_cap_entity_chunk_count",
    "evidence_cap_entity_chunk_count",
    "cap_entity_jaccard",
    "claim_cap_entity_coverage_in_evidence",
)
NEUTRAL_PREFIX = "The following passage provides broader context before the relevant statement."
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CAP_CHUNK_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    warnings: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
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
            row["_stage117_gold_label"] = label
            row["_stage117_row_index"] = len(rows)
            rows.append(row)
    return rows, warnings


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
    text = str(value).strip().upper()
    text = text.replace("-", "_")
    return LABEL_ALIASES.get(text)


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def cap_entities(text: str) -> set[str]:
    chunks: set[str] = set()
    for match in CAP_CHUNK_RE.finditer(text):
        chunk = " ".join(match.group(0).split())
        if len(chunk) > 1:
            chunks.add(chunk.lower())
    return chunks


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def surface_features(claim: str, evidence: str) -> dict[str, float]:
    claim_tokens = token_set(claim)
    evidence_tokens = token_set(evidence)
    claim_entities = cap_entities(claim)
    evidence_entities = cap_entities(evidence)
    token_union = claim_tokens | evidence_tokens
    entity_union = claim_entities | evidence_entities
    return {
        "claim_token_len": len(tokens(claim)),
        "evidence_token_len": len(tokens(evidence)),
        "length_ratio_evidence_to_claim": safe_ratio(len(tokens(evidence)), len(tokens(claim))),
        "token_jaccard": safe_ratio(len(claim_tokens & evidence_tokens), len(token_union)),
        "claim_token_coverage_in_evidence": safe_ratio(len(claim_tokens & evidence_tokens), len(claim_tokens)),
        "evidence_token_coverage_in_claim": safe_ratio(len(claim_tokens & evidence_tokens), len(evidence_tokens)),
        "claim_cap_entity_chunk_count": len(claim_entities),
        "evidence_cap_entity_chunk_count": len(evidence_entities),
        "cap_entity_jaccard": safe_ratio(len(claim_entities & evidence_entities), len(entity_union)),
        "claim_cap_entity_coverage_in_evidence": safe_ratio(len(claim_entities & evidence_entities), len(claim_entities)),
    }


def feature_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in SURFACE_KEYS}


def truncate_to_token_budget(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    matches = list(TOKEN_RE.finditer(text))
    if len(matches) <= budget:
        return text.strip()
    end = matches[budget - 1].end()
    return text[:end].strip()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or f"row_{row['_stage117_row_index']:06d}")


def source_id_for(row: dict[str, Any]) -> str:
    return row_id(row)


def select_distractor(
    target: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    *,
    rng: random.Random,
) -> dict[str, Any] | None:
    target_claim_tokens = token_set(str(target["claim"]))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    target_source = source_id_for(target)
    target_pair = target.get("pair_id")
    for candidate in candidates:
        if source_id_for(candidate) == target_source:
            continue
        if target_pair is not None and candidate.get("pair_id") == target_pair:
            continue
        overlap = safe_ratio(
            len(target_claim_tokens & token_set(str(candidate["evidence"]))),
            len(target_claim_tokens | token_set(str(candidate["evidence"]))),
        )
        scored.append((overlap, rng.randrange(1_000_000_000), candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1], source_id_for(item[2])))
    return scored[0][2]


def nested_stage117_metadata(
    *,
    family: str,
    source: dict[str, Any],
    surface_before: dict[str, float],
    surface_after: dict[str, float],
    surface_delta: dict[str, float],
    distractor: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "family": family,
        "source_id": source_id_for(source),
        "distractor_id": source_id_for(distractor) if distractor is not None else None,
        "label_preserved": True,
        "uses_external_data": False,
        "source_dataset": SOURCE_DATASET,
        "surface_before": surface_before,
        "surface_after": surface_after,
        "surface_delta": surface_delta,
    }


def make_output_row(
    source: dict[str, Any],
    *,
    family: str,
    evidence: str,
    surface_before: dict[str, float],
    surface_after: dict[str, float],
    distractor: dict[str, Any] | None = None,
    include_hard_marker: bool = True,
) -> dict[str, Any]:
    original_metadata = source.get("metadata")
    metadata = dict(original_metadata) if isinstance(original_metadata, dict) else {}
    surface_delta = feature_deltas(surface_before, surface_after)
    metadata["stage117"] = nested_stage117_metadata(
        family=family,
        source=source,
        surface_before=surface_before,
        surface_after=surface_after,
        surface_delta=surface_delta,
        distractor=distractor,
    )

    source_clean = {
        key: value
        for key, value in source.items()
        if not key.startswith("_stage117_")
    }
    out = dict(source_clean)
    source_id = source_id_for(source)
    out.update(
        {
            "id": f"stage117b_{family}__{source_id}",
            "source_id": source_id,
            "claim": str(source["claim"]),
            "evidence": evidence,
            "gold_label": source["_stage117_gold_label"],
            "stage117_family": family,
            "stage117_is_hard_clean": include_hard_marker,
            "stage117_label_preserved": True,
            "stage117_uses_external_data": False,
            "stage117_source_dataset": SOURCE_DATASET,
            "stage117_original_claim": str(source["claim"]),
            "stage117_original_evidence": str(source["evidence"]),
            "stage117_surface_before": surface_before,
            "stage117_surface_after": surface_after,
            "stage117_surface_delta": surface_delta,
            "metadata": metadata,
        }
    )
    if distractor is not None:
        out["stage117_distractor_id"] = source_id_for(distractor)
        out["stage117_distractor_source_label"] = distractor["_stage117_gold_label"]
    return out


def generate_family_row(
    source: dict[str, Any],
    *,
    family: str,
    distractor: dict[str, Any] | None,
    distractor_token_budget: int,
) -> dict[str, Any] | None:
    claim = str(source["claim"])
    original_evidence = str(source["evidence"])
    before = surface_features(claim, original_evidence)
    if family == "evidence_prefix":
        evidence = f"{NEUTRAL_PREFIX} {original_evidence}"
        after = surface_features(claim, evidence)
        return make_output_row(
            source,
            family=family,
            evidence=evidence,
            surface_before=before,
            surface_after=after,
        )
    if family in {"distractor_append", "distractor_prefix"}:
        if distractor is None:
            return None
        fragment = truncate_to_token_budget(str(distractor["evidence"]), distractor_token_budget)
        if not fragment:
            return None
        if family == "distractor_append":
            evidence = f"{original_evidence} {fragment}"
        else:
            evidence = f"{fragment} {original_evidence}"
        after = surface_features(claim, evidence)
        return make_output_row(
            source,
            family=family,
            evidence=evidence,
            surface_before=before,
            surface_after=after,
            distractor=distractor,
        )
    raise ValueError(f"unknown Stage117 family: {family}")


def include_original_row(source: dict[str, Any]) -> dict[str, Any]:
    claim = str(source["claim"])
    evidence = str(source["evidence"])
    surface = surface_features(claim, evidence)
    return make_output_row(
        source,
        family="original",
        evidence=evidence,
        surface_before=surface,
        surface_after=surface,
        include_hard_marker=False,
    )


def build_rows(
    rows: Sequence[dict[str, Any]],
    *,
    target_labels: set[str],
    families: Sequence[str],
    max_rows: int | None,
    distractor_token_budget: int,
    include_original: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(seed)
    selected_sources = [row for row in rows if row["_stage117_gold_label"] in target_labels]
    selected_sources.sort(key=source_id_for)
    if max_rows is not None:
        selected_sources = selected_sources[:max_rows]

    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for source in selected_sources:
        if include_original:
            output.append(include_original_row(source))
        for family in families:
            distractor = None
            if family.startswith("distractor_"):
                distractor = select_distractor(source, rows, rng=rng)
                if distractor is None:
                    skipped[f"{family}:no_distractor"] += 1
                    continue
            generated = generate_family_row(
                source,
                family=family,
                distractor=distractor,
                distractor_token_budget=distractor_token_budget,
            )
            if generated is None:
                skipped[f"{family}:not_generated"] += 1
                continue
            output.append(generated)
    return output, skipped


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def aggregate_surface(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = f"{record['gold_label']}::{record['stage117_family']}"
        grouped[key].append(record)
    summary: dict[str, Any] = {}
    for key in sorted(grouped):
        label, family = key.split("::", 1)
        group = grouped[key]
        before = {
            feature: median([row["stage117_surface_before"][feature] for row in group])
            for feature in SURFACE_KEYS
        }
        after = {
            feature: median([row["stage117_surface_after"][feature] for row in group])
            for feature in SURFACE_KEYS
        }
        delta = {
            feature: median([row["stage117_surface_delta"][feature] for row in group])
            for feature in SURFACE_KEYS
        }
        summary[key] = {
            "gold_label": label,
            "stage117_family": family,
            "count": len(group),
            "before_median": before,
            "after_median": after,
            "delta_median": delta,
        }
    return summary


def build_summary(
    *,
    input_path: Path,
    output_path: Path,
    records: Sequence[dict[str, Any]],
    input_count: int,
    warning_counts: Counter[str],
    skipped_counts: Counter[str],
    families: Sequence[str],
    target_labels: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "decision": DECISION_READY,
        "seed": seed,
        "target_labels": list(target_labels),
        "families": list(families),
        "row_counts": {
            "valid_input_rows": input_count,
            "output_rows": len(records),
            "malformed_or_skipped_input_rows": sum(warning_counts.values()),
        },
        "warning_counts": dict(sorted(warning_counts.items())),
        "skipped_generation_counts": dict(sorted(skipped_counts.items())),
        "label_counts": dict(sorted(Counter(row["gold_label"] for row in records).items())),
        "family_counts": dict(sorted(Counter(row["stage117_family"] for row in records).items())),
        "before_after_median_surface_features_by_label_family": aggregate_surface(records),
    }


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_feature_row(title: str, values: dict[str, float | None]) -> str:
    cells = [title]
    for feature in SURFACE_KEYS:
        value = values.get(feature)
        cells.append("" if value is None else f"{value:.4f}")
    return "| " + " | ".join(cells) + " |"


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage117-B Hard-Clean Diagnostic Design",
        "",
        "## Purpose",
        "",
        (
            "Create an internal controlled-data diagnostic that stresses lexical overlap, "
            "evidence length, entity coverage, and distractor robustness without using "
            "VitaminC or any external dataset for generation or training."
        ),
        "",
        "## Generation Families",
        "",
        "- `distractor_append`: append a low-overlap evidence fragment from another clean row.",
        "- `distractor_prefix`: prefix a low-overlap evidence fragment from another clean row.",
        "- `evidence_prefix`: add a neutral context sentence before the original evidence.",
        "",
        "## Safety Constraints",
        "",
        "- Claims are never rewritten.",
        "- Gold labels are copied from the original clean row and marked label-preserved.",
        "- Entities, numbers, dates, and aliases are not modified.",
        "- External data is not read or used.",
        "- The output is diagnostic-only and should not be mixed into training by default.",
        "",
        "## Counts",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Input: `{summary['input_path']}`",
        f"- Output: `{summary['output_path']}`",
        f"- Valid input rows: {summary['row_counts']['valid_input_rows']}",
        f"- Output rows: {summary['row_counts']['output_rows']}",
        "",
        "### Label Counts",
        "",
        "| label | count |",
        "| --- | ---: |",
    ]
    for label, count in summary["label_counts"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "### Family Counts", "", "| family | count |", "| --- | ---: |"])
    for family, count in summary["family_counts"].items():
        lines.append(f"| {family} | {count} |")
    lines.extend(["", "## Surface Feature Medians", ""])
    header = "| group | " + " | ".join(SURFACE_KEYS) + " |"
    divider = "| --- | " + " | ".join(["---:"] * len(SURFACE_KEYS)) + " |"
    for key, group in summary["before_after_median_surface_features_by_label_family"].items():
        lines.extend(
            [
                f"### {key}",
                "",
                header,
                divider,
                format_feature_row("before", group["before_median"]),
                format_feature_row("after", group["after_median"]),
                format_feature_row("delta", group["delta_median"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Next-Stage Recommendation",
            "",
            "Stage117-C generate hard-clean diagnostic on Kaggle, then Stage118 evaluate vNext on hard-clean.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_args(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    families = parse_csv(args.families)
    unknown = sorted(set(families) - set(DEFAULT_FAMILIES))
    if unknown:
        raise ValueError(f"unknown families: {unknown}; allowed: {list(DEFAULT_FAMILIES)}")
    target_labels = tuple(normalize_label(label) or label for label in parse_csv(args.target_labels))
    unknown_labels = sorted(set(target_labels) - set(DEFAULT_TARGET_LABELS))
    if unknown_labels:
        raise ValueError(f"unknown target labels: {unknown_labels}; allowed: {list(DEFAULT_TARGET_LABELS)}")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive when provided")
    if args.distractor_token_budget < 1:
        raise ValueError("--distractor-token-budget must be positive")
    return families, target_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage117-B controlled hard-clean diagnostic JSONL from "
            "controlled_v5_v3_without_time_swap.jsonl only."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=117002)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--target-labels", default=",".join(DEFAULT_TARGET_LABELS))
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--distractor-token-budget", type=int, default=40)
    parser.add_argument("--include-original", action="store_true")
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

    rows, warning_counts = load_jsonl(args.input_jsonl)
    if not rows:
        print("ERROR: no valid input rows remain after JSONL parsing and label normalization", file=sys.stderr)
        return 1

    records, skipped_counts = build_rows(
        rows,
        target_labels=set(target_labels),
        families=families,
        max_rows=args.max_rows,
        distractor_token_budget=args.distractor_token_budget,
        include_original=args.include_original,
        seed=args.seed,
    )
    if not records:
        print("ERROR: no Stage117 diagnostic rows were generated", file=sys.stderr)
        return 1

    write_jsonl(args.output_jsonl, records)
    summary = build_summary(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        records=records,
        input_count=len(rows),
        warning_counts=warning_counts,
        skipped_counts=skipped_counts,
        families=families,
        target_labels=target_labels,
        seed=args.seed,
    )
    if args.summary_json is not None:
        write_summary_json(args.summary_json, summary)
    if args.summary_md is not None:
        write_summary_md(args.summary_md, summary)

    print(DECISION_READY)
    print(f"input_rows_valid\t{len(rows)}")
    print(f"output_rows\t{len(records)}")
    print("label_counts")
    for label, count in summary["label_counts"].items():
        print(f"{label}\t{count}")
    print("family_counts")
    for family, count in summary["family_counts"].items():
        print(f"{family}\t{count}")
    if warning_counts:
        print("warning_counts")
        for key, count in sorted(warning_counts.items()):
            print(f"{key}\t{count}")
    if skipped_counts:
        print("skipped_generation_counts")
        for key, count in sorted(skipped_counts.items()):
            print(f"{key}\t{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
