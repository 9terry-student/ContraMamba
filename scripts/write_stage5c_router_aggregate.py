"""Aggregate Stage 5C router seed CSVs into reproducible reports."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

try:  # Use the requested coercion when pandas is available.
    import pandas as pd
except ModuleNotFoundError:  # Keep this reporting utility dependency-light.
    pd = None


SYSTEM_ORDER = (
    "classifier_only", "balanced_only", "strict_only",
    "conservative_balanced_router", "conservative_strict_router",
    "dual_auditor_router",
)
SECTION_SPECS = (
    ("classification", "ROUTER_CLASSIFICATION_AGGREGATE", (
        "final_accuracy", "final_macro_f1", "NOT_ENTITLED_f1", "REFUTE_f1", "SUPPORT_f1",
    )),
    ("pairwise", "ROUTER_PAIRWISE_AGGREGATE", (
        "paraphrase_preserved", "predicate_disentangled",
        "polarity_flip_preserved_and_reversed", "deletion_sufficiency_lower",
        "truncation_sufficiency_lower", "entity_frame_lower", "event_frame_lower",
    )),
    ("internal_faithfulness", "ROUTER_INTERNAL_FAITHFULNESS_AGGREGATE", (
        "entitled_output_gate_violation_rate", "entitled_output_count",
        "entitled_output_gate_violations", "polarity_flip_output_ok",
        "polarity_flip_internal_ok", "polarity_flip_output_internal_gap",
        "polarity_flip_output_ok_but_internal_bad",
    )),
)


def _numeric(values: list[str]) -> list[float | None]:
    if pd is not None:
        converted = pd.to_numeric(values, errors="coerce")
        return [None if math.isnan(float(value)) else float(value) for value in converted]
    result: list[float | None] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            result.append(None)
    return result


def read_seed_csvs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("at least one --input CSV is required")
    rows: list[dict[str, Any]] = []
    required = {"section", "system", "metric", "value"}
    for seed, path in enumerate(paths, start=1):
        with path.open(newline="", encoding="utf-8") as handle:
            seed_rows = list(csv.DictReader(handle))
        missing = required - set(seed_rows[0] if seed_rows else {})
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        converted = _numeric([row["value"] for row in seed_rows])
        for row, value in zip(seed_rows, converted):
            if value is not None:
                rows.append({**row, "seed": seed, "numeric_value": value})
    return rows


def aggregate_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["section"], row["system"], row["metric"])].append(
            float(row["numeric_value"])
        )
    section_rank = {name: i for i, (name, _, _) in enumerate(SECTION_SPECS)}
    system_rank = {name: i for i, name in enumerate(SYSTEM_ORDER)}
    aggregate = []
    for (section, system, metric), values in grouped.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append({
            "section": section, "system": system, "metric": metric,
            "mean": mean, "std": std, "n": len(values),
            "formatted": f"{mean:.3f} ± {std:.3f}",
        })
    return sorted(aggregate, key=lambda row: (
        section_rank.get(row["section"], 999), system_rank.get(row["system"], 999), row["metric"]
    ))


def _find(aggregate: Sequence[dict[str, Any]], system: str, metric: str) -> float | None:
    return next((float(row["mean"]) for row in aggregate
                 if row["system"] == system and row["metric"] == metric), None)


def _table(aggregate: Sequence[dict[str, Any]], section: str, metrics: Sequence[str]) -> list[str]:
    lookup = {(row["system"], row["metric"]): row["formatted"]
              for row in aggregate if row["section"] == section}
    available = [metric for metric in metrics if any(key[1] == metric for key in lookup)]
    if not available:
        return ["_No numeric metrics available._"]
    lines = ["| system | " + " | ".join(available) + " |",
             "|---|" + "---:|" * len(available)]
    for system in SYSTEM_ORDER:
        if not any(key[0] == system for key in lookup):
            continue
        lines.append(f"| {system} | " + " | ".join(
            str(lookup.get((system, metric), "—")) for metric in available
        ) + " |")
    return lines


def _interpretation(aggregate: Sequence[dict[str, Any]]) -> list[str]:
    statements = []
    macro = [row for row in aggregate if row["section"] == "classification"
             and row["metric"] == "final_macro_f1"]
    if macro:
        best = max(macro, key=lambda row: row["mean"])["system"]
        if best == "conservative_balanced_router":
            statements.append("conservative_balanced_router achieves the best final-label performance by mean final_macro_f1.")
        else:
            statements.append(f"{best} achieves the best mean final_macro_f1.")
    classifier_gap = _find(aggregate, "classifier_only", "polarity_flip_output_internal_gap")
    router_gaps = [value for system in SYSTEM_ORDER[3:]
                   if (value := _find(aggregate, system, "polarity_flip_output_internal_gap")) is not None]
    if classifier_gap is not None and router_gaps:
        comparison = "larger" if classifier_gap > max(router_gaps) else "not larger"
        statements.append(f"classifier_only is a strong label predictor, and its output/internal faithfulness gap is {comparison} than the conservative routers.")
    zero = [system for system in SYSTEM_ORDER[3:]
            if _find(aggregate, system, "entitled_output_gate_violation_rate") == 0.0]
    if zero:
        statements.append("Conservative routers eliminate entitled-output gate violations when their violation rate is 0: " + ", ".join(zero) + ".")
    if (_find(aggregate, "balanced_only", "final_macro_f1") is not None
            and _find(aggregate, "balanced_only", "polarity_flip_preserved_and_reversed") is not None):
        statements.append("balanced_only can have strong output-level pairwise consistency while lower final-label macro-F1.")
    return statements or ["No complete comparison metrics were available."]


def render_markdown(aggregate: Sequence[dict[str, Any]]) -> str:
    lines = ["# Stage 5C v2 Router Aggregate", ""]
    for section, title, metrics in SECTION_SPECS:
        lines.extend([f"## {title}", "", *_table(aggregate, section, metrics), ""])
    lines.extend(["## INTERPRETATION", "", *[f"- {text}" for text in _interpretation(aggregate)], ""])
    return "\n".join(lines)


def write_reports(inputs: Sequence[Path], output_md: Path, output_csv: Path) -> list[dict[str, Any]]:
    aggregate = aggregate_results(read_seed_csvs(inputs))
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(aggregate), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ("section", "system", "metric", "mean", "std", "n", "formatted")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in aggregate)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_reports(args.input, args.output_md, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
