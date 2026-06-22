"""Aggregate Stage 6C hybrid expert-router results across seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_hybrid_router_search import SYSTEM_ORDER  # noqa: E402
from scripts.sweep_router_thresholds import METRICS  # noqa: E402


def read_results(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for implicit_seed, path in enumerate(paths, start=1):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append({"seed": int(row.get("seed") or implicit_seed),
                                 "threshold": float(row["threshold"]), "system": row["system"],
                                 "metric": row["metric"], "value": float(row["value"])})
                except (KeyError, TypeError, ValueError):
                    continue
    if not rows:
        raise ValueError("no numeric hybrid-router metrics were found")
    return rows


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["threshold"]), str(row["system"]), str(row["metric"]))].append(float(row["value"]))
    system_rank = {name: index for index, name in enumerate(SYSTEM_ORDER)}
    metric_rank = {name: index for index, name in enumerate(METRICS)}
    result = []
    for (threshold, system, metric), values in grouped.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append({"threshold": threshold, "system": system, "metric": metric,
                       "mean": mean, "std": std, "n": len(values),
                       "formatted": f"{mean:.3f} +/- {std:.3f}"})
    return sorted(result, key=lambda row: (row["threshold"], system_rank.get(row["system"], 999),
                                            metric_rank.get(row["metric"], 999)))


def _value(rows: Sequence[Mapping[str, Any]], threshold: float, system: str, metric: str) -> float | None:
    return next((float(row["mean"]) for row in rows if float(row["threshold"]) == threshold
                 and row["system"] == system and row["metric"] == metric), None)


def interpretation(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    macro_rows = [row for row in rows if row["metric"] == "final_macro_f1"]
    best = max(macro_rows, key=lambda row: float(row["mean"]))
    threshold = float(best["threshold"])
    system = str(best["system"])
    baseline = _value(rows, threshold, "conservative_balanced_router", "final_macro_f1")
    violation = _value(rows, threshold, system, "entitled_output_gate_violation_rate")
    gap = _value(rows, threshold, system, "polarity_flip_output_internal_gap")
    improved = baseline is not None and float(best["mean"]) > baseline
    promotion_systems = ("balanced_override_router", "cautious_promotion_router")
    promotion_best = max((row for row in macro_rows if row["system"] in promotion_systems),
                         key=lambda row: float(row["mean"]), default=None)
    lines = [
        f"The highest hybrid macro-F1 is {float(best['mean']):.3f} from {system} at threshold {threshold:.1f}.",
        ("This improves over" if improved else "This does not improve over")
        + " conservative_balanced_router at the same threshold.",
        f"At that setting, gate violation rate is {violation if violation is not None else 0.0:.3f} "
        f"and output/internal gap is {gap if gap is not None else 0.0:.3f}.",
    ]
    if promotion_best is not None:
        p_system, p_threshold = str(promotion_best["system"]), float(promotion_best["threshold"])
        support = _value(rows, p_threshold, p_system, "SUPPORT_f1")
        refute = _value(rows, p_threshold, p_system, "REFUTE_f1")
        p_violation = _value(rows, p_threshold, p_system, "entitled_output_gate_violation_rate")
        lines.append(f"The strongest promotion rule is {p_system}; SUPPORT/REFUTE F1 are "
                     f"{support if support is not None else 0.0:.3f}/{refute if refute is not None else 0.0:.3f}, "
                     f"with gate violation rate {p_violation if p_violation is not None else 0.0:.3f}.")
    lines.append("The comparison indicates whether a fixed multi-model expert router remains useful relative to single-model self-routing.")
    lines.append("This is a controlled fixed-rule diagnostic, not a final model, SOTA result, deployment claim, or general hallucination-reduction claim.")
    return lines


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lookup = {(float(row["threshold"]), row["system"], row["metric"]): row["formatted"] for row in rows}
    thresholds = sorted({float(row["threshold"]) for row in rows})
    lines = ["# Stage 6C Hybrid Expert Router Aggregate", ""]
    for threshold in thresholds:
        lines.extend([f"## THRESHOLD {threshold:.1f}", "",
                      "| system | " + " | ".join(METRICS) + " |",
                      "|---|" + "---:|" * len(METRICS)])
        for system in SYSTEM_ORDER:
            if not any(key[:2] == (threshold, system) for key in lookup):
                continue
            lines.append(f"| {system} | " + " | ".join(
                lookup.get((threshold, system, metric), "--") for metric in METRICS) + " |")
        lines.append("")
    lines.extend(["## INTERPRETATION", "", *[f"- {line}" for line in interpretation(rows)], ""])
    return "\n".join(lines)


def write_reports(inputs: Sequence[Path], output_csv: Path, output_md: Path) -> None:
    rows = aggregate_results(read_results(inputs))
    output_csv.parent.mkdir(parents=True, exist_ok=True); output_md.parent.mkdir(parents=True, exist_ok=True)
    fields = ("threshold", "system", "metric", "mean", "std", "n", "formatted")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    output_md.write_text(render_markdown(rows), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv); write_reports(args.input, args.output_csv, args.output_md); return 0


if __name__ == "__main__":
    raise SystemExit(main())
