"""Aggregate Stage 6B single-model self-routing results across seeds."""

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

from scripts.evaluate_single_model_self_routing import SYSTEM_ORDER  # noqa: E402
from scripts.sweep_router_thresholds import METRICS  # noqa: E402


def read_results(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for implicit_seed, path in enumerate(paths, start=1):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append(
                        {
                            "seed": int(row.get("seed") or implicit_seed),
                            "threshold": float(row["threshold"]),
                            "system": row["system"],
                            "metric": row["metric"],
                            "value": float(row["value"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    if not rows:
        raise ValueError("no numeric self-routing metrics were found")
    return rows


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["threshold"], row["system"], row["metric"])].append(
            float(row["value"])
        )
    system_rank = {system: index for index, system in enumerate(SYSTEM_ORDER)}
    metric_rank = {metric: index for index, metric in enumerate(METRICS)}
    result = []
    for (threshold, system, metric), values in grouped.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(
            {
                "threshold": threshold,
                "system": system,
                "metric": metric,
                "mean": mean,
                "std": std,
                "n": len(values),
                "formatted": f"{mean:.3f} +/- {std:.3f}",
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["threshold"],
            system_rank.get(row["system"], 999),
            metric_rank.get(row["metric"], 999),
        ),
    )


def _mean(
    rows: Sequence[Mapping[str, Any]], threshold: float, system: str, metric: str
) -> float | None:
    return next(
        (
            float(row["mean"])
            for row in rows
            if row["threshold"] == threshold
            and row["system"] == system
            and row["metric"] == metric
        ),
        None,
    )


def interpretation(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    thresholds = sorted({float(row["threshold"]) for row in rows})
    self_systems = (
        "self_routed_classifier", "self_routed_balanced", "self_routed_strict"
    )
    candidates = [
        (float(row["mean"]), float(row["threshold"]), str(row["system"]))
        for row in rows
        if row["metric"] == "final_macro_f1" and row["system"] in self_systems
    ]
    best_value, best_threshold, best_system = max(candidates) if candidates else (0.0, 0.0, "none")
    statements = [
        f"The best self-routed single-model result is {best_system} at threshold {best_threshold:.1f} "
        f"with macro-F1 {best_value:.3f}; compare this diagnostic value with the Stage 5C "
        "conservative balanced-auditor router macro-F1 of 0.878.",
    ]
    zero_systems = sorted(
        {
            str(row["system"])
            for row in rows
            if row["metric"] == "entitled_output_gate_violation_rate"
            and row["system"] in self_systems
            and float(row["mean"]) == 0.0
        }
    )
    statements.append(
        "Self-routing eliminates entitled-output gate violations at evaluated settings for: "
        + (", ".join(zero_systems) if zero_systems else "none")
        + "."
    )
    if len(thresholds) >= 2:
        low, high = thresholds[0], thresholds[-1]
        decreases = []
        for system in self_systems:
            low_count = _mean(rows, low, system, "entitled_output_count")
            high_count = _mean(rows, high, system, "entitled_output_count")
            if low_count is not None and high_count is not None and high_count < low_count:
                decreases.append(system)
        statements.append(
            "Entitled-output counts decrease as the threshold rises for "
            + (", ".join(decreases) if decreases else "no self-routed system")
            + ", indicating the risk of excessive NOT_ENTITLED downgrades."
        )
    statements.append(
        "self_routed_balanced is the leading single-model candidate."
        if best_system == "self_routed_balanced"
        else "self_routed_balanced is not the leading single-model candidate in this aggregate."
    )
    statements.append(
        "This controlled diagnostic informs whether distillation is necessary; it is not a final-model or real-world hallucination-reduction claim."
    )
    return statements


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lookup = {
        (float(row["threshold"]), str(row["system"]), str(row["metric"])): row["formatted"]
        for row in rows
    }
    thresholds = sorted({float(row["threshold"]) for row in rows})
    lines = ["# Stage 6B Single-Model Self-Routing Aggregate", ""]
    for threshold in thresholds:
        lines.extend(
            [
                f"## THRESHOLD {threshold:.1f}",
                "",
                "| system | " + " | ".join(METRICS) + " |",
                "|---|" + "---:|" * len(METRICS),
            ]
        )
        for system in SYSTEM_ORDER:
            if not any(key[:2] == (threshold, system) for key in lookup):
                continue
            values = [lookup.get((threshold, system, metric), "--") for metric in METRICS]
            lines.append(f"| {system} | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(["## INTERPRETATION", ""])
    lines.extend(f"- {statement}" for statement in interpretation(rows))
    lines.append("")
    return "\n".join(lines)


def write_reports(inputs: Sequence[Path], output_csv: Path, output_md: Path) -> None:
    rows = aggregate_results(read_results(inputs))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    fields = ("threshold", "system", "metric", "mean", "std", "n", "formatted")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    output_md.write_text(render_markdown(rows), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_reports(args.input, args.output_csv, args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
