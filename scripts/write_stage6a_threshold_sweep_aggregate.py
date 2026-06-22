"""Aggregate Stage 6A threshold sweep CSVs across seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sweep_router_thresholds import METRICS, SYSTEMS


def read_sweeps(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, path in enumerate(paths, start=1):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    threshold = float(row["threshold"])
                except (KeyError, TypeError, ValueError):
                    continue
                for metric, raw_value in row.items():
                    if metric in {"threshold", "system"}:
                        continue
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    rows.append(
                        {
                            "seed": seed,
                            "threshold": threshold,
                            "system": row["system"],
                            "metric": metric,
                            "value": value,
                        }
                    )
    if not rows:
        raise ValueError("no numeric threshold sweep metrics were found")
    return rows


def aggregate_sweeps(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["threshold"], row["system"], row["metric"])].append(row["value"])
    system_rank = {name: index for index, name in enumerate(SYSTEMS)}
    metric_rank = {name: index for index, name in enumerate(METRICS)}
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


def render_markdown(rows: Sequence[dict[str, Any]]) -> str:
    lookup = {
        (row["threshold"], row["system"], row["metric"]): row["formatted"]
        for row in rows
    }
    thresholds = sorted({row["threshold"] for row in rows})
    lines = ["# Stage 6A Router Threshold Sweep Aggregate", ""]
    for threshold in thresholds:
        lines.extend(
            [
                f"## THRESHOLD {threshold:.1f}",
                "",
                "| system | " + " | ".join(METRICS) + " |",
                "|---|" + "---:|" * len(METRICS),
            ]
        )
        for system in SYSTEMS:
            if not any(key[:2] == (threshold, system) for key in lookup):
                continue
            values = [lookup.get((threshold, system, metric), "--") for metric in METRICS]
            lines.append(f"| {system} | " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines)


def write_reports(inputs: Sequence[Path], output_csv: Path, output_md: Path) -> None:
    aggregate = aggregate_sweeps(read_sweeps(inputs))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    fields = ("threshold", "system", "metric", "mean", "std", "n", "formatted")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in aggregate)
    output_md.write_text(render_markdown(aggregate), encoding="utf-8")


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
