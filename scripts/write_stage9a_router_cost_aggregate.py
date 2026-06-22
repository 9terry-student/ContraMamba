"""Aggregate Stage 9A router cost-of-conservatism sweeps across seeds."""

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

from scripts.evaluate_router_ensemble import ROUTER_COST_METRIC_NAMES  # noqa: E402
from scripts.sweep_router_thresholds import SYSTEMS  # noqa: E402


REPORT_METRICS = (
    "final_accuracy",
    "final_macro_f1",
    *ROUTER_COST_METRIC_NAMES,
)


def read_seed_sweeps(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, path in enumerate(paths, start=1):
        with path.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                try:
                    threshold = float(source["threshold"])
                    system = source["system"]
                except (KeyError, TypeError, ValueError):
                    continue
                for metric in REPORT_METRICS:
                    try:
                        value = float(source[metric])
                    except (KeyError, TypeError, ValueError):
                        continue
                    rows.append(
                        {
                            "seed": seed,
                            "threshold": threshold,
                            "system": system,
                            "metric": metric,
                            "value": value,
                        }
                    )
    if not rows:
        raise ValueError("no numeric Stage 9A router cost metrics were found")
    return rows


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["threshold"], row["system"], row["metric"])].append(
            row["value"]
        )
    system_rank = {name: index for index, name in enumerate(SYSTEMS)}
    metric_rank = {name: index for index, name in enumerate(REPORT_METRICS)}
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
    lines = ["# Stage 9A Router Cost of Conservatism", ""]
    for threshold in thresholds:
        lines.extend(
            [
                f"## THRESHOLD {threshold:.1f}",
                "",
                "| system | " + " | ".join(REPORT_METRICS) + " |",
                "|---|" + "---:|" * len(REPORT_METRICS),
            ]
        )
        for system in SYSTEMS:
            if not any(key[:2] == (threshold, system) for key in lookup):
                continue
            lines.append(
                f"| {system} | "
                + " | ".join(
                    lookup.get((threshold, system, metric), "--")
                    for metric in REPORT_METRICS
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## INTERPRETATION",
            "",
            "Post-routing retained-output gate violations are an enforced invariant for correctly implemented conservative routers. The empirical tradeoff is described by pre-router gate failures, downgrade rates, SUPPORT recall loss, precision gain, and removed false entitled predictions.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(inputs: Sequence[Path], output_csv: Path, output_md: Path) -> None:
    rows = aggregate(read_seed_sweeps(inputs))
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
