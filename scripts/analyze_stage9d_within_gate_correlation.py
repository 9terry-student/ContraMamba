"""Stage 9D within-stratum and residualized gate-correlation analysis."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_router_ensemble import load_prediction_file  # noqa: E402


FIELDS = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
    "polarity_margin",
)
PAIR_METRICS = tuple(
    f"{method}_{left}__{right}"
    for method in ("pearson", "spearman")
    for index, left in enumerate(FIELDS)
    for right in FIELDS[index + 1 :]
)
NUMERIC_METRICS = (
    "n",
    "pearson_mean_abs_off_diagonal",
    "pearson_max_abs_off_diagonal",
    "spearman_mean_abs_off_diagonal",
    "spearman_max_abs_off_diagonal",
    *PAIR_METRICS,
)
PER_SEED_FIELDS = ("seed", "scope", "intervention_type", *NUMERIC_METRICS)


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[indexed[position][0]] = rank
        start = end
    return result


def correlation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    values = {field: [float(row[field]) for row in rows] for field in FIELDS}
    pairwise: dict[str, float] = {}
    pearson_values, spearman_values = [], []
    for index, left in enumerate(FIELDS):
        for right in FIELDS[index + 1 :]:
            linear = pearson(values[left], values[right])
            rank = pearson(ranks(values[left]), ranks(values[right]))
            pairwise[f"pearson_{left}__{right}"] = linear
            pairwise[f"spearman_{left}__{right}"] = rank
            pearson_values.append(abs(linear))
            spearman_values.append(abs(rank))
    return {
        "n": len(rows),
        "pearson_mean_abs_off_diagonal": mean(pearson_values),
        "pearson_max_abs_off_diagonal": max(pearson_values, default=0.0),
        "spearman_mean_abs_off_diagonal": mean(spearman_values),
        "spearman_max_abs_off_diagonal": max(spearman_values, default=0.0),
        **pairwise,
    }


def residualize_by_intervention(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["intervention_type"])].append(row)
    group_means = {
        intervention: {
            field: mean([float(row[field]) for row in group]) for field in FIELDS
        }
        for intervention, group in grouped.items()
    }
    return [
        {
            **row,
            **{
                field: float(row[field])
                - group_means[str(row["intervention_type"])][field]
                for field in FIELDS
            },
        }
        for row in rows
    ]


def intervention_mean_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["intervention_type"])].append(row)
    return [
        {
            "intervention_type": intervention,
            **{field: mean([float(row[field]) for row in group]) for field in FIELDS},
        }
        for intervention, group in sorted(grouped.items())
    ]


def analyze_correlations(
    predictions: Sequence[Mapping[str, Any]], seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["intervention_type"])].append(row)
    output = [
        {
            "seed": seed,
            "scope": "global",
            "intervention_type": "__ALL__",
            **correlation_summary(predictions),
        }
    ]
    output.extend(
        {
            "seed": seed,
            "scope": "within_intervention",
            "intervention_type": intervention,
            **correlation_summary(rows),
        }
        for intervention, rows in sorted(grouped.items())
    )
    output.append(
        {
            "seed": seed,
            "scope": "between_intervention_means",
            "intervention_type": "__MEANS__",
            **correlation_summary(intervention_mean_rows(predictions)),
        }
    )
    output.append(
        {
            "seed": seed,
            "scope": "residualized",
            "intervention_type": "__ALL_RESIDUALIZED__",
            **correlation_summary(residualize_by_intervention(predictions)),
        }
    )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def aggregate(paths: Sequence[Path]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for metric in NUMERIC_METRICS:
                    grouped[(row["scope"], row["intervention_type"], metric)].append(
                        float(row[metric])
                    )
    output = []
    for (scope, intervention, metric), values in grouped.items():
        value_mean = mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(
            {
                "scope": scope,
                "intervention_type": intervention,
                "metric": metric,
                "mean": value_mean,
                "std": std,
                "n": len(values),
                "formatted": f"{value_mean:.4f} +/- {std:.4f}",
            }
        )
    return sorted(
        output, key=lambda row: (row["scope"], row["intervention_type"], row["metric"])
    )


def _lookup(
    rows: Sequence[Mapping[str, Any]], scope: str, intervention: str, metric: str
) -> str:
    return next(
        (
            str(row["formatted"])
            for row in rows
            if row["scope"] == scope
            and row["intervention_type"] == intervention
            and row["metric"] == metric
        ),
        "--",
    )


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    summary_metrics = (
        "pearson_mean_abs_off_diagonal",
        "pearson_max_abs_off_diagonal",
        "spearman_mean_abs_off_diagonal",
        "spearman_max_abs_off_diagonal",
    )
    summary_scopes = (
        ("global", "__ALL__"),
        ("between_intervention_means", "__MEANS__"),
        ("residualized", "__ALL_RESIDUALIZED__"),
    )
    lines = [
        "# Stage 9D Within-Stratum Gate Correlation",
        "",
        "## Global, between-intervention, and residualized correlation",
        "",
        "| scope | " + " | ".join(summary_metrics) + " |",
        "|---|" + "---:|" * len(summary_metrics),
    ]
    for scope, intervention in summary_scopes:
        lines.append(
            f"| {scope} | "
            + " | ".join(_lookup(rows, scope, intervention, metric) for metric in summary_metrics)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Within-intervention correlation",
            "",
            "| intervention_type | " + " | ".join(summary_metrics) + " |",
            "|---|" + "---:|" * len(summary_metrics),
        ]
    )
    interventions = sorted(
        {row["intervention_type"] for row in rows if row["scope"] == "within_intervention"}
    )
    for intervention in interventions:
        lines.append(
            f"| {intervention} | "
            + " | ".join(
                _lookup(rows, "within_intervention", intervention, metric)
                for metric in summary_metrics
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If global correlation is moderate while residualized and within-intervention correlations are low, the global pattern may be driven by between-intervention difficulty rather than a shared gate mechanism. If within-intervention correlations remain high, the heads move together even within the same perturbation type, supporting partial redundancy or a shared mechanism.",
            "",
            "The gate heads show moderate global correlation, but within-stratum analysis is needed to distinguish shared mechanism from between-intervention difficulty.",
            "",
            "Do not claim four independent gates or one effective gate from global correlations alone. The conclusion must follow the within-intervention and residualized results.",
            "",
            "## Link to Stage 10A",
            "",
            "Stage 9D does not determine whether time_swap is temporal-specific or a broader same-type low-surface-change substitution failure. The number_swap probe tests that distinction directly.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--output-aggregate-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input:
        if not args.output_aggregate_csv or not args.output_md:
            raise ValueError("aggregate mode requires aggregate CSV and Markdown outputs")
        rows = aggregate(args.input)
        fields = ("scope", "intervention_type", "metric", "mean", "std", "n", "formatted")
        write_csv(args.output_aggregate_csv, rows, fields)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(rows), encoding="utf-8")
        return 0
    if args.preds is None or args.seed is None or args.output_csv is None:
        raise ValueError("per-seed mode requires --preds, --seed, and --output-csv")
    predictions = load_prediction_file(args.preds)["predictions"]
    write_csv(args.output_csv, analyze_correlations(predictions, args.seed), PER_SEED_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
