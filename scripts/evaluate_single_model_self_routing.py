"""Evaluate Stage 6B single-model self-routing across gate thresholds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_router_ensemble import (  # noqa: E402
    ENTITLED_LABELS,
    auditor_passes,
    classification_metrics,
    internal_faithfulness_metrics,
    load_prediction_file,
    merge_prediction_files,
    pairwise_prediction_checks,
)
from scripts.sweep_router_thresholds import DEFAULT_THRESHOLDS, METRICS  # noqa: E402


SYSTEM_ORDER = (
    "raw_classifier_only",
    "self_routed_classifier",
    "raw_balanced_only",
    "self_routed_balanced",
    "raw_strict_only",
    "self_routed_strict",
)
MODEL_SYSTEMS = {
    "classifier": ("raw_classifier_only", "self_routed_classifier"),
    "balanced": ("raw_balanced_only", "self_routed_balanced"),
    "strict": ("raw_strict_only", "self_routed_strict"),
}


def _labels(
    merged: Sequence[Mapping[str, Mapping[str, Any]]],
    source: str,
    threshold: float,
    route: bool,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in merged:
        row = item[source]
        label = str(row["pred_final_label"])
        if route and label in ENTITLED_LABELS and not auditor_passes(row, threshold):
            label = "NOT_ENTITLED"
        result[str(row["id"])] = label
    return result


def _flatten(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "final_accuracy": metrics["final_accuracy"],
        "final_macro_f1": metrics["final_macro_f1"],
        **{
            f"{label}_f1": metrics["per_label_f1"][label]
            for label in ("NOT_ENTITLED", "REFUTE", "SUPPORT")
        },
        **metrics["internal_faithfulness"],
        **{
            name: result["pass_rate"]
            for name, result in metrics["pairwise_checks"].items()
        },
    }
    return {metric: values[metric] for metric in METRICS}


def evaluate_self_routing(
    classifier: Mapping[str, Any],
    balanced: Mapping[str, Any],
    strict: Mapping[str, Any],
    seed: int,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    merged = merge_prediction_files(classifier, balanced, strict)
    records = [item["classifier"] for item in merged]
    rows: list[dict[str, Any]] = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {threshold}")
        for source, (raw_system, routed_system) in MODEL_SYSTEMS.items():
            internal_rows = {
                str(item["classifier"]["id"]): (item[source],) for item in merged
            }
            for system, route in ((raw_system, False), (routed_system, True)):
                predictions = _labels(merged, source, threshold, route)
                metrics = {
                    **classification_metrics(records, predictions),
                    "pairwise_checks": pairwise_prediction_checks(records, predictions),
                    "internal_faithfulness": internal_faithfulness_metrics(
                        records, predictions, internal_rows, threshold
                    ),
                }
                for metric, value in _flatten(metrics).items():
                    rows.append(
                        {
                            "seed": seed,
                            "threshold": threshold,
                            "system": system,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("seed", "threshold", "system", "metric", "value")
        )
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lookup = {
        (float(row["threshold"]), str(row["system"]), str(row["metric"])): row["value"]
        for row in rows
    }
    thresholds = sorted({float(row["threshold"]) for row in rows})
    lines = ["# Stage 6B Single-Model Self-Routing", ""]
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
            values = [
                f"{float(lookup[(threshold, system, metric)]):.3f}"
                for metric in METRICS
            ]
            lines.append(f"| {system} | " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-preds", type=Path, required=True)
    parser.add_argument("--balanced-preds", type=Path, required=True)
    parser.add_argument("--strict-preds", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = evaluate_self_routing(
        load_prediction_file(args.classifier_preds),
        load_prediction_file(args.balanced_preds),
        load_prediction_file(args.strict_preds),
        args.seed,
        args.thresholds,
    )
    write_csv(args.output_csv, rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
