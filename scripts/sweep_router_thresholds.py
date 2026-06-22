"""Evaluate conservative Stage 6A routers across gate thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluate_router_ensemble import evaluate_router_systems, load_prediction_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SYSTEMS = (
    "conservative_balanced_router",
    "conservative_strict_router",
    "dual_auditor_router",
)
METRICS = (
    "final_accuracy",
    "final_macro_f1",
    "NOT_ENTITLED_f1",
    "REFUTE_f1",
    "SUPPORT_f1",
    "entitled_output_count",
    "entitled_output_gate_violation_rate",
    "entitled_output_gate_violations",
    "polarity_flip_output_ok",
    "polarity_flip_internal_ok",
    "polarity_flip_output_internal_gap",
    "polarity_flip_output_ok_but_internal_bad",
    "paraphrase_preserved",
    "predicate_disentangled",
    "polarity_flip_preserved_and_reversed",
    "deletion_sufficiency_lower",
    "truncation_sufficiency_lower",
    "entity_frame_lower",
    "event_frame_lower",
)


def flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "final_accuracy": metrics["final_accuracy"],
        "final_macro_f1": metrics["final_macro_f1"],
        **{
            f"{label}_f1": metrics["per_label_f1"][label]
            for label in ("NOT_ENTITLED", "REFUTE", "SUPPORT")
        },
        **metrics["internal_faithfulness"],
    }
    row.update(
        {
            name: value["pass_rate"]
            for name, value in metrics["pairwise_checks"].items()
        }
    )
    return {name: row[name] for name in METRICS}


def sweep_thresholds(
    classifier: Mapping[str, Any],
    balanced: Mapping[str, Any],
    strict: Mapping[str, Any],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {threshold}")
        evaluated = evaluate_router_systems(
            classifier, balanced, strict, threshold=float(threshold)
        )
        for system in SYSTEMS:
            rows.append(
                {
                    "threshold": float(threshold),
                    "system": system,
                    **flatten_metrics(evaluated[system]),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("threshold", "system", *METRICS))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: Sequence[Mapping[str, Any]], seed: int) -> str:
    lines = [f"# Stage 6A Router Threshold Sweep: Seed {seed}", ""]
    table_groups = (
        ("CLASSIFICATION", METRICS[:5]),
        ("INTERNAL_FAITHFULNESS", METRICS[5:12]),
        ("PAIRWISE_CONSISTENCY", METRICS[12:]),
    )
    for title, metrics in table_groups:
        lines.extend(
            [
                f"## {title}",
                "",
                "| threshold | system | " + " | ".join(metrics) + " |",
                "|---:|---|" + "---:|" * len(metrics),
            ]
        )
        for row in rows:
            values = [f"{float(row[name]):.3f}" for name in metrics]
            lines.append(
                f"| {float(row['threshold']):.1f} | {row['system']} | "
                + " | ".join(values)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-preds", type=Path, required=True)
    parser.add_argument("--balanced-preds", type=Path, required=True)
    parser.add_argument("--strict-preds", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = sweep_thresholds(
        load_prediction_file(args.classifier_preds),
        load_prediction_file(args.balanced_preds),
        load_prediction_file(args.strict_preds),
        args.thresholds,
    )
    output_csv = args.output_csv or ROOT / "results" / f"stage6a_router_threshold_sweep_seed{args.seed}.csv"
    output_md = args.output_md or ROOT / "results" / f"stage6a_router_threshold_sweep_seed{args.seed}.md"
    write_csv(output_csv, rows)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(rows, args.seed), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
