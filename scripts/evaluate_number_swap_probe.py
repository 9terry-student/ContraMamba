"""Compare Stage 10A number-swap predictions with the v3 time-swap failure."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_router_ensemble import ENTITLED_LABELS, load_prediction_file  # noqa: E402


GATE_FIELDS = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
    "polarity_margin",
)
OUTPUT_FIELDS = (
    "probe",
    "n",
    "gold_NOT_ENTITLED",
    "gold_SUPPORT",
    "gold_REFUTE",
    "pred_SUPPORT",
    "pred_REFUTE",
    "pred_NOT_ENTITLED",
    "classifier_error",
    "false_entitled",
    "false_entitled_rate",
    "mean_frame_prob",
    "mean_predicate_coverage_prob",
    "mean_sufficiency_prob",
    "mean_entitlement_prob",
    "mean_polarity_margin",
)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_probe(
    predictions: Sequence[Mapping[str, Any]], intervention_type: str, probe_name: str
) -> dict[str, Any]:
    rows = [row for row in predictions if row["intervention_type"] == intervention_type]
    if not rows:
        raise ValueError(f"no {intervention_type!r} predictions were found")
    gold = Counter(str(row["gold_final_label"]) for row in rows)
    predicted = Counter(str(row["pred_final_label"]) for row in rows)
    classifier_error = sum(row["gold_final_label"] != row["pred_final_label"] for row in rows)
    false_entitled = sum(
        row["gold_final_label"] == "NOT_ENTITLED"
        and row["pred_final_label"] in ENTITLED_LABELS
        for row in rows
    )
    return {
        "probe": probe_name,
        "n": len(rows),
        **{f"gold_{label}": gold[label] for label in ("NOT_ENTITLED", "SUPPORT", "REFUTE")},
        **{f"pred_{label}": predicted[label] for label in ("SUPPORT", "REFUTE", "NOT_ENTITLED")},
        "classifier_error": classifier_error,
        "false_entitled": false_entitled,
        "false_entitled_rate": safe_div(false_entitled, len(rows)),
        **{
            f"mean_{field}": sum(float(row[field]) for row in rows) / len(rows)
            for field in GATE_FIELDS
        },
    }


def comparison_conclusion(number: Mapping[str, Any], time: Mapping[str, Any]) -> str:
    number_gold_ne = int(number["gold_NOT_ENTITLED"])
    number_gold_refute = int(number["gold_REFUTE"])
    if number_gold_refute and not number_gold_ne:
        return "value_contradiction_refute_ontology"
    number_high_pass = all(
        float(number[field]) > 0.8
        for field in (
            "mean_frame_prob",
            "mean_predicate_coverage_prob",
            "mean_sufficiency_prob",
            "mean_entitlement_prob",
        )
    )
    if (
        float(number["false_entitled_rate"]) > 0.8
        and number_high_pass
        and float(time["false_entitled_rate"]) > 0.8
    ):
        return "same_type_low_surface_change_slot_value_failure"
    if (
        float(number["false_entitled_rate"]) < 0.2
        and float(time["false_entitled_rate"]) > 0.8
    ):
        return "temporal_specific_failure"
    return "inconclusive_mixed_pattern"


def write_outputs(csv_path: Path, md_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in OUTPUT_FIELDS} for row in rows)
    number = next(row for row in rows if row["probe"] == "number_swap")
    time = next(row for row in rows if row["probe"] == "time_swap")
    conclusion = comparison_conclusion(number, time)
    lines = [
        "# Stage 10A Number-Swap Single-Axis Probe",
        "",
        "| probe | n | gold NOT_ENTITLED | pred SUPPORT | pred REFUTE | pred NOT_ENTITLED | errors | false-entitled rate | frame | predicate | sufficiency | entitlement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['probe']} | {row['n']} | {row['gold_NOT_ENTITLED']} | "
            f"{row['pred_SUPPORT']} | {row['pred_REFUTE']} | {row['pred_NOT_ENTITLED']} | "
            f"{row['classifier_error']} | {float(row['false_entitled_rate']):.4f} | "
            f"{float(row['mean_frame_prob']):.4f} | {float(row['mean_predicate_coverage_prob']):.4f} | "
            f"{float(row['mean_sufficiency_prob']):.4f} | {float(row['mean_entitlement_prob']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Temporal-specific vs same-type-substitution decision",
            "",
            f"Diagnostic classification: `{conclusion}`.",
            "",
            "If number_swap and time_swap both show high false-entitled rates with high pass probabilities, the evidence supports a broader same-type low-surface-change or slot-value comparison failure. If number_swap is rejected while time_swap fails, the evidence favors a temporal-specific failure. If the ontology labels numeric mismatch as REFUTE, it must be analyzed separately as a value contradiction.",
            "",
            "## Interpretation constraints",
            "",
            "Do not generalize the time-swap result into broad presence-vs-match blindness: entity, event, and predicate swaps are already mostly rejected in v3.",
            "",
            "Do not infer shared gate mechanisms from global correlation alone. Stage 9D within-stratum and residualized correlations are the relevant diagnostic.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number-preds", type=Path, required=True)
    parser.add_argument("--time-preds", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    number_predictions = load_prediction_file(args.number_preds)["predictions"]
    time_predictions = load_prediction_file(args.time_preds)["predictions"]
    rows = [
        summarize_probe(number_predictions, "number_swap", "number_swap"),
        summarize_probe(time_predictions, "time_swap", "time_swap"),
    ]
    write_outputs(args.output_csv, args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
