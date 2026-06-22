"""Evaluate fixed Stage 6C hybrid expert-router rules."""

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

from scripts.evaluate_router_ensemble import (  # noqa: E402
    AUDITOR_GATES,
    ENTITLED_LABELS,
    auditor_passes,
    classification_metrics,
    internal_faithfulness_metrics,
    load_prediction_file,
    merge_prediction_files,
    pairwise_prediction_checks,
)
from scripts.sweep_router_thresholds import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    METRICS,
    flatten_metrics,
)


SYSTEM_ORDER = (
    "conservative_balanced_router",
    "self_routed_balanced",
    "agreement_keep_router",
    "balanced_override_router",
    "strict_veto_balanced_router",
    "majority_gate_verified_router",
    "balanced_strict_agreement_router",
    "cautious_promotion_router",
)


def _base_label(item: Mapping[str, Mapping[str, Any]], threshold: float) -> str:
    label = str(item["classifier"]["pred_final_label"])
    if label in ENTITLED_LABELS and not auditor_passes(item["balanced"], threshold):
        return "NOT_ENTITLED"
    return label


def _minimum_gate(row: Mapping[str, Any]) -> float:
    return min(float(row[key]) for key in AUDITOR_GATES)


def build_hybrid_systems(
    merged: Sequence[Mapping[str, Mapping[str, Any]]],
    threshold: float,
    high_threshold: float,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, tuple[Mapping[str, Any], ...]]]]:
    labels = {system: {} for system in SYSTEM_ORDER}
    internals = {system: {} for system in SYSTEM_ORDER}
    for item in merged:
        example_id = str(item["classifier"]["id"])
        classifier = item["classifier"]
        balanced = item["balanced"]
        strict = item["strict"]
        c_label = str(classifier["pred_final_label"])
        b_label = str(balanced["pred_final_label"])
        s_label = str(strict["pred_final_label"])
        base = _base_label(item, threshold)

        labels["conservative_balanced_router"][example_id] = base
        internals["conservative_balanced_router"][example_id] = (balanced,)

        self_balanced = b_label
        if b_label in ENTITLED_LABELS and not auditor_passes(balanced, threshold):
            self_balanced = "NOT_ENTITLED"
        labels["self_routed_balanced"][example_id] = self_balanced
        internals["self_routed_balanced"][example_id] = (balanced,)

        agreement = "NOT_ENTITLED"
        if c_label == b_label:
            if c_label in ENTITLED_LABELS and auditor_passes(balanced, threshold):
                agreement = c_label
            elif c_label == "NOT_ENTITLED":
                agreement = c_label
        labels["agreement_keep_router"][example_id] = agreement
        internals["agreement_keep_router"][example_id] = (balanced,)

        override = base
        if (
            c_label == "NOT_ENTITLED"
            and b_label in ENTITLED_LABELS
            and auditor_passes(balanced, high_threshold)
        ):
            override = b_label
        elif c_label in ENTITLED_LABELS and b_label in ENTITLED_LABELS and c_label != b_label:
            override = "NOT_ENTITLED"
        labels["balanced_override_router"][example_id] = override
        internals["balanced_override_router"][example_id] = (balanced,)

        strict_veto = base
        if strict_veto in ENTITLED_LABELS and not auditor_passes(strict, threshold):
            strict_veto = "NOT_ENTITLED"
        labels["strict_veto_balanced_router"][example_id] = strict_veto
        internals["strict_veto_balanced_router"][example_id] = (balanced, strict)

        counts = Counter((c_label, b_label, s_label))
        majority_label, majority_count = counts.most_common(1)[0]
        selected = classifier
        majority = "NOT_ENTITLED"
        if majority_count >= 2 and majority_label in ENTITLED_LABELS:
            candidates = [row for row in (classifier, balanced, strict)
                          if row["pred_final_label"] == majority_label]
            selected = max(candidates, key=_minimum_gate)
            if auditor_passes(selected, threshold):
                majority = majority_label
        labels["majority_gate_verified_router"][example_id] = majority
        internals["majority_gate_verified_router"][example_id] = (selected,)

        if b_label == s_label:
            if b_label == "NOT_ENTITLED":
                bs_agreement = "NOT_ENTITLED"
            elif (b_label in ENTITLED_LABELS and auditor_passes(balanced, threshold)
                  and auditor_passes(strict, threshold)):
                bs_agreement = b_label
            else:
                bs_agreement = "NOT_ENTITLED"
            bs_internal = (balanced, strict)
        else:
            bs_agreement = base
            bs_internal = (balanced,)
        labels["balanced_strict_agreement_router"][example_id] = bs_agreement
        internals["balanced_strict_agreement_router"][example_id] = bs_internal

        cautious = base
        cautious_internal: tuple[Mapping[str, Any], ...] = (balanced,)
        if (
            c_label == "NOT_ENTITLED"
            and b_label == s_label
            and b_label in ENTITLED_LABELS
            and auditor_passes(balanced, threshold)
            and auditor_passes(strict, threshold)
        ):
            cautious = b_label
            cautious_internal = (balanced, strict)
        labels["cautious_promotion_router"][example_id] = cautious
        internals["cautious_promotion_router"][example_id] = cautious_internal
    return labels, internals


def evaluate_hybrid_search(
    classifier: Mapping[str, Any], balanced: Mapping[str, Any], strict: Mapping[str, Any],
    seed: int, thresholds: Sequence[float] = DEFAULT_THRESHOLDS, high_threshold: float = 0.7,
) -> list[dict[str, Any]]:
    merged = merge_prediction_files(classifier, balanced, strict)
    records = [item["classifier"] for item in merged]
    rows = []
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        if not 0.0 <= threshold <= 1.0 or not 0.0 <= high_threshold <= 1.0:
            raise ValueError("thresholds must be within [0, 1]")
        labels, internals = build_hybrid_systems(merged, threshold, high_threshold)
        for system in SYSTEM_ORDER:
            metrics = {
                **classification_metrics(records, labels[system]),
                "pairwise_checks": pairwise_prediction_checks(records, labels[system]),
                "internal_faithfulness": internal_faithfulness_metrics(
                    records, labels[system], internals[system], threshold
                ),
            }
            for metric, value in flatten_metrics(metrics).items():
                rows.append({"seed": seed, "threshold": threshold, "system": system,
                             "metric": metric, "value": value})
    return rows


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lookup = {(float(row["threshold"]), row["system"], row["metric"]): row["value"] for row in rows}
    thresholds = sorted({float(row["threshold"]) for row in rows})
    lines = ["# Stage 6C Hybrid Expert Router Search", ""]
    for threshold in thresholds:
        lines.extend([f"## THRESHOLD {threshold:.1f}", "",
                      "| system | " + " | ".join(METRICS) + " |",
                      "|---|" + "---:|" * len(METRICS)])
        for system in SYSTEM_ORDER:
            values = [f"{float(lookup[(threshold, system, metric)]):.3f}" for metric in METRICS]
            lines.append(f"| {system} | " + " | ".join(values) + " |")
        lines.append("")
    best = max((row for row in rows if row["metric"] == "final_macro_f1"),
               key=lambda row: float(row["value"]))
    lines.extend(["## INTERPRETATION", "",
                  f"- Highest per-seed macro-F1: {best['system']} at threshold {float(best['threshold']):.1f} ({float(best['value']):.3f}).",
                  "- These are fixed controlled diagnostic rules, not a learned or final router.", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-preds", type=Path, required=True)
    parser.add_argument("--balanced-preds", type=Path, required=True)
    parser.add_argument("--strict-preds", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--high-threshold", type=float, default=0.7)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = evaluate_hybrid_search(load_prediction_file(args.classifier_preds),
                                  load_prediction_file(args.balanced_preds),
                                  load_prediction_file(args.strict_preds), args.seed,
                                  args.thresholds, args.high_threshold)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "threshold", "system", "metric", "value"))
        writer.writeheader(); writer.writerows(rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
