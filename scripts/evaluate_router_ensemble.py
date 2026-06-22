"""Evaluate classifier/auditor routers from controlled-v5 prediction exports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
ENTITLED_LABELS = {"REFUTE", "SUPPORT"}
PAIRWISE_CHECKS = (
    "paraphrase_preserved",
    "predicate_disentangled",
    "polarity_flip_preserved_and_reversed",
    "deletion_sufficiency_lower",
    "truncation_sufficiency_lower",
    "entity_frame_lower",
    "event_frame_lower",
)
AUDITOR_GATES = (
    "entitlement_prob",
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
)


def load_prediction_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("predictions"), list):
        raise ValueError(f"{path} must contain a predictions list")
    ids = [row.get("id") for row in payload["predictions"]]
    if any(value is None for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains missing or duplicate prediction ids")
    return payload


def _index_predictions(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in payload["predictions"]}


def merge_prediction_files(
    classifier: Mapping[str, Any],
    balanced: Mapping[str, Any],
    strict: Mapping[str, Any],
) -> list[dict[str, dict[str, Any]]]:
    indexed = [_index_predictions(value) for value in (classifier, balanced, strict)]
    id_sets = [set(value) for value in indexed]
    if id_sets[0] != id_sets[1] or id_sets[0] != id_sets[2]:
        raise ValueError("prediction files must contain exactly the same ids")
    merged = []
    for example_id in sorted(id_sets[0]):
        rows = [value[example_id] for value in indexed]
        identity = (
            rows[0].get("pair_id"),
            rows[0].get("intervention_type"),
            rows[0].get("gold_final_label"),
        )
        if any(
            (row.get("pair_id"), row.get("intervention_type"), row.get("gold_final_label"))
            != identity
            for row in rows[1:]
        ):
            raise ValueError(f"prediction metadata mismatch for id {example_id!r}")
        merged.append(
            {"classifier": rows[0], "balanced": rows[1], "strict": rows[2]}
        )
    return merged


def auditor_passes(row: Mapping[str, Any], threshold: float = 0.5) -> bool:
    missing = [key for key in AUDITOR_GATES if key not in row]
    if missing:
        raise ValueError(f"auditor prediction is missing gates: {missing}")
    return all(float(row[key]) >= threshold for key in AUDITOR_GATES)


def build_system_predictions(
    merged: Sequence[Mapping[str, Mapping[str, Any]]], threshold: float = 0.5
) -> dict[str, dict[str, str]]:
    systems = {
        name: {}
        for name in (
            "classifier_only",
            "balanced_only",
            "strict_only",
            "conservative_balanced_router",
            "conservative_strict_router",
            "dual_auditor_router",
        )
    }
    for item in merged:
        classifier = item["classifier"]
        example_id = str(classifier["id"])
        classifier_label = classifier["pred_final_label"]
        systems["classifier_only"][example_id] = classifier_label
        systems["balanced_only"][example_id] = item["balanced"]["pred_final_label"]
        systems["strict_only"][example_id] = item["strict"]["pred_final_label"]
        balanced_passes = auditor_passes(item["balanced"], threshold)
        strict_passes = auditor_passes(item["strict"], threshold)

        def routed(*passes: bool) -> str:
            if classifier_label in ENTITLED_LABELS and not all(passes):
                return "NOT_ENTITLED"
            return classifier_label

        systems["conservative_balanced_router"][example_id] = routed(balanced_passes)
        systems["conservative_strict_router"][example_id] = routed(strict_passes)
        systems["dual_auditor_router"][example_id] = routed(
            balanced_passes, strict_passes
        )
    return systems


def _f1(gold: Sequence[str], predicted: Sequence[str], label: str) -> float:
    true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
    false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
    false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def classification_metrics(
    records: Sequence[Mapping[str, Any]], predictions: Mapping[str, str]
) -> dict[str, Any]:
    gold = [row["gold_final_label"] for row in records]
    predicted = [predictions[str(row["id"])] for row in records]
    per_label = {label: _f1(gold, predicted, label) for label in LABELS}
    return {
        "final_accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold),
        "final_macro_f1": sum(per_label.values()) / len(LABELS),
        "per_label_f1": per_label,
        "prediction_distribution": dict(sorted(Counter(predicted).items())),
    }


def pairwise_prediction_checks(
    records: Sequence[Mapping[str, Any]], predictions: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for row in records:
        groups[str(row["pair_id"])][str(row["intervention_type"])] = predictions[
            str(row["id"])
        ]
    predicates = {
        "paraphrase_preserved": lambda values: values["none"] == values["paraphrase"],
        "predicate_disentangled": lambda values: values["none"] in ENTITLED_LABELS
        and values["predicate_swap"] == "NOT_ENTITLED",
        "polarity_flip_preserved_and_reversed": lambda values: values["none"]
        in ENTITLED_LABELS
        and values["polarity_flip"] in ENTITLED_LABELS
        and values["none"] != values["polarity_flip"],
        "deletion_sufficiency_lower": lambda values: values["evidence_deletion"]
        == "NOT_ENTITLED",
        "truncation_sufficiency_lower": lambda values: values["evidence_truncation"]
        == "NOT_ENTITLED",
        "entity_frame_lower": lambda values: values["entity_swap"] == "NOT_ENTITLED",
        "event_frame_lower": lambda values: values["event_swap"] == "NOT_ENTITLED",
    }
    results: dict[str, dict[str, Any]] = {}
    for name, predicate in predicates.items():
        passed = 0
        total = 0
        for pair_id, values in groups.items():
            try:
                outcome = predicate(values)
            except KeyError as error:
                raise ValueError(
                    f"pair {pair_id!r} lacks intervention {error.args[0]!r} for {name}"
                ) from error
            total += 1
            passed += int(outcome)
        results[name] = {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total else 0.0,
        }
    return results


def internal_faithfulness_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, str],
    internal_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Measure whether final outputs are supported by their internal gates/signs."""

    entitled_count = 0
    violations = 0
    for row in records:
        example_id = str(row["id"])
        if predictions[example_id] not in ENTITLED_LABELS:
            continue
        entitled_count += 1
        if not all(auditor_passes(internal) for internal in internal_rows[example_id]):
            violations += 1

    groups: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        groups[str(row["pair_id"])][str(row["intervention_type"])] = row
    output_ok_count = 0
    internal_ok_count = 0
    output_ok_internal_bad = 0
    polarity_pair_count = 0

    def direction_matches(example_id: str, label: str) -> bool:
        for internal in internal_rows[example_id]:
            margin = float(internal["polarity_margin"])
            if label == "SUPPORT" and margin <= 0:
                return False
            if label == "REFUTE" and margin >= 0:
                return False
        return True

    for values in groups.values():
        if "none" not in values or "polarity_flip" not in values:
            continue
        polarity_pair_count += 1
        none_id = str(values["none"]["id"])
        flip_id = str(values["polarity_flip"]["id"])
        none_label = predictions[none_id]
        flip_label = predictions[flip_id]
        output_ok = (
            none_label in ENTITLED_LABELS
            and flip_label in ENTITLED_LABELS
            and none_label != flip_label
        )
        internal_ok = (
            output_ok
            and all(auditor_passes(row) for row in internal_rows[none_id])
            and all(auditor_passes(row) for row in internal_rows[flip_id])
            and direction_matches(none_id, none_label)
            and direction_matches(flip_id, flip_label)
        )
        output_ok_count += int(output_ok)
        internal_ok_count += int(internal_ok)
        output_ok_internal_bad += int(output_ok and not internal_ok)

    output_rate = output_ok_count / polarity_pair_count if polarity_pair_count else 0.0
    internal_rate = (
        internal_ok_count / polarity_pair_count if polarity_pair_count else 0.0
    )
    return {
        "entitled_output_gate_violation_rate": (
            violations / entitled_count if entitled_count else 0.0
        ),
        "entitled_output_gate_violations": violations,
        "entitled_output_count": entitled_count,
        "polarity_flip_output_ok": output_rate,
        "polarity_flip_internal_ok": internal_rate,
        "polarity_flip_output_internal_gap": output_rate - internal_rate,
        "polarity_flip_output_ok_but_internal_bad": output_ok_internal_bad,
    }


def evaluate_router_systems(
    classifier: Mapping[str, Any],
    balanced: Mapping[str, Any],
    strict: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    merged = merge_prediction_files(classifier, balanced, strict)
    records = [item["classifier"] for item in merged]
    predictions = build_system_predictions(merged)
    internal_sources = {
        "classifier_only": ("classifier",),
        "balanced_only": ("balanced",),
        "strict_only": ("strict",),
        "conservative_balanced_router": ("balanced",),
        "conservative_strict_router": ("strict",),
        "dual_auditor_router": ("balanced", "strict"),
    }
    return {
        system: {
            **classification_metrics(records, labels),
            "pairwise_checks": pairwise_prediction_checks(records, labels),
            "internal_faithfulness": internal_faithfulness_metrics(
                records,
                labels,
                {
                    str(item["classifier"]["id"]): tuple(
                        item[source] for source in internal_sources[system]
                    )
                    for item in merged
                },
            ),
        }
        for system, labels in predictions.items()
    }


def _format(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(results: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "## ROUTER_CLASSIFICATION_SUMMARY",
        "",
        "| system | accuracy | macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | prediction distribution |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for system, metrics in results.items():
        distribution = json.dumps(metrics["prediction_distribution"], sort_keys=True)
        f1 = metrics["per_label_f1"]
        lines.append(
            f"| {system} | {_format(metrics['final_accuracy'])} | "
            f"{_format(metrics['final_macro_f1'])} | {_format(f1['NOT_ENTITLED'])} | "
            f"{_format(f1['REFUTE'])} | {_format(f1['SUPPORT'])} | `{distribution}` |"
        )
    lines.extend(
        [
            "",
            "## ROUTER_PAIRWISE_SUMMARY",
            "",
            "| system | paraphrase | predicate-pair | polarity-flip | deletion | truncation | entity | event |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for system, metrics in results.items():
        checks = metrics["pairwise_checks"]
        values = [checks[name]["pass_rate"] for name in PAIRWISE_CHECKS]
        lines.append(f"| {system} | " + " | ".join(map(_format, values)) + " |")
    lines.extend(
        [
            "",
            "## ROUTER_INTERNAL_FAITHFULNESS_SUMMARY",
            "",
            "| system | entitled gate violation rate | entitled violations/count | polarity output ok | polarity internal ok | output-internal gap | output-ok/internal-bad count |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for system, metrics in results.items():
        faithfulness = metrics["internal_faithfulness"]
        lines.append(
            f"| {system} | "
            f"{_format(faithfulness['entitled_output_gate_violation_rate'])} | "
            f"{faithfulness['entitled_output_gate_violations']}/{faithfulness['entitled_output_count']} | "
            f"{_format(faithfulness['polarity_flip_output_ok'])} | "
            f"{_format(faithfulness['polarity_flip_internal_ok'])} | "
            f"{_format(faithfulness['polarity_flip_output_internal_gap'])} | "
            f"{faithfulness['polarity_flip_output_ok_but_internal_bad']} |"
        )
    lines.extend(
        [
            "",
            "## ROUTER_KEY_CONTRAST",
            "",
            "| system | macro-F1 | predicate-pair | polarity-flip | paraphrase |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for system, metrics in results.items():
        checks = metrics["pairwise_checks"]
        lines.append(
            f"| {system} | {_format(metrics['final_macro_f1'])} | "
            f"{_format(checks['predicate_disentangled']['pass_rate'])} | "
            f"{_format(checks['polarity_flip_preserved_and_reversed']['pass_rate'])} | "
            f"{_format(checks['paraphrase_preserved']['pass_rate'])} |"
        )
    best_classifier = max(results, key=lambda name: results[name]["final_macro_f1"])
    best_predicate = max(
        results,
        key=lambda name: results[name]["pairwise_checks"]["predicate_disentangled"][
            "pass_rate"
        ],
    )
    best_flip = max(
        results,
        key=lambda name: results[name]["pairwise_checks"][
            "polarity_flip_preserved_and_reversed"
        ]["pass_rate"],
    )
    lines.extend(
        [
            "",
            "## INTERPRETATION",
            "",
            f"Best final-label macro-F1: **{best_classifier}**. "
            f"Best predicate-pair consistency: **{best_predicate}**. "
            f"Best polarity-flip consistency: **{best_flip}**.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "system", "metric", "value"))
        writer.writeheader()
        for system, metrics in results.items():
            for metric in ("final_accuracy", "final_macro_f1"):
                writer.writerow(
                    {"section": "classification", "system": system, "metric": metric, "value": metrics[metric]}
                )
            for label, value in metrics["per_label_f1"].items():
                writer.writerow(
                    {"section": "classification", "system": system, "metric": f"{label}_f1", "value": value}
                )
            writer.writerow(
                {
                    "section": "classification",
                    "system": system,
                    "metric": "prediction_distribution",
                    "value": json.dumps(metrics["prediction_distribution"], sort_keys=True),
                }
            )
            for metric, value in metrics["pairwise_checks"].items():
                writer.writerow(
                    {"section": "pairwise", "system": system, "metric": metric, "value": value["pass_rate"]}
                )
            for metric, value in metrics["internal_faithfulness"].items():
                writer.writerow(
                    {"section": "internal_faithfulness", "system": system, "metric": metric, "value": value}
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-preds", type=Path, required=True)
    parser.add_argument("--balanced-preds", type=Path, required=True)
    parser.add_argument("--strict-preds", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = evaluate_router_systems(
        load_prediction_file(args.classifier_preds),
        load_prediction_file(args.balanced_preds),
        load_prediction_file(args.strict_preds),
    )
    markdown = render_markdown(results)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    write_csv(args.output_csv, results)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
