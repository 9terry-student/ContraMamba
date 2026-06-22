"""Stage 9B stratified and paired analysis for ContraMamba-CAR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_router_ensemble import (  # noqa: E402
    ENTITLED_LABELS,
    auditor_passes,
    build_system_predictions,
    classification_metrics,
    load_prediction_file,
    merge_prediction_files,
    router_cost_metrics,
)


LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
PAIRWISE_INTERVENTIONS = {
    "paraphrase",
    "predicate_swap",
    "polarity_flip",
    "evidence_deletion",
    "evidence_truncation",
    "entity_swap",
    "event_swap",
}
STRATIFIED_METRICS = (
    "n_examples",
    "gold_SUPPORT_count",
    "gold_REFUTE_count",
    "gold_NOT_ENTITLED_count",
    "classifier_entitled_count",
    "routed_entitled_count",
    "downgraded_count",
    "downgrade_rate_among_classifier_entitled",
    "pre_router_candidate_gate_fail_count",
    "pre_router_candidate_gate_fail_rate",
    "classifier_SUPPORT_count",
    "routed_SUPPORT_count",
    "support_recall_pre_router",
    "support_recall_post_router",
    "support_recall_drop",
    "support_precision_pre_router",
    "support_precision_post_router",
    "support_precision_gain",
    "false_support_removed_count",
    "false_support_removed_rate_among_classifier_false_support",
    "classifier_REFUTE_count",
    "routed_REFUTE_count",
    "refute_recall_pre_router",
    "refute_recall_post_router",
    "refute_recall_drop",
    "refute_precision_pre_router",
    "refute_precision_post_router",
    "refute_precision_gain",
    "false_refute_removed_count",
    "false_refute_removed_rate_among_classifier_false_refute",
    "classifier_accuracy",
    "routed_accuracy",
    "accuracy_delta",
    "classifier_macro_f1",
    "routed_macro_f1",
    "macro_f1_delta",
    "pairwise_applicable_count",
    "raw_classifier_pairwise_success_count",
    "raw_classifier_pairwise_success_rate",
    "router_pairwise_success_count",
    "router_pairwise_success_rate",
    "self_routed_classifier_pairwise_success_count",
    "self_routed_classifier_pairwise_success_rate",
    "self_routed_balanced_pairwise_success_count",
    "self_routed_balanced_pairwise_success_rate",
)
BOOTSTRAP_METRICS = (
    "accuracy_delta",
    "macro_f1_delta",
    "support_precision_gain",
    "support_recall_drop",
    "downgrade_rate",
    "pre_router_candidate_gate_fail_rate",
)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_analysis_bundle(
    data_path: Path, classifier_path: Path, balanced_path: Path, strict_path: Path
) -> list[dict[str, dict[str, Any]]]:
    metadata = {row["id"]: row for row in load_jsonl(data_path)}
    merged = merge_prediction_files(
        load_prediction_file(classifier_path),
        load_prediction_file(balanced_path),
        load_prediction_file(strict_path),
    )
    normalized = []
    for item in merged:
        example_id = str(item["classifier"]["id"])
        if example_id not in metadata:
            raise ValueError(f"prediction id {example_id!r} is absent from {data_path}")
        source = metadata[example_id]
        normalized_item = {}
        for model_name, row in item.items():
            normalized_item[model_name] = {
                **row,
                "pair_id": source["pair_id"],
                "intervention_type": source["intervention_type"],
                "gold_final_label": source["final_label"],
            }
        normalized.append(normalized_item)
    return normalized


def _self_route(
    merged: Sequence[Mapping[str, Mapping[str, Any]]], source: str, threshold: float
) -> dict[str, str]:
    labels = {}
    for item in merged:
        row = item[source]
        label = str(row["pred_final_label"])
        if label in ENTITLED_LABELS and not auditor_passes(row, threshold):
            label = "NOT_ENTITLED"
        labels[str(row["id"])] = label
    return labels


def system_predictions(
    merged: Sequence[Mapping[str, Mapping[str, Any]]], threshold: float
) -> dict[str, dict[str, str]]:
    router = build_system_predictions(merged, threshold)[
        "conservative_balanced_router"
    ]
    return {
        "raw_classifier_only": {
            str(item["classifier"]["id"]): str(
                item["classifier"]["pred_final_label"]
            )
            for item in merged
        },
        "conservative_balanced_router": router,
        "self_routed_classifier": _self_route(merged, "classifier", threshold),
        "self_routed_balanced": _self_route(merged, "balanced", threshold),
    }


def _label_statistics(
    records: Sequence[Mapping[str, Any]], labels: Mapping[str, str], label: str
) -> tuple[int, int, int, float, float]:
    gold_count = sum(row["gold_final_label"] == label for row in records)
    predicted_count = sum(labels[str(row["id"])] == label for row in records)
    true_positive = sum(
        row["gold_final_label"] == label and labels[str(row["id"])] == label
        for row in records
    )
    return (
        gold_count,
        predicted_count,
        true_positive,
        safe_div(true_positive, gold_count),
        safe_div(true_positive, predicted_count),
    )


def _pairwise_success(intervention: str, none_label: str, changed_label: str) -> bool:
    if intervention == "paraphrase":
        return none_label == changed_label
    if intervention == "predicate_swap":
        return none_label in ENTITLED_LABELS and changed_label == "NOT_ENTITLED"
    if intervention == "polarity_flip":
        return (
            none_label in ENTITLED_LABELS
            and changed_label in ENTITLED_LABELS
            and none_label != changed_label
        )
    return changed_label == "NOT_ENTITLED"


def pairwise_summaries(
    records: Sequence[Mapping[str, Any]], systems: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, tuple[int, int, float]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        grouped[str(row["pair_id"])][str(row["intervention_type"])] = row
    result: dict[str, dict[str, tuple[int, int, float]]] = {}
    for intervention in PAIRWISE_INTERVENTIONS:
        result[intervention] = {}
        for system, labels in systems.items():
            successes = []
            for variants in grouped.values():
                if "none" not in variants or intervention not in variants:
                    continue
                none_id = str(variants["none"]["id"])
                changed_id = str(variants[intervention]["id"])
                successes.append(
                    _pairwise_success(
                        intervention, labels[none_id], labels[changed_id]
                    )
                )
            passed = sum(successes)
            result[intervention][system] = (
                passed,
                len(successes),
                safe_div(passed, len(successes)),
            )
    return result


def stratified_metrics(
    records: Sequence[Mapping[str, Any]],
    classifier_labels: Mapping[str, str],
    routed_labels: Mapping[str, str],
    auditor_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    threshold: float,
    pairwise: Mapping[str, Mapping[str, tuple[int, int, float]]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["intervention_type"])].append(row)
    output = []
    for intervention, subset in sorted(grouped.items()):
        costs = router_cost_metrics(
            subset, classifier_labels, routed_labels, auditor_rows, threshold
        )
        c_support = _label_statistics(subset, classifier_labels, "SUPPORT")
        r_support = _label_statistics(subset, routed_labels, "SUPPORT")
        c_refute = _label_statistics(subset, classifier_labels, "REFUTE")
        r_refute = _label_statistics(subset, routed_labels, "REFUTE")
        classifier_false_support = c_support[1] - c_support[2]
        classifier_false_refute = c_refute[1] - c_refute[2]
        c_metrics = classification_metrics(subset, classifier_labels)
        r_metrics = classification_metrics(subset, routed_labels)
        pair = pairwise.get(intervention, {}) if pairwise else {}

        def pair_values(system: str) -> tuple[int, int, float]:
            return pair.get(system, (0, 0, 0.0))

        raw_pair = pair_values("raw_classifier_only")
        router_pair = pair_values("conservative_balanced_router")
        self_c_pair = pair_values("self_routed_classifier")
        self_b_pair = pair_values("self_routed_balanced")
        values = {
            "n_examples": len(subset),
            **{
                f"gold_{label}_count": sum(
                    row["gold_final_label"] == label for row in subset
                )
                for label in LABELS
            },
            **{key: costs[key] for key in (
                "classifier_entitled_count", "routed_entitled_count",
                "downgraded_count", "downgrade_rate_among_classifier_entitled",
                "pre_router_candidate_gate_fail_count",
                "pre_router_candidate_gate_fail_rate",
                "support_recall_pre_router", "support_recall_post_router",
                "support_recall_drop", "support_precision_pre_router",
                "support_precision_post_router", "support_precision_gain",
                "false_support_removed_count", "false_refute_removed_count",
            )},
            "classifier_SUPPORT_count": c_support[1],
            "routed_SUPPORT_count": r_support[1],
            "false_support_removed_rate_among_classifier_false_support": safe_div(
                costs["false_support_removed_count"], classifier_false_support
            ),
            "classifier_REFUTE_count": c_refute[1],
            "routed_REFUTE_count": r_refute[1],
            "refute_recall_pre_router": c_refute[3],
            "refute_recall_post_router": r_refute[3],
            "refute_recall_drop": c_refute[3] - r_refute[3],
            "refute_precision_pre_router": c_refute[4],
            "refute_precision_post_router": r_refute[4],
            "refute_precision_gain": r_refute[4] - c_refute[4],
            "false_refute_removed_rate_among_classifier_false_refute": safe_div(
                costs["false_refute_removed_count"], classifier_false_refute
            ),
            "classifier_accuracy": c_metrics["final_accuracy"],
            "routed_accuracy": r_metrics["final_accuracy"],
            "accuracy_delta": r_metrics["final_accuracy"] - c_metrics["final_accuracy"],
            "classifier_macro_f1": c_metrics["final_macro_f1"],
            "routed_macro_f1": r_metrics["final_macro_f1"],
            "macro_f1_delta": r_metrics["final_macro_f1"] - c_metrics["final_macro_f1"],
            "pairwise_applicable_count": router_pair[1],
            "raw_classifier_pairwise_success_count": raw_pair[0],
            "raw_classifier_pairwise_success_rate": raw_pair[2],
            "router_pairwise_success_count": router_pair[0],
            "router_pairwise_success_rate": router_pair[2],
            "self_routed_classifier_pairwise_success_count": self_c_pair[0],
            "self_routed_classifier_pairwise_success_rate": self_c_pair[2],
            "self_routed_balanced_pairwise_success_count": self_b_pair[0],
            "self_routed_balanced_pairwise_success_rate": self_b_pair[2],
        }
        output.append({"intervention_type": intervention, **values})
    return output


def mcnemar_exact(
    records: Sequence[Mapping[str, Any]],
    classifier_labels: Mapping[str, str],
    routed_labels: Mapping[str, str],
) -> dict[str, float | int]:
    both_correct = classifier_only = router_only = both_wrong = 0
    for row in records:
        example_id = str(row["id"])
        gold = row["gold_final_label"]
        c_ok = classifier_labels[example_id] == gold
        r_ok = routed_labels[example_id] == gold
        both_correct += int(c_ok and r_ok)
        classifier_only += int(c_ok and not r_ok)
        router_only += int(not c_ok and r_ok)
        both_wrong += int(not c_ok and not r_ok)
    discordant = classifier_only + router_only
    statistic = (
        (max(0, abs(classifier_only - router_only) - 1) ** 2) / discordant
        if discordant
        else 0.0
    )
    if discordant:
        tail = sum(
            math.comb(discordant, k) for k in range(min(classifier_only, router_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "n": len(records),
        "both_correct": both_correct,
        "classifier_only_correct": classifier_only,
        "router_only_correct": router_only,
        "both_wrong": both_wrong,
        "mcnemar_statistic": statistic,
        "p_value": p_value,
        "accuracy_delta": safe_div(router_only - classifier_only, len(records)),
    }


def _macro_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    scores = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        scores.append(safe_div(2 * tp, 2 * tp + fp + fn))
    return sum(scores) / len(scores)


def _bootstrap_values(
    records: Sequence[Mapping[str, Any]], classifier: Mapping[str, str],
    routed: Mapping[str, str], auditors: Mapping[str, Sequence[Mapping[str, Any]]],
    threshold: float,
) -> dict[str, float]:
    gold = [str(row["gold_final_label"]) for row in records]
    c = [classifier[str(row["id"])] for row in records]
    r = [routed[str(row["id"])] for row in records]
    accuracy_delta = safe_div(sum(g == p for g, p in zip(gold, r)), len(gold)) - safe_div(
        sum(g == p for g, p in zip(gold, c)), len(gold)
    )
    support_gold = sum(g == "SUPPORT" for g in gold)
    c_support_true = sum(g == "SUPPORT" and p == "SUPPORT" for g, p in zip(gold, c))
    r_support_true = sum(g == "SUPPORT" and p == "SUPPORT" for g, p in zip(gold, r))
    c_support = sum(p == "SUPPORT" for p in c)
    r_support = sum(p == "SUPPORT" for p in r)
    entitled = [index for index, label in enumerate(c) if label in ENTITLED_LABELS]
    downgraded = sum(c[index] in ENTITLED_LABELS and r[index] == "NOT_ENTITLED" for index in range(len(c)))
    gate_fails = sum(
        not all(auditor_passes(row, threshold) for row in auditors[str(records[index]["id"])])
        for index in entitled
    )
    return {
        "accuracy_delta": accuracy_delta,
        "macro_f1_delta": _macro_f1(gold, r) - _macro_f1(gold, c),
        "support_precision_gain": safe_div(r_support_true, r_support) - safe_div(c_support_true, c_support),
        "support_recall_drop": safe_div(c_support_true, support_gold) - safe_div(r_support_true, support_gold),
        "downgrade_rate": safe_div(downgraded, len(entitled)),
        "pre_router_candidate_gate_fail_rate": safe_div(gate_fails, len(entitled)),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    records: Sequence[Mapping[str, Any]], classifier: Mapping[str, str],
    routed: Mapping[str, str], auditors: Mapping[str, Sequence[Mapping[str, Any]]],
    threshold: float, n_bootstrap: int = 1000, seed: int = 17,
) -> list[dict[str, Any]]:
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_pair[str(row["pair_id"])].append(row)
    pair_ids = sorted(by_pair)
    estimate = _bootstrap_values(records, classifier, routed, auditors, threshold)
    samples = {metric: [] for metric in BOOTSTRAP_METRICS}
    generator = random.Random(seed)
    for _ in range(n_bootstrap):
        sampled = [generator.choice(pair_ids) for _ in pair_ids]
        bootstrap_records = [row for pair_id in sampled for row in by_pair[pair_id]]
        values = _bootstrap_values(
            bootstrap_records, classifier, routed, auditors, threshold
        )
        for metric in BOOTSTRAP_METRICS:
            samples[metric].append(values[metric])
    return [
        {
            "metric": metric,
            "estimate": estimate[metric],
            "ci_low": _quantile(samples[metric], 0.025),
            "ci_high": _quantile(samples[metric], 0.975),
            "n_bootstrap": n_bootstrap,
            "resampling_unit": "pair_id",
        }
        for metric in BOOTSTRAP_METRICS
    ]


def write_stratified_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("seed", "threshold", "system", "intervention_type", *STRATIFIED_METRICS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def render_stratified_markdown(rows: Sequence[Mapping[str, Any]], seed: int) -> str:
    focus = (
        "n_examples", "downgraded_count", "downgrade_rate_among_classifier_entitled",
        "pre_router_candidate_gate_fail_rate", "support_recall_drop",
        "support_precision_gain", "false_support_removed_count", "false_refute_removed_count",
        "accuracy_delta", "macro_f1_delta", "router_pairwise_success_rate",
    )
    lines = [f"# Stage 9B Stratified Analysis: Seed {seed}", ""]
    for threshold in sorted({float(row["threshold"]) for row in rows}):
        lines.extend([f"## THRESHOLD {threshold:.1f}", "",
                      "| intervention_type | " + " | ".join(focus) + " |",
                      "|---|" + "---:|" * len(focus)])
        for row in rows:
            if float(row["threshold"]) != threshold:
                continue
            lines.append(f"| {row['intervention_type']} | " + " | ".join(
                f"{float(row[metric]):.3f}" for metric in focus) + " |")
        lines.append("")
    return "\n".join(lines)


def aggregate_stratified(paths: Sequence[Path]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, str, str], list[float]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for metric in STRATIFIED_METRICS:
                    grouped[(float(row["threshold"]), row["system"], row["intervention_type"], metric)].append(float(row[metric]))
    output = []
    for (threshold, system, intervention, metric), values in grouped.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append({"threshold": threshold, "system": system, "intervention_type": intervention,
                       "metric": metric, "mean": mean, "std": std, "n": len(values),
                       "formatted": f"{mean:.3f} +/- {std:.3f}"})
    return sorted(output, key=lambda row: (row["threshold"], row["intervention_type"], row["metric"]))


def write_aggregate(paths: Sequence[Path], csv_path: Path, md_path: Path) -> None:
    rows = aggregate_stratified(paths)
    csv_path.parent.mkdir(parents=True, exist_ok=True); md_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("threshold", "system", "intervention_type", "metric", "mean", "std", "n", "formatted")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    focus = ("downgrade_rate_among_classifier_entitled", "pre_router_candidate_gate_fail_rate",
             "support_recall_drop", "support_precision_gain", "accuracy_delta", "router_pairwise_success_rate")
    lookup = {(row["threshold"], row["intervention_type"], row["metric"]): row["formatted"] for row in rows}
    lines = ["# Stage 9B Stratified Aggregate", ""]
    for threshold in sorted({row["threshold"] for row in rows}):
        lines.extend([f"## THRESHOLD {threshold:.1f}", "",
                      "| intervention_type | " + " | ".join(focus) + " |",
                      "|---|" + "---:|" * len(focus)])
        interventions = sorted({row["intervention_type"] for row in rows if row["threshold"] == threshold})
        for intervention in interventions:
            lines.append(f"| {intervention} | " + " | ".join(
                lookup[(threshold, intervention, metric)] for metric in focus) + " |")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def paired_rows_for_bundle(
    merged: Sequence[Mapping[str, Mapping[str, Any]]], seed: int,
    threshold: float, n_bootstrap: int,
) -> list[dict[str, Any]]:
    records = [item["classifier"] for item in merged]
    systems = system_predictions(merged, threshold)
    classifier = systems["raw_classifier_only"]
    router = systems["conservative_balanced_router"]
    auditors = {str(item["classifier"]["id"]): (item["balanced"],) for item in merged}
    output = []
    scopes = [("global", "all", records)]
    scopes.extend(("intervention", intervention, [row for row in records if row["intervention_type"] == intervention])
                  for intervention in sorted({row["intervention_type"] for row in records}))
    for scope, intervention, subset in scopes:
        test = mcnemar_exact(subset, classifier, router)
        output.append({"analysis_type": "mcnemar", "seed": seed, "scope": scope,
                       "intervention_type": intervention, "metric": "final_accuracy", **test,
                       "estimate": test["accuracy_delta"], "ci_low": 0.0, "ci_high": 0.0,
                       "n_bootstrap": 0, "resampling_unit": "example"})
    for result in paired_bootstrap(records, classifier, router, auditors, threshold,
                                   n_bootstrap=n_bootstrap, seed=seed + 9000):
        output.append({"analysis_type": "bootstrap", "seed": seed, "scope": "global",
                       "intervention_type": "all", "n": len(records), "both_correct": 0,
                       "classifier_only_correct": 0, "router_only_correct": 0, "both_wrong": 0,
                       "mcnemar_statistic": 0.0, "p_value": 0.0, "accuracy_delta": 0.0, **result})
    return output


def write_paired(rows: Sequence[Mapping[str, Any]], csv_path: Path, md_path: Path) -> None:
    fields = ("analysis_type", "seed", "scope", "intervention_type", "metric", "n",
              "both_correct", "classifier_only_correct", "router_only_correct", "both_wrong",
              "mcnemar_statistic", "p_value", "accuracy_delta", "estimate", "ci_low", "ci_high",
              "n_bootstrap", "resampling_unit")
    csv_path.parent.mkdir(parents=True, exist_ok=True); md_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    lines = ["# Stage 9B Paired Tests", "", "## McNemar tests", "",
             "| seed | scope | intervention | n | classifier only correct | router only correct | statistic | p_value | accuracy_delta |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["analysis_type"] == "mcnemar":
            lines.append(f"| {row['seed']} | {row['scope']} | {row['intervention_type']} | {row['n']} | {row['classifier_only_correct']} | {row['router_only_correct']} | {float(row['mcnemar_statistic']):.4f} | {float(row['p_value']):.4f} | {float(row['accuracy_delta']):.4f} |")
    lines.extend(["", "## Pair-ID bootstrap", "",
                  "| seed | metric | estimate | 95% CI | samples | unit |",
                  "|---:|---|---:|---:|---:|---|"])
    for row in rows:
        if row["analysis_type"] == "bootstrap":
            lines.append(f"| {row['seed']} | {row['metric']} | {float(row['estimate']):.4f} | [{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | {row['n_bootstrap']} | {row['resampling_unit']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--classifier-preds", type=Path)
    parser.add_argument("--balanced-preds", type=Path)
    parser.add_argument("--strict-preds", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.3, 0.4, 0.5, 0.6, 0.7))
    parser.add_argument("--output-stratified-csv", type=Path)
    parser.add_argument("--output-stratified-md", type=Path)
    parser.add_argument("--aggregate-input", type=Path, action="append")
    parser.add_argument("--output-aggregate-csv", type=Path)
    parser.add_argument("--output-aggregate-md", type=Path)
    parser.add_argument("--paired-inputs", nargs="+", help="classifier,balanced,strict bundles")
    parser.add_argument("--output-paired-csv", type=Path)
    parser.add_argument("--output-paired-md", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.aggregate_input:
        if not args.output_aggregate_csv or not args.output_aggregate_md:
            raise ValueError("aggregate mode requires both aggregate outputs")
        write_aggregate(args.aggregate_input, args.output_aggregate_csv, args.output_aggregate_md)
        return 0
    if args.paired_inputs:
        if not args.data or not args.output_paired_csv or not args.output_paired_md:
            raise ValueError("paired mode requires --data and both paired outputs")
        rows = []
        for seed, bundle in enumerate(args.paired_inputs, start=1):
            paths = bundle.split(",")
            if len(paths) != 3:
                raise ValueError("each --paired-inputs value must be classifier,balanced,strict")
            merged = load_analysis_bundle(args.data, *(Path(path) for path in paths))
            rows.extend(paired_rows_for_bundle(merged, seed, 0.5, args.n_bootstrap))
        write_paired(rows, args.output_paired_csv, args.output_paired_md)
        return 0
    required = (args.data, args.classifier_preds, args.balanced_preds, args.strict_preds,
                args.seed, args.output_stratified_csv, args.output_stratified_md)
    if any(value is None for value in required):
        raise ValueError("per-seed mode requires data, three predictions, seed, and two outputs")
    merged = load_analysis_bundle(args.data, args.classifier_preds, args.balanced_preds, args.strict_preds)
    records = [item["classifier"] for item in merged]
    auditors = {str(item["classifier"]["id"]): (item["balanced"],) for item in merged}
    all_rows = []
    for threshold in args.thresholds:
        systems = system_predictions(merged, threshold)
        pairwise = pairwise_summaries(records, systems)
        rows = stratified_metrics(records, systems["raw_classifier_only"],
                                  systems["conservative_balanced_router"], auditors,
                                  threshold, pairwise)
        all_rows.extend({"seed": args.seed, "threshold": threshold,
                         "system": "conservative_balanced_router", **row} for row in rows)
    write_stratified_csv(args.output_stratified_csv, all_rows)
    args.output_stratified_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_stratified_md.write_text(render_stratified_markdown(all_rows, args.seed), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
