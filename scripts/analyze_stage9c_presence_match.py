"""Stage 9C presence-vs-match and gate-signal diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_router_ensemble import ENTITLED_LABELS, load_prediction_file  # noqa: E402


LABELS = ("SUPPORT", "REFUTE", "NOT_ENTITLED")
GATE_FIELDS = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
)
ALL_SIGNAL_FIELDS = (*GATE_FIELDS, "polarity_margin")
FAILURE_SCORES = (
    "frame_fail_score",
    "predicate_fail_score",
    "sufficiency_fail_score",
    "entitlement_fail_score",
    "polarity_weak_score",
)
TARGETS = ("classifier_error", "false_entitled", "false_support", "false_refute")
INTERVENTION_GROUPS = {
    "evidence_deletion": "presence_removed",
    "evidence_truncation": "presence_removed",
    "irrelevant_evidence": "presence_removed",
    "time_swap": "match_perturbed",
    "entity_swap": "match_perturbed",
    "event_swap": "match_perturbed",
    "location_swap": "match_perturbed",
    "role_swap": "match_perturbed",
    "title_name_swap": "match_perturbed",
    "predicate_swap": "match_perturbed",
    "none": "controls",
    "paraphrase": "controls",
    "polarity_flip": "controls",
}
OPPORTUNITY_NUMERIC = (
    "n",
    "gold_SUPPORT",
    "gold_REFUTE",
    "gold_NOT_ENTITLED",
    "pred_SUPPORT",
    "pred_REFUTE",
    "pred_NOT_ENTITLED",
    "classifier_entitled",
    "classifier_error",
    "false_entitled",
    "false_support",
    "false_refute",
    "opportunity_rate",
    "false_entitled_rate",
    "not_entitled_recall",
    "entitled_accuracy",
    "mean_frame_prob",
    "mean_predicate_coverage_prob",
    "mean_sufficiency_prob",
    "mean_entitlement_prob",
    "mean_polarity_margin",
    "mean_frame_fail_score",
    "mean_predicate_fail_score",
    "mean_sufficiency_fail_score",
    "mean_entitlement_fail_score",
    "mean_polarity_weak_score",
)
STATE_FIELDS = tuple(f"{field}_state" for field in GATE_FIELDS)
SIGNAL_NUMERIC = (
    "n",
    "positives",
    "negatives",
    "mean_score_positive",
    "mean_score_negative",
    "mean_pass_probability_positive",
    "raw_auc",
    "expected_direction_auc",
    "inverted_auc",
    "auc_defined",
)
CORRELATION_NAMES = tuple(
    f"{method}_{left}__{right}"
    for method in ("pearson", "spearman")
    for index, left in enumerate(ALL_SIGNAL_FIELDS)
    for right in ALL_SIGNAL_FIELDS[index + 1 :]
)
CORRELATION_NUMERIC = (
    "n",
    "pearson_mean_abs_off_diagonal",
    "pearson_max_abs_off_diagonal",
    "spearman_mean_abs_off_diagonal",
    "spearman_max_abs_off_diagonal",
    *CORRELATION_NAMES,
)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def classify_gate_state(
    gold_not_entitled: int, classifier_entitled: int, mean_pass_probability: float
) -> str:
    if gold_not_entitled == 0 or classifier_entitled == 0:
        return "no_opportunity"
    if mean_pass_probability < 0.2:
        return "correct_rejection"
    if 0.4 <= mean_pass_probability <= 0.6:
        return "uncertain_no_signal"
    if mean_pass_probability > 0.8:
        return "confidently_inverted"
    return "mixed"


def _failure_scores(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "frame_fail_score": 1.0 - float(row["frame_prob"]),
        "predicate_fail_score": 1.0 - float(row["predicate_coverage_prob"]),
        "sufficiency_fail_score": 1.0 - float(row["sufficiency_prob"]),
        "entitlement_fail_score": 1.0 - float(row["entitlement_prob"]),
        "polarity_weak_score": -float(row["polarity_margin"]),
    }


def _targets(row: Mapping[str, Any]) -> dict[str, bool]:
    gold = str(row["gold_final_label"])
    predicted = str(row["pred_final_label"])
    return {
        "classifier_error": predicted != gold,
        "false_entitled": predicted in ENTITLED_LABELS and gold == "NOT_ENTITLED",
        "false_support": predicted == "SUPPORT" and gold != "SUPPORT",
        "false_refute": predicted == "REFUTE" and gold != "REFUTE",
    }


def opportunity_rows(predictions: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["intervention_type"])].append(row)
    output = []
    for intervention, rows in sorted(grouped.items()):
        gold = Counter(str(row["gold_final_label"]) for row in rows)
        predicted = Counter(str(row["pred_final_label"]) for row in rows)
        classifier_entitled = sum(row["pred_final_label"] in ENTITLED_LABELS for row in rows)
        target_values = [_targets(row) for row in rows]
        gates = {field: mean([float(row[field]) for row in rows]) for field in ALL_SIGNAL_FIELDS}
        failure_means = {
            score: mean([_failure_scores(row)[score] for row in rows])
            for score in FAILURE_SCORES
        }
        entitled_gold = gold["SUPPORT"] + gold["REFUTE"]
        entitled_correct = sum(
            row["gold_final_label"] in ENTITLED_LABELS
            and row["pred_final_label"] == row["gold_final_label"]
            for row in rows
        )
        values = {
            "n": len(rows),
            **{f"gold_{label}": gold[label] for label in LABELS},
            **{f"pred_{label}": predicted[label] for label in LABELS},
            "classifier_entitled": classifier_entitled,
            **{target: sum(item[target] for item in target_values) for target in TARGETS},
            "opportunity_rate": safe_div(classifier_entitled, len(rows)),
            "false_entitled_rate": safe_div(
                sum(item["false_entitled"] for item in target_values), len(rows)
            ),
            "not_entitled_recall": safe_div(predicted["NOT_ENTITLED"], gold["NOT_ENTITLED"]),
            "entitled_accuracy": safe_div(entitled_correct, entitled_gold),
            **{f"mean_{field}": value for field, value in gates.items()},
            **{f"mean_{score}": value for score, value in failure_means.items()},
        }
        states = {
            f"{field}_state": classify_gate_state(
                gold["NOT_ENTITLED"], classifier_entitled, gates[field]
            )
            for field in GATE_FIELDS
        }
        output.append(
            {
                "row_type": "opportunity",
                "seed": seed,
                "intervention_type": intervention,
                "intervention_group": INTERVENTION_GROUPS.get(intervention, "other"),
                "target": "not_applicable",
                "score": "not_applicable",
                "direction_interpretation": "not_applicable",
                **values,
                **states,
                **{field: 0.0 for field in SIGNAL_NUMERIC if field not in values},
            }
        )
    return output


def raw_auc(scores: Sequence[float], targets: Sequence[bool]) -> tuple[float, int]:
    positives = [score for score, target in zip(scores, targets) if target]
    negatives = [score for score, target in zip(scores, targets) if not target]
    if not positives or not negatives:
        return 0.5, 0
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives)), 1


def interpret_auc(raw_value: float, mean_pass_positive: float, auc_defined: int) -> str:
    if not auc_defined:
        return "uninformative"
    if raw_value > 0.6:
        return "expected_positive"
    if 0.4 <= raw_value <= 0.6:
        return "uninformative"
    if mean_pass_positive > 0.8:
        return "confidently_inverted"
    return "inverted_low_confidence"


def signal_rows(predictions: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["intervention_type"])].append(row)
    output = []
    score_to_pass = {
        "frame_fail_score": "frame_prob",
        "predicate_fail_score": "predicate_coverage_prob",
        "sufficiency_fail_score": "sufficiency_prob",
        "entitlement_fail_score": "entitlement_prob",
    }
    for intervention, rows in sorted(grouped.items()):
        for target in TARGETS:
            target_values = [_targets(row)[target] for row in rows]
            for score_name in FAILURE_SCORES:
                scores = [_failure_scores(row)[score_name] for row in rows]
                positive_scores = [score for score, value in zip(scores, target_values) if value]
                negative_scores = [score for score, value in zip(scores, target_values) if not value]
                auc, defined = raw_auc(scores, target_values)
                pass_field = score_to_pass.get(score_name)
                positive_pass = (
                    [float(row[pass_field]) for row, value in zip(rows, target_values) if value]
                    if pass_field
                    else []
                )
                mean_pass_positive = mean(positive_pass)
                values = {
                    "n": len(rows),
                    "positives": sum(target_values),
                    "negatives": len(rows) - sum(target_values),
                    "mean_score_positive": mean(positive_scores),
                    "mean_score_negative": mean(negative_scores),
                    "mean_pass_probability_positive": mean_pass_positive,
                    "raw_auc": auc,
                    "expected_direction_auc": auc,
                    "inverted_auc": 1.0 - auc,
                    "auc_defined": defined,
                }
                output.append(
                    {
                        "row_type": "signal",
                        "seed": seed,
                        "intervention_type": intervention,
                        "intervention_group": INTERVENTION_GROUPS.get(intervention, "other"),
                        "target": target,
                        "score": score_name,
                        "direction_interpretation": interpret_auc(
                            auc, mean_pass_positive, defined
                        ),
                        **{field: 0.0 for field in OPPORTUNITY_NUMERIC},
                        **{field: "not_applicable" for field in STATE_FIELDS},
                        **values,
                    }
                )
    return output


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[indexed[position][0]] = average_rank
        index = end
    return ranks


def correlation_rows(predictions: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    grouped["__GLOBAL__"] = list(predictions)
    for row in predictions:
        grouped[str(row["intervention_type"])].append(row)
    output = []
    for intervention, rows in sorted(grouped.items()):
        values = {field: [float(row[field]) for row in rows] for field in ALL_SIGNAL_FIELDS}
        pearson_values, spearman_values = [], []
        pairwise = {}
        for index, left in enumerate(ALL_SIGNAL_FIELDS):
            for right in ALL_SIGNAL_FIELDS[index + 1 :]:
                pearson = _pearson(values[left], values[right])
                spearman = _pearson(_ranks(values[left]), _ranks(values[right]))
                pairwise[f"pearson_{left}__{right}"] = pearson
                pairwise[f"spearman_{left}__{right}"] = spearman
                pearson_values.append(abs(pearson))
                spearman_values.append(abs(spearman))
        output.append(
            {
                "seed": seed,
                "intervention_type": intervention,
                "n": len(rows),
                "pearson_mean_abs_off_diagonal": mean(pearson_values),
                "pearson_max_abs_off_diagonal": max(pearson_values, default=0.0),
                "spearman_mean_abs_off_diagonal": mean(spearman_values),
                "spearman_max_abs_off_diagonal": max(spearman_values, default=0.0),
                **pairwise,
            }
        )
    return output


PRESENCE_FIELDS = (
    "row_type",
    "seed",
    "intervention_type",
    "intervention_group",
    "target",
    "score",
    "direction_interpretation",
    *OPPORTUNITY_NUMERIC,
    *STATE_FIELDS,
    *tuple(field for field in SIGNAL_NUMERIC if field not in OPPORTUNITY_NUMERIC),
)
CORRELATION_FIELDS = ("seed", "intervention_type", *CORRELATION_NUMERIC)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _aggregate_numeric(values: Sequence[float]) -> tuple[float, float, int, str]:
    value_mean = mean(values)
    value_std = statistics.stdev(values) if len(values) > 1 else 0.0
    return value_mean, value_std, len(values), f"{value_mean:.4f} +/- {value_std:.4f}"


def aggregate_presence(paths: Sequence[Path]) -> list[dict[str, Any]]:
    opportunity: dict[tuple[str, str], list[float]] = defaultdict(list)
    states: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    signal: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["row_type"] == "opportunity":
                    for metric in OPPORTUNITY_NUMERIC:
                        opportunity[(row["intervention_type"], metric)].append(float(row[metric]))
                    for state in STATE_FIELDS:
                        states[(row["intervention_type"], state)][row[state]] += 1
                else:
                    for metric in SIGNAL_NUMERIC:
                        signal[(row["intervention_type"], row["target"], row["score"], metric)].append(float(row[metric]))
    output = []
    for (intervention, metric), values in opportunity.items():
        value_mean, std, n, formatted = _aggregate_numeric(values)
        output.append({"row_type": "opportunity", "intervention_type": intervention,
                       "target": "not_applicable", "score": "not_applicable", "metric": metric,
                       "mean": value_mean, "std": std, "n": n, "formatted": formatted,
                       "state_summary": "not_applicable"})
    for (intervention, metric), counts in states.items():
        output.append({"row_type": "state", "intervention_type": intervention,
                       "target": "not_applicable", "score": "not_applicable", "metric": metric,
                       "mean": 0.0, "std": 0.0, "n": sum(counts.values()), "formatted": "not_applicable",
                       "state_summary": ";".join(f"{key}:{counts[key]}" for key in sorted(counts))})
    for (intervention, target, score, metric), values in signal.items():
        value_mean, std, n, formatted = _aggregate_numeric(values)
        output.append({"row_type": "signal", "intervention_type": intervention,
                       "target": target, "score": score, "metric": metric,
                       "mean": value_mean, "std": std, "n": n, "formatted": formatted,
                       "state_summary": "not_applicable"})
    return sorted(output, key=lambda row: (row["row_type"], row["intervention_type"], row["target"], row["score"], row["metric"]))


def aggregate_correlations(paths: Sequence[Path]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for metric in CORRELATION_NUMERIC:
                    grouped[(row["intervention_type"], metric)].append(float(row[metric]))
    output = []
    for (intervention, metric), values in grouped.items():
        value_mean, std, n, formatted = _aggregate_numeric(values)
        output.append({"intervention_type": intervention, "metric": metric,
                       "mean": value_mean, "std": std, "n": n, "formatted": formatted})
    return sorted(output, key=lambda row: (row["intervention_type"], row["metric"]))


def _lookup_presence(rows: Sequence[Mapping[str, Any]], intervention: str, metric: str) -> str:
    return next((str(row["formatted"]) for row in rows if row["row_type"] == "opportunity"
                 and row["intervention_type"] == intervention and row["metric"] == metric), "--")


def _lookup_state(rows: Sequence[Mapping[str, Any]], intervention: str, metric: str) -> str:
    return next((str(row["state_summary"]) for row in rows if row["row_type"] == "state"
                 and row["intervention_type"] == intervention and row["metric"] == metric), "--")


def _lookup_correlation(rows: Sequence[Mapping[str, Any]], intervention: str, metric: str) -> str:
    return next((str(row["formatted"]) for row in rows if row["intervention_type"] == intervention
                 and row["metric"] == metric), "--")


def _lookup_signal(
    rows: Sequence[Mapping[str, Any]], intervention: str, target: str, score: str, metric: str
) -> str:
    return next(
        (str(row["formatted"]) for row in rows if row["row_type"] == "signal"
         and row["intervention_type"] == intervention and row["target"] == target
         and row["score"] == score and row["metric"] == metric),
        "--",
    )


def render_markdown(presence: Sequence[Mapping[str, Any]], correlations: Sequence[Mapping[str, Any]]) -> str:
    gate_metrics = tuple(f"mean_{field}" for field in GATE_FIELDS)
    lines = [
        "# Stage 9C Presence-vs-Match Gate Diagnostics",
        "",
        "## Why Stage 9C was needed",
        "",
        "Stage 9B thresholded downgrades cannot distinguish no-opportunity cases, no-signal gates, threshold artifacts, and confidently inverted gates. Stage 9C analyzes gate probabilities before routing and preserves the expected score direction.",
        "",
        "## Main mechanism question: evidence presence or claim-evidence match?",
        "",
        "The central contrast is whether gate probabilities fall only when evidence is absent or truncated, or also when evidence remains present but mismatches the claim.",
        "",
        "## Evidence-presence diagnostic",
        "",
        "| intervention | gold NOT_ENTITLED | pred NOT_ENTITLED | classifier error | mean sufficiency | mean entitlement | sufficiency state | entitlement state |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for intervention in ("evidence_deletion", "evidence_truncation", "time_swap"):
        lines.append(
            f"| {intervention} | {_lookup_presence(presence, intervention, 'gold_NOT_ENTITLED')} | "
            f"{_lookup_presence(presence, intervention, 'pred_NOT_ENTITLED')} | "
            f"{_lookup_presence(presence, intervention, 'classifier_error')} | "
            f"{_lookup_presence(presence, intervention, 'mean_sufficiency_prob')} | "
            f"{_lookup_presence(presence, intervention, 'mean_entitlement_prob')} | "
            f"{_lookup_state(presence, intervention, 'sufficiency_prob_state')} | "
            f"{_lookup_state(presence, intervention, 'entitlement_prob_state')} |"
        )
    lines.extend([
        "",
        "Low sufficiency/entitlement probabilities for deletion or truncation indicate evidence-presence sensitivity even when the classifier already predicts NOT_ENTITLED and no downgrade opportunity remains. High probabilities for a mismatched but present-evidence condition indicate a match-detection failure.",
        "",
        "## Confidently inverted diagnostic",
        "",
        "| intervention | false entitled | frame | predicate | sufficiency | entitlement | frame state | predicate state | sufficiency state | entitlement state |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ])
    for intervention in ("time_swap", "entity_swap", "event_swap", "location_swap", "role_swap", "title_name_swap", "predicate_swap"):
        lines.append(f"| {intervention} | {_lookup_presence(presence, intervention, 'false_entitled')} | "
                     + " | ".join(_lookup_presence(presence, intervention, metric) for metric in gate_metrics)
                     + " | " + " | ".join(_lookup_state(presence, intervention, f"{field}_state") for field in GATE_FIELDS) + " |")
    lines.extend([
        "",
        "A confidently inverted state means gold NOT_ENTITLED cases receive high pass probabilities. If replicated across seeds, time_swap is the sharpest candidate for an auditor that detects evidence presence but endorses a mismatched claim-evidence relation.",
        "",
        "Expected-direction signal for time_swap false-entitled cases:",
        "",
        "| score | raw AUC | inverted AUC | mean pass probability on positives |",
        "|---|---:|---:|---:|",
    ])
    for score in FAILURE_SCORES:
        lines.append(
            f"| {score} | {_lookup_signal(presence, 'time_swap', 'false_entitled', score, 'raw_auc')} | "
            f"{_lookup_signal(presence, 'time_swap', 'false_entitled', score, 'inverted_auc')} | "
            f"{_lookup_signal(presence, 'time_swap', 'false_entitled', score, 'mean_pass_probability_positive')} |"
        )
    lines.extend([
        "",
        "## Polarity diagnostic",
        "",
        f"For polarity_flip, gold SUPPORT={_lookup_presence(presence, 'polarity_flip', 'gold_SUPPORT')}, gold REFUTE={_lookup_presence(presence, 'polarity_flip', 'gold_REFUTE')}, pred SUPPORT={_lookup_presence(presence, 'polarity_flip', 'pred_SUPPORT')}, pred REFUTE={_lookup_presence(presence, 'polarity_flip', 'pred_REFUTE')}, and classifier error={_lookup_presence(presence, 'polarity_flip', 'classifier_error')}. Polarity flips are entitled SUPPORT/REFUTE distinctions when gold NOT_ENTITLED is zero; they are not downgrade targets by default.",
        "",
        "## Gate-correlation and independence diagnostic",
        "",
        f"Across all examples, mean absolute off-diagonal Pearson correlation is {_lookup_correlation(correlations, '__GLOBAL__', 'pearson_mean_abs_off_diagonal')} and the maximum is {_lookup_correlation(correlations, '__GLOBAL__', 'pearson_max_abs_off_diagonal')}. Mean absolute Spearman correlation is {_lookup_correlation(correlations, '__GLOBAL__', 'spearman_mean_abs_off_diagonal')} and the maximum is {_lookup_correlation(correlations, '__GLOBAL__', 'spearman_max_abs_off_diagonal')}.",
        "",
        "Global pairwise gate correlations:",
        "",
        "| pair | Pearson | Spearman |",
        "|---|---:|---:|",
    ])
    for index, left in enumerate(ALL_SIGNAL_FIELDS):
        for right in ALL_SIGNAL_FIELDS[index + 1 :]:
            lines.append(
                f"| {left} / {right} | "
                f"{_lookup_correlation(correlations, '__GLOBAL__', f'pearson_{left}__{right}')} | "
                f"{_lookup_correlation(correlations, '__GLOBAL__', f'spearman_{left}__{right}')} |"
            )
    lines.extend([
        "",
        "High correlations would argue against describing the named heads as four independent mechanisms; they should instead be described as correlated gate heads over a shared representation.",
        "",
        "## Paper implication",
        "",
        "If the deletion/truncation versus time-swap contrast and expected-direction diagnostics replicate, the appropriate mechanism-level wording is: Structured entitlement heads detect evidence presence but fail to reliably detect claim-evidence match, with time_swap exposing a confidently inverted entitlement judgment.",
        "",
        "This remains a controlled diagnostic. It does not establish the root cause of temporal failure, independent gate mechanisms, broad entitlement checking, or real-world factuality performance.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-presence-csv", type=Path)
    parser.add_argument("--output-correlation-csv", type=Path)
    parser.add_argument("--presence-input", type=Path, action="append")
    parser.add_argument("--correlation-input", type=Path, action="append")
    parser.add_argument("--output-presence-aggregate", type=Path)
    parser.add_argument("--output-correlation-aggregate", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.presence_input or args.correlation_input:
        required = (args.presence_input, args.correlation_input, args.output_presence_aggregate,
                    args.output_correlation_aggregate, args.output_md)
        if any(value is None for value in required):
            raise ValueError("aggregate mode requires both input sets and all three outputs")
        presence = aggregate_presence(args.presence_input)
        correlations = aggregate_correlations(args.correlation_input)
        aggregate_fields = ("row_type", "intervention_type", "target", "score", "metric",
                            "mean", "std", "n", "formatted", "state_summary")
        write_csv(args.output_presence_aggregate, presence, aggregate_fields)
        correlation_fields = ("intervention_type", "metric", "mean", "std", "n", "formatted")
        write_csv(args.output_correlation_aggregate, correlations, correlation_fields)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(presence, correlations), encoding="utf-8")
        return 0
    required = (args.preds, args.seed, args.output_presence_csv, args.output_correlation_csv)
    if any(value is None for value in required):
        raise ValueError("per-seed mode requires preds, seed, and both CSV outputs")
    predictions = load_prediction_file(args.preds)["predictions"]
    presence = [*opportunity_rows(predictions, args.seed), *signal_rows(predictions, args.seed)]
    correlations = correlation_rows(predictions, args.seed)
    write_csv(args.output_presence_csv, presence, PRESENCE_FIELDS)
    write_csv(args.output_correlation_csv, correlations, CORRELATION_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
