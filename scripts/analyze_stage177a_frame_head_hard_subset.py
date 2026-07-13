"""Stage177-A: diagnostic-only clean-dev frame-head failure audit.

This script performs no training, fitting, calibration, threshold search, or
checkpoint selection.  It validates the frozen Stage176 inputs before model
construction/forward, then audits native frame-head outputs on the deterministic
seed-174 clean-dev split.
"""

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
from typing import Any, Callable, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import analyze_stage176a_support_boundary_attribution as stage176a  # noqa: E402
from scripts import analyze_stage176b_native_structural_separability as stage176b  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402


STAGE = "Stage177-A"
CALIBRATION = "STAGE177A_FRAME_CALIBRATION_FAILURE_WITH_PRESERVED_RANKING"
PAIRWISE = "STAGE177A_FRAME_PAIRWISE_SIGNAL_PRESENT_ABSOLUTE_DISCRIMINATION_WEAK"
NO_SIGNAL = "STAGE177A_FRAME_HEAD_NO_USABLE_CLEAN_SIGNAL"
BLOCKED = "STAGE177A_FRAME_HEAD_AUDIT_BLOCKED"
STAGE176B_DECISION = "STAGE176B_NO_ROBUST_NATIVE_STRUCTURAL_SEPARABILITY_SIGNAL"
EXPECTED_FAMILIES = {
    "entity_swap", "event_swap", "evidence_deletion", "evidence_truncation",
    "irrelevant_evidence", "location_swap", "none", "paraphrase",
    "polarity_flip", "predicate_swap", "role_swap", "title_name_swap",
}
FRAME_INCOMPATIBLE_FAMILIES = {
    "entity_swap", "event_swap", "location_swap", "role_swap", "title_name_swap"
}
FRAME_COMPATIBLE_FAMILIES = {"none", "paraphrase", "polarity_flip"}
OUTPUT_NAMES = {
    "json": "stage177a_frame_head_hard_subset_report.json",
    "md": "stage177a_frame_head_hard_subset_report.md",
    "dev": "stage177a_dev_frame_scores.csv",
    "overall": "stage177a_overall_frame_metrics.csv",
    "hard": "stage177a_hard_subset_metrics.csv",
    "families": "stage177a_intervention_frame_metrics.csv",
    "pairwise": "stage177a_pairwise_ranking.csv",
    "anchors": "stage177a_anchor_ranking.csv",
    "errors": "stage177a_frame_error_rows.csv",
    "drift": "stage177a_frame_drift_summary.csv",
}


class AuditBlocked(ValueError):
    """Hard input or semantic-contract failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditBlocked(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditBlocked(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as error:
        raise AuditBlocked(f"cannot read CSV {path}: {error}") from error
    _require(bool(rows), f"CSV is empty: {path}")
    return rows


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditBlocked(f"{name} is not numeric: {value!r}") from error
    _require(math.isfinite(result), f"{name} is NaN or infinite")
    return result


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.median(materialized) if materialized else None


def _variance(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.pvariance(materialized) if materialized else None


def _percentile(values: list[float], fraction: float) -> float:
    _require(bool(values), "percentile requires nonempty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _quantiles(values: list[float]) -> dict[str, float]:
    return {f"q{int(q * 100):02d}": _percentile(values, q) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)}


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    ranks = _rankdata(scores)
    u = sum(rank for rank, label in zip(ranks, labels) if label == 1) - positive * (positive + 1) / 2.0
    return u / (positive * negative)


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positive = sum(labels)
    if not positive:
        return None
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    hits, total = 0, 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            hits += 1
            total += hits / rank
    return total / positive


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else None


def _confusion(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))

    def division(a: int, b: int) -> float | None:
        return a / b if b else None

    recall1, recall0 = division(tp, tp + fn), division(tn, tn + fp)
    precision1, precision0 = division(tp, tp + fp), division(tn, tn + fn)
    f1_1 = division(2 * tp, 2 * tp + fp + fn)
    f1_0 = division(2 * tn, 2 * tn + fp + fn)
    recalls = [item for item in (recall0, recall1) if item is not None]
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / len(labels) if labels else None,
        "balanced_accuracy": statistics.fmean(recalls) if recalls else None,
        "label_0": {"precision": precision0, "recall": recall0, "f1": f1_0},
        "label_1": {"precision": precision1, "recall": recall1, "f1": f1_1},
    }


def _ece(labels: list[int], probabilities: list[float], bins: int = 10) -> dict[str, Any]:
    records, total = [], len(labels)
    value = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [i for i, p in enumerate(probabilities) if lower <= p < upper or (index == bins - 1 and p == 1.0)]
        confidence = _mean(probabilities[i] for i in members)
        accuracy = _mean(labels[i] for i in members)
        contribution = (len(members) / total) * abs(float(confidence) - float(accuracy)) if members else 0.0
        value += contribution
        records.append({"lower": lower, "upper": upper, "count": len(members), "mean_probability": confidence, "positive_rate": accuracy, "ece_contribution": contribution})
    return {"equal_width_bins": bins, "ece": value, "bins": records}


def _binary_metrics(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    labels = [int(row["gold_frame_label"]) for row in rows]
    scores = [float(row[f"{side}_frame_score"]) for row in rows]
    probabilities = [float(row[f"{side}_frame_prob"]) for row in rows]
    predictions = [int(row[f"{side}_frame_prediction"]) for row in rows]
    confusion = _confusion(labels, predictions)
    return {
        "row_count": len(rows),
        "frame_label_distribution": dict(sorted(Counter(labels).items())),
        "score": {"mean": _mean(scores), "std": math.sqrt(_variance(scores) or 0.0), "minimum": min(scores), "maximum": max(scores), "quantiles": _quantiles(scores)},
        "probability": {"mean": _mean(probabilities), "std": math.sqrt(_variance(probabilities) or 0.0), "minimum": min(probabilities), "maximum": max(probabilities)},
        "auroc": _auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "fixed_threshold": 0.5,
        "confusion": confusion,
        "brier_score": _mean((p - y) ** 2 for p, y in zip(probabilities, labels)),
        "ece": _ece(labels, probabilities),
        "exact_zero_probability_rate": sum(p == 0.0 for p in probabilities) / len(probabilities),
        "exact_one_probability_rate": sum(p == 1.0 for p in probabilities) / len(probabilities),
        "near_zero_probability_rate": sum(p <= 1e-6 for p in probabilities) / len(probabilities),
        "near_one_probability_rate": sum(p >= 1.0 - 1e-6 for p in probabilities) / len(probabilities),
    }


def _bootstrap_rows(
    rows: list[dict[str, Any]], samples: int, seed: int,
    statistic: Callable[[list[dict[str, Any]]], float | None],
) -> list[float]:
    rng = random.Random(seed)
    output = []
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            output.append(float(value))
    return output


def _ci(values: list[float]) -> list[float] | None:
    return [_percentile(values, .025), _percentile(values, .975)] if values else None


def _hard_metrics(rows: list[dict[str, Any]], side: str, samples: int, seed: int) -> dict[str, Any]:
    labels = [int(row["gold_frame_label"]) for row in rows]
    scores = [float(row[f"{side}_frame_score"]) for row in rows]
    zero = [score for score, label in zip(scores, labels) if label == 0]
    one = [score for score, label in zip(scores, labels) if label == 1]
    auc = _auc(labels, scores)
    cliff = 2.0 * float(auc) - 1.0 if auc is not None else None
    overlap = [max(min(zero), min(one)), min(max(zero), max(one))]
    auc_boot = _bootstrap_rows(rows, samples, seed, lambda sample: _auc([int(r["gold_frame_label"]) for r in sample], [float(r[f"{side}_frame_score"]) for r in sample]))
    median_boot = _bootstrap_rows(rows, samples, seed + 1, lambda sample: float(_median(float(r[f"{side}_frame_score"]) for r in sample if int(r["gold_frame_label"]) == 1) or 0.0) - float(_median(float(r[f"{side}_frame_score"]) for r in sample if int(r["gold_frame_label"]) == 0) or 0.0))
    ordering_errors = []
    for positive in (row for row in rows if int(row["gold_frame_label"]) == 1):
        for negative in (row for row in rows if int(row["gold_frame_label"]) == 0):
            if float(positive[f"{side}_frame_score"]) <= float(negative[f"{side}_frame_score"]):
                ordering_errors.append({"compatible_row_id": positive["row_id"], "incompatible_row_id": negative["row_id"], "tie": float(positive[f"{side}_frame_score"]) == float(negative[f"{side}_frame_score"])})
    return {
        **_binary_metrics(rows, side),
        "mean_score_frame_0": _mean(zero), "median_score_frame_0": _median(zero),
        "mean_score_frame_1": _mean(one), "median_score_frame_1": _median(one),
        "score_overlap_range": overlap if overlap[0] <= overlap[1] else None,
        "cliffs_delta_frame_1_over_frame_0": cliff,
        "bootstrap_auroc_ci95": _ci(auc_boot),
        "bootstrap_median_difference_ci95": _ci(median_boot),
        "score_ordering_error_count": len(ordering_errors),
        "score_ordering_errors": ordering_errors,
    }


def _validate_stage176b_report(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("decision") == STAGE176B_DECISION, "Stage176-B decision mismatch")
    beneficial = stage176b._first(report, ("cohort_counts.beneficial_correction", "scope.beneficial_corrections", "cohort_findings.beneficial_correction.rows"))
    harmful = stage176b._first(report, ("cohort_counts.harmful_regression", "scope.harmful_regressions", "cohort_findings.harmful_regression.rows"))
    _require(int(beneficial) == 25 and int(harmful) == 14, "Stage176-B cohort counts must be 25/14")
    available = stage176b._first(report, ("signals.available", "signal_availability.signals"), {})
    if isinstance(available, dict):
        frame_available = bool(available.get("frame_prob", {}).get("available"))
    else:
        frame_available = "frame_prob" in available
    _require(frame_available, "Stage176-B frame signal is unavailable")
    learned = stage176b._first(report, ("scope.learned_probe", "safety_policy.learned_probe"), False)
    threshold = stage176b._first(report, ("scope.threshold_search", "safety_policy.threshold_search"), False)
    external = stage176b._first(report, ("scope.external_evaluation", "safety_policy.external_evaluation"), False)
    _require(not learned and not threshold and not external, "Stage176-B safety contract mismatch")
    return {"status": "passed", "decision": report["decision"], "beneficial": 25, "harmful": 14, "frame_signal_available": True, "no_learned_probe": True, "no_threshold_search": True, "no_external_data": True}


def _validate_stage176b_scores(path: Path, dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    required = {"stable_row_index", "row_id", "pair_id", "intervention_type", "gold_frame_label", "cohort", "baseline_frame_prob", "treatment_frame_prob"}
    _require(required <= set(rows[0]), f"Stage176-B dev scores missing columns: {sorted(required - set(rows[0]))}")
    _require(len(rows) == 720, "Stage176-B dev scores must contain 720 rows")
    ordered = sorted(rows, key=lambda row: int(row["stable_row_index"]))
    _require([int(row["stable_row_index"]) for row in ordered] == list(range(720)), "Stage176-B stable indexes must be 0..719")
    for index, (row, record) in enumerate(zip(ordered, dev_rows)):
        expected_id = str(record.get("id", f"dev_row_{index}"))
        _require(str(row["row_id"]) == expected_id, f"Stage176-B row id mismatch at {index}")
        _require(str(row["pair_id"]) == str(record["pair_id"]), f"Stage176-B pair mismatch at {index}")
        _require(str(row["intervention_type"]) == str(record["intervention_type"]), f"Stage176-B family mismatch at {index}")
        _require(int(row["gold_frame_label"]) == int(record["frame_compatible_label"]), f"Stage176-B frame label mismatch at {index}")
        _finite(row["baseline_frame_prob"], f"Stage176-B baseline frame prob {index}")
        _finite(row["treatment_frame_prob"], f"Stage176-B treatment frame prob {index}")
    counts = Counter(row["cohort"] for row in ordered)
    _require(counts["beneficial_correction"] == 25 and counts["harmful_regression"] == 14, "Stage176-B score cohort counts must be 25/14")
    return ordered


def _discover_frame_signal(output: dict[str, Any], row_count: int) -> dict[str, Any]:
    candidates = (
        ("frame_logit", "native_frame_logit", "identity"),
        ("frame_prob", "native_frame_probability", "identity"),
        ("frame_prediction", "native_frame_prediction", "identity"),
    )
    selected = None
    for key, kind, normalization in candidates:
        if key in output and output.get(key) is not None:
            selected = (key, kind, normalization)
            break
    _require(selected is not None, "model output has no native frame logit, probability, or prediction")
    key, kind, normalization = selected
    score = stage176b._tensor_vector(output, key, row_count)
    probability = stage176b._tensor_vector(output, "frame_prob", row_count) if output.get("frame_prob") is not None else None
    prediction = stage176b._tensor_vector(output, "frame_prediction", row_count) if output.get("frame_prediction") is not None else None
    if probability is None:
        _require(kind == "native_frame_prediction", "native frame probability is required for fixed-0.5 calibration metrics")
        probability = score
    _require(all(0.0 <= value <= 1.0 for value in probability), "native frame probability is outside [0,1]")
    if prediction is None:
        prediction = [int(value >= 0.5) for value in probability]
        prediction_source = "fixed_0.5_on_native_frame_prob"
    else:
        _require(all(value in (0.0, 1.0) for value in prediction), "native frame predictions are not binary")
        prediction = [int(value) for value in prediction]
        prediction_source = "native_frame_prediction"
    availability = {
        "frame_hidden_representation": {"key": "frame_pair_repr", "available": output.get("frame_pair_repr") is not None, "used_as_score": False},
        "pre_sigmoid_frame_logit": {"key": "frame_logit", "available": output.get("frame_logit") is not None},
        "frame_head_raw_energy": {"key": "frame_energy", "available": output.get("frame_energy") is not None, "used_as_score": False},
    }
    return {"key": key, "kind": kind, "normalization": normalization, "score": score, "probability": probability, "prediction": prediction, "prediction_source": prediction_source, "availability": availability}


def _build_rows(
    dev_rows: list[dict[str, Any]], transitions: list[dict[str, Any]], stage176b_rows: list[dict[str, Any]],
    baseline: dict[str, Any], treatment: dict[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for index, (record, transition, old) in enumerate(zip(dev_rows, transitions, stage176b_rows)):
        cohort = stage176b._cohort(transition)
        row = {
            "stable_row_index": index,
            "row_id": str(record.get("id", f"dev_row_{index}")),
            "pair_id": str(record["pair_id"]),
            "intervention_type": str(record["intervention_type"]),
            "gold_final_label": str(record["final_label"]),
            "gold_frame_label": int(record["frame_compatible_label"]),
            "stage176a_category": transition["category"], "cohort": cohort,
            "baseline_final_prediction": transition["baseline_predicted_label"],
            "treatment_final_prediction": transition["treatment_predicted_label"],
            "baseline_support_margin": float(transition["baseline_support_margin"]),
            "treatment_support_margin": float(transition["treatment_support_margin"]),
            "support_margin_delta": float(transition["support_margin_delta"]),
            "baseline_frame_score": baseline["score"][index], "treatment_frame_score": treatment["score"][index],
            "frame_score_delta": treatment["score"][index] - baseline["score"][index],
            "baseline_frame_prob": baseline["probability"][index], "treatment_frame_prob": treatment["probability"][index],
            "frame_prob_delta": treatment["probability"][index] - baseline["probability"][index],
            "baseline_frame_prediction": baseline["prediction"][index], "treatment_frame_prediction": treatment["prediction"][index],
        }
        _require(math.isclose(row["baseline_frame_prob"], float(old["baseline_frame_prob"]), rel_tol=1e-5, abs_tol=1e-6), f"baseline frame probability disagrees with Stage176-B at {index}")
        _require(math.isclose(row["treatment_frame_prob"], float(old["treatment_frame_prob"]), rel_tol=1e-5, abs_tol=1e-6), f"treatment frame probability disagrees with Stage176-B at {index}")
        output.append(row)
    return output


def _family_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for family in sorted({row["intervention_type"] for row in rows}):
        group = [row for row in rows if row["intervention_type"] == family]
        labels = [int(row["gold_frame_label"]) for row in group]
        record: dict[str, Any] = {"intervention_type": family, "row_count": len(group), "frame_label_distribution": dict(sorted(Counter(labels).items()))}
        for side in ("baseline", "treatment"):
            metrics = _binary_metrics(group, side)
            record[f"{side}_auroc"] = metrics["auroc"]
            record[f"{side}_confusion"] = metrics["confusion"]
            record[f"{side}_accuracy"] = metrics["confusion"]["accuracy"]
            record[f"{side}_balanced_accuracy"] = metrics["confusion"]["balanced_accuracy"]
            record[f"{side}_mean_score_frame_0"] = _mean(float(row[f"{side}_frame_score"]) for row in group if int(row["gold_frame_label"]) == 0)
            record[f"{side}_mean_score_frame_1"] = _mean(float(row[f"{side}_frame_score"]) for row in group if int(row["gold_frame_label"]) == 1)
            record[f"{side}_false_compatible"] = metrics["confusion"]["fp"]
            record[f"{side}_false_incompatible"] = metrics["confusion"]["fn"]
        record["treatment_minus_baseline_error_delta"] = (record["treatment_false_compatible"] + record["treatment_false_incompatible"]) - (record["baseline_false_compatible"] + record["baseline_false_incompatible"])
        record["declared_semantic_group"] = "frame_incompatible" if family in FRAME_INCOMPATIBLE_FAMILIES else "frame_compatible" if family in FRAME_COMPATIBLE_FAMILIES else "other"
        record["declared_semantics_verified_by_gold_labels"] = (all(label == 0 for label in labels) if family in FRAME_INCOMPATIBLE_FAMILIES else all(label == 1 for label in labels) if family in FRAME_COMPATIBLE_FAMILIES else None)
        result.append(record)
    return result


def _pair_comparisons(rows: list[dict[str, Any]], side: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)
    comparisons, pair_records = [], []
    for pair_id, group in sorted(grouped.items()):
        compatible = [row for row in group if int(row["gold_frame_label"]) == 1]
        incompatible = [row for row in group if int(row["gold_frame_label"]) == 0]
        _require(compatible and incompatible, f"pair {pair_id} lacks both frame labels")
        outcomes, gaps = [], []
        for pos in compatible:
            for neg in incompatible:
                gap = float(pos[f"{side}_frame_score"]) - float(neg[f"{side}_frame_score"])
                outcome = "correct" if gap > 0 else "tie" if gap == 0 else "violation"
                outcomes.append(outcome); gaps.append(gap)
                comparisons.append({"side": side, "pair_id": pair_id, "compatible_row_id": pos["row_id"], "incompatible_row_id": neg["row_id"], "compatible_family": pos["intervention_type"], "incompatible_family": neg["intervention_type"], "compatible_score": pos[f"{side}_frame_score"], "incompatible_score": neg[f"{side}_frame_score"], "gap": gap, "outcome": outcome})
        pair_records.append({"pair_id": pair_id, "correct": outcomes.count("correct"), "tie": outcomes.count("tie"), "violation": outcomes.count("violation"), "comparison_count": len(outcomes), "accuracy": outcomes.count("correct") / len(outcomes), "mean_gap": _mean(gaps), "mean_compatible": _mean(float(r[f"{side}_frame_score"]) for r in compatible), "mean_incompatible": _mean(float(r[f"{side}_frame_score"]) for r in incompatible)})
    return comparisons, {"pair_records": pair_records}


def _pair_summary(comparisons: list[dict[str, Any]], pair_records: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    correct = sum(row["outcome"] == "correct" for row in comparisons)
    tie = sum(row["outcome"] == "tie" for row in comparisons)
    violation = len(comparisons) - correct - tie
    rng, boot = random.Random(seed), []
    for _ in range(samples):
        sample = [pair_records[rng.randrange(len(pair_records))] for _ in pair_records]
        boot.append(sum(row["correct"] for row in sample) / sum(row["comparison_count"] for row in sample))
    return {
        "total_comparable_pairwise_comparisons": len(comparisons), "correct_ordering_count": correct,
        "tie_count": tie, "violation_count": violation, "pairwise_ranking_accuracy": correct / len(comparisons),
        "mean_score_gap": _mean(float(row["gap"]) for row in comparisons), "median_score_gap": _median(float(row["gap"]) for row in comparisons),
        "pair_level_mean_compatible_score": _mean(float(row["mean_compatible"]) for row in pair_records),
        "pair_level_mean_incompatible_score": _mean(float(row["mean_incompatible"]) for row in pair_records),
        "pair_level_separation_gap": _mean(float(row["mean_compatible"]) - float(row["mean_incompatible"]) for row in pair_records),
        "fully_ordered_pair_count": sum(row["violation"] == 0 and row["tie"] == 0 for row in pair_records),
        "partially_violated_pair_count": sum(row["correct"] > 0 and (row["violation"] > 0 or row["tie"] > 0) for row in pair_records),
        "fully_reversed_pair_count": sum(row["correct"] == 0 for row in pair_records),
        "pair_bootstrap_ranking_accuracy_ci95": _ci(boot), "bootstrap_unit": "pair_id",
    }


def _anchor_metrics(rows: list[dict[str, Any]], side: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)
    records, malformed = [], []
    for pair_id, group in sorted(grouped.items()):
        incompatible = [row for row in group if int(row["gold_frame_label"]) == 0]
        for anchor_name in ("none", "paraphrase"):
            anchors = [row for row in group if row["intervention_type"] == anchor_name]
            if len(anchors) != 1 or int(anchors[0]["gold_frame_label"]) != 1:
                malformed.append({"pair_id": pair_id, "anchor": anchor_name, "reason": "missing_duplicate_or_noncompatible_anchor"})
                continue
            anchor = anchors[0]
            for target in incompatible:
                gap = float(anchor[f"{side}_frame_score"]) - float(target[f"{side}_frame_score"])
                records.append({"side": side, "anchor": anchor_name, "pair_id": pair_id, "anchor_row_id": anchor["row_id"], "target_row_id": target["row_id"], "target_family": target["intervention_type"], "gap": gap, "correct": gap > 0, "tie": gap == 0})
    summary = {"malformed_pairs": malformed}
    for anchor in ("none", "paraphrase"):
        group = [row for row in records if row["anchor"] == anchor]
        summary[anchor] = {"comparison_count": len(group), "correct_rate": sum(row["correct"] for row in group) / len(group) if group else None, "violation_count": sum(not row["correct"] for row in group), "mean_gap": _mean(float(row["gap"]) for row in group), "violations_by_family": dict(sorted(Counter(row["target_family"] for row in group if not row["correct"]).items()))}
    return records, summary


def _error_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output, summary = [], {}
    changed_ids = {row["row_id"] for row in rows if row["baseline_final_prediction"] != row["treatment_final_prediction"]}
    for side in ("baseline", "treatment"):
        side_errors = []
        for row in rows:
            error_type = None
            if int(row["gold_frame_label"]) == 0 and int(row[f"{side}_frame_prediction"]) == 1:
                error_type = "false_compatible"
            elif int(row["gold_frame_label"]) == 1 and int(row[f"{side}_frame_prediction"]) == 0:
                error_type = "false_incompatible"
            if error_type:
                item = {**row, "model_side": side, "frame_error_type": error_type, "overlaps_stage176a_changed": row["row_id"] in changed_ids, "overlaps_beneficial": row["cohort"] == "beneficial_correction", "overlaps_harmful": row["cohort"] == "harmful_regression"}
                output.append(item); side_errors.append(item)
        summary[side] = {}
        for error_type in ("false_compatible", "false_incompatible"):
            group = [row for row in side_errors if row["frame_error_type"] == error_type]
            summary[side][error_type] = {"row_count": len(group), "intervention_distribution": dict(sorted(Counter(r["intervention_type"] for r in group).items())), "pair_distribution": dict(sorted(Counter(r["pair_id"] for r in group).items())), "final_gold_label_distribution": dict(sorted(Counter(r["gold_final_label"] for r in group).items())), "final_prediction_distribution": dict(sorted(Counter(r[f"{side}_final_prediction"] for r in group).items())), "support_margin": {"mean": _mean(float(r[f"{side}_support_margin"]) for r in group), "median": _median(float(r[f"{side}_support_margin"]) for r in group)}, "frame_score": {"mean": _mean(float(r[f"{side}_frame_score"]) for r in group), "median": _median(float(r[f"{side}_frame_score"]) for r in group)}, "overlap_beneficial": sum(r["cohort"] == "beneficial_correction" for r in group), "overlap_harmful": sum(r["cohort"] == "harmful_regression" for r in group), "overlap_stage176a_changed": sum(r["row_id"] in changed_ids for r in group)}
    summary["requested_cohort_checks"] = {"beneficial_25_baseline_false_compatible": sum(r["cohort"] == "beneficial_correction" and r["frame_error_type"] == "false_compatible" and r["model_side"] == "baseline" for r in output), "harmful_14_false_incompatible_baseline": sum(r["cohort"] == "harmful_regression" and r["frame_error_type"] == "false_incompatible" and r["model_side"] == "baseline" for r in output), "harmful_14_false_incompatible_treatment": sum(r["cohort"] == "harmful_regression" and r["frame_error_type"] == "false_incompatible" and r["model_side"] == "treatment" for r in output)}
    return output, summary


def _drift(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def summarize(name: str, group: list[dict[str, Any]]) -> dict[str, Any]:
        deltas = [float(row["frame_score_delta"]) for row in group]
        return {"subset": name, "row_count": len(group), "mean_delta": _mean(deltas), "median_delta": _median(deltas), "negative_delta_count": sum(v < 0 for v in deltas), "positive_delta_count": sum(v > 0 for v in deltas), "zero_delta_count": sum(v == 0 for v in deltas), "baseline_score_delta_correlation": _pearson([float(row["baseline_frame_score"]) for row in group], deltas)}
    records = [summarize("full_dev", rows)]
    for label in (0, 1): records.append(summarize(f"frame_label_{label}", [r for r in rows if int(r["gold_frame_label"]) == label]))
    for family in sorted({r["intervention_type"] for r in rows}): records.append(summarize(f"family:{family}", [r for r in rows if r["intervention_type"] == family]))
    for cohort in ("beneficial_correction", "harmful_regression"): records.append(summarize(f"cohort:{cohort}", [r for r in rows if r["cohort"] == cohort]))
    transitions = Counter(f"{row['baseline_frame_prediction']}->{row['treatment_frame_prediction']}" for row in rows)
    overall = {"full_dev": records[0], "by_frame_label": {str(label): next(r for r in records if r["subset"] == f"frame_label_{label}") for label in (0, 1)}, "by_family": {r["subset"].split(":", 1)[1]: r for r in records if r["subset"].startswith("family:")}, "by_hard_cohort": {r["subset"].split(":", 1)[1]: r for r in records if r["subset"].startswith("cohort:")}, "prediction_transitions": dict(sorted(transitions.items()))}
    return records, overall


def _collapse(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    scores = [float(row[f"{side}_frame_score"]) for row in rows]
    zero = [float(row[f"{side}_frame_score"]) for row in rows if int(row["gold_frame_label"]) == 0]
    one = [float(row[f"{side}_frame_score"]) for row in rows if int(row["gold_frame_label"]) == 1]
    hard = [float(row[f"{side}_frame_score"]) for row in rows if row["cohort"] in {"beneficial_correction", "harmful_regression"}]
    pooled = math.sqrt(((len(zero) - 1) * (statistics.variance(zero) if len(zero) > 1 else 0) + (len(one) - 1) * (statistics.variance(one) if len(one) > 1 else 0)) / max(len(zero) + len(one) - 2, 1))
    separation = float(_mean(one) or 0.0) - float(_mean(zero) or 0.0)
    iqr_overlap = [max(_percentile(zero, .25), _percentile(one, .25)), min(_percentile(zero, .75), _percentile(one, .75))]
    variance = float(_variance(scores) or 0.0)
    return {"full_dev_score_variance": variance, "score_variance_by_frame_label": {"0": _variance(zero), "1": _variance(one)}, "hard_39_variance": _variance(hard), "compatible_minus_incompatible_mean_separation": separation, "standardized_mean_difference": separation / pooled if pooled else None, "interquartile_overlap": iqr_overlap if iqr_overlap[0] <= iqr_overlap[1] else None, "unique_rounded_score_count_3dp": len({round(value, 3) for value in scores}), "near_constant_signal": variance < 1e-8 or len({round(value, 6) for value in scores}) <= 2, "saturated_probability_signal": all(float(row[f"{side}_frame_prob"]) >= .99 or float(row[f"{side}_frame_prob"]) <= .01 for row in rows), "diagnostic_term": "frame_output_discrimination_collapse", "hidden_state_representation_collapse_claimed": False}


def _diagnose(overall: dict[str, Any], hard: dict[str, Any], families: list[dict[str, Any]], pairwise: dict[str, Any], anchor: dict[str, Any], drift: dict[str, Any], collapse: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for side in ("baseline", "treatment"):
        om, hm, pm, cm = overall[side], hard[side], pairwise[side], collapse[side]
        high_ranking = float(om["auroc"] or 0) >= .80 and float(pm["pairwise_ranking_accuracy"]) >= .70
        weak_fixed = float(om["confusion"]["balanced_accuracy"] or 0) < .70 or float(om["ece"]["ece"]) > .10
        result[f"A_global_calibration_failure_{side}"] = {"supported": high_ranking and weak_fixed, "rule": "full AUROC >= .80 and pairwise >= .70, with balanced accuracy < .70 or ECE > .10", "evidence": {"auroc": om["auroc"], "pairwise": pm["pairwise_ranking_accuracy"], "balanced_accuracy": om["confusion"]["balanced_accuracy"], "ece": om["ece"]["ece"]}}
        family_errors = {row["intervention_type"]: row[f"{side}_false_compatible"] + row[f"{side}_false_incompatible"] for row in families}
        max_family = max(family_errors, key=family_errors.get)
        result[f"B_family_specific_classification_failure_{side}"] = {"supported": family_errors[max_family] >= 3 and family_errors[max_family] / 60 > (om["confusion"]["fp"] + om["confusion"]["fn"]) / 720 + .10, "rule": "a 60-row family has >=3 errors and error rate exceeds overall by >.10", "evidence": {"highest_error_family": max_family, "family_errors": family_errors[max_family], "all_family_errors": family_errors}}
        result[f"C_within_pair_ranking_failure_{side}"] = {"supported": pm["pairwise_ranking_accuracy"] < .65 or pm["pair_bootstrap_ranking_accuracy_ci95"][0] <= .50, "rule": "pairwise accuracy < .65 or pair-bootstrap lower bound <= .50", "evidence": {"accuracy": pm["pairwise_ranking_accuracy"], "ci95": pm["pair_bootstrap_ranking_accuracy_ci95"], "anchor": anchor[side]}}
        result[f"D_hard_subset_discrimination_collapse_{side}"] = {"supported": float(hm["auroc"] or 0) < .70 and (float(om["auroc"] or 0) - float(hm["auroc"] or 0) >= .10 or hm["score_overlap_range"] is not None), "rule": "hard AUROC < .70 plus >=.10 full-to-hard drop or score overlap", "evidence": {"full_auroc": om["auroc"], "hard_auroc": hm["auroc"], "overlap": hm["score_overlap_range"]}}
        result[f"E_global_frame_output_discrimination_collapse_{side}"] = {"supported": float(om["auroc"] or 0) < .70 and abs(float(cm["standardized_mean_difference"] or 0)) < .50 and pm["pairwise_ranking_accuracy"] < .65, "rule": "full AUROC < .70, |SMD| < .50, and pairwise < .65", "evidence": {"auroc": om["auroc"], "smd": cm["standardized_mean_difference"], "pairwise": pm["pairwise_ranking_accuracy"]}, "terminology_limit": "frame output only; no hidden-state collapse claim"}
    result["F_stage175_induced_frame_degradation"] = {"supported": float(overall["baseline"]["auroc"] or 0) - float(overall["treatment"]["auroc"] or 0) > .02 or pairwise["baseline"]["pairwise_ranking_accuracy"] - pairwise["treatment"]["pairwise_ranking_accuracy"] > .02 or float(drift["by_frame_label"]["1"]["mean_delta"] or 0) < -.02, "rule": "treatment loses >.02 AUROC or pairwise accuracy, or compatible mean score declines by >.02", "evidence": {"baseline_auroc": overall["baseline"]["auroc"], "treatment_auroc": overall["treatment"]["auroc"], "baseline_pairwise": pairwise["baseline"]["pairwise_ranking_accuracy"], "treatment_pairwise": pairwise["treatment"]["pairwise_ranking_accuracy"], "compatible_mean_delta": drift["by_frame_label"]["1"]["mean_delta"]}}
    return result


def _gate(overall: dict[str, Any], hard: dict[str, Any], pairwise: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    gate1, gate2 = [], []
    for side in ("baseline", "treatment"):
        fixed_weak = float(overall[side]["confusion"]["balanced_accuracy"] or 0) < .70 or float(overall[side]["ece"]["ece"]) > .10
        if float(overall[side]["auroc"] or 0) >= .80 and float(hard[side]["auroc"] or 0) >= .70 and pairwise[side]["pairwise_ranking_accuracy"] >= .70 and fixed_weak:
            gate1.append(side)
        ci = pairwise[side]["pair_bootstrap_ranking_accuracy_ci95"]
        if pairwise[side]["pairwise_ranking_accuracy"] >= .65 and ci[0] > .50 and pairwise[side]["mean_score_gap"] > 0 and float(hard[side]["auroc"] or 0) < .70:
            gate2.append(side)
    decision = CALIBRATION if gate1 else PAIRWISE if gate2 else NO_SIGNAL
    return decision, {"gate_1_calibration_only_repair_candidate": {"criteria": {"full_dev_auroc_minimum": .80, "hard_39_auroc_minimum": .70, "pairwise_ranking_minimum": .70, "fixed_0_5_performance_materially_weak": True}, "qualifying_views": gate1, "entered": bool(gate1)}, "gate_2_frame_pairwise_objective_candidate": {"criteria": {"pairwise_ranking_minimum": .65, "pair_bootstrap_ci_lower_strictly_greater_than": .50, "mean_gap_strictly_greater_than": 0, "hard_39_absolute_discrimination_weak": True}, "qualifying_views": gate2, "entered": bool(gate2)}, "gate_3_no_usable_frame_signal": {"entered": not gate1 and not gate2}, "decision": decision, "threshold_tuning_authorized": False, "training_authorized": False}


def _csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows: writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {STAGE} frame-head hard-subset failure audit", "", f"**Decision:** `{report['decision']}`", "", "## Signal source", "", f"Primary ranking/discrimination score: `{report['signal_source']['primary_key']}` ({report['signal_source']['normalization']}). Fixed-0.5, Brier, and ECE use native `{report['signal_source']['probability_key']}`. No score is reconstructed from final classifier logits.", "", "## Overall and hard-subset performance", "", "| View | Full AUROC | Full AP | Full balanced accuracy | Hard-39 AUROC | Pairwise accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for side in ("baseline", "treatment"):
        o, h, p = report["overall_frame_performance"][side], report["hard_subset_performance"][side], report["within_pair_ranking"][side]
        lines.append(f"| {side} | {o['auroc']:.6f} | {o['average_precision']:.6f} | {o['confusion']['balanced_accuracy']:.6f} | {h['auroc']:.6f} | {p['pairwise_ranking_accuracy']:.6f} |")
    lines += ["", "## Family, ranking, errors, drift, and collapse", "", "All 12 intervention families are reported with actual gold-frame distributions and fixed-0.5 errors. Within-pair evidence compares every compatible row only with incompatible rows sharing its `pair_id`; bootstrap resampling is by pair. Anchor analyses use valid gold-frame-1 `none` and `paraphrase` rows and record malformed pairs instead of imputing them.", "", "False-compatible and false-incompatible rows are decomposed by family, pair, final gold/prediction, SUPPORT margin, frame score, and Stage176-A cohort overlap. Baseline-to-treatment drift is reported globally, by frame label, family, and hard cohort. Collapse language is restricted to `frame_output_discrimination_collapse`; hidden-state representation collapse is not claimed.", "", "## Stage177-B gate", "", f"Gate decision: `{report['stage177b_gate']['decision']}`. Threshold tuning, training, loss changes, checkpoint selection, and external tuning remain unauthorized.", ""]
    return "\n".join(lines)


def _blocked_outputs(output_dir: Path, error: Exception) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"stage": STAGE, "decision": BLOCKED, "scope": None, "input_validation": {"status": "blocked", "error": f"{type(error).__name__}: {error}", "blocked_before_model_forward_if_input_failure": True}, "signal_source": None, "overall_frame_performance": None, "hard_subset_performance": None, "intervention_performance": None, "within_pair_ranking": None, "anchor_ranking": None, "error_decomposition": None, "frame_drift": None, "score_collapse_diagnostics": None, "diagnosis": None, "stage177b_gate": None, "limitations": [], "safety_policy": {"training": False, "threshold_search": False, "calibration_fitting": False, "checkpoint_selection": False}}
    _write_json(output_dir / OUTPUT_NAMES["json"], report)
    (output_dir / OUTPUT_NAMES["md"]).write_text(f"# Stage177-A blocked\n\n**Decision:** `{BLOCKED}`\n\n`{type(error).__name__}: {error}`\n", encoding="utf-8")
    for key in ("dev", "overall", "hard", "families", "pairwise", "anchors", "errors", "drift"): _write_csv(output_dir / OUTPUT_NAMES[key], [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/controlled_v5_v3_without_time_swap.jsonl"))
    parser.add_argument("--stage176a-report", type=Path, required=True)
    parser.add_argument("--stage176a-row-transitions", type=Path, required=True)
    parser.add_argument("--stage176b-report", type=Path, required=True)
    parser.add_argument("--stage176b-dev-native-scores", type=Path, required=True)
    parser.add_argument("--baseline-provenance", type=Path, required=True)
    parser.add_argument("--treatment-provenance", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=177)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        _require(args.device == "cuda", "Stage177-A requires --device cuda")
        _require(args.eval_batch_size > 0 and args.bootstrap_samples > 0, "batch size and bootstrap samples must be positive")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        paths = {name: getattr(args, name.replace("_", "-") if False else name).resolve() for name in ("data", "stage176a_report", "stage176a_row_transitions", "stage176b_report", "stage176b_dev_native_scores", "baseline_provenance", "treatment_provenance", "baseline_checkpoint", "treatment_checkpoint")}
        for name, path in paths.items(): _require(path.is_file(), f"{name} file does not exist: {path}")

        stage176a_report = _read_json(paths["stage176a_report"])
        stage176a_validation = stage176b._validate_stage176a_report(stage176a_report)
        transitions = stage176b._read_transition_csv(paths["stage176a_row_transitions"])
        transition_validation = stage176b._validate_transition_counts(transitions, stage176a_report)
        stage176b_report = _read_json(paths["stage176b_report"])
        stage176b_validation = _validate_stage176b_report(stage176b_report)
        baseline_prov, treatment_prov = _read_json(paths["baseline_provenance"]), _read_json(paths["treatment_provenance"])
        provenance_validation = stage176b._validate_provenance_pair(baseline_prov, treatment_prov, paths["data"])
        baseline_path = stage176b._resolve_checkpoint(paths["baseline_provenance"], baseline_prov, paths["baseline_checkpoint"])
        treatment_path = stage176b._resolve_checkpoint(paths["treatment_provenance"], treatment_prov, paths["treatment_checkpoint"])
        _require(baseline_path != treatment_path, "baseline and treatment checkpoints are identical")
        baseline_state, baseline_metadata, baseline_payload = stage176b._load_checkpoint(baseline_path)
        treatment_state, treatment_metadata, treatment_payload = stage176b._load_checkpoint(treatment_path)
        baseline_checkpoint_validation = stage176b._validate_checkpoint("baseline", baseline_path, baseline_prov, baseline_metadata, baseline_payload)
        treatment_checkpoint_validation = stage176b._validate_checkpoint("treatment", treatment_path, treatment_prov, treatment_metadata, treatment_payload)
        _train_rows, dev_rows, split = stage176b._build_split(paths["data"], baseline_prov)
        transitions = stage176b._validate_row_alignment(transitions, dev_rows)
        _require({str(row["intervention_type"]) for row in dev_rows} == EXPECTED_FAMILIES, "dev split must contain all 12 intervention families")
        stage176b_scores = _validate_stage176b_scores(paths["stage176b_dev_native_scores"], dev_rows)
        input_validation = {"status": "passed", "stage176a": stage176a_validation, "stage176a_rows": transition_validation, "stage176b": stage176b_validation, "provenance": provenance_validation, "baseline_checkpoint": baseline_checkpoint_validation, "treatment_checkpoint": treatment_checkpoint_validation, "split": split, "stable_row_identity": True, "stage176b_score_alignment": True, "completed_before_model_construction_and_forward": True}

        from transformers import AutoTokenizer
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(str(stage176a._runtime(baseline_prov, "model_name")))
        if tokenizer.pad_token_id is None:
            _require(tokenizer.eos_token_id is not None, "tokenizer has no pad/eos token")
            tokenizer.pad_token = tokenizer.eos_token
        max_length = int(stage176a._arg(baseline_prov, "max_length", baseline_metadata.get("max_length", 128)))
        _require(max_length == int(stage176a._arg(treatment_prov, "max_length", treatment_metadata.get("max_length", 128))), "max_length mismatch")
        bundle = v5.encode_mamba_records(dev_rows, tokenizer, max_length)
        dev_inputs = v5.move_inputs(bundle["model_inputs"], device)
        baseline_model = stage176a._construct_model(baseline_prov, baseline_metadata, baseline_state, device)
        treatment_model = stage176a._construct_model(treatment_prov, treatment_metadata, treatment_state, device)
        _require(not baseline_model.training and not treatment_model.training, "models must remain in eval mode")
        baseline_output = stage176a._forward(baseline_model, dev_inputs, dev_rows, baseline_prov, device, args.eval_batch_size)
        treatment_output = stage176a._forward(treatment_model, dev_inputs, dev_rows, treatment_prov, device, args.eval_batch_size)
        baseline_signal = _discover_frame_signal(baseline_output, 720)
        treatment_signal = _discover_frame_signal(treatment_output, 720)
        _require(baseline_signal["key"] == treatment_signal["key"], "baseline/treatment frame signal source mismatch")
        rows = _build_rows(dev_rows, transitions, stage176b_scores, baseline_signal, treatment_signal)
        hard_rows = [row for row in rows if row["cohort"] in {"beneficial_correction", "harmful_regression"}]
        _require(len(hard_rows) == 39, "hard subset must contain 39 rows")
        _require(all(int(row["gold_frame_label"]) == 0 for row in hard_rows if row["cohort"] == "beneficial_correction"), "beneficial cohort frame labels must all be 0")
        _require(all(int(row["gold_frame_label"]) == 1 for row in hard_rows if row["cohort"] == "harmful_regression"), "harmful cohort frame labels must all be 1")

        overall = {side: _binary_metrics(rows, side) for side in ("baseline", "treatment")}
        hard = {side: _hard_metrics(hard_rows, side, args.bootstrap_samples, args.bootstrap_seed + index * 10) for index, side in enumerate(("baseline", "treatment"))}
        families = _family_metrics(rows)
        pair_csv, pairwise = [], {}
        anchor_csv, anchors = [], {}
        for index, side in enumerate(("baseline", "treatment")):
            comparisons, pair_data = _pair_comparisons(rows, side)
            pair_csv.extend(comparisons)
            pairwise[side] = _pair_summary(comparisons, pair_data["pair_records"], args.bootstrap_samples, args.bootstrap_seed + 100 + index)
            anchor_records, anchor_summary = _anchor_metrics(rows, side)
            anchor_csv.extend(anchor_records); anchors[side] = anchor_summary
        errors_csv, errors = _error_rows(rows)
        drift_csv, drift = _drift(rows)
        collapse = {side: _collapse(rows, side) for side in ("baseline", "treatment")}
        diagnosis = _diagnose(overall, hard, families, pairwise, anchors, drift, collapse)
        decision, gate = _gate(overall, hard, pairwise)
        report = {"stage": STAGE, "decision": decision, "scope": {"data": str(paths["data"]), "clean_dev_only": True, "seed": 174, "selected_epoch": 20, "dev_rows": 720, "hard_subset_rows": 39, "diagnostic_only": True, "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed}, "input_validation": input_validation, "signal_source": {"priority": ["native frame logit", "native frame probability", "frame prediction"], "primary_key": baseline_signal["key"], "primary_kind": baseline_signal["kind"], "normalization": baseline_signal["normalization"], "probability_key": "frame_prob", "prediction_source": baseline_signal["prediction_source"], "baseline_availability": baseline_signal["availability"], "treatment_availability": treatment_signal["availability"], "final_classifier_logits_used_to_reconstruct_frame_score": False, "logit_to_probability_transform_for_primary_result": False}, "overall_frame_performance": overall, "hard_subset_performance": hard, "intervention_performance": families, "within_pair_ranking": pairwise, "anchor_ranking": anchors, "error_decomposition": errors, "frame_drift": drift, "score_collapse_diagnostics": collapse, "diagnosis": diagnosis, "stage177b_gate": gate, "limitations": ["Single-seed observational comparison on internal clean dev.", "Fixed 0.5 is descriptive; no threshold was searched or optimized.", "Frame-output discrimination collapse does not establish hidden-state representation collapse.", "Bootstrap intervals capture row or pair resampling as stated, not training-seed uncertainty."], "safety_policy": {"clean_dev_only": True, "training": False, "optimizer_created": False, "backward": False, "train_mode_called": False, "fitted_probe": False, "classifier_fitting": False, "threshold_search": False, "calibration_fitting": False, "checkpoint_selection": False, "external_evaluation": False, "external_labels": False, "time_swap": False, "architecture_modified": False, "stage175_loss_modified": False, "hidden_state_hook_added": False}}
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / OUTPUT_NAMES["json"], report)
        (output_dir / OUTPUT_NAMES["md"]).write_text(_render_markdown(report), encoding="utf-8")
        _write_csv(output_dir / OUTPUT_NAMES["dev"], rows)
        _write_csv(output_dir / OUTPUT_NAMES["overall"], [{"view": side, **overall[side]} for side in ("baseline", "treatment")])
        _write_csv(output_dir / OUTPUT_NAMES["hard"], [{"view": side, **hard[side]} for side in ("baseline", "treatment")])
        _write_csv(output_dir / OUTPUT_NAMES["families"], families)
        _write_csv(output_dir / OUTPUT_NAMES["pairwise"], pair_csv)
        _write_csv(output_dir / OUTPUT_NAMES["anchors"], anchor_csv)
        _write_csv(output_dir / OUTPUT_NAMES["errors"], errors_csv)
        _write_csv(output_dir / OUTPUT_NAMES["drift"], drift_csv)
        print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except (AuditBlocked, stage176b.AuditBlocked, OSError, ValueError, KeyError, TypeError) as error:
        _blocked_outputs(output_dir, error)
        print(json.dumps({"decision": BLOCKED, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
