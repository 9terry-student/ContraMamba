"""Stage178-A native frame-logit absolute-comparability offset audit.

This is an evaluation-only, single-seed diagnostic.  Pair-centering and
leave-one-row-out centering are transductive diagnostics, not authorized
inference-time mechanisms.  The module never trains, calls backward, fits a
probe, calibrates, searches a threshold, or selects a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import analyze_stage176a_support_boundary_attribution as stage176a  # noqa: E402
from scripts import analyze_stage176b_native_structural_separability as stage176b  # noqa: E402
from scripts import analyze_stage177e_frame_pairwise_pilot_attribution as stage177e  # noqa: E402


STAGE = "Stage178-A"
BLOCKED = "STAGE178A_FRAME_ABSOLUTE_COMPARABILITY_OFFSET_AUDIT_BLOCKED"
OBSERVED = "STAGE178A_OBSERVED_NUISANCE_ASSOCIATED_PAIR_OFFSET_IDENTIFIED"
LATENT = "STAGE178A_LATENT_PAIR_OFFSET_EXPLAINS_ABSOLUTE_COMPARABILITY_FAILURE"
WEAK = "STAGE178A_PAIR_OFFSET_EXPLANATION_WEAK_PATH_CLOSED"
STAGE177E = "STAGE177E_FRAME_PAIRWISE_OBJECTIVE_REDUNDANT_PATH_CLOSED"
MODEL_NAME = "state-spaces/mamba-130m-hf"
SCHEMA = "stage176a0_selected_checkpoint_v1"
OUTPUTS = {
    "json": "stage178a_frame_absolute_comparability_offset_report.json",
    "md": "stage178a_frame_absolute_comparability_offset_report.md",
    "rows": "stage178a_row_offset_decomposition.csv",
    "pairs": "stage178a_pair_offset_summary.csv",
    "variance": "stage178a_variance_decomposition.csv",
    "threshold": "stage178a_global_threshold_feasibility.csv",
    "ranking": "stage178a_within_cross_pair_ranking.csv",
    "metrics": "stage178a_raw_centered_metric_comparison.csv",
    "hard39": "stage178a_hard39_offset_attribution.csv",
    "transitions": "stage178a_frame_transition_attribution.csv",
    "numeric": "stage178a_surface_numeric_associations.csv",
    "categorical": "stage178a_categorical_offset_associations.csv",
    "intervention": "stage178a_intervention_residual_summary.csv",
    "stability": "stage178a_baseline_pilot_offset_stability.csv",
}


class AuditBlocked(ValueError):
    """A hard input or semantic-contract failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditBlocked(message)


def _required_int(value: Any, name: str) -> int:
    if value is None or isinstance(value, bool):
        raise AuditBlocked(f"{name} is missing or not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AuditBlocked(f"{name} is not an integer: {value!r}") from error
    return result


def _finite(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise AuditBlocked(f"{name} is missing or not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditBlocked(f"{name} is not numeric: {value!r}") from error
    _require(math.isfinite(result), f"{name} is non-finite")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditBlocked(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as error:
        raise AuditBlocked(f"cannot read CSV {path}: {error}") from error
    _require(bool(rows), f"CSV is empty: {path}")
    return rows


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, ensure_ascii=False,
                               sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                                 if isinstance(value, (dict, list, tuple)) else value
                                 for key, value in row.items()})


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _variance(values: Iterable[float], sample: bool = False) -> float | None:
    items = list(values)
    if len(items) <= int(sample):
        return None
    return statistics.variance(items) if sample else statistics.pvariance(items)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def _distribution(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "mean": _mean(values), "std": math.sqrt(_variance(values) or 0.0),
            "minimum": min(values) if values else None, "p25": _percentile(values, .25),
            "median": _percentile(values, .5), "p75": _percentile(values, .75),
            "maximum": max(values) if values else None}


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + end - 1) / 2 + 1
        for index in order[cursor:end]:
            result[index] = average
        cursor = end
    return result


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _rho_pvalue(rho: float | None, count: int) -> float | None:
    if rho is None or count < 4:
        return None
    clipped = max(-.999999, min(.999999, rho))
    z = abs(math.atanh(clipped)) * math.sqrt(max(1, count - 3))
    return math.erfc(z / math.sqrt(2))


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels); negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = _rank(scores)
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ordered = sorted(zip(scores, labels), reverse=True)
    correct = 0
    return sum((correct := correct + label) / rank for rank, (_, label) in enumerate(ordered, 1) if label) / positives


def _binary_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    labels = [int(row["gold_frame_label"]) for row in rows]
    scores = [float(row[score_key]) for row in rows]
    predictions = [int(score >= 0) for score in scores]
    tp = sum(y == p == 1 for y, p in zip(labels, predictions)); tn = sum(y == p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions)); fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tpr = tp / (tp + fn) if tp + fn else 0.0; tnr = tn / (tn + fp) if tn + fp else 0.0
    return {"rows": len(rows), "auroc": _auc(labels, scores), "average_precision": _average_precision(labels, scores),
            "balanced_accuracy": (tpr + tnr) / 2, "accuracy": (tp + tn) / len(rows) if rows else None,
            "false_compatible_count": fp, "false_incompatible_count": fn,
            "threshold": 0.0, "probability_calibration_metrics_computed": False}


def _bootstrap(values: list[Any], samples: int, seed: int,
               statistic: Callable[[list[Any]], float | None]) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed); estimates = []
    for _ in range(samples):
        estimate = statistic([values[rng.randrange(len(values))] for _ in values])
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return None
    return [float(_percentile(estimates, .025)), float(_percentile(estimates, .975))]


def _validate_stage177e(report: dict[str, Any]) -> dict[str, Any]:
    _require(report.get("decision") == STAGE177E, "Stage177-E decision mismatch")
    final = report.get("final_decision_attribution") or {}
    cohort = report.get("stage176_cohort_attribution") or {}
    parameter = report.get("parameter_delta_audit") or {}
    checks = {
        "changed_rows": _required_int(final.get("changed_rows"), "Stage177-E changed_rows"),
        "recovered_errors": _required_int(final.get("recovered_errors"), "Stage177-E recovered_errors"),
        "introduced_errors": _required_int(final.get("introduced_errors"), "Stage177-E introduced_errors"),
        "cohort_net": _required_int(cohort.get("net_cohort_benefit"), "Stage177-E cohort net"),
        "changed_tensors": _required_int(parameter.get("changed_tensor_count"), "Stage177-E changed tensors"),
        "unchanged_tensors": _required_int(parameter.get("unchanged_tensor_count"), "Stage177-E unchanged tensors"),
        "global_l2_delta": _finite(parameter.get("global_checkpoint_l2_delta"), "Stage177-E global L2"),
    }
    _require((checks["changed_rows"], checks["recovered_errors"], checks["introduced_errors"], checks["cohort_net"]) == (0, 0, 0, 0), "Stage177-E final/cohort evidence mismatch")
    _require((checks["changed_tensors"], checks["unchanged_tensors"]) == (37, 243), "Stage177-E tensor counts mismatch")
    _require(math.isclose(checks["global_l2_delta"], .116048, abs_tol=1e-6), "Stage177-E L2 mismatch")
    _require(report.get("authorized_next_stage") == "STAGE178A_FRAME_ABSOLUTE_COMPARABILITY_OFFSET_AUDIT", "Stage177-E next-stage authorization mismatch")
    return {"status": "passed", "decision": STAGE177E, **checks}


def _validate_stage177e_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    required = {"stable_row_index", "row_id", "pair_id", "baseline_prediction", "pilot_prediction",
                "baseline_support_margin", "pilot_support_margin"}
    _require(not (required - set(rows[0])), f"Stage177-E comparison columns missing: {sorted(required - set(rows[0]))}")
    result = {}
    for line, row in enumerate(rows, 2):
        index = _required_int(row.get("stable_row_index"), f"Stage177-E line {line} index")
        identity = (index, str(row.get("row_id")))
        _require(identity not in result, f"duplicate Stage177-E row identity: {identity}")
        _require(str(row.get("baseline_prediction")) == str(row.get("pilot_prediction")), f"Stage177-E prediction changed at {identity}")
        for field in ("baseline_support_margin", "pilot_support_margin"):
            _finite(row.get(field), f"Stage177-E line {line} {field}")
        result[identity] = row
    _require(len(result) == 720, "Stage177-E comparison must contain 720 dev rows")
    return result


def _validate_topology(records: list[dict[str, Any]], split: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(records):
        _require(row.get("pair_id") is not None, f"{split} row {index} lacks pair_id")
        _require(row.get("frame_compatible_label") is not None, f"{split} row {index} lacks native frame target")
        label = _required_int(row["frame_compatible_label"], f"{split} row {index} frame target")
        _require(label in (0, 1), f"{split} row {index} native frame target is not binary")
        groups[str(row["pair_id"])].append(row)
    malformed = []
    for pair_id, group in groups.items():
        counts = Counter(_required_int(row["frame_compatible_label"], f"{split}.{pair_id}.target") for row in group)
        if len(group) != 12 or counts != Counter({0: 6, 1: 6}):
            malformed.append({"pair_id": pair_id, "rows": len(group), "labels": dict(counts)})
    _require(not malformed, f"{split} contains malformed pairs: {malformed[:3]}")
    intervention_sets = {tuple(sorted(str(row.get("intervention_type")) for row in group)) for group in groups.values()}
    _require(len(intervention_sets) == 1, f"{split} pairs do not share one intervention topology")
    return {"split": split, "pairs": len(groups), "rows": len(records), "rows_per_pair": 12,
            "compatible_rows_per_pair": 6, "incompatible_rows_per_pair": 6,
            "malformed_pairs": 0, "native_target_field": "frame_compatible_label",
            "pair_ids_preserved_as_strings": True, "intervention_topology_equal": True,
            "interventions": list(next(iter(intervention_sets)))}

def _decompose(records, outputs, split, model):
    logits = stage177e._tensor(outputs, "frame_logit", len(records)).reshape(-1).tolist()
    probs = stage177e._tensor(outputs, "frame_prob", len(records)).reshape(-1).tolist()
    groups = defaultdict(list)
    for i, record in enumerate(records): groups[str(record["pair_id"])].append(i)
    row_rows, pair_rows = [], []
    for pair_id, indices in sorted(groups.items()):
        comp = [logits[i] for i in indices if int(records[i]["frame_compatible_label"]) == 1]
        incomp = [logits[i] for i in indices if int(records[i]["frame_compatible_label"]) == 0]
        cm, im = statistics.fmean(comp), statistics.fmean(incomp); center = (cm + im) / 2; gap = cm - im
        category = ("reversed_or_non_separated" if cm <= im else "globally_feasible" if im < 0 < cm
                    else "positive_offset_misaligned" if 0 <= im < cm else "negative_offset_misaligned")
        pair_rows.append({"split": split, "model": model, "pair_id": pair_id, "compatible_mean": cm,
                          "incompatible_mean": im, "pair_center": center, "absolute_pair_center": abs(center),
                          "pair_gap": gap, "threshold_feasibility_category": category})
        total = sum(logits[i] for i in indices)
        for i in indices:
            r = records[i]; loo = (total - logits[i]) / 11
            row_rows.append({"split": split, "model": model, "stable_row_index": i, "row_id": str(r["id"]),
                "pair_id": pair_id, "intervention_type": str(r["intervention_type"]),
                "gold_final_label": str(r["final_label"]), "gold_frame_label": int(r["frame_compatible_label"]),
                "frame_logit": logits[i], "frame_prob": probs[i], "pair_center": center, "loo_pair_center": loo,
                "centered_frame_logit": logits[i] - center, "loo_centered_frame_logit": logits[i] - loo,
                "raw_frame_prediction": int(logits[i] >= 0), "centered_frame_prediction": int(logits[i] - center >= 0),
                "loo_centered_frame_prediction": int(logits[i] - loo >= 0),
                "threshold_feasibility_category": category, "centering_scope": "transductive diagnostic only",
                "authorized_inference_time_mechanism": False})
    return row_rows, pair_rows


def _threshold_summary(pairs):
    counts = Counter(r["threshold_feasibility_category"] for r in pairs); total = len(pairs)
    categories = ("globally_feasible", "positive_offset_misaligned", "negative_offset_misaligned", "reversed_or_non_separated")
    return {"pairs": total, "categories": {k: {"count": counts[k], "rate": counts[k] / total} for k in categories},
            "absolute_pair_center_distribution": _distribution([abs(r["pair_center"]) for r in pairs]),
            "pair_gap_distribution": _distribution([r["pair_gap"] for r in pairs])}


def _variance_summary(rows, pairs, samples, seed):
    scores = [r["frame_logit"] for r in rows]; grand = statistics.fmean(scores); groups = defaultdict(list)
    for r in rows: groups[r["pair_id"]].append(r["frame_logit"])
    total = sum((x - grand) ** 2 for x in scores)
    between = sum(len(v) * (statistics.fmean(v) - grand) ** 2 for v in groups.values())
    within = sum(sum((x - statistics.fmean(v)) ** 2 for x in v) for v in groups.values()); ids = sorted(groups)
    def share(chosen):
        values = [x for p in chosen for x in groups[p]]; mean = statistics.fmean(values)
        t = sum((x - mean) ** 2 for x in values)
        b = sum(len(groups[p]) * (statistics.fmean(groups[p]) - mean) ** 2 for p in chosen)
        return b / t if t else None
    centers = [r["pair_center"] for r in pairs]; gaps = [r["pair_gap"] for r in pairs]
    cs, gs = math.sqrt(_variance(centers) or 0), math.sqrt(_variance(gaps) or 0)
    return {"row_count": len(rows), "pair_count": len(pairs), "total_variance": _variance(scores),
        "pair_center_variance": _variance(centers), "within_pair_residual_variance": within / len(scores),
        "total_sum_of_squares": total, "weighted_between_pair_sum_of_squares": between,
        "weighted_within_pair_sum_of_squares": within, "between_pair_variance_share": between / total if total else None,
        "intraclass_correlation_style_pair_offset_ratio": between / (between + within) if between + within else None,
        "between_pair_variance_share_pair_bootstrap_ci": _bootstrap(ids, samples, seed, share),
        "compatible_pair_mean_variance": _variance(r["compatible_mean"] for r in pairs),
        "incompatible_pair_mean_variance": _variance(r["incompatible_mean"] for r in pairs),
        "pair_gap_variance": _variance(gaps), "pair_center_std_over_pair_gap_std": cs / gs if gs else None,
        "sum_of_squares_identity_error": total - between - within}


def _ranking_summary(rows, samples, seed):
    groups = defaultdict(list)
    for r in rows: groups[r["pair_id"]].append(r)
    ids = sorted(groups); keys = ("frame_logit", "centered_frame_logit", "loo_centered_frame_logit")
    within, source = {}, {k: {} for k in keys}
    for pair_id in ids:
        pos = [r for r in groups[pair_id] if r["gold_frame_label"] == 1]; neg = [r for r in groups[pair_id] if r["gold_frame_label"] == 0]
        within[pair_id] = statistics.fmean(1.0 if a["frame_logit"] > b["frame_logit"] else .5 if a["frame_logit"] == b["frame_logit"] else 0.0 for a in pos for b in neg)
        for key in keys:
            source[key][pair_id] = statistics.fmean(1.0 if a[key] > b[key] else .5 if a[key] == b[key] else 0.0
                for target in ids if target != pair_id for a in pos for b in groups[target] if b["gold_frame_label"] == 0)
    means = {k: statistics.fmean(v.values()) for k, v in source.items()}; raw = means["frame_logit"]
    cis = {k: _bootstrap(ids, samples, seed + n, lambda chosen, key=k: statistics.fmean(source[key][p] for p in chosen)) for n, k in enumerate(keys)}
    return {"within_pair_comparison_ranking_accuracy": statistics.fmean(within.values()),
        "within_pair_pair_normalized_ranking_accuracy": statistics.fmean(within.values()),
        "cross_pair_raw_ranking_accuracy": raw, "cross_pair_centered_ranking_accuracy": means["centered_frame_logit"],
        "cross_pair_loo_centered_ranking_accuracy": means["loo_centered_frame_logit"],
        "cross_pair_pair_normalized": means, "raw_to_centered_delta": means["centered_frame_logit"] - raw,
        "raw_to_loo_centered_delta": means["loo_centered_frame_logit"] - raw,
        "pair_bootstrap_ci": cis, "same_pair_excluded_from_cross_pair": True}


def _metric_views(rows, subset, samples, seed):
    views = {"raw": "frame_logit", "pair_centered": "centered_frame_logit", "loo_pair_centered": "loo_centered_frame_logit"}
    csv_rows, report = [], {}
    groups = defaultdict(list)
    for r in rows: groups[r["pair_id"]].append(r)
    for n, (view, key) in enumerate(views.items()):
        metrics = _binary_metrics(rows, key)
        if subset == "hard_39":
            ids = sorted(groups)
            metrics["pair_bootstrap_auroc_ci"] = _bootstrap(ids, samples, seed + n, lambda chosen: _auc(
                [x["gold_frame_label"] for p in chosen for x in groups[p]], [x[key] for p in chosen for x in groups[p]]))
        csv_rows.append({"subset": subset, "score_view": view, **metrics}); report[view] = metrics
    before = [r["raw_frame_prediction"] == r["gold_frame_label"] for r in rows]
    after = [r["loo_centered_frame_prediction"] == r["gold_frame_label"] for r in rows]
    corrected = sum(not a and b for a, b in zip(before, after)); introduced = sum(a and not b for a, b in zip(before, after))
    report["raw_to_loo_transition"] = {"corrected_rows": corrected, "introduced_rows": introduced, "net_correction": corrected - introduced}
    return csv_rows, report


def _transitions(rows):
    dimensions = ("all", "gold_frame_label", "gold_final_label", "intervention_type", "threshold_feasibility_category")
    output = []
    for dimension in dimensions:
        values = ["all"] if dimension == "all" else sorted({str(r[dimension]) for r in rows})
        for value in values:
            group = rows if dimension == "all" else [r for r in rows if str(r[dimension]) == value]
            counter = Counter()
            for r in group:
                a = r["raw_frame_prediction"] == r["gold_frame_label"]; b = r["loo_centered_frame_prediction"] == r["gold_frame_label"]
                counter[(a, b)] += 1
            output.append({"dimension": dimension, "value": value, "rows": len(group),
                "correct_to_correct": counter[(True, True)], "wrong_to_correct": counter[(False, True)],
                "correct_to_wrong": counter[(True, False)], "wrong_to_wrong": counter[(False, False)],
                "net_correctness": counter[(False, True)] - counter[(True, False)]})
    return output


def _surface_features(records):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    groups = defaultdict(list)
    for r in records: groups[str(r["pair_id"])].append(r)
    result = {}
    word = re.compile(r"\b\w+\b", re.UNICODE); number = re.compile(r"\b\d+(?:[.,]\d+)?\b")
    for pair_id, rows in groups.items():
        features = defaultdict(list)
        for r in rows:
            claim, evidence = str(r["claim"]), str(r["evidence"]); ct, et = claim.split(), evidence.split()
            cs, es = set(x.lower() for x in word.findall(claim)), set(x.lower() for x in word.findall(evidence))
            features["claim_character_length"].append(len(claim)); features["evidence_character_length"].append(len(evidence))
            features["claim_whitespace_token_length"].append(len(ct)); features["evidence_whitespace_token_length"].append(len(et))
            features["tokenizer_wordpiece_length"].append(len(tokenizer(claim, evidence, add_special_tokens=True)["input_ids"]))
            features["claim_evidence_lexical_jaccard_overlap"].append(len(cs & es) / len(cs | es) if cs | es else 0)
            features["claim_evidence_token_count_ratio"].append(len(ct) / len(et) if et else 0)
            features["punctuation_count"].append(sum(not c.isalnum() and not c.isspace() for c in claim + evidence))
            features["numeric_token_count"].append(len(number.findall(claim)) + len(number.findall(evidence)))
            features["capitalized_token_count"].append(sum(x[:1].isupper() for x in ct + et))
        result[pair_id] = {k: statistics.fmean(v) for k, v in features.items()}
    return result


def _numeric_associations(records_by_split, pairs_by_key, samples, seed):
    output = []
    for split, records in records_by_split.items():
        features = _surface_features(records)
        for model in ("baseline", "pilot"):
            pairs = pairs_by_key[(split, model)]; lookup = {r["pair_id"]: r["pair_center"] for r in pairs}; ids = sorted(lookup)
            for n, feature in enumerate(sorted(next(iter(features.values())))):
                x = [features[p][feature] for p in ids]; y = [lookup[p] for p in ids]; rho = _spearman(x, y)
                ci = _bootstrap(ids, samples, seed + n, lambda chosen: _spearman([features[p][feature] for p in chosen], [lookup[p] for p in chosen]))
                output.append({"split": split, "model": model, "feature": feature, "pairs": len(ids), "spearman_rho": rho,
                    "pair_bootstrap_ci_low": ci[0] if ci else None, "pair_bootstrap_ci_high": ci[1] if ci else None,
                    "p_value": _rho_pvalue(rho, len(ids)), "target": "pair_center", "fitted_probe": False})
    ordered = sorted((r for r in output if r["p_value"] is not None), key=lambda r: r["p_value"]); m = len(ordered); running = 1.0
    for rank, row in reversed(list(enumerate(ordered, 1))): running = min(running, row["p_value"] * m / rank); row["bh_corrected_p_value"] = running
    for row in output:
        row.setdefault("bh_corrected_p_value", None)
        peers = {(r["split"], r["model"]): r for r in output if r["feature"] == row["feature"]}
        rho = row["spearman_rho"] or 0; row["train_dev_sign_consistent"] = rho * (peers.get(("dev" if row["split"] == "train" else "train", row["model"]), {}).get("spearman_rho") or 0) > 0
        row["baseline_pilot_sign_consistent"] = rho * (peers.get((row["split"], "pilot" if row["model"] == "baseline" else "baseline"), {}).get("spearman_rho") or 0) > 0
    return output


def _categorical(records_by_split, pairs_by_key):
    output = []
    for split, records in records_by_split.items():
        grouped = defaultdict(list)
        for r in records: grouped[str(r["pair_id"])].append(r)
        metadata = {"original_gold_label_composition": {p: "|".join(f"{k}:{v}" for k, v in sorted(Counter(x["final_label"] for x in rows).items())) for p, rows in grouped.items()},
                    "primary_failure_type_composition": {p: "|".join(sorted({str(x.get("primary_failure_type")) for x in rows})) for p, rows in grouped.items()}}
        for model in ("baseline", "pilot"):
            centers = {r["pair_id"]: r["pair_center"] for r in pairs_by_key[(split, model)]}
            for field, values in metadata.items():
                valid = {g for g, n in Counter(values.values()).items() if n >= 5}; grand = statistics.fmean(centers.values())
                ss_total = sum((x - grand) ** 2 for x in centers.values()); ss_between = sum(sum(1 for v in values.values() if v == g) * (statistics.fmean(centers[p] for p, v in values.items() if v == g) - grand) ** 2 for g in valid)
                effect = ss_between / ss_total if ss_total and len(valid) >= 2 else None
                for group, count in sorted(Counter(values.values()).items()):
                    vals = [centers[p] for p, v in values.items() if v == group]
                    output.append({"split": split, "model": model, "metadata_field": field, "group": group, "group_count": count,
                        "pair_center_mean": statistics.fmean(vals), "pair_center_median": statistics.median(vals),
                        "included_in_effect_size": count >= 5, "eta_squared": effect, "causal_claim": False})
    effect_lookup = {}
    for row in output:
        key = (row["split"], row["model"], row["metadata_field"])
        if row["eta_squared"] is not None:
            effect_lookup[key] = row["eta_squared"]
    for row in output:
        effect = row["eta_squared"]
        other_split = "dev" if row["split"] == "train" else "train"
        other_model = "pilot" if row["model"] == "baseline" else "baseline"
        row["train_dev_consistent"] = effect is not None and effect_lookup.get((other_split, row["model"], row["metadata_field"])) is not None
        row["baseline_pilot_consistent"] = effect is not None and effect_lookup.get((row["split"], other_model, row["metadata_field"])) is not None
    return output


def _intervention(rows_by_key):
    output = []
    for (split, model), rows in rows_by_key.items():
        for family in sorted({r["intervention_type"] for r in rows}):
            group = [r for r in rows if r["intervention_type"] == family]
            output.append({"split": split, "model": model, "intervention_type": family, "rows": len(group),
                "mean_raw_frame_logit": statistics.fmean(r["frame_logit"] for r in group),
                "mean_pair_centered_frame_logit": statistics.fmean(r["centered_frame_logit"] for r in group),
                "frame_error_rate": statistics.fmean(r["raw_frame_prediction"] != r["gold_frame_label"] for r in group),
                "raw_logit_pair_center_correlation": _pearson([r["frame_logit"] for r in group], [r["pair_center"] for r in group]),
                "intervention_is_not_treated_as_offset_cause": True})
    lookup = {(r["split"], r["intervention_type"], r["model"]): r for r in output}
    for r in output:
        other = lookup.get((r["split"], r["intervention_type"], "pilot" if r["model"] == "baseline" else "baseline"))
        r["pilot_minus_baseline_mean_raw_delta"] = (lookup[(r["split"], r["intervention_type"], "pilot")]["mean_raw_frame_logit"] - lookup[(r["split"], r["intervention_type"], "baseline")]["mean_raw_frame_logit"]) if other else None
    return output


def _stability(pairs_by_key):
    output, report = [], {}
    for split in ("train", "dev"):
        b = {r["pair_id"]: r for r in pairs_by_key[(split, "baseline")]}; p = {r["pair_id"]: r for r in pairs_by_key[(split, "pilot")]}; ids = sorted(b)
        bc, pc = [b[x]["pair_center"] for x in ids], [p[x]["pair_center"] for x in ids]; bg, pg = [b[x]["pair_gap"] for x in ids], [p[x]["pair_gap"] for x in ids]
        cd, gd = [y - x for x, y in zip(bc, pc)], [y - x for x, y in zip(bg, pg)]
        summary = {"split": split, "pairs": len(ids), "pair_center_pearson": _pearson(bc, pc), "pair_center_spearman": _spearman(bc, pc),
            "pair_gap_pearson": _pearson(bg, pg), "pair_gap_spearman": _spearman(bg, pg), "mean_absolute_center_delta": statistics.fmean(map(abs, cd)),
            "mean_absolute_gap_delta": statistics.fmean(map(abs, gd)), "center_delta_variance": _variance(cd), "gap_delta_variance": _variance(gd),
            "pilot_change_magnitude_ratio": statistics.fmean(map(abs, gd)) / statistics.fmean(map(abs, cd)) if statistics.fmean(map(abs, cd)) else None,
            "threshold_category_transition_counts": dict(Counter(f"{b[x]['threshold_feasibility_category']}->{p[x]['threshold_feasibility_category']}" for x in ids))}
        output.append(summary); report[split] = summary
    report["interpretation"] = {"pilot_primarily_changes_gap": report["dev"]["mean_absolute_gap_delta"] > report["dev"]["mean_absolute_center_delta"],
        "pilot_removes_center_offset": statistics.fmean(abs(r["pair_center"]) for r in pairs_by_key[("dev", "pilot")]) < statistics.fmean(abs(r["pair_center"]) for r in pairs_by_key[("dev", "baseline")]),
        "offset_stable_across_checkpoints": (report["dev"]["pair_center_spearman"] or 0) >= .7}
    return output, report

def _hard_attribution(baseline_rows, pilot_rows, transitions, stage177_rows):
    b = {(r["stable_row_index"], r["row_id"]): r for r in baseline_rows}; p = {(r["stable_row_index"], r["row_id"]): r for r in pilot_rows}
    output = []
    for old in transitions:
        cohort = stage176b._cohort(old)
        if cohort not in ("beneficial_correction", "harmful_regression"): continue
        key = (int(old["stable_row_index"]), str(old["row_id"])); _require(key in b and key in p and key in stage177_rows, f"hard-39 identity missing: {key}")
        br, pr, context = b[key], p[key], stage177_rows[key]; gold = br["gold_frame_label"]
        raw_ok = br["raw_frame_prediction"] == gold; centered_ok = br["loo_centered_frame_prediction"] == gold
        output.append({"stable_row_index": key[0], "row_id": key[1], "pair_id": br["pair_id"], "intervention_type": br["intervention_type"],
            "gold_final_label": br["gold_final_label"], "gold_frame_label": gold, "stage176_cohort": cohort,
            **{f"baseline_{k}": br[k] for k in ("frame_logit", "frame_prob", "pair_center", "loo_pair_center", "centered_frame_logit", "loo_centered_frame_logit", "raw_frame_prediction", "centered_frame_prediction", "loo_centered_frame_prediction")},
            **{f"pilot_{k}": pr[k] for k in ("frame_logit", "frame_prob", "pair_center", "loo_pair_center", "centered_frame_logit", "loo_centered_frame_logit", "raw_frame_prediction", "centered_frame_prediction", "loo_centered_frame_prediction")},
            "baseline_raw_error_corrected_by_loo_centering": not raw_ok and centered_ok,
            "baseline_correct_raw_damaged_by_loo_centering": raw_ok and not centered_ok,
            "pair_threshold_feasibility_category": br["threshold_feasibility_category"],
            "final_prediction_changed": str(context["baseline_prediction"]) != str(context["pilot_prediction"]),
            "baseline_support_margin_context_only": _finite(context["baseline_support_margin"], "Stage177-E support margin"),
            "pilot_support_margin_context_only": _finite(context["pilot_support_margin"], "Stage177-E support margin")})
    _require(len(output) == 39 and Counter(r["stage176_cohort"] for r in output) == Counter({"beneficial_correction": 25, "harmful_regression": 14}), "hard-39 cohort mismatch")
    return output


def _decision(variance, threshold, ranking, hard_report, numeric, categorical, stability):
    misaligned = sum(threshold["categories"][k]["rate"] for k in ("positive_offset_misaligned", "negative_offset_misaligned"))
    hard_net = hard_report["raw_to_loo_transition"]["net_correction"]; cross_delta = ranking["raw_to_loo_centered_delta"]
    explanation = {"between_pair_variance_share_at_least_0_20": (variance["between_pair_variance_share"] or 0) >= .2,
        "offset_misaligned_dev_pair_rate_at_least_0_20": misaligned >= .2,
        "hard39_loo_net_correction_at_least_3": hard_net >= 3, "cross_pair_loo_improvement_at_least_0_05": cross_delta >= .05}
    strong_numeric = [r for r in numeric if r["split"] == "dev" and r["model"] == "baseline" and abs(r["spearman_rho"] or 0) >= .4
                      and r["pair_bootstrap_ci_low"] is not None and r["pair_bootstrap_ci_low"] * r["pair_bootstrap_ci_high"] > 0
                      and r["train_dev_sign_consistent"] and r["baseline_pilot_sign_consistent"]]
    strong_cat = [r for r in categorical if r["split"] == "dev" and r["model"] == "baseline" and (r["eta_squared"] or 0) >= .14
                  and r["train_dev_consistent"] and r["baseline_pilot_consistent"]
                  and any(x["split"] == "dev" and x["model"] == "pilot" and x["metadata_field"] == r["metadata_field"] and (x["eta_squared"] or 0) >= .14 for x in categorical)]
    count = sum(explanation.values()); stable = stability["interpretation"]["offset_stable_across_checkpoints"]
    meaningful = hard_net >= 3 or cross_delta >= .05
    if count >= 2 and (strong_numeric or strong_cat) and stable:
        decision, next_stage = OBSERVED, "STAGE178B_DATA_OR_SURFACE_OFFSET_LOCALIZATION_AUDIT"
    elif count >= 2 and not (strong_numeric or strong_cat) and stable and meaningful:
        decision, next_stage = LATENT, "STAGE178B_PAIR_INVARIANT_FRAME_REPRESENTATION_FEASIBILITY_AUDIT"
    elif hard_net < 3 and cross_delta < .05:
        decision, next_stage = WEAK, "STAGE179_FRAME_LABEL_SEMANTICS_AND_INPUT_REPRESENTATION_AUDIT"
    else:
        decision, next_stage = BLOCKED, None
    gate = {"decision": decision, "priority_order_applied": [OBSERVED, LATENT, WEAK, BLOCKED], "offset_explanation_conditions": explanation,
            "offset_explanation_condition_count": count, "observed_numeric_associations": [r["feature"] for r in strong_numeric],
            "observed_categorical_associations": sorted({r["metadata_field"] for r in strong_cat}), "pilot_offset_structure_stable": stable,
            "hard39_or_cross_pair_meaningful_improvement": meaningful, "authorized_next_stage": next_stage,
            "architecture_or_training_authorized": False}
    return decision, gate, {"baseline_dev_offset_misaligned_pair_rate": misaligned, "hard39_loo_net_correction": hard_net,
                            "cross_pair_loo_improvement": cross_delta, "observed_association_found": bool(strong_numeric or strong_cat)}


def _render_markdown(report):
    threshold = report["global_threshold_feasibility"]["dev"]["baseline"]; ranking = report["within_cross_pair_ranking"]["dev"]["baseline"]
    hard = report["hard39_offset_attribution"]["baseline_metrics"]; transitions = report["full_dev_transition_attribution"]["baseline"][0]
    stability = report["baseline_pilot_offset_stability"]["interpretation"]
    misaligned = threshold["categories"]["positive_offset_misaligned"]["count"] + threshold["categories"]["negative_offset_misaligned"]["count"]
    return "\n".join(["# Stage178-A frame absolute-comparability offset audit", "", f"**Decision:** `{report['decision']}`", "",
        "## Diagnostic scope", "", "Pair-centering and leave-one-row-out centering are **transductive diagnostics only** and are not authorized inference-time mechanisms. Native `frame_logit` and `frame_prob` are used; centered scores are not calibrated probabilities.", "",
        "## Absolute comparability", "", f"Baseline dev same-pair ranking is {ranking['within_pair_pair_normalized_ranking_accuracy']:.6f}, cross-pair raw ranking is {ranking['cross_pair_raw_ranking_accuracy']:.6f}, and LOO-centered cross-pair ranking is {ranking['cross_pair_loo_centered_ranking_accuracy']:.6f}. This separates strong relative ordering from cross-pair absolute comparability.", "",
        f"The global zero threshold is offset-misaligned for {misaligned} of {threshold['pairs']} dev pairs. Hard-39 LOO centering corrected {hard['raw_to_loo_transition']['corrected_rows']} raw errors, introduced {hard['raw_to_loo_transition']['introduced_rows']}, and produced net {hard['raw_to_loo_transition']['net_correction']}. Full-dev collateral transition has {transitions['wrong_to_correct']} wrong-to-correct and {transitions['correct_to_wrong']} correct-to-wrong rows.", "",
        "## Observed association and checkpoint stability", "", f"Observed surface/construction association meeting the gate: {report['diagnosis']['observed_association_found']}. Pair-center structure stable across baseline and pilot: {stability['offset_stable_across_checkpoints']}. Stage177-C primarily changes the gap: {stability['pilot_primarily_changes_gap']}; it reduces mean absolute center offset: {stability['pilot_removes_center_offset']}.", "",
        "These are observational associations, not causal claims. Intervention families are summarized descriptively and are never used to infer frame targets or asserted as offset causes.", "",
        "## Gate and safety", "", f"Authorized next stage: `{report['stage178b_gate']['authorized_next_stage']}`. Training, architecture changes, pair-centering deployment, offset loss, normalization sweeps, threshold fitting, calibration, probes, external evaluation, time-swap, weight sweeps, and multi-seed expansion remain forbidden.", ""])


def _blocked(output_dir, error, failure_stage):
    output_dir.mkdir(parents=True, exist_ok=True); detail = {"error_type": type(error).__name__, "error": str(error), "failure_stage": failure_stage, "traceback": traceback.format_exc()}
    report = {"stage": STAGE, "decision": BLOCKED, **detail, "scope": {"evaluation_only": True}, "input_validation": {"status": "blocked", **detail},
        "checkpoint_contract": None, "pair_topology": None, "offset_definition": None, "global_threshold_feasibility": None,
        "variance_decomposition": None, "within_cross_pair_ranking": None, "raw_centered_metric_comparison": None,
        "hard39_offset_attribution": None, "full_dev_transition_attribution": None, "observed_nuisance_association": None,
        "intervention_residual_analysis": None, "baseline_pilot_offset_stability": None, "diagnosis": detail,
        "stage178b_gate": {"decision": BLOCKED, "authorized_next_stage": None}, "limitations": ["Validation or analysis failed."],
        "safety_policy": {"training": False, "optimizer": False, "backward": False, "train_mode": False, "external_evaluation": False, "time_swap": False}}
    _write_json(output_dir / OUTPUTS["json"], report); (output_dir / OUTPUTS["md"]).write_text(f"# Stage178-A blocked\n\n`{BLOCKED}`\n\n```text\n{detail['traceback']}\n```\n", encoding="utf-8")
    for key, name in OUTPUTS.items():
        if key not in ("json", "md"): _write_csv(output_dir / name, [])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/controlled_v5_v3_without_time_swap.jsonl"))
    for name in ("stage176a-report", "stage176a-row-transitions", "stage177a-report", "stage177e-report", "stage177e-dev-row-comparison", "baseline-provenance", "baseline-checkpoint", "pilot-provenance", "pilot-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000); parser.add_argument("--bootstrap-seed", type=int, default=178)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv); output_dir = args.output_dir.resolve(); current = "argument_validation"
    try:
        _require(args.device == "cuda" and args.eval_batch_size > 0 and args.bootstrap_samples > 0, "cuda and positive batch/bootstrap counts are required")
        names = ("data", "stage176a_report", "stage176a_row_transitions", "stage177a_report", "stage177e_report", "stage177e_dev_row_comparison", "baseline_provenance", "baseline_checkpoint", "pilot_provenance", "pilot_checkpoint")
        paths = {n: getattr(args, n).resolve() for n in names}; current = "input_path_validation"
        for name, path in paths.items(): _require(path.is_file(), f"{name} does not exist: {path}")
        _require(paths["baseline_checkpoint"] != paths["pilot_checkpoint"], "checkpoints resolve to the same path")
        current = "upstream_report_validation"; stage_reports = stage177e._validate_stage_reports(_read_json(paths["stage176a_report"]), _read_json(paths["stage177a_report"])); closure = _validate_stage177e(_read_json(paths["stage177e_report"])); stage177_rows = _validate_stage177e_rows(_read_csv(paths["stage177e_dev_row_comparison"]))
        transitions = stage177e._validate_stage176_csv_identifiers(_read_csv(paths["stage176a_row_transitions"]))
        current = "provenance_validation"; bprov, pprov = _read_json(paths["baseline_provenance"]), _read_json(paths["pilot_provenance"])
        bpv = stage177e._validate_provenance("baseline", bprov, paths["data"]); ppv = stage177e._validate_provenance("pilot", pprov, paths["data"])
        _require(bpv["data_sha256"] == ppv["data_sha256"], "provenance data hashes differ")
        current = "checkpoint_validation"; bs, bm, bcheck = stage177e._load_and_validate_checkpoint("baseline", paths["baseline_checkpoint"], bprov, "off", 0.0); ps, pm, pcheck = stage177e._load_and_validate_checkpoint("pilot", paths["pilot_checkpoint"], pprov, "pair_softplus", .05)
        current = "deterministic_split_validation"; train, dev, split = stage177e._validate_split(paths["data"], bprov); ptrain, pdev, _ = stage177e._validate_split(paths["data"], pprov)
        _require([r["id"] for r in train] == [r["id"] for r in ptrain] and [r["id"] for r in dev] == [r["id"] for r in pdev], "baseline/pilot deterministic splits differ")
        transitions = stage176b._validate_row_alignment(transitions, dev); train_top, dev_top = _validate_topology(train, "train"), _validate_topology(dev, "dev")
        input_validation = {"status": "passed", "upstream_reports": {**stage_reports, "stage177e": closure}, "stage177e_row_comparison_rows": 720,
            "provenance": {"baseline": bpv, "pilot": ppv}, "checkpoints": {"baseline": bcheck, "pilot": pcheck}, "split": split,
            "completed_before_model_construction_and_forward": True}
        _require(torch.cuda.is_available(), "CUDA is unavailable"); device = torch.device("cuda")
        current = "model_forward"; btrain, ptrain_out = stage177e._evaluate(train, bprov, pprov, bm, pm, bs, ps, device, args.eval_batch_size); bdev, pdev_out = stage177e._evaluate(dev, bprov, pprov, bm, pm, bs, ps, device, args.eval_batch_size)
        current = "offset_analysis"; rows_by, pairs_by = {}, {}
        for split_name, records, outputs in (("train", train, (btrain, ptrain_out)), ("dev", dev, (bdev, pdev_out))):
            for model, out in zip(("baseline", "pilot"), outputs): rows_by[(split_name, model)], pairs_by[(split_name, model)] = _decompose(records, out, split_name, model)
        threshold = {s: {m: _threshold_summary(pairs_by[(s, m)]) for m in ("baseline", "pilot")} for s in ("train", "dev")}
        variance = {s: {m: _variance_summary(rows_by[(s, m)], pairs_by[(s, m)], args.bootstrap_samples, args.bootstrap_seed) for m in ("baseline", "pilot")} for s in ("train", "dev")}
        ranking = {s: {m: _ranking_summary(rows_by[(s, m)], args.bootstrap_samples, args.bootstrap_seed + 100) for m in ("baseline", "pilot")} for s in ("train", "dev")}
        metric_csv, metrics = [], {"full_dev": {}, "hard_39": {}}
        hard_rows = {}
        for model in ("baseline", "pilot"):
            hard_rows[model] = [r for r in rows_by[("dev", model)] if stage176b._cohort(transitions[r["stable_row_index"]]) in ("beneficial_correction", "harmful_regression")]
            csv1, metrics["full_dev"][model] = _metric_views(rows_by[("dev", model)], "full_dev", args.bootstrap_samples, args.bootstrap_seed + 200)
            csv2, metrics["hard_39"][model] = _metric_views(hard_rows[model], "hard_39", args.bootstrap_samples, args.bootstrap_seed + 300); metric_csv += [{"model": model, **r} for r in csv1 + csv2]
        hard_csv = _hard_attribution(rows_by[("dev", "baseline")], rows_by[("dev", "pilot")], transitions, stage177_rows)
        hard_summary = {"rows": 39, "baseline_metrics": metrics["hard_39"]["baseline"], "pilot_metrics": metrics["hard_39"]["pilot"],
            "raw_errors_explained_by_positive_pair_offset": sum(r["baseline_raw_error_corrected_by_loo_centering"] and r["pair_threshold_feasibility_category"] == "positive_offset_misaligned" for r in hard_csv),
            "raw_errors_explained_by_negative_pair_offset": sum(r["baseline_raw_error_corrected_by_loo_centering"] and r["pair_threshold_feasibility_category"] == "negative_offset_misaligned" for r in hard_csv),
            "offset_corrections_without_final_decision_change": sum(r["baseline_raw_error_corrected_by_loo_centering"] and not r["final_prediction_changed"] for r in hard_csv),
            "centered_corrections_not_linked_to_final_support_boundary": sum(r["baseline_raw_error_corrected_by_loo_centering"] and not r["final_prediction_changed"] for r in hard_csv),
            "beneficial_25": {"rows": 25, "corrected": sum(r["baseline_raw_error_corrected_by_loo_centering"] for r in hard_csv if r["stage176_cohort"] == "beneficial_correction"), "introduced": sum(r["baseline_correct_raw_damaged_by_loo_centering"] for r in hard_csv if r["stage176_cohort"] == "beneficial_correction")},
            "harmful_14": {"rows": 14, "corrected": sum(r["baseline_raw_error_corrected_by_loo_centering"] for r in hard_csv if r["stage176_cohort"] == "harmful_regression"), "introduced": sum(r["baseline_correct_raw_damaged_by_loo_centering"] for r in hard_csv if r["stage176_cohort"] == "harmful_regression")}}
        transition_csv, transition_report = [], {}
        for model in ("baseline", "pilot"): transition_report[model] = _transitions(rows_by[("dev", model)]); transition_csv += [{"model": model, **r} for r in transition_report[model]]
        numeric = _numeric_associations({"train": train, "dev": dev}, pairs_by, args.bootstrap_samples, args.bootstrap_seed + 400); categorical = _categorical({"train": train, "dev": dev}, pairs_by)
        intervention = _intervention(rows_by); stability_csv, stability = _stability(pairs_by)
        decision, gate, diagnosis = _decision(variance["dev"]["baseline"], threshold["dev"]["baseline"], ranking["dev"]["baseline"], metrics["hard_39"]["baseline"], numeric, categorical, stability)
        report = {"stage": STAGE, "decision": decision, "scope": {"data": str(paths["data"]), "seed": 174, "epochs": 20, "clean_controlled_data_only": True, "evaluation_only": True, "single_seed": True, "external_or_time_swap": False},
            "input_validation": input_validation, "checkpoint_contract": {"schema": SCHEMA, "model": MODEL_NAME, "baseline": bcheck, "pilot": pcheck, "native_outputs": ["frame_logit", "frame_prob"], "final_classifier_score_substitution": False},
            "pair_topology": {"total_pairs": 300, "train": train_top, "dev": dev_top, "train_dev_pair_overlap": 0},
            "offset_definition": {"pair_center": "(compatible_mean + incompatible_mean) / 2", "pair_gap": "compatible_mean - incompatible_mean", "centered": "frame_logit - pair_center", "loo_centered": "frame_logit - mean(other 11 pair rows)", "scope": "transductive diagnostic only", "authorized_inference_time_mechanism": False},
            "global_threshold_feasibility": threshold, "variance_decomposition": variance, "within_cross_pair_ranking": ranking,
            "raw_centered_metric_comparison": metrics, "hard39_offset_attribution": hard_summary, "full_dev_transition_attribution": transition_report,
            "observed_nuisance_association": {"numeric": numeric, "categorical": categorical, "supervised_probe": False, "regression_fitting": False, "causal_claim": False},
            "intervention_residual_analysis": {"rows": intervention, "intervention_topology_validated": True, "causal_claim": False},
            "baseline_pilot_offset_stability": stability, "diagnosis": diagnosis, "stage178b_gate": gate,
            "limitations": ["Single-seed internal controlled-data audit.", "Pair centering is transductive and not deployable.", "Observed univariate associations are not causal.", "Bootstrap intervals do not capture training-seed uncertainty."],
            "safety_policy": {"training": False, "optimizer": False, "backward": False, "train_mode": False, "calibration": False, "threshold_search": False, "fitted_probe": False, "regression": False, "checkpoint_selection": False, "external_evaluation": False, "external_labels": False, "time_swap": False, "weight_sweep": False, "multi_seed": False, "pair_centering_inference_proposal": False, "architecture_implementation": False}}
        output_dir.mkdir(parents=True, exist_ok=True); _write_json(output_dir / OUTPUTS["json"], report); (output_dir / OUTPUTS["md"]).write_text(_render_markdown(report), encoding="utf-8")
        csv_map = {"rows": [r for v in rows_by.values() for r in v], "pairs": [r for v in pairs_by.values() for r in v],
            "variance": [{"split": s, "model": m, **variance[s][m]} for s in variance for m in variance[s]],
            "threshold": [{"split": s, "model": m, **threshold[s][m]} for s in threshold for m in threshold[s]],
            "ranking": [{"split": s, "model": m, **ranking[s][m]} for s in ranking for m in ranking[s]], "metrics": metric_csv,
            "hard39": hard_csv, "transitions": transition_csv, "numeric": numeric, "categorical": categorical, "intervention": intervention, "stability": stability_csv}
        for key, rows in csv_map.items(): _write_csv(output_dir / OUTPUTS[key], rows)
        return 0
    except Exception as error:
        _blocked(output_dir, error, current); return 2


if __name__ == "__main__":
    raise SystemExit(main())
