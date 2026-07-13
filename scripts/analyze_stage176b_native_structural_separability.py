"""Stage176-B: clean-dev native structural separability audit.

This evaluation-only audit compares the 25 beneficial and 14 harmful
Stage176-A SUPPORT->NOT_ENTITLED transitions using tensors emitted natively by
the two selected models.  It fits no probe, searches no threshold, and never
enters training mode.
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
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import analyze_stage176a_support_boundary_attribution as stage176a  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402


STAGE = "Stage176-B"
IDENTIFIED = "STAGE176B_NATIVE_STRUCTURAL_SEPARABILITY_SIGNAL_IDENTIFIED"
NO_SIGNAL = "STAGE176B_NO_ROBUST_NATIVE_STRUCTURAL_SEPARABILITY_SIGNAL"
BLOCKED_INPUT = "STAGE176B_NATIVE_STRUCTURAL_SEPARABILITY_BLOCKED_INPUT_VALIDATION"
BLOCKED_SIGNALS = "STAGE176B_NATIVE_STRUCTURAL_SEPARABILITY_BLOCKED_INSUFFICIENT_SIGNALS"
STAGE176A_COMPLETE = "STAGE176A_CLEAN_DEV_SUPPORT_BOUNDARY_ATTRIBUTION_COMPLETE"
STAGE176A_CLOSURE = (
    "STAGE176A_GLOBAL_CONSERVATIVE_SHIFT_WITH_USEFUL_FRAME_MISMATCH_CORRECTION_"
    "AND_SUPPORT_COLLATERAL_DAMAGE"
)
CHECKPOINT_SCHEMA = "stage176a0_selected_checkpoint_v1"
EXPECTED = {
    "dev_rows": 720,
    "changed_rows": 39,
    "beneficial_correction": 25,
    "harmful_regression": 14,
    "train_dev_overlap": 0,
}
EXPECTED_BENEFICIAL_FAMILIES = {
    "location_swap": 13,
    "role_swap": 6,
    "title_name_swap": 3,
    "entity_swap": 2,
    "event_swap": 1,
}
EXPECTED_HARMFUL_FAMILIES = {"none": 6, "paraphrase": 1, "polarity_flip": 7}
OUTPUT_NAMES = {
    "json": "stage176b_native_structural_separability_report.json",
    "md": "stage176b_native_structural_separability_report.md",
    "dev": "stage176b_dev_native_scores.csv",
    "changed": "stage176b_changed_row_native_scores.csv",
    "cohorts": "stage176b_cohort_summary.csv",
    "separability": "stage176b_signal_separability.csv",
    "interventions": "stage176b_intervention_cohort_summary.csv",
    "availability": "stage176b_signal_availability.json",
}
REQUIRED_TRANSITION_COLUMNS = {
    "stable_row_index",
    "row_id",
    "pair_id",
    "intervention_type",
    "category",
    "gold_final_label",
    "baseline_predicted_label",
    "treatment_predicted_label",
    "baseline_support_margin",
    "treatment_support_margin",
    "support_margin_delta",
}

# These are static, model-native output keys.  Availability is still checked
# against the actual runtime output dictionaries; absent tensors are never
# synthesized or recovered from final logits.
NATIVE_CANDIDATES = {
    "frame_prob": {"family": "frame_compatibility", "normalized": True, "gate_eligible": True},
    "predicate_coverage_prob": {"family": "predicate_coverage", "normalized": True, "gate_eligible": True},
    "sufficiency_prob": {"family": "sufficiency", "normalized": True, "gate_eligible": True},
    "entitlement_prob": {"family": "entitlement", "normalized": True, "gate_eligible": True},
    "entitlement_for_decision": {"family": "entitlement", "normalized": True, "gate_eligible": True},
    "compositional_entitlement_prob": {"family": "entitlement", "normalized": True, "gate_eligible": True},
    "learned_entitlement_prob": {"family": "entitlement", "normalized": True, "gate_eligible": True},
    "learned_entitlement_logit": {"family": "entitlement", "normalized": False, "gate_eligible": True},
    "polarity_margin": {"family": "polarity", "normalized": False, "gate_eligible": False},
}
PRIMARY_FAMILIES = {"frame_compatibility", "predicate_coverage", "sufficiency", "entitlement"}


class AuditBlocked(ValueError):
    """Raised for a hard validation failure or insufficient native signals."""

    def __init__(self, message: str, decision: str = BLOCKED_INPUT):
        super().__init__(message)
        self.decision = decision


def _require(condition: bool, message: str, decision: str = BLOCKED_INPUT) -> None:
    if not condition:
        raise AuditBlocked(message, decision)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditBlocked(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _get(mapping: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _first(mapping: dict[str, Any], paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        value = _get(mapping, path)
        if value is not None:
            return value
    return default


def _as_int(value: Any, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    return int(value)


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditBlocked(f"{name} is not numeric: {value!r}") from error
    _require(math.isfinite(result), f"{name} is NaN or infinite")
    return result


def _read_transition_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_TRANSITION_COLUMNS - columns)
            _require(not missing, f"row-transition CSV missing columns: {missing}")
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise AuditBlocked(f"cannot read row-transition CSV {path}: {error}") from error
    _require(bool(rows), "row-transition CSV is empty")
    for row_number, row in enumerate(rows, start=2):
        try:
            row["stable_row_index"] = int(row["stable_row_index"])
        except (TypeError, ValueError) as error:
            raise AuditBlocked(f"invalid stable_row_index at CSV line {row_number}") from error
        for key in ("baseline_support_margin", "treatment_support_margin", "support_margin_delta"):
            row[key] = _finite(row[key], f"CSV line {row_number} {key}")
    identities = [(row["stable_row_index"], row["row_id"]) for row in rows]
    _require(len(identities) == len(set(identities)), "duplicate stable row identity in transition CSV")
    return rows


def _validate_stage176a_report(report: dict[str, Any]) -> dict[str, Any]:
    decision = report.get("decision")
    completed = decision == STAGE176A_COMPLETE or (
        decision == STAGE176A_CLOSURE
        and _get(report, "closure.attribution_completion_decision") == STAGE176A_COMPLETE
    )
    _require(completed, f"Stage176-A report is not completed: {decision!r}")
    dev_rows = _first(report, ("split.dev_rows", "scope.dev_rows"))
    overlap = _first(report, ("split.train_dev_overlap", "scope.train_dev_overlap"))
    changed = _first(report, ("overall_transition.support_to_not_entitled",))
    beneficial = _first(
        report,
        (
            "correctness_tradeoff.gold_not_entitled_false_support_removed",
            "correctness_tradeoff.beneficial_correction.rows",
            "correctness_tradeoff.overall.recovered_errors",
        ),
    )
    harmful = _first(
        report,
        (
            "correctness_tradeoff.gold_support_true_support_lost",
            "correctness_tradeoff.harmful_regression.rows",
            "correctness_tradeoff.overall.regressed_errors",
        ),
    )
    checks = {
        "decision": decision,
        "attribution_completion_decision": STAGE176A_COMPLETE,
        "dev_rows": _as_int(dev_rows, "Stage176-A dev rows"),
        "support_to_not_entitled": _as_int(changed, "Stage176-A SUPPORT->NOT_ENTITLED"),
        "beneficial_correction": _as_int(beneficial, "Stage176-A beneficial corrections"),
        "harmful_regression": _as_int(harmful, "Stage176-A harmful regressions"),
        "train_dev_overlap": _as_int(overlap, "Stage176-A train/dev overlap"),
    }
    _require(checks["dev_rows"] == EXPECTED["dev_rows"], "Stage176-A dev row count must be 720")
    _require(checks["support_to_not_entitled"] == EXPECTED["changed_rows"], "Stage176-A changed count must be 39")
    _require(checks["beneficial_correction"] == EXPECTED["beneficial_correction"], "Stage176-A beneficial count must be 25")
    _require(checks["harmful_regression"] == EXPECTED["harmful_regression"], "Stage176-A harmful count must be 14")
    _require(checks["train_dev_overlap"] == EXPECTED["train_dev_overlap"], "Stage176-A train/dev overlap must be zero")
    checks["status"] = "passed"
    return checks


def _cohort(row: dict[str, Any]) -> str:
    gold = row["gold_final_label"]
    baseline = row["baseline_predicted_label"]
    treatment = row["treatment_predicted_label"]
    if gold == "NOT_ENTITLED" and baseline == "SUPPORT" and treatment == "NOT_ENTITLED":
        return "beneficial_correction"
    if gold == "SUPPORT" and baseline == "SUPPORT" and treatment == "NOT_ENTITLED":
        return "harmful_regression"
    if gold == "SUPPORT" and baseline == treatment == "SUPPORT":
        return "stable_gold_support"
    if gold == "NOT_ENTITLED" and baseline == treatment == "NOT_ENTITLED":
        return "stable_gold_not_entitled"
    return "other"


def _validate_transition_counts(rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    changed = [row for row in rows if row["baseline_predicted_label"] != row["treatment_predicted_label"]]
    _require(len(changed) == EXPECTED["changed_rows"], f"expected 39 changed CSV rows, got {len(changed)}")
    _require(
        all(row["baseline_predicted_label"] == "SUPPORT" and row["treatment_predicted_label"] == "NOT_ENTITLED" for row in changed),
        "the 39 changed rows must all be SUPPORT->NOT_ENTITLED",
    )
    counts = Counter(_cohort(row) for row in rows)
    _require(counts["beneficial_correction"] == EXPECTED["beneficial_correction"], "CSV beneficial count must be 25")
    _require(counts["harmful_regression"] == EXPECTED["harmful_regression"], "CSV harmful count must be 14")
    _require(_get(report, "overall_transition.support_to_not_entitled") == len(changed), "CSV changed count disagrees with report")
    return {
        "status": "passed",
        "csv_rows": len(rows),
        "changed_rows": len(changed),
        "beneficial_correction": counts["beneficial_correction"],
        "harmful_regression": counts["harmful_regression"],
    }


def _selected_checkpoint_record(prov: dict[str, Any]) -> dict[str, Any]:
    value = _first(prov, ("finalization.selected_checkpoint", "selected_checkpoint"))
    _require(isinstance(value, dict), "provenance lacks selected-checkpoint record")
    return value


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise AuditBlocked(f"cannot load checkpoint header {path}: {error}") from error
    _require(isinstance(payload, dict), f"checkpoint payload is not a dictionary: {path}")
    _require(payload.get("schema_version") == CHECKPOINT_SCHEMA, f"checkpoint schema must be {CHECKPOINT_SCHEMA}: {path}")
    state = payload.get("model_state_dict")
    metadata = payload.get("metadata")
    _require(isinstance(state, dict) and bool(state), f"checkpoint state is empty: {path}")
    _require(isinstance(metadata, dict), f"checkpoint metadata is missing: {path}")
    _require(all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()), f"invalid state_dict tensors: {path}")
    return state, metadata, payload


def _normalized_label_mapping(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {str(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError):
        return None


def _validate_checkpoint(
    role: str,
    path: Path,
    prov: dict[str, Any],
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    record = _selected_checkpoint_record(prov)
    actual_sha = stage176a._sha256(path)
    _require(record.get("schema_version") == CHECKPOINT_SCHEMA, f"{role} provenance checkpoint schema mismatch")
    _require(record.get("sha256") == actual_sha, f"{role} checkpoint SHA-256 mismatch")
    _require(payload.get("schema_version") == CHECKPOINT_SCHEMA, f"{role} payload schema mismatch")
    selected_epoch = stage176a._selected_epoch(prov)
    _require(selected_epoch == 20 and metadata.get("selected_epoch") == selected_epoch, f"{role} selected epoch mismatch")
    for key in ("architecture", "backbone", "model_name", "seed"):
        expected = stage176a._runtime(prov, key)
        _require(metadata.get(key) == expected, f"{role} checkpoint {key} mismatch")
    final_to_id = _normalized_label_mapping(metadata.get("final_label_to_id"))
    _require(final_to_id == {str(key): int(value) for key, value in v5.FINAL_LABEL_TO_ID.items()}, f"{role} final label mapping mismatch")
    id_to_final = metadata.get("final_id_to_label")
    _require(isinstance(id_to_final, dict), f"{role} inverse label mapping missing")
    normalized_inverse = {int(key): str(value) for key, value in id_to_final.items()}
    _require(normalized_inverse == v5.ID_TO_FINAL_LABEL, f"{role} inverse label mapping mismatch")
    _require(metadata.get("stage174c_clean_pairwise_mode") == "off", f"{role} checkpoint Stage174-C must be off")
    _require(float(metadata.get("stage174c_clean_pairwise_weight", -1.0)) == 0.0, f"{role} checkpoint Stage174-C weight must be zero")
    _require(metadata.get("final_ce_logits_source") in ('output["logits"]', "output['logits']"), f"{role} checkpoint CE source mismatch")
    _require(metadata.get("loss_logits_used_for_final_classifier_ce") is False, f"{role} checkpoint used loss_logits")
    _require(metadata.get("clean_dev_only_checkpoint_selection") is True, f"{role} checkpoint was not clean-dev selected")
    _require(metadata.get("external_data_used") is False and metadata.get("external_labels_used") is False, f"{role} checkpoint used external data")
    _require(metadata.get("time_swap_used") is False, f"{role} checkpoint used time_swap")
    mode = metadata.get("stage175b_support_anchor_mode")
    weight = float(metadata.get("stage175b_support_anchor_weight", 0.0))
    if role == "baseline":
        _require(mode == "off" or weight == 0.0, "baseline checkpoint Stage175-B must be off or zero-weight")
    else:
        _require(mode == "paraphrase_margin", "treatment checkpoint Stage175-B mode mismatch")
        _require(math.isclose(weight, 0.05), "treatment checkpoint Stage175-B weight mismatch")
        _require(math.isclose(float(metadata.get("stage175b_support_anchor_tolerance", -1.0)), 0.10), "treatment checkpoint Stage175-B tolerance mismatch")
    return {
        "status": "passed",
        "path": str(path),
        "sha256": actual_sha,
        "schema_version": CHECKPOINT_SCHEMA,
        "selected_epoch": selected_epoch,
        "label_mapping": "passed",
    }


def _validate_provenance_pair(
    baseline: dict[str, Any], treatment: dict[str, Any], data_path: Path
) -> dict[str, Any]:
    try:
        validation = stage176a._validate_provenances(baseline, treatment, data_path)
    except Exception as error:
        raise AuditBlocked(f"provenance validation failed: {error}") from error
    for role, prov in (("baseline", baseline), ("treatment", treatment)):
        _require(stage176a._selected_epoch(prov) == 20, f"{role} selected epoch must be 20")
        _require(stage176a._runtime(prov, "seed") == 174, f"{role} seed must be 174")
    validation["deterministic_pair_split"] = True
    validation["selected_epoch"] = 20
    return validation


def _resolve_checkpoint(provenance_path: Path, prov: dict[str, Any], explicit: Path | None) -> Path:
    try:
        return stage176a._resolve_checkpoint(provenance_path, prov, explicit)
    except Exception as error:
        raise AuditBlocked(str(error)) from error


def _build_split(data_path: Path, baseline: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = v5.load_jsonl(data_path)
    seed = int(stage176a._runtime(baseline, "seed"))
    dev_ratio = float(stage176a._arg(baseline, "dev_ratio", 0.2))
    train_rows, dev_rows = v5.split_by_pair_id(records, dev_ratio=dev_ratio, seed=seed)
    train_pairs = {str(row["pair_id"]) for row in train_rows}
    dev_pairs = {str(row["pair_id"]) for row in dev_rows}
    summary = {
        "method": "scripts.build_controlled_v5.split_by_pair_id",
        "seed": seed,
        "train_fraction": 1.0 - dev_ratio,
        "total_pair_groups": len(train_pairs | dev_pairs),
        "train_pair_groups": len(train_pairs),
        "dev_pair_groups": len(dev_pairs),
        "dev_rows": len(dev_rows),
        "train_dev_overlap": len(train_pairs & dev_pairs),
    }
    _require(summary["total_pair_groups"] == 300, "split must contain 300 pair groups")
    _require(summary["train_pair_groups"] == 240, "split must contain 240 train pairs")
    _require(summary["dev_pair_groups"] == 60, "split must contain 60 dev pairs")
    _require(summary["dev_rows"] == EXPECTED["dev_rows"], "split must contain 720 dev rows")
    _require(summary["train_dev_overlap"] == 0, "train/dev pair overlap detected")
    return train_rows, dev_rows, summary


def _validate_row_alignment(csv_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _require(len(csv_rows) == len(dev_rows), "transition CSV must contain all 720 dev rows")
    by_index = {int(row["stable_row_index"]): row for row in csv_rows}
    _require(set(by_index) == set(range(len(dev_rows))), "transition CSV stable indexes are not exactly 0..719")
    ordered: list[dict[str, Any]] = []
    for index, record in enumerate(dev_rows):
        row = by_index[index]
        expected_id = str(record.get("id", f"dev_row_{index}"))
        _require(str(row["row_id"]) == expected_id, f"row identity mismatch at dev index {index}")
        for csv_key, data_key in (
            ("pair_id", "pair_id"),
            ("intervention_type", "intervention_type"),
            ("gold_final_label", "final_label"),
        ):
            _require(str(row[csv_key]) == str(record.get(data_key)), f"{csv_key} mismatch at dev index {index}")
        ordered.append(row)
    return ordered


def _tensor_vector(output: dict[str, Any], key: str, row_count: int) -> list[float]:
    value = output[key]
    _require(torch.is_tensor(value), f"native output {key} is not a tensor", BLOCKED_SIGNALS)
    tensor = value.detach().float().cpu().reshape(-1)
    _require(tensor.numel() == row_count, f"native output {key} row count mismatch", BLOCKED_SIGNALS)
    values = [float(item) for item in tensor.tolist()]
    _require(all(math.isfinite(item) for item in values), f"native output {key} contains NaN/Infinity", BLOCKED_SIGNALS)
    return values


def _discover_signals(
    baseline_output: dict[str, Any], treatment_output: dict[str, Any], row_count: int
) -> tuple[dict[str, dict[str, Any]], dict[str, list[float]], dict[str, list[float]]]:
    availability: dict[str, dict[str, Any]] = {}
    baseline_values: dict[str, list[float]] = {}
    treatment_values: dict[str, list[float]] = {}
    for key, spec in NATIVE_CANDIDATES.items():
        in_baseline = key in baseline_output and baseline_output.get(key) is not None
        in_treatment = key in treatment_output and treatment_output.get(key) is not None
        available = in_baseline and in_treatment
        availability[key] = {
            **spec,
            "source_key": key,
            "normalization": "identity" if available else None,
            "baseline_present": in_baseline,
            "treatment_present": in_treatment,
            "available": available,
        }
        if available:
            baseline_values[key] = _tensor_vector(baseline_output, key, row_count)
            treatment_values[key] = _tensor_vector(treatment_output, key, row_count)
            if spec["normalized"]:
                _require(all(0.0 <= item <= 1.0 for item in baseline_values[key] + treatment_values[key]), f"normalized signal {key} is outside [0,1]", BLOCKED_SIGNALS)
    available_families = {availability[key]["family"] for key in baseline_values}
    primary_count = len(available_families & PRIMARY_FAMILIES)
    _require(primary_count >= 3, f"only {primary_count} primary native structural signal families are available", BLOCKED_SIGNALS)
    return availability, baseline_values, treatment_values


def _add_composites(
    availability: dict[str, dict[str, Any]],
    baseline_values: dict[str, list[float]],
    treatment_values: dict[str, list[float]],
) -> None:
    components = ("frame_prob", "predicate_coverage_prob", "sufficiency_prob")
    if not all(key in baseline_values for key in components):
        return
    for side in (baseline_values, treatment_values):
        triples = zip(*(side[key] for key in components))
        rows = list(triples)
        side["structural_minimum"] = [min(values) for values in rows]
        side["structural_product"] = [math.prod(values) for values in rows]
        if "entitlement_prob" in side:
            side["entitlement_conditioned_product"] = [
                product * entitlement
                for product, entitlement in zip(side["structural_product"], side["entitlement_prob"])
            ]
    availability["structural_minimum"] = {"family": "fixed_composite", "normalized": True, "gate_eligible": True, "source_key": None, "normalization": "fixed_minimum", "available": True, "components": list(components)}
    availability["structural_product"] = {"family": "fixed_composite", "normalized": True, "gate_eligible": True, "source_key": None, "normalization": "fixed_product", "available": True, "components": list(components)}
    if "entitlement_conditioned_product" in baseline_values:
        availability["entitlement_conditioned_product"] = {"family": "fixed_composite", "normalized": True, "gate_eligible": True, "source_key": None, "normalization": "fixed_product", "available": True, "components": [*components, "entitlement_prob"]}


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


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


def _auc_and_u(positive: list[float], negative: list[float]) -> tuple[float, float]:
    _require(bool(positive) and bool(negative), "both primary cohorts are required")
    ranks = _rankdata(positive + negative)
    n_positive = len(positive)
    u = sum(ranks[:n_positive]) - n_positive * (n_positive + 1) / 2.0
    auc = u / (n_positive * len(negative))
    return auc, u


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(
    positive: list[float], negative: list[float], samples: int, seed: int
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    adjusted_aucs: list[float] = []
    median_differences: list[float] = []
    for _ in range(samples):
        pos = [positive[rng.randrange(len(positive))] for _ in positive]
        neg = [negative[rng.randrange(len(negative))] for _ in negative]
        auc, _u = _auc_and_u(pos, neg)
        adjusted_aucs.append(max(auc, 1.0 - auc))
        median_differences.append(float(statistics.median(pos) - statistics.median(neg)))
    return {"adjusted_auc": adjusted_aucs, "median_difference": median_differences}


def _separability_record(
    signal: str,
    view: str,
    positive: list[float],
    negative: list[float],
    samples: int,
    seed: int,
    availability: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    auc, u = _auc_and_u(positive, negative)
    adjusted = max(auc, 1.0 - auc)
    orientation = "beneficial_associated_with_higher_values" if auc >= 0.5 else "beneficial_associated_with_lower_values"
    cliff_delta = 2.0 * auc - 1.0
    median_difference = float(statistics.median(positive) - statistics.median(negative))
    boot = _bootstrap(positive, negative, samples, seed)
    auc_ci = [_percentile(boot["adjusted_auc"], 0.025), _percentile(boot["adjusted_auc"], 0.975)]
    median_ci = [_percentile(boot["median_difference"], 0.025), _percentile(boot["median_difference"], 0.975)]
    spec = availability[signal]
    robust = (
        bool(spec.get("gate_eligible"))
        and view in {"baseline", "treatment"}
        and adjusted >= 0.75
        and auc_ci[0] > 0.50
        and abs(cliff_delta) >= 0.50
    )
    return {
        "signal": signal,
        "signal_family": spec["family"],
        "source": "fixed_composite" if spec["family"] == "fixed_composite" else "native_model_output",
        "view": view,
        "positive_cohort": "beneficial_correction",
        "positive_count": len(positive),
        "negative_cohort": "harmful_regression",
        "negative_count": len(negative),
        "raw_auroc": auc,
        "orientation": orientation,
        "direction_adjusted_auroc": adjusted,
        "beneficial_mean": _mean(positive),
        "beneficial_median": _median(positive),
        "harmful_mean": _mean(negative),
        "harmful_median": _median(negative),
        "median_difference_beneficial_minus_harmful": median_difference,
        "mann_whitney_u": u,
        "rank_biserial_correlation": cliff_delta,
        "cliffs_delta": cliff_delta,
        "bootstrap_direction_adjusted_auroc_ci95_lower": auc_ci[0],
        "bootstrap_direction_adjusted_auroc_ci95_upper": auc_ci[1],
        "bootstrap_median_difference_ci95_lower": median_ci[0],
        "bootstrap_median_difference_ci95_upper": median_ci[1],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "robust_stage177_gate_signal": robust,
    }


def _build_scored_rows(
    transition_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    baseline_values: dict[str, list[float]],
    treatment_values: dict[str, list[float]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, (transition, record) in enumerate(zip(transition_rows, dev_rows)):
        row: dict[str, Any] = {
            "stable_row_index": index,
            "row_id": transition["row_id"],
            "pair_id": transition["pair_id"],
            "intervention_type": transition["intervention_type"],
            "stage176a_category": transition["category"],
            "cohort": _cohort(transition),
            "gold_final_label": transition["gold_final_label"],
            "gold_frame_label": record.get("frame_compatible_label"),
            "gold_predicate_label": record.get("predicate_covered_label"),
            "gold_sufficiency_label": record.get("sufficiency_label"),
            "polarity_label": record.get("polarity_label"),
            "baseline_predicted_label": transition["baseline_predicted_label"],
            "treatment_predicted_label": transition["treatment_predicted_label"],
            "baseline_support_margin": transition["baseline_support_margin"],
            "treatment_support_margin": transition["treatment_support_margin"],
            "support_margin_delta": transition["support_margin_delta"],
        }
        for signal in baseline_values:
            baseline = baseline_values[signal][index]
            treatment = treatment_values[signal][index]
            row[f"baseline_{signal}"] = baseline
            row[f"treatment_{signal}"] = treatment
            row[f"delta_{signal}"] = treatment - baseline
        result.append(row)
    return result


def _signal_separability(
    rows: list[dict[str, Any]],
    signals: list[str],
    availability: dict[str, dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    beneficial = [row for row in rows if row["cohort"] == "beneficial_correction"]
    harmful = [row for row in rows if row["cohort"] == "harmful_regression"]
    _require(len(beneficial) == 25 and len(harmful) == 14, "primary cohort count mismatch")
    output: list[dict[str, Any]] = []
    for signal_index, signal in enumerate(signals):
        for view_index, (view, prefix) in enumerate((
            ("baseline", "baseline_"),
            ("treatment", "treatment_"),
            ("treatment_minus_baseline", "delta_"),
        )):
            positive = [_finite(row[f"{prefix}{signal}"], f"{signal} beneficial") for row in beneficial]
            negative = [_finite(row[f"{prefix}{signal}"], f"{signal} harmful") for row in harmful]
            output.append(_separability_record(signal, view, positive, negative, samples, seed + signal_index * 3 + view_index, availability))
    # Final SUPPORT margin is context-only and is explicitly ineligible for the gate.
    context_availability = {"support_margin": {"family": "final_logit_context", "gate_eligible": False}}
    for view_index, (view, key) in enumerate((
        ("baseline", "baseline_support_margin"),
        ("treatment", "treatment_support_margin"),
        ("treatment_minus_baseline", "support_margin_delta"),
    )):
        positive = [_finite(row[key], f"support margin beneficial {view}") for row in beneficial]
        negative = [_finite(row[key], f"support margin harmful {view}") for row in harmful]
        output.append(_separability_record("support_margin", view, positive, negative, samples, seed + 10000 + view_index, context_availability))
    return output


def _cohort_summary(rows: list[dict[str, Any]], signals: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cohort_order = ("beneficial_correction", "harmful_regression", "stable_gold_support", "stable_gold_not_entitled", "other")
    for cohort in cohort_order:
        group = [row for row in rows if row["cohort"] == cohort]
        item: dict[str, Any] = {"cohort": cohort, "row_count": len(group)}
        for signal in signals:
            for view, prefix in (("baseline", "baseline_"), ("treatment", "treatment_"), ("delta", "delta_")):
                values = [float(row[f"{prefix}{signal}"]) for row in group]
                item[f"{view}_{signal}_mean"] = _mean(values)
                item[f"{view}_{signal}_median"] = _median(values)
        output.append(item)
    return output


def _intervention_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = [row for row in rows if row["cohort"] in {"beneficial_correction", "harmful_regression"}]
    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        key = (
            row["cohort"], row["intervention_type"], row["stage176a_category"],
            str(row["polarity_label"]), str(row["gold_frame_label"]),
            str(row["gold_predicate_label"]), str(row["gold_sufficiency_label"]),
        )
        grouped[key].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        result.append({
            "cohort": key[0], "intervention_type": key[1], "stage176a_category": key[2],
            "polarity_label": key[3], "gold_frame_label": key[4],
            "gold_predicate_label": key[5], "gold_sufficiency_label": key[6],
            "row_count": len(group),
            "pair_ids": sorted({row["pair_id"] for row in group}),
        })
    family_counts = {
        cohort: dict(sorted(Counter(row["intervention_type"] for row in primary if row["cohort"] == cohort).items()))
        for cohort in ("beneficial_correction", "harmful_regression")
    }
    _require(family_counts["beneficial_correction"] == EXPECTED_BENEFICIAL_FAMILIES, f"beneficial intervention counts mismatch: {family_counts['beneficial_correction']}")
    _require(family_counts["harmful_regression"] == EXPECTED_HARMFUL_FAMILIES, f"harmful intervention counts mismatch: {family_counts['harmful_regression']}")
    return result


def _fixed_threshold_summary(rows: list[dict[str, Any]], signals: list[str]) -> dict[str, Any]:
    threshold_signals = [key for key in ("frame_prob", "predicate_coverage_prob", "sufficiency_prob", "entitlement_prob", "structural_minimum") if key in signals]
    result: dict[str, Any] = {"threshold": 0.5, "purpose": "descriptive_count_only", "cohorts": {}}
    for cohort in ("beneficial_correction", "harmful_regression"):
        group = [row for row in rows if row["cohort"] == cohort]
        result["cohorts"][cohort] = {
            signal: {
                "baseline_below_0_5": sum(float(row[f"baseline_{signal}"]) < 0.5 for row in group),
                "treatment_below_0_5": sum(float(row[f"treatment_{signal}"]) < 0.5 for row in group),
            }
            for signal in threshold_signals
        }
    return result


def _native_score_summary(rows: list[dict[str, Any]], signals: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for signal in signals:
        result[signal] = {}
        for view, prefix in (("baseline", "baseline_"), ("treatment", "treatment_"), ("treatment_minus_baseline", "delta_")):
            values = [float(row[f"{prefix}{signal}"]) for row in rows]
            result[signal][view] = {"mean": _mean(values), "median": _median(values), "minimum": min(values), "maximum": max(values)}
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _render_markdown(report: dict[str, Any]) -> str:
    gate = report["stage177_gate"]
    counts = report["cohort_counts"]
    available = [name for name, value in report["signal_availability"]["signals"].items() if value.get("available")]
    lines = [
        "# Stage176-B native structural separability audit",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        "## Scope",
        "",
        "Clean-dev-only, evaluation-only audit of native model outputs. No training, fitted probe, threshold search, calibration, checkpoint selection, external evaluation, external labels, or time-swap data was used.",
        "",
        f"Primary cohorts: {counts['beneficial_correction']} beneficial corrections and {counts['harmful_regression']} harmful regressions.",
        "",
        "## Native signals",
        "",
        ", ".join(f"`{name}`" for name in available),
        "",
        "All native signals use their exact model-output key with identity normalization. Fixed composites use only the predeclared minimum/product formulas. No structural score was reconstructed from `output[\"logits\"]`.",
        "",
        "## Stage177 gate",
        "",
        f"Robust qualifying signals: {len(gate['qualifying_signals'])}. Recommendation: {gate['recommendation']}",
        "",
    ]
    if gate["qualifying_signals"]:
        lines.extend(["| Signal | View | Adjusted AUROC | CI lower | |effect| |", "|---|---|---:|---:|---:|"])
        for item in gate["qualifying_signals"]:
            lines.append(f"| {item['signal']} | {item['view']} | {item['direction_adjusted_auroc']:.6f} | {item['bootstrap_direction_adjusted_auroc_ci95_lower']:.6f} | {abs(item['rank_biserial_correlation']):.6f} |")
        lines.append("")
    lines.extend([
        "Final SUPPORT margin is reported only as context and cannot satisfy the gate. Multiple qualifying signals, if any, do not authorize model selection or threshold tuning in this stage.",
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def _empty_outputs(output_dir: Path, decision: str, error: Exception) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": STAGE,
        "decision": decision,
        "scope": {"diagnostic_only": True, "model_forward_executed": False},
        "input_validation": {"status": "blocked", "error": f"{type(error).__name__}: {error}"},
        "checkpoint_validation": None,
        "signal_availability": None,
        "cohort_definition": None,
        "cohort_counts": None,
        "native_score_summary": None,
        "signal_separability": None,
        "intervention_context": None,
        "fixed_threshold_summary": None,
        "stage177_gate": {"entered": False, "recommendation": "blocked"},
        "limitations": ["Hard validation failed; no separability result was computed."],
        "safety_policy": {"training": False, "optimizer": False, "backward": False, "model_forward": False, "threshold_search": False, "external_evaluation": False, "time_swap": False},
    }
    _write_json(output_dir / OUTPUT_NAMES["json"], report)
    (output_dir / OUTPUT_NAMES["md"]).write_text(f"# Stage176-B blocked\n\n**Decision:** `{decision}`\n\nValidation stopped before model forward: `{type(error).__name__}: {error}`\n", encoding="utf-8")
    _write_json(output_dir / OUTPUT_NAMES["availability"], {"stage": STAGE, "decision": decision, "status": "blocked", "signals": {}, "error": str(error)})
    for key in ("dev", "changed", "cohorts", "separability", "interventions"):
        _write_csv(output_dir / OUTPUT_NAMES[key], [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/controlled_v5_v3_without_time_swap.jsonl"))
    parser.add_argument("--stage176a-report", type=Path, required=True)
    parser.add_argument("--stage176a-row-transitions", type=Path, required=True)
    parser.add_argument("--baseline-provenance", type=Path, required=True)
    parser.add_argument("--treatment-provenance", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True, default="cuda")
    parser.add_argument("--eval-batch-size", required=True, type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=176)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    forward_started = False
    try:
        _require(args.device == "cuda", "Stage176-B requires --device cuda")
        _require(args.eval_batch_size > 0, "--eval-batch-size must be positive")
        _require(args.bootstrap_samples > 0, "--bootstrap-samples must be positive")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        paths = {
            "data": args.data.resolve(),
            "stage176a_report": args.stage176a_report.resolve(),
            "stage176a_row_transitions": args.stage176a_row_transitions.resolve(),
            "baseline_provenance": args.baseline_provenance.resolve(),
            "treatment_provenance": args.treatment_provenance.resolve(),
        }
        for name, path in paths.items():
            _require(path.is_file(), f"{name} file does not exist: {path}")

        stage176a_report = _read_json(paths["stage176a_report"])
        report_validation = _validate_stage176a_report(stage176a_report)
        transition_rows = _read_transition_csv(paths["stage176a_row_transitions"])
        transition_validation = _validate_transition_counts(transition_rows, stage176a_report)
        baseline_prov = _read_json(paths["baseline_provenance"])
        treatment_prov = _read_json(paths["treatment_provenance"])
        provenance_validation = _validate_provenance_pair(baseline_prov, treatment_prov, paths["data"])

        baseline_checkpoint = _resolve_checkpoint(paths["baseline_provenance"], baseline_prov, args.baseline_checkpoint.resolve())
        treatment_checkpoint = _resolve_checkpoint(paths["treatment_provenance"], treatment_prov, args.treatment_checkpoint.resolve())
        _require(baseline_checkpoint != treatment_checkpoint, "baseline and treatment checkpoints are identical")
        baseline_state, baseline_metadata, baseline_payload = _load_checkpoint(baseline_checkpoint)
        treatment_state, treatment_metadata, treatment_payload = _load_checkpoint(treatment_checkpoint)
        baseline_checkpoint_validation = _validate_checkpoint("baseline", baseline_checkpoint, baseline_prov, baseline_metadata, baseline_payload)
        treatment_checkpoint_validation = _validate_checkpoint("treatment", treatment_checkpoint, treatment_prov, treatment_metadata, treatment_payload)

        _train_rows, dev_rows, split = _build_split(paths["data"], baseline_prov)
        transition_rows = _validate_row_alignment(transition_rows, dev_rows)
        input_validation = {
            "status": "passed",
            "stage176a_report": report_validation,
            "row_transitions": transition_validation,
            "provenance": provenance_validation,
            "split": split,
            "row_identity_one_to_one": True,
            "completed_before_model_construction_and_forward": True,
        }
        checkpoint_validation = {
            "status": "passed",
            "baseline": baseline_checkpoint_validation,
            "treatment": treatment_checkpoint_validation,
            "same_seed_architecture_backbone_model_data_and_split": True,
            "completed_before_model_construction_and_forward": True,
        }

        # No model is constructed until every input, provenance, checkpoint,
        # split, count, and row-identity check above has passed.
        from transformers import AutoTokenizer

        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(str(stage176a._runtime(baseline_prov, "model_name")))
        if tokenizer.pad_token_id is None:
            _require(tokenizer.eos_token_id is not None, "Mamba tokenizer has no pad/eos token")
            tokenizer.pad_token = tokenizer.eos_token
        max_length = int(stage176a._arg(baseline_prov, "max_length", baseline_metadata.get("max_length", 128)))
        treatment_max_length = int(stage176a._arg(treatment_prov, "max_length", treatment_metadata.get("max_length", 128)))
        _require(max_length == treatment_max_length, "baseline/treatment max_length mismatch")
        bundle = v5.encode_mamba_records(dev_rows, tokenizer, max_length)
        dev_inputs = v5.move_inputs(bundle["model_inputs"], device)

        baseline_model = stage176a._construct_model(baseline_prov, baseline_metadata, baseline_state, device)
        treatment_model = stage176a._construct_model(treatment_prov, treatment_metadata, treatment_state, device)
        _require(not baseline_model.training and not treatment_model.training, "models must remain in eval mode")
        forward_started = True
        baseline_output = stage176a._forward(baseline_model, dev_inputs, dev_rows, baseline_prov, device, args.eval_batch_size)
        treatment_output = stage176a._forward(treatment_model, dev_inputs, dev_rows, treatment_prov, device, args.eval_batch_size)

        availability, baseline_values, treatment_values = _discover_signals(baseline_output, treatment_output, len(dev_rows))
        _add_composites(availability, baseline_values, treatment_values)
        signals = list(baseline_values)
        scored_rows = _build_scored_rows(transition_rows, dev_rows, baseline_values, treatment_values)
        cohort_counts = dict(sorted(Counter(row["cohort"] for row in scored_rows).items()))
        _require(cohort_counts.get("beneficial_correction") == 25, "scored beneficial cohort must contain 25 rows")
        _require(cohort_counts.get("harmful_regression") == 14, "scored harmful cohort must contain 14 rows")
        intervention_rows = _intervention_summary(scored_rows)
        separability = _signal_separability(scored_rows, signals, availability, args.bootstrap_samples, args.bootstrap_seed)
        qualifying = [row for row in separability if row["robust_stage177_gate_signal"]]
        decision = IDENTIFIED if qualifying else NO_SIGNAL
        recommendation = (
            "Proceed only to a separate Stage177-A clean-only asymmetric-gating design audit; do not select an exact threshold here."
            if qualifying
            else "Terminate the Stage175/176 final-logit auxiliary path; do not run weight/tolerance sweeps or external tuning."
        )
        availability_report = {
            "stage": STAGE,
            "decision": decision,
            "minimum_primary_family_requirement": 3,
            "available_primary_families": sorted({value["family"] for value in availability.values() if value.get("available")} & PRIMARY_FAMILIES),
            "signals": availability,
            "logit_derived_structural_signals": False,
        }
        report = {
            "stage": STAGE,
            "decision": decision,
            "scope": {
                "data": str(paths["data"]),
                "clean_dev_only": True,
                "diagnostic_only": True,
                "device": str(device),
                "eval_batch_size": args.eval_batch_size,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
                "final_support_margin_role": "context_only",
            },
            "input_validation": input_validation,
            "checkpoint_validation": checkpoint_validation,
            "signal_availability": availability_report,
            "cohort_definition": {
                "positive": {"name": "beneficial_correction", "gold": "NOT_ENTITLED", "baseline_prediction": "SUPPORT", "treatment_prediction": "NOT_ENTITLED"},
                "negative": {"name": "harmful_regression", "gold": "SUPPORT", "baseline_prediction": "SUPPORT", "treatment_prediction": "NOT_ENTITLED"},
                "controls": ["stable_gold_support", "stable_gold_not_entitled", "other"],
            },
            "cohort_counts": cohort_counts,
            "native_score_summary": _native_score_summary(scored_rows, signals),
            "signal_separability": separability,
            "intervention_context": {
                "expected_beneficial_family_counts": EXPECTED_BENEFICIAL_FAMILIES,
                "expected_harmful_family_counts": EXPECTED_HARMFUL_FAMILIES,
                "validation": "passed",
            },
            "fixed_threshold_summary": _fixed_threshold_summary(scored_rows, signals),
            "stage177_gate": {
                "criteria": {"direction_adjusted_auroc_minimum": 0.75, "bootstrap_ci95_lower_strictly_greater_than": 0.50, "absolute_rank_biserial_or_cliffs_delta_minimum": 0.50, "allowed_views": ["baseline", "treatment"], "final_support_margin_disallowed": True},
                "qualifying_signals": qualifying,
                "entered": bool(qualifying),
                "recommendation": recommendation,
                "multiple_signals_do_not_authorize_model_selection_or_threshold_tuning": True,
            },
            "limitations": [
                "Single-seed observational comparison of two selected checkpoints on internal clean dev.",
                "The gate is evidence for Stage177 design entry, not causal proof or a deployable decision rule.",
                "Bootstrap intervals use cohort-stratified row resampling and do not model pair-level or training-seed uncertainty.",
                "The fixed 0.5 counts are descriptive and were not used to choose a threshold.",
            ],
            "safety_policy": {
                "clean_dev_only": True,
                "training": False,
                "optimizer_created": False,
                "backward": False,
                "train_mode_called": False,
                "learned_probe": False,
                "classifier_fitting": False,
                "threshold_search": False,
                "calibration": False,
                "checkpoint_selection": False,
                "external_evaluation": False,
                "external_labels": False,
                "time_swap": False,
                "architecture_modified": False,
                "stage175_loss_modified": False,
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / OUTPUT_NAMES["json"], report)
        (output_dir / OUTPUT_NAMES["md"]).write_text(_render_markdown(report), encoding="utf-8")
        _write_csv(output_dir / OUTPUT_NAMES["dev"], scored_rows)
        _write_csv(output_dir / OUTPUT_NAMES["changed"], [row for row in scored_rows if row["cohort"] in {"beneficial_correction", "harmful_regression"}])
        _write_csv(output_dir / OUTPUT_NAMES["cohorts"], _cohort_summary(scored_rows, signals))
        _write_csv(output_dir / OUTPUT_NAMES["separability"], separability)
        _write_csv(output_dir / OUTPUT_NAMES["interventions"], intervention_rows)
        _write_json(output_dir / OUTPUT_NAMES["availability"], availability_report)
        print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except AuditBlocked as error:
        if forward_started and error.decision == BLOCKED_INPUT:
            error.decision = BLOCKED_SIGNALS
        _empty_outputs(output_dir, error.decision, error)
        print(json.dumps({"decision": error.decision, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
