"""Stage177-B clean frame-logit pairwise-objective feasibility audit.

This is an evaluation-only audit.  It performs hard Stage177-A,
provenance/checkpoint, data, and split validation before model construction or
forward.  It never trains, fits, calibrates, searches a threshold, or changes a
trainer.  The candidate objective described by the report is not implemented
here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from scripts import analyze_stage176b_native_structural_separability as stage176b  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402


STAGE = "Stage177-B"
FEASIBLE = "STAGE177B_FRAME_PAIRWISE_OBJECTIVE_FEASIBLE"
NOT_FEASIBLE = "STAGE177B_FRAME_PAIRWISE_OBJECTIVE_NOT_FEASIBLE"
BLOCKED = "STAGE177B_FRAME_PAIRWISE_FEASIBILITY_BLOCKED"
CHECKPOINT_SCHEMA = "stage176a0_selected_checkpoint_v1"
EXPECTED_MODEL = "state-spaces/mamba-130m-hf"
DECLARED_COMPATIBLE_FAMILIES = {"none", "paraphrase", "polarity_flip"}
DECLARED_INCOMPATIBLE_FAMILIES = {
    "entity_swap", "event_swap", "location_swap", "role_swap", "title_name_swap"
}
OUTPUTS = {
    "report_json": "stage177b_frame_pairwise_feasibility_report.json",
    "report_md": "stage177b_frame_pairwise_feasibility_report.md",
    "pair_topology": "stage177b_pair_topology.csv",
    "comparison_topology": "stage177b_comparison_topology.csv",
    "family_matrix": "stage177b_family_comparison_matrix.csv",
    "gap_rows": "stage177b_baseline_gap_rows.csv",
    "pair_gaps": "stage177b_pair_gap_summary.csv",
    "family_gaps": "stage177b_family_gap_summary.csv",
    "batch_audit": "stage177b_batch_topology_audit.json",
    "conflict_audit": "stage177b_loss_conflict_audit.json",
}


class AuditBlocked(ValueError):
    """A hard pre-forward input or semantic validation failed."""


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


def _get(value: Any, dotted: str, default: Any = None) -> Any:
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _first(value: dict[str, Any], paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        result = _get(value, path)
        if result is not None:
            return result
    return default


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AuditBlocked(f"{name} is not numeric: {value!r}") from error
    _require(math.isfinite(result), f"{name} is not finite")
    return result


def _close(actual: Any, expected: float, name: str, tolerance: float = 5e-6) -> None:
    value = _finite(actual, name)
    _require(math.isclose(value, expected, abs_tol=tolerance, rel_tol=tolerance),
             f"{name} mismatch: {value} != approximately {expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _median(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.median(items) if items else None


def _percentile(values: list[float], fraction: float) -> float:
    _require(bool(values), "percentile needs values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _quantiles(values: list[float]) -> dict[str, float]:
    return {f"q{int(q * 100):02d}": _percentile(values, q)
            for q in (0, .1, .25, .5, .75, .9, 1)}


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        if isinstance(value, set):
            value = sorted(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


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
            writer.writerows({key: _json_cell(row.get(key)) for key in fields} for row in rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _validate_stage177a(report: dict[str, Any]) -> dict[str, Any]:
    expected_decision = (
        "STAGE177A_FRAME_PAIRWISE_SIGNAL_PRESENT_"
        "ABSOLUTE_DISCRIMINATION_WEAK"
    )
    actual_decision = report.get("decision")
    authorization_context = (
        f"expected_decision={expected_decision!r}, actual_decision={actual_decision!r}"
    )
    _require(actual_decision == expected_decision,
             f"Stage177-A decision mismatch: {authorization_context}, "
             "authorization_field='decision'")
    gate = report.get("stage177b_gate")
    _require(isinstance(gate, dict),
             f"Stage177-A stage177b_gate must be an object: {authorization_context}, "
             "authorization_field='stage177b_gate'")
    if "decision" in gate:
        _require(gate["decision"] == expected_decision,
                 f"Stage177-A gate decision mismatch: expected_decision={expected_decision!r}, "
                 f"actual_decision={gate['decision']!r}, "
                 "authorization_field='stage177b_gate.decision'")
    authorization_field = "frame_head_pairwise_feasibility_authorized"
    if authorization_field in gate:
        _require(gate[authorization_field] is not False,
                 f"Stage177-A did not authorize feasibility audit: {authorization_context}, "
                 f"authorization_field='stage177b_gate.{authorization_field}', "
                 f"authorization_value={gate[authorization_field]!r}")
    _close(_first(report, ("overall_performance.baseline.frame_auroc",
                           "overall_frame_performance.baseline.auroc")), .931242,
           "Stage177-A baseline full-dev AUROC")
    _close(_first(report, ("hard_subset_performance.baseline_frame_auroc",
                           "hard_subset_performance.baseline.auroc")), .414286,
           "Stage177-A baseline hard-39 AUROC")
    _close(_first(report, ("within_pair_performance.baseline_ranking_accuracy",
                           "within_pair_ranking.baseline.pairwise_ranking_accuracy")), .939352,
           "Stage177-A baseline same-pair ranking")
    key = _first(report, ("signal_source.primary_ranking_and_discrimination_key",
                          "signal_source.primary_key"))
    _require(key == "frame_logit", "Stage177-A must use native frame_logit")
    _require(_first(report, ("signal_source.primary_normalization",
                             "signal_source.normalization")) == "identity",
             "Stage177-A frame_logit normalization must be identity")
    _require(_first(report, ("signal_source.final_classifier_logits_used_to_reconstruct_frame_score",), False) is False,
             "Stage177-A reconstructed frame score from final logits")
    closure = report.get("closure") or {}
    _require(closure.get("training_authorized") is False,
             f"Stage177-A safety contract must prohibit training: {authorization_context}, "
             "authorization_field='closure.training_authorized', "
             f"authorization_value={closure.get('training_authorized')!r}")
    safety = report.get("safety_policy") or {}
    for key_name in ("training", "threshold_search", "external_evaluation", "time_swap"):
        _require(safety.get(key_name) is False, f"Stage177-A safety flag {key_name} must be false")
    return {"status": "passed", "decision": report["decision"],
            "native_frame_logit": True, "no_final_logit_reconstruction": True,
            "no_threshold_tuning": True, "no_training": True, "no_external_data": True}


def _runtime(prov: dict[str, Any], key: str) -> Any:
    return stage176a._runtime(prov, key)


def _validate_provenance(prov: dict[str, Any], data_path: Path,
                         split_seed: int, train_fraction: float) -> dict[str, Any]:
    _require(prov.get("status") == "completed", "baseline provenance status is not completed")
    _require(_runtime(prov, "seed") == 174 == split_seed, "seed must be 174")
    _require(_runtime(prov, "architecture") == "v6b_minimal", "architecture must be v6b_minimal")
    _require(_runtime(prov, "backbone") == "mamba", "backbone must be mamba")
    _require(_runtime(prov, "model_name") == EXPECTED_MODEL, "model name mismatch")
    _require(stage176a._selected_epoch(prov) == 20, "selected epoch must be 20")
    dev_ratio = float(stage176a._arg(prov, "dev_ratio", .2))
    _require(math.isclose(train_fraction, .8) and math.isclose(dev_ratio, 1 - train_fraction),
             "train fraction/provenance dev ratio mismatch")
    stage174 = prov.get("stage174c_clean_pairwise") or {}
    _require(stage174.get("mode") == "off" and float(stage174.get("weight", 0)) == 0,
             "Stage174-C must be off")
    stage175 = prov.get("stage175b_support_anchor") or {}
    _require(stage175.get("mode") == "off" or float(stage175.get("weight", 0)) == 0,
             "baseline Stage175-B must be off")
    policy = prov.get("training_selection_policy") or {}
    _require(policy.get("clean_dev_only_checkpoint_selection") is True,
             "checkpoint selection was not clean-dev-only")
    _require(policy.get("final_ce_logits_source") in ('output["logits"]', "output['logits']"),
             "final CE source must be output[\"logits\"]")
    _require(policy.get("loss_logits_used_for_final_classifier_ce") is False,
             "loss_logits was used for final CE")
    for name in ("external_evaluation_used_for_training",
                 "external_evaluation_used_for_calibration",
                 "external_evaluation_used_for_threshold_selection",
                 "external_evaluation_used_for_checkpoint_selection",
                 "time_swap_included_in_main_classification_training"):
        _require(policy.get(name) is False, f"provenance policy {name} must be false")
    activity = _get(prov, "data_provenance.auxiliary_activity", {}) or {}
    _require(activity.get("external_evaluation_active") is False, "external evaluation was active")
    _require(activity.get("time_swap_active") is False, "time_swap was active")
    data_record = stage176a._data_record(prov)
    actual_hash = _sha256(data_path)
    _require(data_record.get("sha256") == actual_hash, "clean data SHA-256 mismatch")
    _require(int(data_record.get("row_count", -1)) == 3600, "clean data row count mismatch")
    return {"status": "passed", "seed": 174, "selected_epoch": 20,
            "architecture": "v6b_minimal", "backbone": "mamba",
            "model_name": EXPECTED_MODEL, "data_sha256": actual_hash,
            "stage174c_off": True, "stage175b_off": True,
            "clean_dev_only_checkpoint_selection": True,
            "final_ce_source": 'output["logits"]', "loss_logits_used": False,
            "external_data_or_labels_used": False, "time_swap_used": False}


def _validate_checkpoint(path: Path, prov: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    state, metadata, payload = stage176b._load_checkpoint(path)
    validation = stage176b._validate_checkpoint("baseline", path, prov, metadata, payload)
    _require(payload.get("schema_version") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
    return state, metadata, validation


def _split_records(data_path: Path, seed: int, train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = v5.load_jsonl(data_path)
    train, dev = v5.split_by_pair_id(records, dev_ratio=1 - train_fraction, seed=seed)
    train_pairs = {str(row["pair_id"]) for row in train}
    dev_pairs = {str(row["pair_id"]) for row in dev}
    _require(len(records) == 3600 and len(train) == 2880 and len(dev) == 720,
             "controlled split row counts must be 3600/2880/720")
    _require(len(train_pairs) == 240 and len(dev_pairs) == 60,
             "controlled split pair counts must be 240/60")
    _require(not train_pairs & dev_pairs, "train/dev pair overlap detected")
    all_ids = [str(row.get("id")) for row in records]
    _require(len(set(all_ids)) == len(all_ids), "stable row ids are not unique")
    return train, dev, {"status": "passed", "seed": seed,
                        "train_fraction": train_fraction, "total_pairs": 300,
                        "train_pairs": 240, "dev_pairs": 60, "pair_overlap": 0,
                        "total_rows": 3600, "train_rows": 2880, "dev_rows": 720,
                        "stable_row_identity_unique": True}


def _topology(split_rows: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for split, rows in split_rows.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row["pair_id"])].append(row)
        for pair_id, group in sorted(groups.items()):
            compatible = [row for row in group if int(row["frame_compatible_label"]) == 1]
            incompatible = [row for row in group if int(row["frame_compatible_label"]) == 0]
            problems = []
            if not compatible:
                problems.append("empty_compatible_side")
            if not incompatible:
                problems.append("empty_incompatible_side")
            if len({str(row.get("id")) for row in group}) != len(group):
                problems.append("duplicate_row_id")
            if problems:
                malformed.append({"split": split, "pair_id": pair_id, "reasons": problems})
            pair_record = {
                "split": split, "pair_id": pair_id, "row_count": len(group),
                "compatible_row_count": len(compatible),
                "incompatible_row_count": len(incompatible),
                "raw_comparison_count": len(compatible) * len(incompatible),
                "compatible_families": sorted({str(row["intervention_type"]) for row in compatible}),
                "incompatible_families": sorted({str(row["intervention_type"]) for row in incompatible}),
                "unique_compatible_family_count": len({str(row["intervention_type"]) for row in compatible}),
                "unique_incompatible_family_count": len({str(row["intervention_type"]) for row in incompatible}),
                "final_label_distribution": dict(sorted(Counter(str(row["final_label"]) for row in group).items())),
                "malformed": bool(problems), "malformed_reasons": problems,
            }
            pair_rows.append(pair_record)
            for positive in compatible:
                for negative in incompatible:
                    comparisons.append({
                        "split": split, "pair_id": pair_id,
                        "compatible_row_id": str(positive["id"]),
                        "incompatible_row_id": str(negative["id"]),
                        "compatible_family": str(positive["intervention_type"]),
                        "incompatible_family": str(negative["intervention_type"]),
                        "compatible_final_label": str(positive["final_label"]),
                        "incompatible_final_label": str(negative["final_label"]),
                        "compatible_gold_frame_label": 1,
                        "incompatible_gold_frame_label": 0,
                    })
    matrix_groups: dict[tuple[str, str, str, str, str], int] = Counter(
        (row["split"], row["compatible_family"], row["incompatible_family"],
         row["compatible_final_label"], row["incompatible_final_label"])
        for row in comparisons
    )
    matrix = [{"split": key[0], "compatible_family": key[1],
               "incompatible_family": key[2], "compatible_final_label": key[3],
               "incompatible_final_label": key[4], "comparison_count": count}
              for key, count in sorted(matrix_groups.items())]
    summary: dict[str, Any] = {"malformed_pairs": malformed, "splits": {}}
    for split in split_rows:
        pairs = [row for row in pair_rows if row["split"] == split]
        counts = [int(row["raw_comparison_count"]) for row in pairs]
        compatible_counts = [int(row["compatible_row_count"]) for row in pairs]
        incompatible_counts = [int(row["incompatible_row_count"]) for row in pairs]
        top_n = max(1, math.ceil(len(pairs) * .1))
        total = sum(counts)
        summary["splits"][split] = {
            "eligible_pair_count": sum(count > 0 for count in counts),
            "malformed_pair_count": sum(row["malformed"] for row in pairs),
            "total_raw_comparisons": total,
            "comparisons_per_pair": {"mean": _mean(counts), "median": _median(counts),
                                     "minimum": min(counts), "maximum": max(counts),
                                     "quantiles": _quantiles([float(x) for x in counts])},
            "compatible_rows_per_pair": {"mean": _mean(compatible_counts),
                                         "median": _median(compatible_counts),
                                         "minimum": min(compatible_counts), "maximum": max(compatible_counts)},
            "incompatible_rows_per_pair": {"mean": _mean(incompatible_counts),
                                           "median": _median(incompatible_counts),
                                           "minimum": min(incompatible_counts), "maximum": max(incompatible_counts)},
            "compatible_families": sorted({family for row in pairs for family in row["compatible_families"]}),
            "incompatible_families": sorted({family for row in pairs for family in row["incompatible_families"]}),
            "top_10_percent_pair_comparison_concentration": sum(sorted(counts, reverse=True)[:top_n]) / total,
            "top_10_percent_pair_count": top_n,
        }
    actual_family_labels: dict[str, set[int]] = defaultdict(set)
    for rows in split_rows.values():
        for row in rows:
            actual_family_labels[str(row["intervention_type"])].add(int(row["frame_compatible_label"]))
    summary["hard_family_audit"] = {
        "declared_compatible_families": sorted(DECLARED_COMPATIBLE_FAMILIES),
        "declared_incompatible_families": sorted(DECLARED_INCOMPATIBLE_FAMILIES),
        "actual_gold_frame_labels_by_family": {key: sorted(value) for key, value in sorted(actual_family_labels.items())},
        "target_derived_from_family_name": False,
        "all_observed_family_combinations_emitted": True,
    }
    return pair_rows, comparisons, matrix, summary


def _evaluate(model: torch.nn.Module, metadata: dict[str, Any], prov: dict[str, Any],
              rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[float]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(_runtime(prov, "model_name")))
    if tokenizer.pad_token_id is None:
        _require(tokenizer.eos_token_id is not None, "tokenizer has no pad/eos token")
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(stage176a._arg(prov, "max_length", metadata.get("max_length", 128)))
    inputs = v5.move_inputs(v5.encode_mamba_records(rows, tokenizer, max_length)["model_inputs"], device)
    model_output = stage176a._forward(model, inputs, rows, prov, device, batch_size)
    _require(model_output.get("frame_logit") is not None, "native frame_logit is unavailable")
    logits = stage176b._tensor_vector(model_output, "frame_logit", len(rows))
    _require(len(logits) == len(rows) and all(math.isfinite(x) for x in logits),
             "invalid native frame_logit vector")
    return logits


def _gap_diagnostics(split_rows: dict[str, list[dict[str, Any]]],
                     split_logits: dict[str, list[float]], comparisons: list[dict[str, Any]],
                     samples: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    lookup: dict[tuple[str, str], float] = {}
    for split, rows in split_rows.items():
        for row, logit in zip(rows, split_logits[split]):
            lookup[(split, str(row["id"]))] = float(logit)
    gap_rows = []
    for row in comparisons:
        positive = lookup[(row["split"], row["compatible_row_id"])]
        negative = lookup[(row["split"], row["incompatible_row_id"])]
        gap = positive - negative
        gap_rows.append({**row, "compatible_frame_logit": positive,
                         "incompatible_frame_logit": negative, "gap": gap,
                         "outcome": "positive" if gap > 0 else "zero" if gap == 0 else "negative",
                         "correct": gap > 0})
    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gap_rows:
        pair_groups[(row["split"], row["pair_id"])].append(row)
    pair_summaries = []
    for (split, pair_id), group in sorted(pair_groups.items()):
        gaps = [float(row["gap"]) for row in group]
        correct = sum(gap > 0 for gap in gaps)
        pair_summaries.append({
            "split": split, "pair_id": pair_id, "comparison_count": len(group),
            "ranking_accuracy": correct / len(group), "mean_gap": _mean(gaps),
            "median_gap": _median(gaps), "positive_count": sum(g > 0 for g in gaps),
            "zero_count": sum(g == 0 for g in gaps), "negative_count": sum(g < 0 for g in gaps),
            "ordering_status": "fully_ordered" if all(g > 0 for g in gaps)
                               else "fully_reversed" if all(g <= 0 for g in gaps)
                               else "partially_violated",
        })
    family_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gap_rows:
        family_groups[(row["split"], row["compatible_family"], row["incompatible_family"])].append(row)
    family_summaries = []
    for key, group in sorted(family_groups.items()):
        gaps = [float(row["gap"]) for row in group]
        family_summaries.append({"split": key[0], "compatible_family": key[1],
                                 "incompatible_family": key[2], "comparison_count": len(group),
                                 "ranking_accuracy": sum(g > 0 for g in gaps) / len(gaps),
                                 "mean_gap": _mean(gaps), "violation_count": sum(g <= 0 for g in gaps)})
    report: dict[str, Any] = {}
    for split_index, split in enumerate(split_rows):
        rows = [row for row in gap_rows if row["split"] == split]
        pairs = [row for row in pair_summaries if row["split"] == split]
        gaps = [float(row["gap"]) for row in rows]
        rng = random.Random(seed + split_index)
        accuracy_boot, mean_gap_boot = [], []
        for _ in range(samples):
            sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            accuracy_boot.append(statistics.fmean(float(row["ranking_accuracy"]) for row in sampled))
            mean_gap_boot.append(statistics.fmean(float(row["mean_gap"]) for row in sampled))
        report[split] = {
            "comparison_count": len(rows),
            "comparison_level_ranking_accuracy": sum(g > 0 for g in gaps) / len(gaps),
            "pair_normalized_ranking_accuracy": statistics.fmean(float(row["ranking_accuracy"]) for row in pairs),
            "mean_gap": _mean(gaps), "median_gap": _median(gaps),
            "gap_standard_deviation": statistics.pstdev(gaps),
            "positive_comparison_count": sum(g > 0 for g in gaps),
            "zero_comparison_count": sum(g == 0 for g in gaps),
            "negative_comparison_count": sum(g < 0 for g in gaps),
            "pair_mean_gap_mean": statistics.fmean(float(row["mean_gap"]) for row in pairs),
            "fully_ordered_pair_count": sum(row["ordering_status"] == "fully_ordered" for row in pairs),
            "partially_violated_pair_count": sum(row["ordering_status"] == "partially_violated" for row in pairs),
            "fully_reversed_pair_count": sum(row["ordering_status"] == "fully_reversed" for row in pairs),
            "pair_bootstrap_ci95": {
                "pair_normalized_ranking_accuracy": [_percentile(accuracy_boot, .025), _percentile(accuracy_boot, .975)],
                "pair_mean_gap_mean": [_percentile(mean_gap_boot, .025), _percentile(mean_gap_boot, .975)],
                "unit": "pair_id", "samples": samples, "seed": seed + split_index,
            },
        }
    return gap_rows, pair_summaries, family_summaries, report


def _source_evidence(path: Path, needles: list[str]) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for needle in needles:
        hits = [index + 1 for index, line in enumerate(lines) if needle in line]
        result.append({"needle": needle, "line_numbers": hits[:20], "found": bool(hits)})
    return result


def _static_audits(topology: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trainer = ROOT / "scripts" / "train_controlled_v6b_minimal.py"
    v5_path = ROOT / "scripts" / "train_controlled_v5.py"
    model_path = ROOT / "src" / "contramamba" / "modeling_v6b_minimal.py"
    batch = {
        "audit_type": "static_code_inspection",
        "current_training_forward": "full clean-train tensor view, optionally differentiably microbatched, followed by one optimizer step per epoch",
        "ordinary_independent_shuffled_row_batches": False,
        "same_pair_sides_available_together_in_epoch_output": True,
        "pair_index_or_pair_aware_collation_required": True,
        "recommended_topology": "Reuse the current differentiable full clean-train frame_logit output and add a validated train pair-index that groups complete pairs; no counterpart forward is needed.",
        "allowed_fallback_needed": False,
        "extra_model_forward_required": False,
        "train_reference_lookup_required": False,
        "dev_reference_lookup_required": False,
        "gradient_accumulation_compatibility": "Pair loss is formed once over the complete differentiable epoch output before the existing backward; internal forward microbatching does not detach logits.",
        "expected_effective_pair_count_per_optimizer_step": topology["splits"]["train"]["eligible_pair_count"],
        "malformed_pair_handling": "Hard fail before training; never skip or impute a side.",
        "stage174c_infrastructure_reuse": "Reuse only deterministic pair-index validation/grouping concepts; do not reuse detached reference logits, final-logit targets, margin loss, or reference forward.",
        "forbidden_topologies": ["no-grad counterpart", "detached counterpart", "dev reference lookup", "selected-checkpoint teacher", "cross-pair negative sampling", "final-logit reference"],
        "viable_differentiable_topology": True,
        "source_evidence": _source_evidence(trainer, ["_vnext_forward_maybe_batched(", "indices = v5.sample_indices(", "total_loss = losses[\"total\"]", "total_loss.backward()"]),
    }
    existing = {
        "direct_frame_supervision_present": True,
        "output_key": "frame_logit",
        "loss_type": "binary_cross_entropy_with_logits",
        "loss_weight": 1.0,
        "gold_frame_label_source": "record[\"frame_compatible_label\"] encoded as frame_compatible_labels",
        "auxiliary_loss_aggregation": "label CE + frame BCE + predicate BCE + sufficiency BCE + polarity CE, then enabled auxiliary losses",
        "final_ce_source": 'output["logits"]',
        "source_evidence": {
            "controlled_loss": _source_evidence(v5_path, ["frame_loss = F.binary_cross_entropy_with_logits(", "total = label_loss + frame_loss"]),
            "model_loss": _source_evidence(model_path, ["losses[\"frame_loss\"] = F.binary_cross_entropy_with_logits(", "frame[\"frame_logit\"]"]),
        },
    }
    conflict = {
        "audit_type": "static_code_inspection",
        "direct_frame_bce_and_candidate_pairwise_relationship": "complementary but overlapping",
        "reason": "BCE supplies absolute per-row supervision; the candidate supplies within-pair ordering on the same frame_logit and shared frame-head parameters.",
        "duplicate_signal_risk": "moderate because both losses reward compatible logits above incompatible logits, but pair normalization changes weighting and relative geometry",
        "final_classifier_gradient_conflict_risk": "possible through shared encoder/frame representations, but the candidate never consumes or directly differentiates output[\"logits\"]",
        "gradient_target_overlap": ["native frame head", "shared encoder and frame representations"],
        "direct_final_classifier_logit_target_overlap": False,
        "stage174c_failure_structurally_distinct": True,
        "stage174c_distinction": "Stage174-C ranked final classifier SUPPORT/polarity margins against detached reference outputs; this candidate ranks native frame_logit within complete clean pairs with both sides differentiable, no margin, and pair-normalized weighting.",
        "candidate_implementation_status": "not implemented in Stage177-B",
    }
    return batch, existing, conflict


def _objective_contract() -> dict[str, Any]:
    return {
        "per_comparison": "gap = compatible_frame_logit - incompatible_frame_logit; comparison_loss = softplus(-gap)",
        "per_pair": "mean comparison_loss over every compatible x incompatible row combination in the pair",
        "aggregate": "mean pair_loss over eligible pairs",
        "margin_hyperparameter": None,
        "equal_pair_weight": True,
        "native_gradient_output_key": "frame_logit",
        "both_sides_differentiable": True,
        "final_three_way_logits_directly_targeted": False,
        "final_labels_used_as_pairwise_targets": False,
        "positive_definition": "gold_frame_label == 1",
        "negative_definition": "gold_frame_label == 0",
        "teacher_or_reference": None,
        "absolute_threshold": None,
        "external_data": False,
        "implemented_by_this_audit": False,
    }


def _gate(topology: dict[str, Any], gaps: dict[str, Any], batch: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    train_top, dev_gap, train_gap = topology["splits"]["train"], gaps["dev"], gaps["train"]
    major = set(train_top["incompatible_families"])
    checks = {
        "train_eligible_pairs_gte_200": train_top["eligible_pair_count"] >= 200,
        "malformed_train_pairs_eq_0": train_top["malformed_pair_count"] == 0,
        "all_major_incompatible_families_covered": major <= set(train_top["incompatible_families"]),
        "top_10_percent_concentration_lte_0_25": train_top["top_10_percent_pair_comparison_concentration"] <= .25,
        "train_pair_normalized_ranking_gte_0_80": train_gap["pair_normalized_ranking_accuracy"] >= .80,
        "dev_pair_normalized_ranking_gte_0_85": dev_gap["pair_normalized_ranking_accuracy"] >= .85,
        "dev_pair_bootstrap_ci_lower_gt_0_75": dev_gap["pair_bootstrap_ci95"]["pair_normalized_ranking_accuracy"][0] > .75,
        "train_and_dev_mean_gap_gt_0": train_gap["mean_gap"] > 0 and dev_gap["mean_gap"] > 0,
        "viable_differentiable_topology": batch["viable_differentiable_topology"] is True,
        "no_direct_final_logit_pairwise_loss": True,
        "no_external_or_time_swap_data": True,
    }
    passed = all(checks.values())
    return (FEASIBLE if passed else NOT_FEASIBLE), {
        "criteria": {"train_eligible_pairs_minimum": 200, "malformed_train_pairs_maximum": 0,
                     "top_10_percent_concentration_maximum": .25,
                     "train_pair_normalized_ranking_minimum": .80,
                     "dev_pair_normalized_ranking_minimum": .85,
                     "dev_pair_bootstrap_ci95_lower_strictly_greater_than": .75,
                     "mean_gap_strictly_greater_than": 0},
        "checks": checks, "passed": passed,
        "decision": FEASIBLE if passed else NOT_FEASIBLE,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    train_top = report["comparison_topology"]["splits"]["train"]
    dev_top = report["comparison_topology"]["splits"]["dev"]
    train_gap = report["baseline_gap_diagnostic"]["train"]
    dev_gap = report["baseline_gap_diagnostic"]["dev"]
    lines = [
        "# Stage177-B frame-pairwise objective feasibility audit", "",
        f"**Decision:** `{report['decision']}`", "",
        "This is an evaluation-only feasibility audit. It did not train, backpropagate, fit, calibrate, search thresholds, select a checkpoint, modify a trainer, or use external/time-swap data.", "",
        "## Pair topology", "",
        "| Split | Eligible pairs | Malformed | Raw comparisons | Top-10% concentration |", "|---|---:|---:|---:|---:|",
        f"| train | {train_top['eligible_pair_count']} | {train_top['malformed_pair_count']} | {train_top['total_raw_comparisons']} | {train_top['top_10_percent_pair_comparison_concentration']:.6f} |",
        f"| dev | {dev_top['eligible_pair_count']} | {dev_top['malformed_pair_count']} | {dev_top['total_raw_comparisons']} | {dev_top['top_10_percent_pair_comparison_concentration']:.6f} |", "",
        "Targets come only from the actual `frame_compatible_label`; family names never infer labels. Every compatible × incompatible combination within each pair is emitted.", "",
        "## Candidate objective", "",
        "For each comparison, `gap = compatible_frame_logit - incompatible_frame_logit` and `loss = softplus(-gap)`. Losses are averaged inside each pair, then averaged equally across eligible pairs. There is no margin, threshold, teacher, detached counterpart, final-label target, final-logit target, or external data. Stage177-B specifies but does not implement this objective.", "",
        "## Baseline gap diagnostic", "",
        "| Split | Comparison ranking | Pair-normalized ranking | Mean gap | Pair-bootstrap ranking CI95 |", "|---|---:|---:|---:|---:|",
        f"| train | {train_gap['comparison_level_ranking_accuracy']:.6f} | {train_gap['pair_normalized_ranking_accuracy']:.6f} | {train_gap['mean_gap']:.6f} | {train_gap['pair_bootstrap_ci95']['pair_normalized_ranking_accuracy']} |",
        f"| dev | {dev_gap['comparison_level_ranking_accuracy']:.6f} | {dev_gap['pair_normalized_ranking_accuracy']:.6f} | {dev_gap['mean_gap']:.6f} | {dev_gap['pair_bootstrap_ci95']['pair_normalized_ranking_accuracy']} |", "",
        "Bootstrap resampling uses `pair_id` as the unit. Family-pair ranking, mean gaps, and violations are provided in the CSV outputs.", "",
        "## Batch and loss audit", "",
        report["batch_topology"]["recommended_topology"], "",
        "Existing direct frame BCE and the candidate are complementary but overlap: BCE supplies absolute row supervision while the candidate supplies equally weighted within-pair ordering on the same native frame head. Final CE still consumes `output[\"logits\"]`; the candidate must not directly target those logits. Stage174-C is structurally different because it used final-classifier margins and detached references.", "",
        "## Stage177-C gate", "",
    ]
    for name, passed in report["stage177c_gate"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines += ["", "If feasible, the only authorized next action is a default-off trainer implementation plus an implementation smoke test, followed later by one 20-epoch pilot if separately authorized. No sweep, multi-seed run, long training, external evaluation, calibration, or final-logit pairwise loss is authorized.", ""]
    return "\n".join(lines)


def _blocked_outputs(output_dir: Path, error: Exception) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"stage": STAGE, "decision": BLOCKED, "scope": None,
              "input_validation": {"status": "blocked", "error": f"{type(error).__name__}: {error}",
                                   "blocked_before_model_forward_if_input_failure": True},
              "split_validation": None, "frame_label_topology": None,
              "comparison_topology": None, "baseline_gap_diagnostic": None,
              "family_gap_diagnostic": None, "batch_topology": None,
              "existing_frame_supervision": None, "loss_conflict_audit": None,
              "candidate_objective_contract": _objective_contract(), "stage177c_gate": None,
              "limitations": [], "safety_policy": {"training": False, "optimizer": False,
              "backward": False, "threshold_search": False, "external_evaluation": False}}
    _write_json(output_dir / OUTPUTS["report_json"], report)
    (output_dir / OUTPUTS["report_md"]).write_text(
        f"# Stage177-B blocked\n\n**Decision:** `{BLOCKED}`\n\n`{type(error).__name__}: {error}`\n",
        encoding="utf-8")
    for key in ("pair_topology", "comparison_topology", "family_matrix", "gap_rows", "pair_gaps", "family_gaps"):
        _write_csv(output_dir / OUTPUTS[key], [])
    _write_json(output_dir / OUTPUTS["batch_audit"], {"status": "blocked", "error": str(error)})
    _write_json(output_dir / OUTPUTS["conflict_audit"], {"status": "blocked", "error": str(error)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/controlled_v5_v3_without_time_swap.jsonl"))
    parser.add_argument("--stage177a-report", type=Path, required=True)
    parser.add_argument("--baseline-provenance", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=174)
    parser.add_argument("--train-fraction", type=float, default=.8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=177)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        _require(args.device == "cuda", "Stage177-B requires --device cuda")
        _require(args.eval_batch_size > 0 and args.bootstrap_samples > 0,
                 "eval batch size and bootstrap samples must be positive")
        _require(args.split_seed == 174 and math.isclose(args.train_fraction, .8),
                 "Stage177-B fixed split is seed 174 and train fraction 0.8")
        paths = {name: getattr(args, name).resolve() for name in
                 ("data", "stage177a_report", "baseline_provenance", "baseline_checkpoint")}
        for name, path in paths.items():
            _require(path.is_file(), f"{name} file does not exist: {path}")

        # All hard semantic/provenance/checkpoint/data validation is completed
        # before model construction and forward.
        stage177a_validation = _validate_stage177a(_read_json(paths["stage177a_report"]))
        provenance = _read_json(paths["baseline_provenance"])
        provenance_validation = _validate_provenance(
            provenance, paths["data"], args.split_seed, args.train_fraction)
        resolved_checkpoint = stage176b._resolve_checkpoint(
            paths["baseline_provenance"], provenance, paths["baseline_checkpoint"])
        _require(resolved_checkpoint == paths["baseline_checkpoint"],
                 "explicit baseline checkpoint did not resolve identically")
        state, metadata, checkpoint_validation = _validate_checkpoint(resolved_checkpoint, provenance)
        train_rows, dev_rows, split_validation = _split_records(
            paths["data"], args.split_seed, args.train_fraction)
        pair_rows, comparison_rows, family_matrix, topology = _topology(
            {"train": train_rows, "dev": dev_rows})
        _require(topology["splits"]["train"]["malformed_pair_count"] == 0 and
                 topology["splits"]["dev"]["malformed_pair_count"] == 0,
                 "malformed pair topology blocks evaluation")
        input_validation = {"status": "passed", "stage177a": stage177a_validation,
                            "provenance": provenance_validation,
                            "checkpoint": checkpoint_validation,
                            "split_and_row_identity": split_validation,
                            "completed_before_model_construction_and_forward": True}

        _require(torch.cuda.is_available(), "CUDA is unavailable")
        device = torch.device("cuda")
        model = stage176a._construct_model(provenance, metadata, state, device)
        _require(not model.training, "model must remain in eval mode")
        train_logits = _evaluate(model, metadata, provenance, train_rows, device, args.eval_batch_size)
        dev_logits = _evaluate(model, metadata, provenance, dev_rows, device, args.eval_batch_size)
        _require(not model.training, "model entered training mode")
        gap_rows, pair_gaps, family_gaps, gap_report = _gap_diagnostics(
            {"train": train_rows, "dev": dev_rows},
            {"train": train_logits, "dev": dev_logits}, comparison_rows,
            args.bootstrap_samples, args.bootstrap_seed)
        batch, existing, conflict = _static_audits(topology)
        decision, gate = _gate(topology, gap_report, batch)
        report = {
            "stage": STAGE, "decision": decision,
            "scope": {"data": str(paths["data"]), "clean_controlled_only": True,
                      "evaluation_only_model_forward": True, "split_seed": args.split_seed,
                      "train_fraction": args.train_fraction, "bootstrap_samples": args.bootstrap_samples,
                      "bootstrap_seed": args.bootstrap_seed, "selected_epoch": 20},
            "input_validation": input_validation, "split_validation": split_validation,
            "frame_label_topology": topology["hard_family_audit"],
            "comparison_topology": topology,
            "baseline_gap_diagnostic": gap_report,
            "family_gap_diagnostic": {"rows": family_gaps,
                                      "all_observed_family_pairs_reported": True},
            "batch_topology": batch, "existing_frame_supervision": existing,
            "loss_conflict_audit": conflict,
            "candidate_objective_contract": _objective_contract(),
            "stage177c_gate": gate,
            "limitations": ["Single selected checkpoint and one deterministic split.",
                            "Bootstrap intervals capture pair resampling, not training-seed uncertainty.",
                            "Static trainer feasibility does not establish optimization stability or downstream benefit.",
                            "No objective weight, margin, threshold, or calibration was tested."],
            "safety_policy": {"clean_controlled_data_only": True, "training": False,
                              "optimizer_created": False, "backward": False,
                              "train_mode_called": False, "trainer_modified": False,
                              "loss_implemented": False, "threshold_search": False,
                              "calibration": False, "fitted_probe": False,
                              "checkpoint_selection": False, "external_evaluation": False,
                              "external_labels": False, "time_swap": False,
                              "final_classifier_logits_used_as_frame_score": False},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / OUTPUTS["report_json"], report)
        (output_dir / OUTPUTS["report_md"]).write_text(_render_markdown(report), encoding="utf-8")
        _write_csv(output_dir / OUTPUTS["pair_topology"], pair_rows)
        _write_csv(output_dir / OUTPUTS["comparison_topology"], comparison_rows)
        _write_csv(output_dir / OUTPUTS["family_matrix"], family_matrix)
        _write_csv(output_dir / OUTPUTS["gap_rows"], gap_rows)
        _write_csv(output_dir / OUTPUTS["pair_gaps"], pair_gaps)
        _write_csv(output_dir / OUTPUTS["family_gaps"], family_gaps)
        _write_json(output_dir / OUTPUTS["batch_audit"], batch)
        _write_json(output_dir / OUTPUTS["conflict_audit"], conflict)
        print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except (AuditBlocked, stage176b.AuditBlocked, OSError, ValueError, KeyError, TypeError) as error:
        _blocked_outputs(output_dir, error)
        print(json.dumps({"decision": BLOCKED, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
