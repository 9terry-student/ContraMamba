"""Stage179-A frame-label semantics and input-representation audit.

Evaluation only.  This audit validates the canonical Stage174 baseline, reads
native frame targets and native model outputs, reconstructs the exact linear
frame readout, and emits descriptive/transductive diagnostics.  It never
trains, fits a probe, calibrates, searches a threshold, or calls backward().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import analyze_stage176a_support_boundary_attribution as stage176a  # noqa: E402
from scripts import analyze_stage177e_frame_pairwise_pilot_attribution as stage177e  # noqa: E402
from scripts import train_controlled_v5 as v5  # noqa: E402


STAGE = "Stage179-A"
BLOCKED = "STAGE179A_FRAME_SEMANTICS_INPUT_REPRESENTATION_AUDIT_BLOCKED"
DIRECT = "STAGE179A_DIRECT_FRAME_LABEL_INCONSISTENCY_IDENTIFIED"
INTERFERENCE = "STAGE179A_FRAME_SEMANTIC_CROSS_CHANNEL_INTERFERENCE_IDENTIFIED"
INSENSITIVE = "STAGE179A_FRAME_INPUT_REPRESENTATION_INSENSITIVITY_IDENTIFIED"
READOUT = "STAGE179A_FRAME_READOUT_ALIGNMENT_FAILURE_IDENTIFIED"
MIXED = "STAGE179A_FRAME_SEMANTICS_REPRESENTATION_CAUSE_MIXED_OR_INSUFFICIENT"
S176 = "STAGE176A_CLEAN_DEV_SUPPORT_BOUNDARY_ATTRIBUTION_COMPLETE"
S177A = "STAGE177A_FRAME_PAIRWISE_SIGNAL_PRESENT_ABSOLUTE_DISCRIMINATION_WEAK"
S177E = "STAGE177E_FRAME_PAIRWISE_OBJECTIVE_REDUNDANT_PATH_CLOSED"
S178A = "STAGE178A_PAIR_OFFSET_EXPLANATION_WEAK_PATH_CLOSED"
MODEL_NAME = "state-spaces/mamba-130m-hf"
OUTPUTS = {
    "json": "stage179a_frame_semantics_input_representation_report.json",
    "md": "stage179a_frame_semantics_input_representation_report.md",
    "rows": "stage179a_dev_row_semantics_representation.csv",
    "contingency": "stage179a_cross_channel_contingency.csv",
    "strata": "stage179a_semantic_stratum_metrics.csv",
    "duplicates": "stage179a_exact_duplicate_consistency.csv",
    "near": "stage179a_near_duplicate_review_queue.csv",
    "intervention": "stage179a_intervention_response_summary.csv",
    "readout": "stage179a_frame_head_readout_decomposition.csv",
    "hard39": "stage179a_hard39_semantics_representation_attribution.csv",
    "centroid": "stage179a_centroid_diagnostic_summary.csv",
    "sensitivity": "stage179a_frame_sensitivity_pair_summary.csv",
    "interference": "stage179a_semantic_interference_gate.csv",
    "decision": "stage179a_decision_evidence.csv",
}


class AuditBlocked(ValueError):
    """A hard validation or model-contract failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditBlocked(message)


def _required_int(value: Any, name: str) -> int:
    if value is None or isinstance(value, bool):
        raise AuditBlocked(f"required integer is missing/invalid: {name}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AuditBlocked(f"invalid integer {name}: {value!r}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditBlocked(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise AuditBlocked(f"cannot read CSV {path}: {exc}") from exc
    _require(bool(rows), f"CSV is empty: {path}")
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in rows)


def _mean(values: Iterable[float]) -> float | None:
    data = list(values)
    return statistics.fmean(data) if data else None


def _median(values: Iterable[float]) -> float | None:
    data = list(values)
    return statistics.median(data) if data else None


def _percentile(values: Iterable[float], q: float) -> float | None:
    data = sorted(values)
    if not data:
        return None
    position = (len(data) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return data[lo] if lo == hi else data[lo] * (hi - position) + data[hi] * (position - lo)


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    data = list(values)
    return {"count": len(data), "mean": _mean(data), "median": _median(data),
            "p25": _percentile(data, .25), "p75": _percentile(data, .75),
            "minimum": min(data) if data else None, "maximum": max(data) if data else None}


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    ranks = _rank(scores)
    return (sum(rank for rank, label in zip(ranks, labels) if label) - positives * (positives + 1) / 2) / (positives * negatives)


def _ap(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    return sum(sum(label for _, label in ranked[:i + 1]) / (i + 1)
               for i, (_, label) in enumerate(ranked) if label) / positives


def _norm_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).lower().split())


def _tokens(value: Any) -> set[str]:
    return set(_norm_text(value).split())


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left | right else 1.0


def _decision_from(report: dict[str, Any]) -> Any:
    return report.get("decision") or (report.get("closure") or {}).get("attribution_completion_decision")


def _validate_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {"stage176a": S176, "stage177a": S177A, "stage177e": S177E, "stage178a": S178A}
    actual = {name: _decision_from(report) for name, report in reports.items()}
    actual["stage176a"] = ((reports["stage176a"].get("closure") or {})
                            .get("attribution_completion_decision") or actual["stage176a"])
    for name, wanted in expected.items():
        _require(actual[name] == wanted, f"{name} decision mismatch: {actual[name]!r}")
    return {"status": "passed", "decisions": actual}


def _validate_hard_rows(transitions: list[dict[str, Any]], offset_rows: list[dict[str, Any]],
                        dev: list[dict[str, Any]]) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    transitions = stage177e._validate_stage176_csv_identifiers(transitions)
    transitions = stage177e.stage176b._validate_row_alignment(transitions, dev)
    hard: dict[tuple[int, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in transitions:
        cohort = stage177e.stage176b._cohort(row)
        if cohort in ("beneficial_correction", "harmful_regression"):
            key = (int(row["stable_row_index"]), str(row["row_id"]))
            hard[key] = {**row, "stage176_cohort": cohort}
            counts[cohort] += 1
    _require(counts == Counter({"beneficial_correction": 25, "harmful_regression": 14}),
             f"hard cohort must be 25/14, got {dict(counts)}")
    required = {"stable_row_index", "row_id"}
    _require(required <= set(offset_rows[0]), "Stage178-A hard39 attribution lacks stable identity")
    offset_keys = {(int(row["stable_row_index"]), str(row["row_id"])) for row in offset_rows}
    _require(offset_keys == set(hard), "Stage178-A hard39 identities disagree with Stage176-A")
    return hard, {"status": "passed", "beneficial": 25, "harmful": 14, "total": 39,
                  "stable_identity_matches_stage178a": True}


def _validate_topology(records: list[dict[str, Any]], split: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(records):
        _require(row.get("pair_id") is not None, f"{split} row {index} lacks pair_id")
        label = _required_int(row.get("frame_compatible_label"), f"{split}.{index}.frame_compatible_label")
        _require(label in (0, 1), f"{split} row {index} has non-binary native frame target")
        groups[str(row["pair_id"])].append(row)
    for pair_id, rows in groups.items():
        counts = Counter(int(row["frame_compatible_label"]) for row in rows)
        _require(len(rows) == 12 and counts == Counter({0: 6, 1: 6}),
                 f"malformed pair {split}.{pair_id}: rows={len(rows)}, labels={dict(counts)}")
        anchors = [row for row in rows if str(row.get("intervention_type")) == "none"]
        _require(len(anchors) == 1, f"pair {split}.{pair_id} must have exactly one none anchor")
    return {"pairs": len(groups), "rows": len(records), "rows_per_pair": 12,
            "compatible_rows_per_pair": 6, "incompatible_rows_per_pair": 6,
            "canonical_none_per_pair": 1, "native_target_field": "frame_compatible_label",
            "pair_ids_preserved_as_strings": True, "malformed_pairs": 0}


def _evaluate(dev: list[dict[str, Any]], prov: dict[str, Any], metadata: dict[str, Any],
              state: dict[str, torch.Tensor], device: torch.device, batch_size: int
              ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        _require(tokenizer.eos_token_id is not None, "tokenizer has no pad/eos token")
        tokenizer.pad_token = tokenizer.eos_token
    max_length = _required_int(stage177e._first_non_none(
        (prov.get("parsed_args") or {}).get("max_length"), metadata.get("max_length")), "max_length")
    bundle = v5.encode_mamba_records(dev, tokenizer, max_length)
    inputs = v5.move_inputs(bundle["model_inputs"], device)
    model = stage176a._construct_model(prov, metadata, state, device)
    _require(not model.training, "model must be in eval mode")
    gate = getattr(model, "frame_gate", None)
    classifier = getattr(gate, "frame_classifier", None)
    _require(isinstance(classifier, nn.Linear) and classifier.out_features == 1,
             "canonical frame head must be one scalar Linear layer")
    output = stage176a._forward(model, inputs, dev, prov, device, batch_size)
    count = len(dev)
    logits = stage177e._tensor(output, "frame_logit", count).reshape(-1)
    probs = stage177e._tensor(output, "frame_prob", count).reshape(-1)
    representation = stage177e._tensor(output, "frame_pair_repr", count)
    _require(representation.ndim == 2 and representation.shape[1] == classifier.in_features,
             "frame_pair_repr is not the exact frame classifier input")
    weight = classifier.weight.detach().float().cpu().reshape(-1)
    bias = float(classifier.bias.detach().float().cpu().reshape(-1)[0]) if classifier.bias is not None else 0.0
    reconstructed = representation @ weight + bias
    errors = (reconstructed - logits).abs()
    _require(float(errors.max()) <= 1e-5, f"frame-logit reconstruction error exceeds 1e-5: {float(errors.max())}")
    norm = torch.linalg.vector_norm(weight)
    _require(float(norm) > 0, "frame classifier has zero weight norm")
    unit = weight / norm
    contract = {"representation_source": 'output["frame_pair_repr"]', "hook_used": False,
                "hooked_module_name": None, "tensor_shape": list(representation.shape),
                "frame_head_module": "frame_gate.frame_classifier", "head_structure": "Linear",
                "linear_or_mlp": "linear", "representation_immediately_precedes_frame_logit": True,
                "row_order_verified": True, "readout_equation": "frame_logit = w dot h + b",
                "weight_norm": float(norm), "bias": bias,
                "maximum_reconstruction_absolute_error": float(errors.max()),
                "reconstruction_tolerance": 1e-5}
    return {"frame_logit": logits, "frame_prob": probs, "representation": representation,
            "weight": weight, "unit": unit, "bias": torch.tensor(bias),
            "projection": representation @ unit, "reconstructed": reconstructed,
            "reconstruction_error": errors}, contract


def _label_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = {"frame": "frame_compatible_label", "predicate": "predicate_covered_label",
                  "sufficiency": "sufficiency_label", "entitlement": "entitlement_label",
                  "temporal": "temporal_label", "final": "final_label"}
    schema = {axis: {"field": field, "available": all(field in row for row in records),
                     "derived": False} for axis, field in candidates.items()}
    for axis in ("frame", "predicate", "sufficiency", "entitlement", "temporal"):
        info = schema[axis]
        if info["available"]:
            values = [_required_int(row.get(info["field"]), f"semantic_schema.{axis}.{index}")
                      for index, row in enumerate(records)]
            _require(set(values) <= {0, 1}, f"semantic axis {axis} is not binary")
            info["values"] = sorted(set(values))
    if schema["final"]["available"]:
        _require(all(row.get("final_label") is not None and str(row.get("final_label")) != "" for row in records),
                 "native final labels contain missing values")
        schema["final"]["values"] = sorted({str(row["final_label"]) for row in records})
    return schema


def _vector_metrics(rows: list[dict[str, Any]], reps: torch.Tensor, outputs: dict[str, torch.Tensor],
                    hard: dict[tuple[int, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["pair_id"])].append(index)
    result: list[dict[str, Any]] = []
    for pair_id, indices in groups.items():
        anchor = [i for i in indices if str(rows[i]["intervention_type"]) == "none"][0]
        anchor_repr = reps[anchor]
        anchor_projection = float(outputs["projection"][anchor])
        anchor_logit = float(outputs["frame_logit"][anchor])
        for i in indices:
            vector = reps[i]
            projection = float(outputs["projection"][i])
            orthogonal = vector - outputs["unit"] * projection
            displacement = vector - anchor_repr
            projection_delta = projection - anchor_projection
            orthogonal_displacement = displacement - outputs["unit"] * float(displacement @ outputs["unit"])
            stable = (i, str(rows[i]["id"]))
            cohort = hard.get(stable, {}).get("stage176_cohort", "none")
            gold = int(rows[i]["frame_compatible_label"])
            logit = float(outputs["frame_logit"][i])
            item = {"stable_row_index": i, "row_id": str(rows[i]["id"]), "pair_id": pair_id,
                    "intervention_type": str(rows[i]["intervention_type"]),
                    "gold_final_label": str(rows[i]["final_label"]), "final_label": str(rows[i]["final_label"]),
                    "gold_frame_label": gold, "frame_compatible_label": gold,
                    "predicate_covered_label": rows[i].get("predicate_covered_label"),
                    "sufficiency_label": rows[i].get("sufficiency_label"),
                    "entitlement_label": rows[i].get("entitlement_label"),
                    "temporal_label": rows[i].get("temporal_label"),
                    "frame_logit": logit, "frame_prob": float(outputs["frame_prob"][i]),
                    "frame_prediction": int(logit >= 0), "frame_correct": int(logit >= 0) == gold,
                    "representation_norm": float(torch.linalg.vector_norm(vector)),
                    "head_direction_projection": projection,
                    "orthogonal_component_norm": float(torch.linalg.vector_norm(orthogonal)),
                    "representation_displacement_from_none": float(torch.linalg.vector_norm(displacement)),
                    "frame_logit_delta_from_none": logit - anchor_logit,
                    "head_projection_delta_from_none": projection_delta,
                    "orthogonal_displacement_from_none": float(torch.linalg.vector_norm(orthogonal_displacement)),
                    "stage176_cohort": cohort, "hard39": cohort != "none"}
            result.append(item)
    result.sort(key=lambda row: int(row["stable_row_index"]))
    return result, {"rows": len(result), "exact_head_direction": True,
                    "canonical_anchor_from_metadata_only": True}


def _centroids(rows: list[dict[str, Any]], reps: torch.Tensor) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row["pair_id"])].append(i)
    output: dict[int, dict[str, Any]] = {}
    for indices in groups.values():
        for i in indices:
            gold = int(rows[i]["frame_compatible_label"])
            centroids = {}
            for label in (0, 1):
                members = [j for j in indices if int(rows[j]["frame_compatible_label"]) == label and j != i]
                _require(bool(members), "LOO centroid has no member")
                centroids[label] = reps[members].mean(dim=0)
            d0 = float(torch.linalg.vector_norm(reps[i] - centroids[0]))
            d1 = float(torch.linalg.vector_norm(reps[i] - centroids[1]))
            score = d0 - d1
            output[i] = {"distance_to_compatible_centroid": d1,
                         "distance_to_incompatible_centroid": d0,
                         "signed_centroid_score": score, "centroid_prediction": int(score >= 0),
                         "centroid_correct": int(score >= 0) == gold}
    return output, {"gold_conditioned": True, "transductive": True, "leave_one_row_out": True,
                    "inference_proposal": False, "fitted_classifier": False}


def _contingency(rows: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [(axis, info["field"]) for axis, info in schema.items()
              if axis != "frame" and info["available"]]
    output = []
    for axis, field in fields:
        grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["gold_frame_label"], row[field])].append(row)
        for (frame, other), group in sorted(grouped.items(), key=lambda item: str(item[0])):
            output.append({"axis": axis, "axis_field": field, "frame_label": frame,
                           "axis_label": other, "row_count": len(group),
                           "frame_error_count": sum(not row["frame_correct"] for row in group),
                           "frame_error_rate": _mean(not row["frame_correct"] for row in group),
                           "mean_frame_logit": _mean(row["frame_logit"] for row in group),
                           "median_frame_logit": _median(row["frame_logit"] for row in group),
                           "hard39_count": sum(row["hard39"] for row in group),
                           "beneficial_count": sum(row["stage176_cohort"] == "beneficial_correction" for row in group),
                           "harmful_count": sum(row["stage176_cohort"] == "harmful_regression" for row in group)})
    return output


def _semantic_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axes = (("sufficiency", "sufficiency_label", "sufficient", "insufficient"),
            ("predicate", "predicate_covered_label", "predicate-covered", "predicate-uncovered"))
    output = []
    for axis, field, positive, negative in axes:
        for frame in (1, 0):
            for other in (1, 0):
                group = [row for row in rows if int(row["gold_frame_label"]) == frame and int(row[field]) == other]
                name = f"frame-{'compatible' if frame else 'incompatible'} + {positive if other else negative}"
                errors = [row for row in group if not row["frame_correct"]]
                output.append({"semantic_stratum": name, "axis": axis, "frame_label": frame,
                               "axis_label": other, "support_count": len(group),
                               "frame_accuracy": _mean(row["frame_correct"] for row in group),
                               "false_compatible_rate": _mean(row["frame_prediction"] == 1 and row["gold_frame_label"] == 0 for row in group),
                               "false_incompatible_rate": _mean(row["frame_prediction"] == 0 and row["gold_frame_label"] == 1 for row in group),
                               "frame_logit_distribution": _distribution(row["frame_logit"] for row in group),
                               "frame_error_count": len(errors), "hard39_count": sum(row["hard39"] for row in group),
                               "hard39_rate": _mean(row["hard39"] for row in group),
                               "beneficial_count": sum(row["stage176_cohort"] == "beneficial_correction" for row in group),
                               "harmful_count": sum(row["stage176_cohort"] == "harmful_regression" for row in group),
                               "annotation_contradiction_claimed": False})
    return output


def _duplicates(all_rows: list[dict[str, Any]], dev_ids: set[str], hard_ids: set[str]
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[(_norm_text(row["claim"]), _norm_text(row["evidence"]))].append(row)
    duplicate_rows, conflict_groups = [], []
    auxiliaries = ("predicate_covered_label", "sufficiency_label", "entitlement_label", "temporal_label")
    for key, group in groups.items():
        if len(group) < 2:
            continue
        frame_values = sorted({str(row.get("frame_compatible_label")) for row in group})
        aux_conflicts = {field: sorted({str(row.get(field)) for row in group}) for field in auxiliaries
                         if field in group[0] and len({str(row.get(field)) for row in group}) > 1}
        final_values = sorted({str(row.get("final_label")) for row in group})
        item = {"normalized_claim": key[0], "normalized_evidence": key[1], "row_count": len(group),
                "row_ids": [str(row["id"]) for row in group], "pair_ids": sorted({str(row["pair_id"]) for row in group}),
                "frame_labels": frame_values, "conflicting_frame_label": len(frame_values) > 1,
                "conflicting_auxiliary_labels": aux_conflicts, "conflicting_final_label": len(final_values) > 1,
                "dev_overlap_count": sum(str(row["id"]) in dev_ids for row in group),
                "hard39_overlap_count": sum(str(row["id"]) in hard_ids for row in group),
                "metadata_only_difference": len({str(row["claim"]) + "\0" + str(row["evidence"]) for row in group}) == 1}
        duplicate_rows.append(item)
        if item["conflicting_frame_label"]:
            conflict_groups.append(item)
    near = []
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_pair[str(row["pair_id"])].append(row)
    for pair_id, group in sorted(by_pair.items()):
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                claim_j = _jaccard(_tokens(left["claim"]), _tokens(right["claim"]))
                evidence_j = _jaccard(_tokens(left["evidence"]), _tokens(right["evidence"]))
                if (claim_j + evidence_j) / 2 >= .8 and (_norm_text(left["claim"]), _norm_text(left["evidence"])) != (_norm_text(right["claim"]), _norm_text(right["evidence"])):
                    near.append({"pair_id": pair_id, "left_row_id": left["id"], "right_row_id": right["id"],
                                 "claim_token_jaccard": claim_j, "evidence_token_jaccard": evidence_j,
                                 "mean_token_jaccard": (claim_j + evidence_j) / 2,
                                 "claim_token_length_left": len(_tokens(left["claim"])),
                                 "claim_token_length_right": len(_tokens(right["claim"])),
                                 "evidence_token_length_left": len(_tokens(left["evidence"])),
                                 "evidence_token_length_right": len(_tokens(right["evidence"])),
                                 "frame_labels_equal": left["frame_compatible_label"] == right["frame_compatible_label"],
                                 "descriptive_review_only": True, "label_error_claimed": False})
    summary = {"duplicate_group_count": len(duplicate_rows),
               "conflicting_frame_label_group_count": len(conflict_groups),
               "conflicting_auxiliary_label_group_count": sum(bool(row["conflicting_auxiliary_labels"]) for row in duplicate_rows),
               "conflicting_final_label_group_count": sum(row["conflicting_final_label"] for row in duplicate_rows),
               "conflicting_frame_hard39_overlap": sum(row["hard39_overlap_count"] for row in conflict_groups),
               "conflicting_frame_dev_overlap": sum(row["dev_overlap_count"] for row in conflict_groups),
               "near_duplicate_queue_count": len(near), "near_duplicate_is_error_evidence": False}
    return duplicate_rows, near, summary


def _interventions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for family in sorted({row["intervention_type"] for row in rows}):
        group = [row for row in rows if row["intervention_type"] == family]
        output.append({"intervention_type": family, "row_count": len(group),
                       "mean_frame_logit": _mean(row["frame_logit"] for row in group),
                       "frame_error_rate": _mean(not row["frame_correct"] for row in group),
                       "mean_head_direction_projection": _mean(row["head_direction_projection"] for row in group),
                       "mean_representation_norm": _mean(row["representation_norm"] for row in group),
                       "mean_canonical_displacement": _mean(row["representation_displacement_from_none"] for row in group),
                       "target_inferred_from_family": False})
    return output


def _sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
    output = []
    for pair_id, group in sorted(groups.items()):
        compatible = [row for row in group if row["gold_frame_label"] == 1]
        incompatible = [row for row in group if row["gold_frame_label"] == 0]
        cm, im = _mean(row["frame_logit"] for row in compatible), _mean(row["frame_logit"] for row in incompatible)
        cs = statistics.pstdev(row["frame_logit"] for row in compatible)
        ins = statistics.pstdev(row["frame_logit"] for row in incompatible)
        pooled = math.sqrt((cs * cs + ins * ins) / 2)
        hard = any(row["hard39"] for row in group)
        movement = _median(row["representation_displacement_from_none"] for row in group if row["intervention_type"] != "none") or 0.0
        projection = _median(abs(row["head_projection_delta_from_none"]) for row in group if row["intervention_type"] != "none") or 0.0
        orthogonal = _median(row["orthogonal_displacement_from_none"] for row in group if row["intervention_type"] != "none") or 0.0
        gap = float(cm) - float(im)
        category = "adequate_head_direction_movement_bias_or_readout_ambiguous"
        if movement <= 1e-8:
            category = "representation_movement_small"
        elif orthogonal > 2 * projection:
            category = "movement_mostly_head_orthogonal"
        elif gap < 0:
            category = "head_direction_movement_wrong_direction"
        elif min(abs(row["frame_logit"]) for row in group) < .1:
            category = "small_margin_near_native_threshold"
        output.append({"pair_id": pair_id, "hard_error_pair": hard,
                       "compatible_logit_range": max(row["frame_logit"] for row in compatible) - min(row["frame_logit"] for row in compatible),
                       "incompatible_logit_range": max(row["frame_logit"] for row in incompatible) - min(row["frame_logit"] for row in incompatible),
                       "between_label_centroid_gap": gap, "within_compatible_spread": cs,
                       "within_incompatible_spread": ins, "pooled_within_spread": pooled,
                       "gap_over_pooled_within_spread": gap / pooled if pooled else None,
                       "median_canonical_projection_response": projection,
                       "median_canonical_orthogonal_response": orthogonal,
                       "diagnostic_category": category})
    return output


def _matched_percentiles(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    metrics = ("representation_displacement_from_none", "head_projection_delta_from_none",
               "orthogonal_displacement_from_none", "representation_norm", "frame_logit")
    hard_errors = [row for row in rows if row["hard39"] and not row["frame_correct"]]
    for row in hard_errors:
        pool = [other for other in rows if other["frame_correct"] and
                other["intervention_type"] == row["intervention_type"] and
                other["gold_frame_label"] == row["gold_frame_label"]]
        level = "intervention_and_gold_frame_label"
        if len(pool) < 5:
            pool = [other for other in rows if other["frame_correct"] and
                    other["gold_frame_label"] == row["gold_frame_label"]]
            level = "gold_frame_label_only"
        _require(len(pool) >= 5, f"matched comparison pool below five for {row['row_id']}")
        row["matching_level"] = level
        row["matching_pool_size"] = len(pool)
        for metric in metrics:
            value = abs(row[metric]) if metric == "head_projection_delta_from_none" else row[metric]
            reference = [abs(other[metric]) if metric == "head_projection_delta_from_none" else other[metric] for other in pool]
            row[f"{metric}_percentile"] = sum(candidate <= value for candidate in reference) / len(reference)
    medians = {metric: _median(row[f"{metric}_percentile"] for row in hard_errors) for metric in metrics}
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hard_errors:
        pair_groups[row["pair_id"]].append(row)
    pair_ids, rng = sorted(pair_groups), random.Random(seed)
    boot: dict[str, list[float]] = {metric: [] for metric in metrics}
    if pair_ids:
        for _ in range(samples):
            sampled = [row for _pair in (rng.choice(pair_ids) for _ in pair_ids) for row in pair_groups[_pair]]
            for metric in metrics:
                value = _median(row[f"{metric}_percentile"] for row in sampled)
                if value is not None:
                    boot[metric].append(value)
    return {"hard39_frame_head_error_count": len(hard_errors), "median_percentiles": medians,
            "matching_levels": dict(Counter(row["matching_level"] for row in hard_errors)),
            "bootstrap_unit": "pair_id",
            "bootstrap_ci": {metric: {"low": _percentile(values, .025), "high": _percentile(values, .975)}
                             for metric, values in boot.items()}}


def _centroid_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full = rows
    hard = [row for row in rows if row["hard39"]]
    def view(group: list[dict[str, Any]]) -> dict[str, Any]:
        head_wrong = [row for row in group if not row["frame_correct"]]
        return {"rows": len(group), "centroid_sign_accuracy": _mean(row["centroid_correct"] for row in group),
                "centroid_auroc": _auc([row["gold_frame_label"] for row in group], [row["signed_centroid_score"] for row in group]),
                "centroid_average_precision": _ap([row["gold_frame_label"] for row in group], [row["signed_centroid_score"] for row in group]),
                "native_frame_logit_auroc": _auc([row["gold_frame_label"] for row in group], [row["frame_logit"] for row in group]),
                "corrected_frame_head_errors": sum(row["centroid_correct"] for row in head_wrong),
                "introduced_errors": sum(row["frame_correct"] and not row["centroid_correct"] for row in group),
                "net_corrections": sum(row["centroid_correct"] for row in head_wrong) - sum(row["frame_correct"] and not row["centroid_correct"] for row in group),
                "representation_failure_count": sum(not row["frame_correct"] and not row["centroid_correct"] for row in group),
                "readout_alignment_candidate_count": sum(not row["frame_correct"] and row["centroid_correct"] for row in group),
                "centroid_collateral_risk_count": sum(row["frame_correct"] and not row["centroid_correct"] for row in group)}
    summary = {"full_dev": view(full), "hard39": view(hard), "gold_conditioned_transductive_diagnostic": True,
               "deployment_or_classifier_proposal": False}
    csv_rows = [{"subset": name, **metrics} for name, metrics in (("full_dev", summary["full_dev"]), ("hard39", summary["hard39"]))]
    return csv_rows, summary


def _bh(rows: list[dict[str, Any]]) -> None:
    candidates = sorted([(float(row["p_value"]), index) for index, row in enumerate(rows)
                         if row.get("p_value") is not None])
    adjusted = 1.0
    for reverse_index in range(len(candidates) - 1, -1, -1):
        p_value, original = candidates[reverse_index]
        adjusted = min(adjusted, p_value * len(candidates) / (reverse_index + 1))
        rows[original]["bh_adjusted_p_value"] = adjusted
        rows[original]["bh_significant"] = adjusted < .05


def _interference(strata: list[dict[str, Any]], rows: list[dict[str, Any]], samples: int, seed: int
                 ) -> list[dict[str, Any]]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    ids = sorted(pairs)
    output = []
    for stratum in strata:
        frame, axis, label = int(stratum["frame_label"]), stratum["axis"], int(stratum["axis_label"])
        field = "sufficiency_label" if axis == "sufficiency" else "predicate_covered_label"
        inside = [row for row in rows if row["gold_frame_label"] == frame and int(row[field]) == label]
        outside = [row for row in rows if not (row["gold_frame_label"] == frame and int(row[field]) == label)]
        rate_in = _mean(row["hard39"] for row in inside) or 0.0
        rate_out = _mean(row["hard39"] for row in outside) or 0.0
        rd = rate_in - rate_out
        rr = rate_in / rate_out if rate_out else (math.inf if rate_in else None)
        rng, boot = random.Random(seed + len(output)), []
        for _ in range(samples):
            sampled = [row for _pair in (rng.choice(ids) for _ in ids) for row in pairs[_pair]]
            a = [row for row in sampled if row["gold_frame_label"] == frame and int(row[field]) == label]
            b = [row for row in sampled if not (row["gold_frame_label"] == frame and int(row[field]) == label)]
            if a and b:
                boot.append((_mean(row["hard39"] for row in a) or 0) - (_mean(row["hard39"] for row in b) or 0))
        p_value = 2 * min(_mean(value <= 0 for value in boot) or 0, _mean(value >= 0 for value in boot) or 0) if boot else None
        output.append({"semantic_stratum": stratum["semantic_stratum"], "axis": axis,
                       "support_count": len(inside), "hard_error_count": sum(row["hard39"] for row in inside),
                       "hard_error_rate": rate_in, "non_stratum_error_rate": rate_out,
                       "risk_ratio": rr, "risk_difference": rd,
                       "risk_difference_ci_low": _percentile(boot, .025),
                       "risk_difference_ci_high": _percentile(boot, .975), "p_value": p_value,
                       "minimum_support_met": len(inside) >= 10,
                       "beneficial_count": sum(row["stage176_cohort"] == "beneficial_correction" for row in inside),
                       "harmful_count": sum(row["stage176_cohort"] == "harmful_regression" for row in inside)})
    _bh(output)
    for row in output:
        row["gate_pass"] = bool(row["minimum_support_met"] and row["risk_difference"] >= .20 and
                                row["risk_ratio"] is not None and row["risk_ratio"] >= 2 and
                                row["risk_difference_ci_low"] is not None and row["risk_difference_ci_low"] > 0 and
                                row.get("bh_significant") and
                                (row["beneficial_count"] >= 13 or row["harmful_count"] >= 7))
        row["annotation_error_claimed"] = False
    return output


def _diagnose(duplicates: dict[str, Any], interference: list[dict[str, Any]], rows: list[dict[str, Any]],
              centroid: dict[str, Any], matched: dict[str, Any], sensitivity: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    hard_errors = [row for row in rows if row["hard39"] and not row["frame_correct"]]
    direct = duplicates["conflicting_frame_label_group_count"] >= 1 and (
        duplicates["conflicting_frame_hard39_overlap"] > 0 or duplicates["conflicting_frame_dev_overlap"] > 0)
    semantic = not direct and any(row["gate_pass"] for row in interference)
    representation_failure_fraction = (_mean(not row["centroid_correct"] for row in hard_errors)
                                       if hard_errors else None)
    hard_pairs = [row for row in sensitivity if row["hard_error_pair"]]
    full_ratio = _median(row["gap_over_pooled_within_spread"] for row in sensitivity
                         if row["gap_over_pooled_within_spread"] is not None)
    hard_ratio = _median(row["gap_over_pooled_within_spread"] for row in hard_pairs
                         if row["gap_over_pooled_within_spread"] is not None)
    displacement_low = (matched["median_percentiles"]["representation_displacement_from_none"] or 1) <= .25
    gap_weaker = hard_ratio is not None and full_ratio is not None and hard_ratio < full_ratio
    insensitive = bool(not direct and not semantic and hard_errors and representation_failure_fraction >= .70 and displacement_low and gap_weaker)
    hard_view = centroid["hard39"]
    native_auc, centroid_auc = hard_view["native_frame_logit_auroc"], hard_view["centroid_auroc"]
    candidate_fraction = hard_view["readout_alignment_candidate_count"] / len(hard_errors) if hard_errors else 0
    readout = bool(not direct and not semantic and not insensitive and
                   hard_view["corrected_frame_head_errors"] >= 5 and hard_view["introduced_errors"] <= 1 and
                   centroid_auc is not None and centroid_auc >= .70 and native_auc is not None and
                   centroid_auc - native_auc >= .10 and candidate_fraction >= .40)
    gates = [("direct_label_inconsistency", direct, DIRECT, "STAGE179B_FRAME_LABEL_CONSISTENCY_ADJUDICATION_AUDIT"),
             ("semantic_cross_channel_interference", semantic, INTERFERENCE, "STAGE179B_FRAME_AXIS_DEFINITION_AND_ROUTING_FEASIBILITY_AUDIT"),
             ("input_representation_insensitivity", insensitive, INSENSITIVE, "STAGE179B_FRAME_INPUT_CONTRAST_LOCALIZATION_FEASIBILITY_AUDIT"),
             ("readout_alignment_failure", readout, READOUT, "STAGE179B_FRAME_READOUT_REPARAMETERIZATION_FEASIBILITY_AUDIT")]
    selected = next((item for item in gates if item[1]), ("mixed_or_insufficient", True, MIXED, "STAGE180_HARD_FRAME_CASE_MANUAL_TAXONOMY_AND_DATA_DESIGN_AUDIT"))
    evidence = [{"priority": index + 1, "gate": name, "passed": passed, "decision_if_selected": decision,
                 "next_stage_if_selected": next_stage} for index, (name, passed, decision, next_stage) in enumerate(gates)]
    evidence.append({"priority": 5, "gate": "mixed_or_insufficient", "passed": selected[2] == MIXED,
                     "decision_if_selected": MIXED,
                     "next_stage_if_selected": "STAGE180_HARD_FRAME_CASE_MANUAL_TAXONOMY_AND_DATA_DESIGN_AUDIT"})
    diagnosis = {"selected_gate": selected[0], "decision": selected[2], "authorized_next_stage": selected[3],
                 "direct_duplicate_conflict": direct, "semantic_interference": semantic,
                 "hard39_frame_head_errors": len(hard_errors),
                 "representation_failure_fraction_among_hard39_frame_errors": representation_failure_fraction,
                 "hard_error_displacement_median_at_or_below_matched_p25": displacement_low,
                 "hard_pair_gap_spread_weaker_than_full_dev": gap_weaker,
                 "readout_candidate_fraction_among_hard39_frame_errors": candidate_fraction,
                 "priority_order_enforced": True}
    return selected[2], evidence, diagnosis


def _render_markdown(report: dict[str, Any]) -> str:
    c = report["centroid_diagnostic"]["hard39"]
    return "\n".join([
        "# Stage179-A frame semantics and input-representation audit", "",
        "## Decision", "", f"`{report['decision']}`", "",
        "## Scope and contracts", "",
        "The audit uses only the canonical seed-174, epoch-20 Stage174 baseline on the deterministic clean controlled dev split. All provenance, checkpoint, upstream-decision, split, and hard-39 identity checks complete before model construction and forward.", "",
        "The native frame target is read only from `frame_compatible_label`; it is not inferred from intervention family or final label. Other epistemic axes are retained as separate descriptive channels. A cross-channel combination is not, by itself, an annotation contradiction.", "",
        "## Exact frame-head input and readout", "",
        f"The exact input is native `output[\"frame_pair_repr\"]`, shape `{report['frame_head_contract']['tensor_shape']}`. The head is `frame_gate.frame_classifier`, a scalar linear layer immediately preceding `frame_logit`. Maximum `w·h+b` reconstruction error was `{report['frame_head_contract']['maximum_reconstruction_absolute_error']:.9g}` (required ≤ `1e-5`). No hook or model modification was used.", "",
        "## Semantic structure and duplicate consistency", "",
        f"Exact duplicate groups: {report['duplicate_consistency']['duplicate_group_count']}; conflicting frame-label groups: {report['duplicate_consistency']['conflicting_frame_label_group_count']}. Only normalized exact claim–evidence conflicts are treated as direct label inconsistency. Near duplicates are a review queue and do not establish label error.", "",
        "Semantic strata describe structural interaction between frame compatibility and other native axes; they do not re-judge labels or claim annotation error.", "",
        "## Intervention and representation/readout decomposition", "",
        "Intervention families are used only for descriptive grouping. Every pair has exactly one metadata `none` anchor. Canonical displacement is decomposed into exact head-direction and orthogonal components; `none` is never used to infer a target.", "",
        "## Gold-conditioned centroid diagnostic", "",
        f"On hard-39, centroid AUROC was `{c['centroid_auroc']}`, corrected head errors were `{c['corrected_frame_head_errors']}`, introduced errors were `{c['introduced_errors']}`, and net corrections were `{c['net_corrections']}`. This is a pair-level, leave-one-row-out, gold-conditioned transductive diagnostic—not a deployable classifier or inference proposal.", "",
        "Representation failure means both the native frame head and centroid diagnostic are wrong. A readout-alignment candidate means the native frame head is wrong while the centroid diagnostic is correct. These are diagnostic definitions, not causal findings.", "",
        "## Hard-39 attribution and sensitivity", "",
        "The fixed Stage176-A beneficial-25 and harmful-14 cohorts are reported separately with semantic tuple, intervention, native outputs, exact projection, canonical movement, centroid result, and diagnostic class. Pair sensitivity compares between-label gap with within-label spread and separates small movement, head-orthogonal movement, wrong-direction movement, small-margin cases, and ambiguous adequate movement.", "",
        "## Decision gate and limitations", "",
        f"Only the first passing priority gate is selected. Authorized next stage: `{report['stage179b_gate']['authorized_next_stage']}`. No training, relabeling, readout reparameterization, architecture change, calibration, threshold search, fitted probe, or external evaluation is authorized.", "",
        "This is a single-seed internal observational audit. Bootstrap intervals quantify pair resampling, not training-seed uncertainty. No causal mechanism is claimed.", ""
    ])


def _blocked(output_dir: Path, error: Exception, failure_stage: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail = {"error_type": type(error).__name__, "error": str(error),
              "failure_stage": failure_stage, "traceback": traceback.format_exc()}
    report = {"stage": STAGE, "decision": BLOCKED, "scope": None,
              "input_validation": detail, "checkpoint_contract": None, "pair_topology": None,
              "frame_head_contract": None, "label_semantic_structure": None,
              "duplicate_consistency": None, "intervention_response": None,
              "representation_readout_decomposition": None, "centroid_diagnostic": None,
              "hard39_attribution": None, "frame_logit_sensitivity": None,
              "semantic_interference_analysis": None, "diagnosis": detail,
              "stage179b_gate": {"decision": BLOCKED},
              "limitations": ["Validation failed; no diagnostic conclusion is available."],
              "safety_policy": {"training": False, "optimizer": False, "backward": False,
                                "train_mode": False, "local_execution_validation": False}}
    _write_json(output_dir / OUTPUTS["json"], report)
    (output_dir / OUTPUTS["md"]).write_text(
        f"# Stage179-A blocked\n\n**Decision:** `{BLOCKED}`\n\nFailure stage: `{failure_stage}`\n\n"
        f"`{type(error).__name__}: {error}`\n\n```text\n{detail['traceback']}\n```\n", encoding="utf-8")
    for key, filename in OUTPUTS.items():
        if key not in ("json", "md"):
            _write_csv(output_dir / filename, [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/controlled_v5_v3_without_time_swap.jsonl"))
    for name in ("stage176a-report", "stage176a-row-transitions", "stage177a-report",
                 "stage177e-report", "stage178a-report", "stage178a-hard39-offset-attribution",
                 "baseline-provenance", "baseline-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=179)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    current = "argument_validation"
    try:
        _require(args.device == "cuda", "Stage179-A requires --device cuda")
        args.eval_batch_size = _required_int(args.eval_batch_size, "cli.eval_batch_size")
        args.bootstrap_samples = _required_int(args.bootstrap_samples, "cli.bootstrap_samples")
        args.bootstrap_seed = _required_int(args.bootstrap_seed, "cli.bootstrap_seed")
        _require(args.eval_batch_size > 0 and args.bootstrap_samples > 0, "batch/bootstrap counts must be positive")
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        names = ("data", "stage176a_report", "stage176a_row_transitions", "stage177a_report",
                 "stage177e_report", "stage178a_report", "stage178a_hard39_offset_attribution",
                 "baseline_provenance", "baseline_checkpoint")
        current = "input_path_validation"
        paths = {name: getattr(args, name).resolve() for name in names}
        for name, path in paths.items():
            _require(path.is_file(), f"{name} does not exist: {path}")
        current = "upstream_decision_validation"
        upstream = _validate_reports({name: _read_json(paths[f"{name}_report"])
                                      for name in ("stage176a", "stage177a", "stage177e", "stage178a")})
        current = "provenance_validation"
        provenance = _read_json(paths["baseline_provenance"])
        provenance_check = stage177e._validate_provenance("baseline", provenance, paths["data"])
        current = "checkpoint_validation"
        state, metadata, checkpoint = stage177e._load_and_validate_checkpoint(
            "baseline", paths["baseline_checkpoint"], provenance, "off", 0.0)
        current = "deterministic_split_validation"
        train, dev, split = stage177e._validate_split(paths["data"], provenance)
        pair_topology = {"total": _validate_topology(train + dev, "all"),
                         "train": _validate_topology(train, "train"), "dev": _validate_topology(dev, "dev"),
                         "total_pairs": 300, "train_pairs": 240, "dev_pairs": 60,
                         "pair_overlap": 0}
        current = "hard39_identity_validation"
        hard, hard_check = _validate_hard_rows(_read_csv(paths["stage176a_row_transitions"]),
                                                _read_csv(paths["stage178a_hard39_offset_attribution"]), dev)
        input_validation = {"status": "passed", "upstream": upstream, "provenance": provenance_check,
                            "checkpoint": checkpoint, "split": split, "hard39": hard_check,
                            "completed_before_model_construction_and_forward": True}
        current = "dev_model_forward"
        outputs, head_contract = _evaluate(dev, provenance, metadata, state,
                                           torch.device("cuda"), args.eval_batch_size)
        current = "representation_analysis"
        rows, decomposition = _vector_metrics(dev, outputs["representation"], outputs, hard)
        centroids, centroid_contract = _centroids(dev, outputs["representation"])
        for row in rows:
            row.update(centroids[int(row["stable_row_index"])])
            row["diagnostic_class"] = ("representation_failure" if not row["frame_correct"] and not row["centroid_correct"]
                                        else "readout_alignment_candidate" if not row["frame_correct"] and row["centroid_correct"]
                                        else "centroid_collateral_risk" if row["frame_correct"] and not row["centroid_correct"]
                                        else "both_correct")
            row["native_label_tuple"] = [row.get(key) for key in ("gold_frame_label", "predicate_covered_label",
                                                                   "sufficiency_label", "entitlement_label",
                                                                   "temporal_label", "gold_final_label")]
        schema = _label_schema(dev)
        contingency = _contingency(rows, schema)
        strata = _semantic_strata(rows)
        stratum_lookup = {(row["frame_label"], row["axis"], row["axis_label"]): row["semantic_stratum"] for row in strata}
        for row in rows:
            row["semantic_stratum"] = "; ".join((
                stratum_lookup[(row["gold_frame_label"], "sufficiency", int(row["sufficiency_label"]))],
                stratum_lookup[(row["gold_frame_label"], "predicate", int(row["predicate_covered_label"]))]))
        current = "duplicate_analysis"
        duplicates, near, duplicate_summary = _duplicates(train + dev, {str(row["id"]) for row in dev},
                                                           {key[1] for key in hard})
        intervention = _interventions(rows)
        readout_rows = [{"stable_row_index": row["stable_row_index"], "row_id": row["row_id"],
                         "representation_norm": row["representation_norm"],
                         "unit_head_direction_projection": row["head_direction_projection"],
                         "bias": head_contract["bias"], "reconstructed_logit": float(outputs["reconstructed"][int(row["stable_row_index"])]),
                         "native_frame_logit": row["frame_logit"],
                         "reconstruction_absolute_error": float(outputs["reconstruction_error"][int(row["stable_row_index"])])}
                        for row in rows]
        matched = _matched_percentiles(rows, args.bootstrap_samples, args.bootstrap_seed + 1000)
        centroid_csv, centroid_summary = _centroid_summary(rows)
        centroid_summary.update(centroid_contract)
        sensitivity = _sensitivity(rows)
        interference = _interference(strata, rows, args.bootstrap_samples, args.bootstrap_seed)
        decision, decision_rows, diagnosis = _diagnose(duplicate_summary, interference, rows,
                                                        centroid_summary, matched, sensitivity)
        hard_rows = [row for row in rows if row["hard39"]]
        hard_attribution = {
            "beneficial": {"rows": 25,
                           "representation_failure_count": sum(row["diagnostic_class"] == "representation_failure" for row in hard_rows if row["stage176_cohort"] == "beneficial_correction"),
                           "readout_candidate_count": sum(row["diagnostic_class"] == "readout_alignment_candidate" for row in hard_rows if row["stage176_cohort"] == "beneficial_correction")},
            "harmful": {"rows": 14,
                        "representation_failure_count": sum(row["diagnostic_class"] == "representation_failure" for row in hard_rows if row["stage176_cohort"] == "harmful_regression"),
                        "readout_candidate_count": sum(row["diagnostic_class"] == "readout_alignment_candidate" for row in hard_rows if row["stage176_cohort"] == "harmful_regression")},
            "semantic_stratum_distribution": dict(Counter(row["semantic_stratum"] for row in hard_rows)),
            "intervention_distribution": dict(Counter(row["intervention_type"] for row in hard_rows))}
        report = {"stage": STAGE, "decision": decision,
                  "scope": {"data": str(paths["data"]), "seed": 174, "epochs": 20,
                            "selected_epoch": 20, "architecture": "v6b_minimal", "backbone": "mamba",
                            "model": MODEL_NAME, "clean_controlled_data_only": True, "evaluation_only": True,
                            "single_seed": True, "device": args.device, "eval_batch_size": args.eval_batch_size,
                            "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed},
                  "input_validation": input_validation,
                  "checkpoint_contract": {**checkpoint, "stage174c_off": True, "stage175b_off": True,
                                          "stage177c_off": True, "final_ce_source": 'output["logits"]',
                                          "loss_logits_used": False, "external_data": False, "time_swap": False},
                  "pair_topology": pair_topology, "frame_head_contract": head_contract,
                  "label_semantic_structure": {"native_schema": schema, "contingency_rows": len(contingency),
                                               "semantic_strata": len(strata), "targets_inferred_from_intervention_or_final_label": False,
                                               "cross_axis_combination_equals_annotation_contradiction": False},
                  "duplicate_consistency": duplicate_summary,
                  "intervention_response": {"families": len(intervention), "canonical_anchor": "metadata intervention_type == none",
                                            "exactly_one_anchor_per_pair": True, "target_inferred_from_none": False},
                  "representation_readout_decomposition": {**decomposition, "matched_hard_error_comparison": matched},
                  "centroid_diagnostic": centroid_summary, "hard39_attribution": hard_attribution,
                  "frame_logit_sensitivity": {"pair_rows": len(sensitivity),
                                              "hard_error_pairs": sum(row["hard_error_pair"] for row in sensitivity),
                                              "categories": dict(Counter(row["diagnostic_category"] for row in sensitivity))},
                  "semantic_interference_analysis": {"rows": len(interference),
                                                     "passing_strata": [row["semantic_stratum"] for row in interference if row["gate_pass"]],
                                                     "minimum_support": 10, "multiple_comparison_correction": "Benjamini-Hochberg",
                                                     "interpretation": "structural cross-axis interaction, not annotation error"},
                  "diagnosis": diagnosis,
                  "stage179b_gate": {"decision": decision, "authorized_next_stage": diagnosis["authorized_next_stage"],
                                     "training_authorized": False, "relabeling_authorized": False,
                                     "architecture_or_readout_implementation_authorized": False},
                  "limitations": ["Single-seed observational internal audit; no causal claim.",
                                  "Centroids are gold-conditioned transductive diagnostics, not deployment proposals.",
                                  "Near-duplicate similarity is descriptive and cannot establish label error.",
                                  "Pair bootstrap does not measure training-seed uncertainty."],
                  "safety_policy": {"clean_controlled_data_only": True, "evaluation_only": True,
                                    "training": False, "optimizer_created": False, "backward": False,
                                    "train_mode_called": False, "calibration": False, "threshold_search": False,
                                    "fitted_probe": False, "regression_fitting": False, "checkpoint_selection": False,
                                    "external_evaluation": False, "external_labels": False, "time_swap": False,
                                    "trainer_modified": False, "model_modified": False, "new_model_output": False,
                                    "relabeling": False, "architecture_implementation": False,
                                    "loss_modified": False, "pair_centering_deployment": False,
                                    "weight_sweep": False, "multi_seed": False}}
        current = "output_serialization"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / OUTPUTS["json"], report)
        (output_dir / OUTPUTS["md"]).write_text(_render_markdown(report), encoding="utf-8")
        _write_csv(output_dir / OUTPUTS["rows"], rows)
        _write_csv(output_dir / OUTPUTS["contingency"], contingency)
        _write_csv(output_dir / OUTPUTS["strata"], strata)
        _write_csv(output_dir / OUTPUTS["duplicates"], duplicates)
        _write_csv(output_dir / OUTPUTS["near"], near)
        _write_csv(output_dir / OUTPUTS["intervention"], intervention)
        _write_csv(output_dir / OUTPUTS["readout"], readout_rows)
        _write_csv(output_dir / OUTPUTS["hard39"], hard_rows)
        _write_csv(output_dir / OUTPUTS["centroid"], centroid_csv)
        _write_csv(output_dir / OUTPUTS["sensitivity"], sensitivity)
        _write_csv(output_dir / OUTPUTS["interference"], interference)
        _write_csv(output_dir / OUTPUTS["decision"], decision_rows)
        print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except (AuditBlocked, stage176a.ValidationBlocked, stage177e.AuditBlocked,
            OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        _blocked(output_dir, error, current)
        print(json.dumps({"decision": BLOCKED, "error_type": type(error).__name__, "error": str(error),
                          "failure_stage": current, "traceback": traceback.format_exc()}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
