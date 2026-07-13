"""Stage174-C clean, train-only structural pairwise entitlement regularization."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch.nn import functional as F


ANCHORS = ("none", "paraphrase")
FRAME_MISMATCH = (
    "entity_swap", "event_swap", "irrelevant_evidence", "location_swap",
    "role_swap", "title_name_swap",
)
PREDICATE_FAILURE = ("predicate_swap",)
SUFFICIENCY_FAILURE = ("evidence_deletion", "evidence_truncation")
POLARITY_FLIP = ("polarity_flip",)
NON_ANCHORS = FRAME_MISMATCH + PREDICATE_FAILURE + SUFFICIENCY_FAILURE + POLARITY_FLIP
EXPECTED_INTERVENTIONS = ANCHORS + NON_ANCHORS
NATIVE_SCORE_KEYS = {
    "frame": "frame_prob",
    "predicate": "predicate_coverage_prob",
    "sufficiency": "sufficiency_prob",
    "entitlement": "entitlement_prob",
}


def taxonomy_record() -> dict[str, Any]:
    return {
        "anchors": list(ANCHORS),
        "frame_mismatch": list(FRAME_MISMATCH),
        "predicate_noncoverage": list(PREDICATE_FAILURE),
        "sufficiency_failure": list(SUFFICIENCY_FAILURE),
        "entitled_polarity_flip": list(POLARITY_FLIP),
        "earliest_gate_failure_semantics": True,
    }


def validation_rules() -> dict[str, Any]:
    return {
        "exactly_one_row_per_expected_intervention": True,
        "anchors_and_polarity_flip": {"frame_compatible_label": 1, "predicate_covered_label": 1, "sufficiency_label": 1},
        "predicate_swap": {"frame_compatible_label": 1, "predicate_covered_label": 0},
        "sufficiency_failures": {"frame_compatible_label": 1, "predicate_covered_label": 1, "sufficiency_label": 0},
        "frame_mismatches": {"frame_compatible_label": 0},
        "paraphrase_matches_none_final_label": True,
        "polarity_flip_opposes_none_final_label": True,
    }


def _require_labels(record: dict[str, Any], expected: dict[str, int], context: str) -> None:
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"[stage174c] {context}: expected {key}={value}, got {record.get(key)!r}")


def build_train_pair_index(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {}
    for index, record in enumerate(records):
        pair_id = record.get("pair_id")
        intervention = record.get("intervention_type")
        if pair_id is None or intervention is None:
            raise ValueError(f"[stage174c] train row {index} lacks pair_id or intervention_type")
        pair_key = str(pair_id)
        if intervention not in EXPECTED_INTERVENTIONS:
            raise ValueError(f"[stage174c] pair_id={pair_key}: unexpected intervention_type={intervention!r}")
        group = groups.setdefault(pair_key, {})
        if intervention in group:
            raise ValueError(f"[stage174c] pair_id={pair_key}: duplicate intervention_type={intervention!r}")
        group[intervention] = index
    for pair_id, group in groups.items():
        missing = sorted(set(EXPECTED_INTERVENTIONS) - set(group))
        extra = sorted(set(group) - set(EXPECTED_INTERVENTIONS))
        if missing or extra or len(group) != len(EXPECTED_INTERVENTIONS):
            raise ValueError(f"[stage174c] pair_id={pair_id}: malformed group missing={missing} extra={extra} row_count={len(group)}")
        for intervention, index in group.items():
            record = records[index]
            context = f"pair_id={pair_id} intervention_type={intervention}"
            if intervention in ANCHORS or intervention in POLARITY_FLIP:
                _require_labels(record, {"frame_compatible_label": 1, "predicate_covered_label": 1, "sufficiency_label": 1}, context)
            elif intervention in FRAME_MISMATCH:
                _require_labels(record, {"frame_compatible_label": 0}, context)
            elif intervention in PREDICATE_FAILURE:
                _require_labels(record, {"frame_compatible_label": 1, "predicate_covered_label": 0}, context)
            else:
                _require_labels(record, {"frame_compatible_label": 1, "predicate_covered_label": 1, "sufficiency_label": 0}, context)
        none_label = records[group["none"]].get("final_label")
        paraphrase_label = records[group["paraphrase"]].get("final_label")
        flip_label = records[group["polarity_flip"]].get("final_label")
        if none_label not in ("SUPPORT", "REFUTE"):
            raise ValueError(f"[stage174c] pair_id={pair_id}: none.final_label must be SUPPORT or REFUTE, got {none_label!r}")
        if paraphrase_label != none_label:
            raise ValueError(f"[stage174c] pair_id={pair_id}: paraphrase.final_label does not match none.final_label")
        opposite = "REFUTE" if none_label == "SUPPORT" else "SUPPORT"
        if flip_label != opposite:
            raise ValueError(f"[stage174c] pair_id={pair_id}: polarity_flip.final_label={flip_label!r}, expected {opposite!r}")
    return groups


def _stable_offset(seed: int, pair_id: str, intervention: str) -> int:
    payload = f"stage174c|{seed}|{pair_id}|{intervention}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_reference_indices(records: list[dict[str, Any]], pair_index: dict[str, dict[str, int]], *, seed: int, epoch: int) -> list[int]:
    references: list[int] = []
    for record in records:
        pair_id = str(record["pair_id"])
        intervention = record["intervention_type"]
        offset = _stable_offset(seed, pair_id, intervention)
        choices = NON_ANCHORS if intervention in ANCHORS else ANCHORS
        selected = choices[(offset + epoch - 1) % len(choices)]
        references.append(pair_index[pair_id][selected])
    return references


def _rank(high: torch.Tensor, low: torch.Tensor, margin: float) -> torch.Tensor:
    return F.relu(float(margin) - (high - low))


def compute_loss(current: dict[str, Any], reference: dict[str, Any], records: list[dict[str, Any]], reference_indices: list[int], *, mode: str, margin: float, polarity_weight: float) -> tuple[torch.Tensor, dict[str, Any]]:
    for key in NATIVE_SCORE_KEYS.values():
        if current.get(key) is None or reference.get(key) is None:
            raise RuntimeError(f"[stage174c] required native differentiable output tensor {key!r} is unavailable")
    buckets: dict[str, list[torch.Tensor]] = {
        "frame_local_ranking": [], "frame_entitlement_ranking": [],
        "predicate_local_ranking": [], "predicate_entitlement_ranking": [],
        "sufficiency_local_ranking": [], "sufficiency_entitlement_ranking": [],
        "polarity_local_preservation": [], "polarity_entitlement_preservation": [],
    }
    row_losses: list[torch.Tensor] = []
    for index, record in enumerate(records):
        intervention = record["intervention_type"]
        reference_intervention = records[reference_indices[index]]["intervention_type"]
        current_is_anchor = intervention in ANCHORS
        failure = reference_intervention if current_is_anchor else intervention
        def ranked(key: str) -> torch.Tensor:
            current_score = current[NATIVE_SCORE_KEYS[key]][index]
            reference_score = reference[NATIVE_SCORE_KEYS[key]][index].detach()
            return _rank(current_score, reference_score, margin) if current_is_anchor else _rank(reference_score, current_score, margin)
        if failure in FRAME_MISMATCH:
            local = ranked("frame"); buckets["frame_local_ranking"].append(local)
            entitlement = ranked("entitlement") if mode == "local_plus_entitlement" else None
            if entitlement is not None: buckets["frame_entitlement_ranking"].append(entitlement)
        elif failure in PREDICATE_FAILURE:
            local = ranked("predicate"); buckets["predicate_local_ranking"].append(local)
            entitlement = ranked("entitlement") if mode == "local_plus_entitlement" else None
            if entitlement is not None: buckets["predicate_entitlement_ranking"].append(entitlement)
        elif failure in SUFFICIENCY_FAILURE:
            local = ranked("sufficiency"); buckets["sufficiency_local_ranking"].append(local)
            entitlement = ranked("entitlement") if mode == "local_plus_entitlement" else None
            if entitlement is not None: buckets["sufficiency_entitlement_ranking"].append(entitlement)
        elif failure in POLARITY_FLIP:
            local_parts = [F.smooth_l1_loss(current[NATIVE_SCORE_KEYS[key]][index], reference[NATIVE_SCORE_KEYS[key]][index].detach()) for key in ("frame", "predicate", "sufficiency")]
            local = torch.stack(local_parts).mean(); buckets["polarity_local_preservation"].append(local)
            entitlement = F.smooth_l1_loss(current[NATIVE_SCORE_KEYS["entitlement"]][index], reference[NATIVE_SCORE_KEYS["entitlement"]][index].detach()) if mode == "local_plus_entitlement" else None
            if entitlement is not None: buckets["polarity_entitlement_preservation"].append(entitlement)
        else:
            raise RuntimeError(f"[stage174c] invalid selected pair family {failure!r}")
        active = [local] + ([entitlement] if entitlement is not None else [])
        row_loss = torch.stack(active).mean()
        if failure in POLARITY_FLIP:
            row_loss = row_loss * float(polarity_weight)
        row_losses.append(row_loss)
    total = torch.stack(row_losses).mean()
    metrics: dict[str, Any] = {}
    for name, values in buckets.items():
        metrics[f"{name}_count"] = len(values)
        metrics[f"{name}_loss"] = float(torch.stack([v.detach() for v in values]).mean().item()) if values else None
    return total, metrics
