"""Stage175-B clean SUPPORT-paraphrase margin preservation objective."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from scripts.stage174c_clean_pairwise import build_train_pair_index


EXPECTED_ELIGIBLE_TRAIN_PAIR_COUNT = 121
CONDITION_FIELDS = (
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
)


def resolved_configuration(*, mode: str, weight: float, tolerance: float) -> dict[str, Any]:
    enabled = mode == "paraphrase_margin" and float(weight) > 0.0
    return {
        "mode": mode,
        "enabled": enabled,
        "weight": float(weight),
        "tolerance": float(tolerance),
        "classifier_source": 'output["logits"]',
        "support_margin_definition": (
            "support_logit - logsumexp(not_entitled_logit, refute_logit)"
        ),
        "current_intervention_type": "paraphrase",
        "reference_intervention_type": "none",
        "detached_reference": True,
        "teacher_checkpoint_used": False,
        "absolute_confidence_floor_used": False,
        "failure_variant_loss_used": False,
    }


def _anchor_has_required_labels(record: dict[str, Any]) -> bool:
    return all(record.get(field) == 1 for field in CONDITION_FIELDS)


def build_train_support_anchor_index(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    *,
    expected_eligible_count: int = EXPECTED_ELIGIBLE_TRAIN_PAIR_COUNT,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Validate the complete clean taxonomy and return train-only anchor rows."""
    groups = build_train_pair_index(train_records)
    train_pair_ids = set(groups)
    dev_pair_ids = {str(record.get("pair_id")) for record in dev_records}
    overlap = train_pair_ids & dev_pair_ids
    if overlap:
        raise ValueError(
            "[stage175b] train/dev pair leakage: " + ", ".join(sorted(overlap))
        )

    eligible: dict[str, dict[str, int]] = {}
    missing_current_count = 0
    missing_reference_count = 0
    ambiguous_reference_count = 0
    same_row_count = 0
    for pair_id, group in groups.items():
        current_indices = [group[name] for name in group if name == "paraphrase"]
        reference_indices = [group[name] for name in group if name == "none"]
        if len(current_indices) != 1:
            missing_current_count += int(len(current_indices) == 0)
            raise ValueError(
                f"[stage175b] pair_id={pair_id}: expected exactly one paraphrase row, "
                f"got {len(current_indices)}"
            )
        if len(reference_indices) != 1:
            missing_reference_count += int(len(reference_indices) == 0)
            ambiguous_reference_count += int(len(reference_indices) > 1)
            raise ValueError(
                f"[stage175b] pair_id={pair_id}: expected exactly one none row, "
                f"got {len(reference_indices)}"
            )
        current_index = current_indices[0]
        reference_index = reference_indices[0]
        if current_index == reference_index:
            same_row_count += 1
            raise ValueError(f"[stage175b] pair_id={pair_id}: current/reference are the same row")

        current = train_records[current_index]
        reference = train_records[reference_index]
        current_is_candidate = (
            current.get("final_label") == "SUPPORT"
            and _anchor_has_required_labels(current)
        )
        reference_is_eligible = (
            reference.get("final_label") == "SUPPORT"
            and reference.get("polarity_label") == current.get("polarity_label")
            and _anchor_has_required_labels(reference)
        )
        if current_is_candidate and not reference_is_eligible:
            raise ValueError(
                f"[stage175b] pair_id={pair_id}: malformed eligible SUPPORT paraphrase "
                "has no label-consistent canonical none reference"
            )
        if current_is_candidate and reference_is_eligible:
            eligible[pair_id] = {
                "current_index": current_index,
                "reference_index": reference_index,
            }

    if len(eligible) != int(expected_eligible_count):
        raise ValueError(
            "[stage175b] eligible train SUPPORT-anchor pair count mismatch: "
            f"expected {expected_eligible_count}, got {len(eligible)}"
        )
    validation = {
        "total_train_pair_groups": len(groups),
        "eligible_train_support_anchor_pair_groups": len(eligible),
        "expected_eligible_count": int(expected_eligible_count),
        "malformed_pair_count": 0,
        "missing_current_count": missing_current_count,
        "missing_reference_count": missing_reference_count,
        "ambiguous_reference_count": ambiguous_reference_count,
        "train_dev_overlap_count": len(overlap),
        "current_reference_same_row_count": same_row_count,
    }
    return eligible, validation


def ordered_anchor_indices(
    anchor_index: dict[str, dict[str, int]],
) -> tuple[list[int], list[int]]:
    """Return current/reference indices in deterministic train-row order."""
    entries = sorted(anchor_index.values(), key=lambda item: item["current_index"])
    return (
        [item["current_index"] for item in entries],
        [item["reference_index"] for item in entries],
    )


def support_margin(logits: torch.Tensor, label_to_id: dict[str, int]) -> torch.Tensor:
    required = ("SUPPORT", "NOT_ENTITLED", "REFUTE")
    missing = [label for label in required if label not in label_to_id]
    if missing:
        raise ValueError(f"[stage175b] final label-index mapping lacks {missing}")
    if logits.ndim != 2:
        raise RuntimeError(f"[stage175b] output logits must be rank 2, got {tuple(logits.shape)}")
    indices = [int(label_to_id[label]) for label in required]
    if len(set(indices)) != 3 or min(indices) < 0 or max(indices) >= logits.shape[1]:
        raise ValueError("[stage175b] invalid final label-index mapping for output logits")
    support = logits[:, indices[0]]
    competitors = logits[:, [indices[1], indices[2]]]
    return support - torch.logsumexp(competitors, dim=-1)


def compute_loss(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    current_indices: list[int],
    *,
    label_to_id: dict[str, int],
    tolerance: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute a one-sided hinge; gradients reach only current output logits."""
    if reference_logits.shape[0] != len(current_indices):
        raise RuntimeError("[stage175b] current/reference anchor ordering is misaligned")
    if not current_indices:
        zero = current_logits.sum() * 0.0
        return zero, {
            "eligible_current_row_count": 0,
            "active_violation_count": 0,
            "zero_violation_count": 0,
            "mean_current_support_margin": None,
            "mean_detached_reference_support_margin": None,
            "mean_raw_margin_gap": None,
            "mean_active_hinge_loss": None,
            "malformed_skipped_count": 0,
        }
    index_tensor = torch.tensor(current_indices, dtype=torch.long, device=current_logits.device)
    current_margin = support_margin(
        current_logits.index_select(0, index_tensor), label_to_id
    )
    reference_margin = support_margin(reference_logits, label_to_id).detach()
    hinge = F.relu(reference_margin - float(tolerance) - current_margin)
    active = hinge > 0
    metrics = {
        "eligible_current_row_count": len(current_indices),
        "active_violation_count": int(active.sum().item()),
        "zero_violation_count": int((~active).sum().item()),
        "mean_current_support_margin": float(current_margin.detach().mean().item()),
        "mean_detached_reference_support_margin": float(reference_margin.mean().item()),
        "mean_raw_margin_gap": float((reference_margin - current_margin.detach()).mean().item()),
        "mean_active_hinge_loss": (
            float(hinge.detach()[active].mean().item()) if active.any() else None
        ),
        "malformed_skipped_count": 0,
    }
    return hinge.mean(), metrics
