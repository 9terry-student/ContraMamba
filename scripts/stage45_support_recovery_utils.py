"""Stage45-C internal-only SUPPORT entitlement recovery utilities.

These helpers compute an optional, internal-only auxiliary training signal and
matching diagnostics targeting two failure modes observed in Stage45-B1 internal
family holdouts: SUPPORT under-recall and entitled-example (SUPPORT/REFUTE)
over-rejection into NOT_ENTITLED. Everything here reads only:

  * the internal training split's `final_labels` tensor, and
  * the model's own `logits` output on that same training split.

It never reads dev/holdout labels, Stage43-B1 artifacts, VitaminC, Climate-FEVER,
or any other external data. When disabled (default), callers should not invoke
these functions at all, so existing training behavior is unaffected.
"""

from __future__ import annotations

from typing import Any

import torch


def compute_support_recovery_terms(
    logits: torch.Tensor,
    final_labels: torch.Tensor,
    *,
    label_to_id: dict[str, int],
    target_label: str = "SUPPORT",
    entitled_labels: tuple[str, ...] = ("SUPPORT", "REFUTE"),
    not_entitled_label: str = "NOT_ENTITLED",
) -> dict[str, Any]:
    """Compute Stage45-C auxiliary loss terms from training logits/labels only.

    Both returned loss tensors are always zero-dim, on `logits`'s device/dtype,
    and connected to the autograd graph (via `logits.sum() * 0.0`) even when a
    term is inactive because no matching gold rows are present in this batch.
    """
    zero = logits.sum() * 0.0
    probs = torch.softmax(logits, dim=-1)

    target_id = label_to_id.get(target_label)
    support_recovery_loss = zero
    support_recovery_active = False
    target_row_count = 0
    if target_id is not None:
        target_mask = final_labels == target_id
        target_row_count = int(target_mask.sum().item())
        if target_row_count > 0:
            target_probs = probs[target_mask][:, target_id]
            support_recovery_loss = (1.0 - target_probs).clamp_min(0.0).mean()
            support_recovery_active = True

    not_entitled_id = label_to_id.get(not_entitled_label)
    entitled_ne_penalty_loss = zero
    entitled_ne_penalty_active = False
    entitled_row_count = 0
    if not_entitled_id is not None:
        entitled_ids = [
            label_to_id[name] for name in entitled_labels if name in label_to_id
        ]
        if entitled_ids:
            entitled_mask = torch.zeros_like(final_labels, dtype=torch.bool)
            for entitled_id in entitled_ids:
                entitled_mask = entitled_mask | (final_labels == entitled_id)
            entitled_row_count = int(entitled_mask.sum().item())
            if entitled_row_count > 0:
                ne_probs = probs[entitled_mask][:, not_entitled_id]
                entitled_ne_penalty_loss = ne_probs.mean()
                entitled_ne_penalty_active = True

    return {
        "support_recovery_loss": support_recovery_loss,
        "support_recovery_active": support_recovery_active,
        "entitled_ne_penalty_loss": entitled_ne_penalty_loss,
        "entitled_ne_penalty_active": entitled_ne_penalty_active,
        "target_row_count": target_row_count,
        "entitled_row_count": entitled_row_count,
    }


def label_counts(final_labels: torch.Tensor, id_to_label: dict[int, str]) -> dict[str, int]:
    """Count training rows per gold label name from the internal `final_labels` tensor."""
    counts: dict[str, int] = {name: 0 for name in id_to_label.values()}
    for value in final_labels.detach().cpu().tolist():
        name = id_to_label.get(int(value))
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


LEAKAGE_POLICY: dict[str, Any] = {
    "scope": "internal_controlled_training_split_only",
    "stage43b1_files_read": False,
    "external_examples_used": False,
    "external_labels_or_metrics_used": False,
    "vitaminc_used": False,
    "climate_fever_used": False,
    "used_for_threshold_selection": False,
    "used_for_calibration": False,
    "used_for_checkpoint_selection": False,
    "used_dev_or_holdout_labels_in_loss": False,
}


def build_stage45c_report(
    *,
    enabled: bool,
    support_recovery_weight: float,
    entitled_ne_penalty_weight: float,
    target_label: str,
    entitled_labels: tuple[str, ...],
    train_label_counts: dict[str, int],
    support_recovery_loss_mean: float | None,
    entitled_ne_penalty_loss_mean: float | None,
) -> dict[str, Any]:
    """Assemble the Stage45-C report fields shared between per-run and standalone reports."""
    loss_terms_active: list[str] = []
    if enabled and support_recovery_weight > 0.0:
        loss_terms_active.append("support_recovery")
    if enabled and entitled_ne_penalty_weight > 0.0:
        loss_terms_active.append("entitled_ne_penalty")

    return {
        "stage45c_enabled": enabled,
        "stage45c_support_recovery_weight": support_recovery_weight,
        "stage45c_entitled_ne_penalty_weight": entitled_ne_penalty_weight,
        "stage45c_target_label": target_label,
        "stage45c_entitled_labels": list(entitled_labels),
        "stage45c_train_support_count": train_label_counts.get("SUPPORT", 0),
        "stage45c_train_refute_count": train_label_counts.get("REFUTE", 0),
        "stage45c_train_not_entitled_count": train_label_counts.get("NOT_ENTITLED", 0),
        "stage45c_loss_terms_active": loss_terms_active,
        "stage45c_support_recovery_loss_mean": support_recovery_loss_mean,
        "stage45c_entitled_ne_penalty_loss_mean": entitled_ne_penalty_loss_mean,
        "stage45c_leakage_policy": LEAKAGE_POLICY,
        "stage45c_recommendation": (
            "Stage45-C is an internal-only auxiliary diagnostic/training scaffold "
            "targeting SUPPORT under-recall and entitled-to-NOT_ENTITLED over-rejection "
            "observed in Stage45-B1 internal family holdouts. Treat any resulting "
            "improvement as an internal robustness signal only; it does not constitute "
            "external validation and must not be claimed as VitaminC/Climate-FEVER "
            "transfer success or naturalistic generalization without a new, separate "
            "held-out external evaluation."
            if enabled
            else (
                "Stage45-C is disabled for this run; no auxiliary loss terms were added "
                "and training behavior is unchanged from the pre-Stage45-C baseline."
            )
        ),
    }
