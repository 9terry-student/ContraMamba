"""Intervention-aware losses for controlled ContraMamba-v5 training."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F


def _pair_index(
    pair_ids: Sequence[str], intervention_types: Sequence[str]
) -> dict[str, dict[str, int]]:
    if len(pair_ids) != len(intervention_types):
        raise ValueError("pair_ids and intervention_types must have equal length")
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for index, (pair_id, intervention) in enumerate(
        zip(pair_ids, intervention_types, strict=True)
    ):
        if intervention in result[pair_id]:
            raise ValueError(
                f"duplicate intervention {intervention!r} for pair_id {pair_id!r}"
            )
        result[pair_id][intervention] = index
    return dict(result)


def _mean_or_zero(terms: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    return torch.stack(terms).mean() if terms else reference.sum() * 0.0


def intervention_pairwise_losses(
    output: dict[str, Any],
    pair_ids: Sequence[str],
    intervention_types: Sequence[str],
    final_labels: Sequence[int] | torch.Tensor,
    *,
    lambda_frame_preserve: float = 1.0,
    lambda_frame_anchor: float = 1.0,
    lambda_predicate_contrast: float = 1.0,
    lambda_predicate_anchor: float = 1.0,
    lambda_sufficiency_contrast: float = 1.0,
    lambda_polarity_flip: float = 1.0,
    lambda_polarity_margin_anchor: float = 1.0,
    lambda_paraphrase_preserve: float = 1.0,
    lambda_entitlement_preserve: float = 1.0,
    lambda_logit_preserve: float = 1.0,
    ranking_margin: float = 0.5,
    polarity_margin_min: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute pairwise losses without exposing intervention metadata to the model."""

    if ranking_margin < 0:
        raise ValueError("ranking_margin must be non-negative")
    if polarity_margin_min < 0:
        raise ValueError("polarity_margin_min must be non-negative")
    if len(final_labels) != len(pair_ids):
        raise ValueError("final_labels and pair_ids must have equal length")
    pairs = _pair_index(pair_ids, intervention_types)
    frame_terms: list[torch.Tensor] = []
    frame_anchor_terms: list[torch.Tensor] = []
    predicate_terms: list[torch.Tensor] = []
    predicate_anchor_terms: list[torch.Tensor] = []
    sufficiency_terms: list[torch.Tensor] = []
    polarity_terms: list[torch.Tensor] = []
    polarity_anchor_terms: list[torch.Tensor] = []
    paraphrase_terms: list[torch.Tensor] = []
    entitlement_terms: list[torch.Tensor] = []
    logit_terms: list[torch.Tensor] = []

    for pair_id, variants in pairs.items():
        if "none" not in variants:
            raise ValueError(f"pair_id {pair_id!r} has no original ('none') record")
        original = variants["none"]

        for intervention in ("paraphrase", "predicate_swap", "polarity_flip"):
            if intervention in variants:
                changed = variants[intervention]
                frame_terms.append(
                    F.mse_loss(
                        output["frame_logit"][changed],
                        output["frame_logit"][original],
                    )
                )
                for index in (original, changed):
                    frame_anchor_terms.append(
                        F.binary_cross_entropy_with_logits(
                            output["frame_logit"][index],
                            torch.ones_like(output["frame_logit"][index]),
                        )
                    )

        if "predicate_swap" in variants:
            changed = variants["predicate_swap"]
            difference = (
                output["predicate_coverage_logit"][original]
                - output["predicate_coverage_logit"][changed]
            )
            predicate_terms.append(F.relu(ranking_margin - difference))
            predicate_anchor_terms.append(
                F.binary_cross_entropy_with_logits(
                    output["predicate_coverage_logit"][original],
                    torch.ones_like(output["predicate_coverage_logit"][original]),
                )
            )

        for intervention in ("evidence_deletion", "evidence_truncation"):
            if intervention in variants:
                changed = variants[intervention]
                difference = (
                    output["sufficiency_logit"][original]
                    - output["sufficiency_logit"][changed]
                )
                sufficiency_terms.append(F.relu(ranking_margin - difference))

        if "polarity_flip" in variants:
            changed = variants["polarity_flip"]
            original_label = int(final_labels[original])
            changed_label = int(final_labels[changed])
            direction_by_label = {0: -1.0, 2: 1.0}
            if original_label not in direction_by_label or changed_label not in direction_by_label:
                raise ValueError(
                    "polarity_flip pairs require REFUTE=0 or SUPPORT=2 labels"
                )
            original_direction = direction_by_label[original_label]
            changed_direction = direction_by_label[changed_label]
            if original_direction == changed_direction:
                raise ValueError("polarity_flip labels must have opposite directions")
            entitlement_terms.append(F.mse_loss(
                output["entitlement_prob"][changed],
                output["entitlement_prob"][original],
            ))
            sign_reversal = (
                output["polarity_margin"][original]
                + output["polarity_margin"][changed]
            ).square()
            original_magnitude_penalty = F.relu(
                polarity_margin_min
                - original_direction * output["polarity_margin"][original]
            ).square()
            flipped_magnitude_penalty = F.relu(
                polarity_margin_min
                - changed_direction * output["polarity_margin"][changed]
            ).square()
            polarity_anchor_terms.append(
                0.5 * (original_magnitude_penalty + flipped_magnitude_penalty)
            )
            predicate_preservation = F.mse_loss(
                output["predicate_coverage_logit"][changed],
                output["predicate_coverage_logit"][original],
            )
            sufficiency_preservation = F.mse_loss(
                output["sufficiency_logit"][changed],
                output["sufficiency_logit"][original],
            )
            polarity_terms.append(
                sign_reversal + predicate_preservation + sufficiency_preservation
            )
            for index in (original, changed):
                predicate_anchor_terms.append(
                    F.binary_cross_entropy_with_logits(
                        output["predicate_coverage_logit"][index],
                        torch.ones_like(output["predicate_coverage_logit"][index]),
                    )
                )

        if "paraphrase" in variants:
            changed = variants["paraphrase"]
            scalar_keys = (
                "predicate_coverage_logit",
                "sufficiency_logit",
            )
            scalar_preservation = torch.stack(
                [
                    F.mse_loss(output[key][changed], output[key][original])
                    for key in scalar_keys
                ]
            ).mean()
            paraphrase_terms.append(scalar_preservation)
            entitlement_terms.append(F.mse_loss(
                output["entitlement_prob"][changed],
                output["entitlement_prob"][original],
            ))
            logit_terms.append(F.mse_loss(
                output["logits"][changed], output["logits"][original]
            ))
            for index in (original, changed):
                predicate_anchor_terms.append(
                    F.binary_cross_entropy_with_logits(
                        output["predicate_coverage_logit"][index],
                        torch.ones_like(output["predicate_coverage_logit"][index]),
                    )
                )

    reference = output["logits"]
    frame_preserve = _mean_or_zero(frame_terms, reference)
    frame_anchor = _mean_or_zero(frame_anchor_terms, reference)
    predicate_contrast = _mean_or_zero(predicate_terms, reference)
    predicate_anchor = _mean_or_zero(predicate_anchor_terms, reference)
    sufficiency_contrast = _mean_or_zero(sufficiency_terms, reference)
    polarity_flip = _mean_or_zero(polarity_terms, reference)
    polarity_margin_anchor = _mean_or_zero(polarity_anchor_terms, reference)
    paraphrase_preserve = _mean_or_zero(paraphrase_terms, reference)
    entitlement_preserve = _mean_or_zero(entitlement_terms, reference)
    logit_preserve = _mean_or_zero(logit_terms, reference)
    total = (
        lambda_frame_preserve * frame_preserve
        + lambda_frame_anchor * frame_anchor
        + lambda_predicate_contrast * predicate_contrast
        + lambda_predicate_anchor * predicate_anchor
        + lambda_sufficiency_contrast * sufficiency_contrast
        + lambda_polarity_flip * polarity_flip
        + lambda_polarity_margin_anchor * polarity_margin_anchor
        + lambda_paraphrase_preserve * paraphrase_preserve
        + lambda_entitlement_preserve * entitlement_preserve
        + lambda_logit_preserve * logit_preserve
    )
    return {
        "total": total,
        "frame_preserve": frame_preserve,
        "frame_anchor": frame_anchor,
        "predicate_contrast": predicate_contrast,
        "predicate_anchor": predicate_anchor,
        "sufficiency_contrast": sufficiency_contrast,
        "polarity_flip": polarity_flip,
        "polarity_margin_anchor": polarity_margin_anchor,
        "paraphrase_preserve": paraphrase_preserve,
        "entitlement_preserve": entitlement_preserve,
        "logit_preserve": logit_preserve,
    }
