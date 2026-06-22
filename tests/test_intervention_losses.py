from __future__ import annotations

import pytest
import torch

from contramamba.losses import intervention_pairwise_losses


PAIR_IDS = ["p"] * 6
INTERVENTIONS = [
    "none",
    "paraphrase",
    "predicate_swap",
    "evidence_deletion",
    "evidence_truncation",
    "polarity_flip",
]


def ideal_output() -> dict[str, torch.Tensor]:
    return {
        "frame_logit": torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, 1.0], requires_grad=True),
        "predicate_coverage_logit": torch.tensor(
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0], requires_grad=True
        ),
        "sufficiency_logit": torch.tensor(
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0], requires_grad=True
        ),
        "entitlement_prob": torch.tensor(
            [0.8, 0.8, 0.2, 0.1, 0.1, 0.8], requires_grad=True
        ),
        "polarity_margin": torch.tensor(
            [2.0, 2.0, 0.0, 0.0, 0.0, -2.0], requires_grad=True
        ),
        "logits": torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            requires_grad=True,
        ),
    }


def test_ideal_intervention_relations_have_zero_loss() -> None:
    losses = intervention_pairwise_losses(
        ideal_output(),
        PAIR_IDS,
        INTERVENTIONS,
        ranking_margin=0.5,
        lambda_frame_anchor=0.0,
        lambda_predicate_anchor=0.0,
    )
    assert torch.equal(losses["total"], torch.tensor(0.0))
    assert set(losses) == {
        "total",
        "frame_preserve",
        "frame_anchor",
        "predicate_contrast",
        "predicate_anchor",
        "sufficiency_contrast",
        "polarity_flip",
        "polarity_margin_anchor",
        "paraphrase_preserve",
        "entitlement_preserve",
        "logit_preserve",
    }


def test_violations_produce_loss_and_gradients() -> None:
    output = ideal_output()
    output["frame_logit"] = torch.tensor(
        [1.0, -1.0, -1.0, -1.0, -1.0, -1.0], requires_grad=True
    )
    output["predicate_coverage_logit"] = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True
    )
    output["sufficiency_logit"] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0], requires_grad=True
    )
    output["entitlement_prob"] = torch.tensor(
        [0.9, 0.2, 0.2, 0.1, 0.1, 0.1], requires_grad=True
    )
    output["polarity_margin"] = torch.tensor(
        [2.0, 0.0, 0.0, 0.0, 0.0, 2.0], requires_grad=True
    )
    losses = intervention_pairwise_losses(output, PAIR_IDS, INTERVENTIONS)
    losses["total"].backward()
    assert losses["total"].item() > 0
    assert output["frame_logit"].grad is not None
    assert output["polarity_margin"].grad is not None


def test_pair_metadata_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        intervention_pairwise_losses(ideal_output(), ["p"], INTERVENTIONS)
    with pytest.raises(ValueError, match="has no original"):
        intervention_pairwise_losses(
            ideal_output(),
            ["p"] * 5,
            INTERVENTIONS[1:],
        )

    with pytest.raises(ValueError, match="polarity_margin_min"):
        intervention_pairwise_losses(
            ideal_output(), PAIR_IDS, INTERVENTIONS, polarity_margin_min=-0.1
        )


def test_frame_anchor_penalizes_jointly_low_preserved_frames() -> None:
    high = ideal_output()
    low = ideal_output()
    low["frame_logit"] = torch.tensor(
        [-5.0, -5.0, -5.0, -1.0, -1.0, -5.0], requires_grad=True
    )
    high_losses = intervention_pairwise_losses(high, PAIR_IDS, INTERVENTIONS)
    low_losses = intervention_pairwise_losses(low, PAIR_IDS, INTERVENTIONS)
    assert low_losses["frame_preserve"].item() == 0.0
    assert low_losses["frame_anchor"] > high_losses["frame_anchor"]
    assert low_losses["total"] > high_losses["total"]


def test_zero_symmetric_polarity_margins_are_anchored() -> None:
    output = ideal_output()
    output["polarity_margin"] = torch.zeros(6, requires_grad=True)
    losses = intervention_pairwise_losses(
        output, PAIR_IDS, INTERVENTIONS, polarity_margin_min=1.0
    )
    assert losses["polarity_flip"].item() == 0.0
    assert losses["polarity_margin_anchor"].item() == pytest.approx(1.0)


def test_opposite_nonzero_margins_are_preferred_to_zero() -> None:
    nonzero = intervention_pairwise_losses(
        ideal_output(), PAIR_IDS, INTERVENTIONS, polarity_margin_min=1.0
    )
    zero_output = ideal_output()
    zero_output["polarity_margin"] = torch.zeros(6, requires_grad=True)
    zero = intervention_pairwise_losses(
        zero_output, PAIR_IDS, INTERVENTIONS, polarity_margin_min=1.0
    )
    assert nonzero["polarity_flip"].item() == 0.0
    assert nonzero["polarity_margin_anchor"].item() == 0.0
    assert zero["total"] > nonzero["total"]
