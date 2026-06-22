from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import contramamba
from contramamba import (
    ContraMambaV5,
    FinalEntitlementDecisionHead,
    FinalLabel,
    FrameGate,
)


class DummyBackbone(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def make_batch(batch_size: int = 3, sequence_length: int = 10):
    input_ids = torch.randint(0, 64, (batch_size, sequence_length))
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    attention_mask[:, -2:] = False
    claim_mask = torch.zeros_like(attention_mask)
    evidence_mask = torch.zeros_like(attention_mask)
    claim_mask[:, 1:3] = True
    evidence_mask[:, 4:7] = True
    return input_ids, attention_mask, claim_mask, evidence_mask


def make_model(diagnostics: bool = False) -> ContraMambaV5:
    return ContraMambaV5(
        backbone=DummyBackbone(),
        frame_size=12,
        predicate_size=12,
        sufficiency_size=12,
        energy_size=8,
        dropout=0.0,
        return_token_diagnostics=diagnostics,
    )


def test_output_shapes_and_optional_diagnostics() -> None:
    model = make_model().eval()
    input_ids, attention, claim, evidence = make_batch()
    output = model(input_ids, attention, claim, evidence)

    assert output["logits"].shape == (3, 3)
    assert output["predictions"].shape == (3,)
    for key in (
        "frame_logit",
        "frame_prob",
        "predicate_coverage_logit",
        "predicate_coverage_prob",
        "sufficiency_logit",
        "sufficiency_prob",
        "positive_energy",
        "negative_energy",
        "entitlement_prob",
    ):
        assert output[key].shape == (3,)
    assert output["frame_pair_repr"].shape == (3, 12)
    assert output["predicate_pair_repr"].shape == (3, 12)
    assert output["polarity_features"].shape == (3, 8)
    assert output["frame_token_scores"] is None
    assert output["predicate_token_scores"] is None


def test_masked_pooling_ignores_unselected_tokens() -> None:
    gate = FrameGate(16, 12, dropout=0.0).eval()
    states = torch.randn(2, 8, 16)
    attention = torch.ones(2, 8, dtype=torch.bool)
    claim = torch.zeros_like(attention)
    evidence = torch.zeros_like(attention)
    claim[:, :2] = True
    evidence[:, 3:5] = True

    first = gate(states, attention, claim, evidence)
    changed = states.clone()
    changed[:, 5:] = changed[:, 5:] + 10_000
    second = gate(changed, attention, claim, evidence)

    assert torch.allclose(first["claim_frame_state"], second["claim_frame_state"])
    assert torch.allclose(
        first["evidence_frame_state"], second["evidence_frame_state"]
    )
    assert torch.allclose(first["frame_logit"], second["frame_logit"])


@pytest.mark.parametrize("failure", ["empty_claim", "empty_evidence", "overlap"])
def test_pair_masks_must_be_non_empty_and_disjoint(failure: str) -> None:
    gate = FrameGate(16, 12)
    states = torch.randn(2, 8, 16)
    attention = torch.ones(2, 8, dtype=torch.bool)
    claim = torch.zeros_like(attention)
    evidence = torch.zeros_like(attention)
    claim[:, :2] = True
    evidence[:, 3:5] = True
    if failure == "empty_claim":
        claim[0] = False
    elif failure == "empty_evidence":
        evidence[0] = False
    else:
        evidence[0, 0] = True

    with pytest.raises(ValueError):
        gate(states, attention, claim, evidence)


def test_explicit_product_and_final_logit_ordering() -> None:
    head = FinalEntitlementDecisionHead(
        not_entitled_bias_init=0.25,
        alpha_init=2.0,
        decision_mode="explicit_product",
    )
    frame = torch.tensor([0.5, 0.8])
    predicate = torch.tensor([0.4, 0.5])
    sufficiency = torch.tensor([0.25, 0.5])
    positive = torch.tensor([3.0, 4.0])
    negative = torch.tensor([7.0, 2.0])
    output = head(frame, predicate, sufficiency, positive, negative)

    expected_entitlement = frame * predicate * sufficiency
    assert torch.equal(output["entitlement_prob"], expected_entitlement)
    assert torch.allclose(output["logits"][:, 0], expected_entitlement * negative)
    assert torch.allclose(output["logits"][:, 2], expected_entitlement * positive)
    assert torch.allclose(output["logits"][:, 1], output["not_entitled_logit"])
    assert FinalLabel.REFUTE == 0
    assert FinalLabel.NOT_ENTITLED == 1
    assert FinalLabel.SUPPORT == 2


def test_polarity_loss_can_mask_not_entitled_examples() -> None:
    model = make_model().eval()
    batch = make_batch(batch_size=2)
    output = model(
        *batch,
        final_labels=torch.tensor([1, 2]),
        polarity_labels=torch.tensor([1, 2]),
    )
    assert output["polarity_loss"] is not None
    assert torch.isfinite(output["polarity_loss"])

    fully_masked = model(
        *batch,
        polarity_labels=torch.tensor([1, 2]),
        polarity_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert fully_masked["polarity_loss"] is None


def test_v5_package_does_not_import_v4_geometric_classifier() -> None:
    source = inspect.getsource(contramamba)
    model_source = inspect.getsource(ContraMambaV5)
    assert "classify_geometric" not in source
    assert "classify_geometric" not in model_source
