from __future__ import annotations

import json

import torch

from scripts.build_controlled_v5 import build_seed_records, split_by_pair_id
from scripts.train_controlled_v5 import (
    MODEL_INPUT_KEYS,
    build_model,
    build_optimizer,
    build_vocab,
    controlled_losses,
    cache_frozen_encoder_states,
    compute_metrics,
    encode_records,
    final_prediction_distribution,
    intervention_diagnostics,
    intervention_objective,
    pairwise_checks,
    sample_indices,
)


def test_controlled_training_step_smoke() -> None:
    records = build_seed_records()[:13]
    vocab = build_vocab(records)
    bundle = encode_records(records, vocab)
    inputs = bundle["model_inputs"]
    model = build_model(len(vocab), inputs["input_ids"].shape[1], hidden_size=24)
    output = model(**inputs)
    loss = output["loss"] + intervention_objective(output, records)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_metrics_and_intervention_diagnostics() -> None:
    records = build_seed_records()[:13]
    vocab = build_vocab(records)
    inputs = encode_records(records, vocab)["model_inputs"]
    model = build_model(len(vocab), inputs["input_ids"].shape[1], hidden_size=24)
    with torch.no_grad():
        output = model(**inputs)
    metrics = compute_metrics(output, inputs)
    diagnostics = intervention_diagnostics(records, output)
    checks = pairwise_checks(records, output)
    distribution = final_prediction_distribution(output)
    assert set(metrics) == {
        "final_accuracy",
        "final_macro_f1",
        "per_label",
        "frame_accuracy",
        "predicate_accuracy",
        "sufficiency_accuracy",
        "polarity_accuracy_entitled",
        "prediction_distribution",
    }
    assert set(diagnostics) == {record["intervention_type"] for record in records}
    scalar_metrics = (
        "final_accuracy",
        "final_macro_f1",
        "frame_accuracy",
        "predicate_accuracy",
        "sufficiency_accuracy",
        "polarity_accuracy_entitled",
    )
    assert all(0.0 <= metrics[key] <= 1.0 for key in scalar_metrics)
    assert set(metrics["per_label"]) == {"REFUTE", "NOT_ENTITLED", "SUPPORT"}
    assert sum(distribution.values()) == len(records)
    json.dumps(
        {
            "metrics": metrics,
            "diagnostics": diagnostics,
            "checks": checks,
            "distribution": distribution,
        }
    )


def test_pair_split_and_intervention_metadata_are_not_model_inputs() -> None:
    records = build_seed_records()
    train, dev = split_by_pair_id(records, dev_ratio=0.2, seed=17)
    train_pairs = {record["pair_id"] for record in train}
    dev_pairs = {record["pair_id"] for record in dev}
    bundle = encode_records(train, build_vocab(records))
    assert train_pairs.isdisjoint(dev_pairs)
    assert set(bundle["model_inputs"]) == MODEL_INPUT_KEYS
    assert "intervention_type" not in bundle["model_inputs"]
    assert "pair_id" not in bundle["model_inputs"]


def test_weighted_balanced_losses_and_optimizer_groups() -> None:
    records = build_seed_records()[:26]
    vocab = build_vocab(records)
    inputs = encode_records(records, vocab)["model_inputs"]
    model = build_model(len(vocab), inputs["input_ids"].shape[1], hidden_size=24)
    output = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        claim_mask=inputs["claim_mask"],
        evidence_mask=inputs["evidence_mask"],
    )
    generator = torch.Generator().manual_seed(3)
    indices = sample_indices(inputs["final_labels"], True, generator)
    losses = controlled_losses(output, inputs, indices, weighted_label_loss=True)
    full_losses = controlled_losses(
        output,
        inputs,
        torch.arange(inputs["final_labels"].shape[0]),
        weighted_label_loss=True,
    )
    optimizer = build_optimizer(model, lr=1e-3, head_lr=2e-3, encoder_lr=1e-4)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    for key in ("frame", "predicate", "sufficiency", "polarity"):
        assert torch.equal(losses[key], full_losses[key])
    assert sorted(group["lr"] for group in optimizer.param_groups) == [1e-4, 2e-3]


def test_frozen_encoder_cache_matches_direct_forward() -> None:
    records = build_seed_records()[:2]
    vocab = build_vocab(records)
    inputs = encode_records(records, vocab)["model_inputs"]
    model = build_model(len(vocab), inputs["input_ids"].shape[1], hidden_size=24).eval()
    for parameter in model.mamba.parameters():
        parameter.requires_grad = False
    with torch.no_grad():
        direct = model(**inputs)
    cache_frozen_encoder_states(model, inputs, batch_size=1)
    with torch.no_grad():
        cached = model(**inputs)
    assert torch.allclose(direct["logits"], cached["logits"])
