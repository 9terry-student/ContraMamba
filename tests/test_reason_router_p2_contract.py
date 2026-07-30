from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import scripts.train_controlled_v6b_minimal as trainer
from contramamba.heads.entitlement_decision import (
    FinalEntitlementDecisionHead,
    _p2_finite_nonnegative,
    _p2_validate_finite_nonnegative,
)


RTOL = 1e-10
ATOL = 1e-12


def _sample_inputs(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    return {
        "frame_prob": torch.tensor([0.05, 0.25, 0.8, 0.99], dtype=dtype),
        "predicate_coverage_prob": torch.tensor([0.2, 0.95, 0.5, 0.99], dtype=dtype),
        "sufficiency_prob": torch.tensor([0.9, 0.4, 0.85, 0.99], dtype=dtype),
        "positive_energy": torch.tensor([0.1, 1.5, 0.7, 2.0], dtype=dtype),
        "negative_energy": torch.tensor([1.1, 0.5, 1.2, 0.1], dtype=dtype),
    }


def _q_only_product_reference(head: FinalEntitlementDecisionHead, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    frame = inputs["frame_prob"]
    predicate = inputs["predicate_coverage_prob"]
    sufficiency = inputs["sufficiency_prob"]
    q_frame = 1.0 - frame
    q_predicate = frame * (1.0 - predicate)
    q_sufficiency = frame * predicate * (1.0 - sufficiency)
    q_authorized = frame * predicate * sufficiency
    entitlement = q_authorized
    support = inputs["positive_energy"] * entitlement
    refute = inputs["negative_energy"] * entitlement
    not_entitled = head.not_entitled_bias + F.softplus(head.raw_alpha) * (q_frame + q_predicate + q_sufficiency)
    return {
        "q_frame": q_frame,
        "q_predicate": q_predicate,
        "q_sufficiency": q_sufficiency,
        "q_authorized": q_authorized,
        "logits": torch.stack([refute, not_entitled, support], dim=-1),
    }


def test_e0_algebraic_equivalence_fp64_gradients_and_predictions() -> None:
    head_current = FinalEntitlementDecisionHead(decision_mode="explicit_product").to(dtype=torch.float64)
    head_reference = copy.deepcopy(head_current)
    inputs_current = {key: value.clone().requires_grad_(True) for key, value in _sample_inputs().items()}
    inputs_reference = {key: value.clone().requires_grad_(True) for key, value in _sample_inputs().items()}

    current = head_current(**inputs_current)
    reference = _q_only_product_reference(head_reference, inputs_reference)
    assert torch.allclose(current["logits"], reference["logits"], rtol=RTOL, atol=ATOL)
    assert torch.equal(current["logits"].argmax(dim=-1), reference["logits"].argmax(dim=-1))

    upstream = torch.randn_like(current["logits"], dtype=torch.float64)
    (current["logits"] * upstream).sum().backward()
    (reference["logits"] * upstream).sum().backward()
    for key in ("frame_prob", "predicate_coverage_prob", "sufficiency_prob"):
        assert torch.allclose(inputs_current[key].grad, inputs_reference[key].grad, rtol=RTOL, atol=ATOL)
    for key in ("positive_energy", "negative_energy"):
        assert torch.allclose(inputs_current[key].grad, inputs_reference[key].grad, rtol=RTOL, atol=ATOL)
    assert torch.allclose(head_current.not_entitled_bias.grad, head_reference.not_entitled_bias.grad, rtol=RTOL, atol=ATOL)
    assert torch.allclose(head_current.raw_alpha.grad, head_reference.raw_alpha.grad, rtol=RTOL, atol=ATOL)


def test_q_normalization_and_boundary_contract() -> None:
    head = FinalEntitlementDecisionHead(decision_mode="conditional_first_blocker")
    inputs = {
        "frame_prob": torch.tensor([0.0, 1.0, 1.0, 1.0, 0.2, 0.7]),
        "predicate_coverage_prob": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.8, 0.3]),
        "sufficiency_prob": torch.tensor([1.0, 1.0, 0.0, 1.0, 0.4, 0.9]),
        "positive_energy": torch.tensor([0.5, 1.7, 0.2, 2.3, 0.9, 3.1]),
        "negative_energy": torch.tensor([1.5, 0.1, 2.2, 0.4, 1.4, 0.8]),
    }
    out = head(**inputs)
    expected_q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.8, 0.04, 0.096, 0.064],
            [0.3, 0.49, 0.021, 0.189],
        ],
        dtype=out["q_masses_4"].dtype,
    )
    assert out["normalization_max_abs_error"].shape == torch.Size([])
    assert out["normalization_ok"].shape == torch.Size([])
    assert torch.all(out["normalization_ok"])
    assert torch.allclose(out["q_masses_4"], expected_q, atol=1e-6)
    assert torch.all(out["q_masses_4"] >= 0)
    assert torch.allclose(out["q_masses_4"].sum(dim=-1), torch.ones(6), atol=1e-6)
    for key in ("logits", "reason_logits_4", "polarity_logits_2", "internal_probs_5", "collapsed_probs_3"):
        assert torch.isfinite(out[key]).all()

    e0 = FinalEntitlementDecisionHead(decision_mode="explicit_product")
    e0_out = e0(**inputs, return_q_diagnostics=True)
    e0_ref = _q_only_product_reference(e0, inputs)
    assert torch.allclose(e0_out["q_masses_4"], expected_q, atol=1e-6)
    assert torch.equal(e0_out["logits"].argmax(dim=-1), e0_ref["logits"].argmax(dim=-1))

def test_a0_return_dictionary_compatibility() -> None:
    head = FinalEntitlementDecisionHead(decision_mode="explicit_product")
    assert set(head(**_sample_inputs(torch.float32))) == {
        "entitlement_prob",
        "support_logit",
        "refute_logit",
        "not_entitled_logit",
        "logits",
    }


def test_five_state_three_way_and_posterior_normalization() -> None:
    out = FinalEntitlementDecisionHead(decision_mode="conditional_first_blocker")(**_sample_inputs(torch.float32))
    assert torch.allclose(out["internal_probs_5"].sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(out["collapsed_probs_3"].sum(dim=-1), torch.ones(4), atol=1e-6)
    active = out["primary_reason_posterior_valid_mask"]
    assert torch.allclose(
        out["primary_reason_posterior"][active].sum(dim=-1),
        torch.ones_like(out["primary_reason_posterior_sum"][active]),
        atol=1e-6,
    )


def test_first_blocker_precedence_reference() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    cases = [
        (0, 1, 1, "FRAME"),
        (1, 0, 1, "PREDICATE"),
        (1, 1, 0, "SUFFICIENCY"),
        (1, 1, 1, "AUTHORIZED"),
        (0, 0, 0, "FRAME"),
    ]
    assert [trainer._p2_primary_reason_from_axes(f, p, s) for f, p, s, _ in cases] == [expected for *_, expected in cases]

def test_secondary_reason_non_causality() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    base = _production_model("A1")
    changed = copy.deepcopy(base)
    indices = torch.arange(4)

    def payload(secondary: torch.Tensor):
        output = _forward_for_arm(base if secondary[0, 0].item() == 0 else changed, "A1")
        inputs = {
            "final_labels": torch.tensor([0, 1, 2, 1]),
            "frame_compatible_labels": torch.tensor([1.0, 0.0, 1.0, 1.0]),
            "predicate_covered_labels": torch.tensor([1.0, 1.0, 0.0, 1.0]),
            "sufficiency_labels": torch.tensor([1.0, 1.0, 1.0, 0.0]),
            "p2_frame_applicability_mask": torch.ones(4, dtype=torch.bool),
            "p2_predicate_applicability_mask": torch.ones(4, dtype=torch.bool),
            "p2_sufficiency_applicability_mask": torch.ones(4, dtype=torch.bool),
            "p2_polarity_applicability_mask": torch.tensor([True, False, True, False]),
            "p2_polarity_targets_2": torch.tensor([0, -100, 1, -100]),
            "p2_primary_reason_targets_4": torch.tensor([3, 0, 1, 2]),
            "p2_reason_supervision_eligible": torch.ones(4, dtype=torch.bool),
            "p2_secondary_reason_targets_3": secondary,
        }
        losses = trainer._p2_reason_router_losses(output, inputs, indices, False, 1.0)
        losses["total"].backward()
        grads = {
            name: None if param.grad is None else param.grad.detach().clone()
            for name, param in (base if secondary[0, 0].item() == 0 else changed).named_parameters()
            if param.requires_grad
        }
        return losses, grads

    losses_a, grads_a = payload(torch.zeros(4, 3, dtype=torch.long))
    losses_b, grads_b = payload(torch.ones(4, 3, dtype=torch.long))
    for key in ("total", "label", "primary_reason"):
        assert torch.allclose(losses_a[key], losses_b[key])
    assert grads_a.keys() == grads_b.keys()
    for key in grads_a:
        if grads_a[key] is None or grads_b[key] is None:
            assert grads_a[key] is None and grads_b[key] is None
        else:
            assert torch.allclose(grads_a[key], grads_b[key])

class _DeterministicBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 6) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(32, hidden_size)
        with torch.no_grad():
            values = torch.arange(32 * hidden_size, dtype=torch.float32).reshape(32, hidden_size)
            self.embedding.weight.copy_(values / values.max())

    def forward(self, input_ids: torch.Tensor):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _production_model(arm: str):
    from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal

    mode = "conditional_first_blocker" if arm in {"A1", "A3"} else "explicit_product"
    model = ContraMambaV6BMinimal(
        backbone=_DeterministicBackbone(hidden_size=6),
        hidden_size=6,
        frame_size=4,
        predicate_size=4,
        sufficiency_size=4,
        energy_size=3,
        dropout=0.0,
        decision_mode=mode,
    )
    for param in model.mamba.parameters():
        param.requires_grad_(False)
    return model


def _production_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8], [3, 5, 7, 9]]),
        "attention_mask": torch.ones(4, 4),
        "claim_mask": torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float32),
        "evidence_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.float32),
    }


def _forward_for_arm(model: torch.nn.Module, arm: str) -> dict[str, torch.Tensor]:
    return model(
        **_production_batch(),
        gradient_ownership_mode="explicit_local" if arm in {"A2", "A3"} else "joint",
        return_q_diagnostics=True,
    )


def _has_grad(module: torch.nn.Module) -> bool:
    return any(param.grad is not None for param in module.parameters())


def _all_grad_none(module: torch.nn.Module) -> bool:
    return all(param.grad is None for param in module.parameters())


def _clear_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def _assert_backbone_grad_none(model: torch.nn.Module) -> None:
    assert _all_grad_none(model.mamba)



def _p2_loss_inputs() -> dict[str, torch.Tensor]:
    return {
        "final_labels": torch.tensor([0, 1, 2, 1]),
        "frame_compatible_labels": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "predicate_covered_labels": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "sufficiency_labels": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "p2_frame_applicability_mask": torch.ones(4, dtype=torch.bool),
        "p2_predicate_applicability_mask": torch.ones(4, dtype=torch.bool),
        "p2_sufficiency_applicability_mask": torch.ones(4, dtype=torch.bool),
        "p2_polarity_applicability_mask": torch.tensor([True, False, True, False]),
        "p2_polarity_targets_2": torch.tensor([0, -100, 1, -100]),
        "p2_primary_reason_targets_4": torch.tensor([3, 0, 1, 2]),
        "p2_reason_supervision_eligible": torch.ones(4, dtype=torch.bool),
        "p2_secondary_reason_targets_3": torch.zeros(4, 3, dtype=torch.long),
    }


def _legacy_polarity_loss(output: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    polarity_logits = torch.stack([output["negative_energy"], output["positive_energy"]], dim=-1)
    return F.cross_entropy(polarity_logits, targets)


def _production_polarity_loss(output: dict[str, torch.Tensor], arm: str, targets: torch.Tensor) -> torch.Tensor:
    if arm in {"A1", "A3"}:
        import scripts.train_controlled_v6b_minimal as trainer

        losses = trainer._p2_reason_router_losses(output, _p2_loss_inputs(), torch.arange(4), False, 1.0)
        return losses["polarity"]
    return _legacy_polarity_loss(output, targets)


def _observed_ownership_matrix(model: torch.nn.Module) -> dict[str, bool]:
    return {
        "backbone": _has_grad(model.mamba),
        "frame_gate": _has_grad(model.frame_gate),
        "predicate_head": _has_grad(model.predicate_coverage_head),
        "sufficiency_gate": _has_grad(model.sufficiency_gate),
        "polarity_head": _has_grad(model.polarity_energy_head),
        "product_composer": (
            model.decision_head.not_entitled_bias.grad is not None
            or model.decision_head.raw_alpha.grad is not None
        ),
        "reason_router": bool(
            getattr(model.decision_head, "reason_bias_3", None) is not None
            and model.decision_head.reason_bias_3.grad is not None
        ),
    }


def _ownership_audit(model: torch.nn.Module, expected: dict[str, bool]) -> dict[str, object]:
    observed = _observed_ownership_matrix(model)
    violations = sum(1 for name, expected_value in expected.items() if observed[name] != expected_value)
    return {
        "expected_nonzero_matrix": expected,
        "observed_nonzero_matrix": observed,
        "ownership_violation_count": violations,
        "ownership_contract_pass": violations == 0,
    }


def _assert_ownership_audit(model: torch.nn.Module, expected: dict[str, bool]) -> None:
    audit = _ownership_audit(model, expected)
    assert set(audit) == {
        "expected_nonzero_matrix",
        "observed_nonzero_matrix",
        "ownership_violation_count",
        "ownership_contract_pass",
    }
    assert audit["ownership_violation_count"] == 0
    assert audit["ownership_contract_pass"] is True

def test_a0_a3_actual_production_autograd_ownership_matrix() -> None:
    labels = torch.tensor([0, 1, 2, 0])
    polarity_targets = torch.tensor([0, 1, 0, 1])
    for arm in ("A0", "A1", "A2", "A3"):
        model = _production_model(arm)
        explicit = arm in {"A2", "A3"}

        out = _forward_for_arm(model, arm)
        F.binary_cross_entropy_with_logits(out["frame_logit"], torch.ones(4)).backward()
        assert _has_grad(model.frame_gate)
        assert not _has_grad(model.predicate_coverage_head)
        assert not _has_grad(model.sufficiency_gate)
        assert not _has_grad(model.polarity_energy_head)
        _assert_backbone_grad_none(model)
        _clear_grads(model)

        out = _forward_for_arm(model, arm)
        F.binary_cross_entropy_with_logits(out["predicate_coverage_logit"], torch.ones(4)).backward()
        assert _has_grad(model.predicate_coverage_head)
        assert _has_grad(model.frame_gate) is (not explicit)
        assert not _has_grad(model.sufficiency_gate)
        assert not _has_grad(model.polarity_energy_head)
        _assert_backbone_grad_none(model)
        _clear_grads(model)

        out = _forward_for_arm(model, arm)
        F.binary_cross_entropy_with_logits(out["sufficiency_logit"], torch.ones(4)).backward()
        assert _has_grad(model.sufficiency_gate)
        assert _has_grad(model.frame_gate) is (not explicit)
        assert _has_grad(model.predicate_coverage_head) is (not explicit)
        assert not _has_grad(model.polarity_energy_head)
        _assert_backbone_grad_none(model)
        _clear_grads(model)

        out = _forward_for_arm(model, arm)
        _production_polarity_loss(out, arm, polarity_targets).backward()
        expected_polarity = {
            "backbone": False,
            "frame_gate": not explicit,
            "predicate_head": not explicit,
            "sufficiency_gate": not explicit,
            "polarity_head": True,
            "product_composer": False,
            "reason_router": False,
        }
        _assert_ownership_audit(model, expected_polarity)
        _assert_backbone_grad_none(model)
        _clear_grads(model)

        out = _forward_for_arm(model, arm)
        F.cross_entropy(out["logits"], labels).backward()
        _assert_backbone_grad_none(model)
        if arm == "A2":
            assert model.decision_head.not_entitled_bias.grad is not None
            assert model.decision_head.raw_alpha.grad is not None
            assert not _has_grad(model.frame_gate)
            assert not _has_grad(model.predicate_coverage_head)
            assert not _has_grad(model.sufficiency_gate)
            assert not _has_grad(model.polarity_energy_head)
        elif arm == "A3":
            assert model.decision_head.reason_bias_3.grad is not None
            assert model.decision_head.not_entitled_bias.grad is None
            assert model.decision_head.raw_alpha.grad is None
            assert not _has_grad(model.frame_gate)
            assert not _has_grad(model.predicate_coverage_head)
            assert not _has_grad(model.sufficiency_gate)
            assert not _has_grad(model.polarity_energy_head)
        else:
            assert _has_grad(model.frame_gate)
            assert _has_grad(model.predicate_coverage_head)
            assert _has_grad(model.sufficiency_gate)
            assert _has_grad(model.polarity_energy_head)
        _clear_grads(model)


def test_a1_a3_primary_reason_ce_production_gradients() -> None:
    targets = torch.tensor([0, 1, 2, 3])

    a1 = _production_model("A1")
    out = _forward_for_arm(a1, "A1")
    F.cross_entropy(out["reason_logits_4"], targets).backward()
    assert a1.decision_head.reason_bias_3.grad is not None
    assert _has_grad(a1.frame_gate)
    assert _has_grad(a1.predicate_coverage_head)
    assert _has_grad(a1.sufficiency_gate)
    assert not _has_grad(a1.polarity_energy_head)
    _assert_backbone_grad_none(a1)

    a3 = _production_model("A3")
    out = _forward_for_arm(a3, "A3")
    F.cross_entropy(out["reason_logits_4"], targets).backward()
    assert a3.decision_head.reason_bias_3.grad is not None
    assert a3.decision_head.not_entitled_bias.grad is None
    assert a3.decision_head.raw_alpha.grad is None
    assert not _has_grad(a3.frame_gate)
    assert not _has_grad(a3.predicate_coverage_head)
    assert not _has_grad(a3.sufficiency_gate)
    assert not _has_grad(a3.polarity_energy_head)
    _assert_backbone_grad_none(a3)


def test_legacy_none_comparator_constructor_config_preserved() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    class Parser:
        def error(self, message: str) -> None:
            raise AssertionError(message)

    none_args = SimpleNamespace(
        reason_router_arm="none",
        use_temporal_comparator=True,
        use_predicate_comparator=True,
    )
    trainer._p2_resolve_arm_contract(none_args, [], Parser())
    assert none_args.resolved_use_temporal_comparator is True
    assert none_args.resolved_use_predicate_comparator is True
    assert none_args.use_temporal_comparator is True
    assert none_args.use_predicate_comparator is True

    p2_args = SimpleNamespace(
        reason_router_arm="A0",
        architecture="v6b_minimal",
        reason_router_mode="auto",
        gradient_ownership_mode="auto",
        reason_loss_weight=0.0,
        freeze_encoder=True,
        frame_downstream_gradient_mode="joint",
        reason_router_epsilon=1e-8,
        reason_min_train_count=1,
        reason_min_dev_count=1,
        use_temporal_comparator=False,
        use_predicate_comparator=False,
    )
    trainer._p2_resolve_arm_contract(p2_args, [], Parser())
    assert p2_args.resolved_use_temporal_comparator is False
    assert p2_args.resolved_use_predicate_comparator is False


def _p2_lineage_record(row_id: str, pair_id: str, kind: str) -> dict:
    configs = {
        "none": ("none", "SUPPORT", 1, 1, 1, "SUPPORT"),
        "entity_swap": ("entity_swap", "NOT_ENTITLED", 0, 1, 1, "NONE"),
        "predicate_swap": ("predicate_swap", "NOT_ENTITLED", 1, 0, 1, "NONE"),
        "evidence_deletion": ("evidence_deletion", "NOT_ENTITLED", 1, 1, 0, "NONE"),
        "polarity_flip": ("polarity_flip", "REFUTE", 1, 1, 1, "REFUTE"),
    }
    intervention_type, final_label, frame, predicate, sufficiency, polarity = configs[kind]
    primary = "polarity" if kind == "polarity_flip" else "frame" if kind == "entity_swap" else "predicate" if kind == "predicate_swap" else "sufficiency" if kind == "evidence_deletion" else "none"
    return {
        "id": row_id,
        "pair_id": pair_id,
        "intervention_type": intervention_type,
        "final_label": final_label,
        "frame_compatible_label": frame,
        "predicate_covered_label": predicate,
        "sufficiency_label": sufficiency,
        "polarity_label": polarity,
        "primary_failure_type": primary,
    }


def _p2_lineage_sidecar(row: dict, split: str, canonical_row_id: str) -> dict:
    import scripts.train_controlled_v6b_minimal as trainer

    return {
        "row_id": row["id"],
        "pair_id": row["pair_id"],
        "split": split,
        "canonical_row_id": canonical_row_id,
        "frame_compatible_label": row["frame_compatible_label"],
        **{field: "PASS" for field in trainer.P2_GENERATOR_COMPONENT_STATUS_FIELDS},
    }


def _p2_lineage_records(pair_id: str, split: str) -> tuple[list[dict], dict[str, dict]]:
    suffixes = ("none", "entity_swap", "predicate_swap", "evidence_deletion", "polarity_flip")
    records = [_p2_lineage_record(f"{pair_id}__{suffix}", pair_id, suffix) for suffix in suffixes]
    canonical_row_id = f"{pair_id}__none"
    return records, {row["id"]: _p2_lineage_sidecar(row, split, canonical_row_id) for row in records}


def test_p2_supervision_integrity_polarity_intervention_and_frame_binary_audit() -> None:
    train_records, train_sidecar = _p2_lineage_records("pair_x", "train")
    dev_records, dev_sidecar = _p2_lineage_records("pair_y", "dev")
    sidecar = {**train_sidecar, **dev_sidecar}
    audit = trainer._p2_prepare_reason_supervision(
        train_records=train_records,
        dev_records=dev_records,
        train_inputs={},
        dev_inputs={},
        train_source_labels=["clean_main"] * len(train_records),
        sidecar_by_id=sidecar,
        require_min_counts=False,
        min_train_count=1,
        min_dev_count=1,
        device=torch.device("cpu"),
    )
    assert audit["target_class_counts"]["train_applicable_binary"]["frame"] == {0: 1, 1: 4}
    assert audit["train_reason_counts"] == {"FRAME": 1, "PREDICATE": 1, "SUFFICIENCY": 1, "AUTHORIZED": 2}
    assert audit["train_exclusion_counts"].get("P2_CANONICAL_ROW_ID_MISMATCH", 0) == 0

    bad = [_p2_lineage_record("bad_pair__polarity_flip", "bad_pair", "polarity_flip")]
    bad_dev = [_p2_lineage_record("bad_pair_dev__polarity_flip", "bad_pair_dev", "polarity_flip")]
    bad[0]["intervention_type"] = "none"
    bad_sidecar = {
        bad[0]["id"]: _p2_lineage_sidecar(bad[0], "train", bad[0]["id"]),
        bad_dev[0]["id"]: _p2_lineage_sidecar(bad_dev[0], "dev", bad_dev[0]["id"]),
    }
    trainer._p2_prepare_reason_supervision(
        train_records=bad,
        dev_records=bad_dev,
        train_inputs={},
        dev_inputs={},
        train_source_labels=["clean_main"],
        sidecar_by_id=bad_sidecar,
        require_min_counts=False,
        min_train_count=1,
        min_dev_count=1,
        device=torch.device("cpu"),
    )
    assert bad[0]["p2_reason_supervision_eligible"] is False
    assert "P2_POLARITY_INTERVENTION_CONTRACT_FAIL" in bad[0]["p2_reason_exclusion_codes"]


def test_p2_canonical_lineage_accepts_shared_pair_anchor() -> None:
    records, sidecar = _p2_lineage_records("pair_x", "train")
    assert trainer._p2_resolve_canonical_lineage_for_split(records=records, sidecar_by_id=sidecar, split="train") == {"pair_x": "pair_x__none"}


def test_p2_canonical_lineage_rejects_sidecar_identity_corruption() -> None:
    records, sidecar = _p2_lineage_records("pair_x", "train")
    sidecar["pair_x__entity_swap"]["row_id"] = "wrong"
    with pytest.raises(ValueError, match="sidecar row_id mismatch"):
        trainer._p2_resolve_canonical_lineage_for_split(records=records, sidecar_by_id=sidecar, split="train")
    records, sidecar = _p2_lineage_records("pair_x", "train")
    sidecar["pair_x__entity_swap"]["pair_id"] = "other_pair"
    with pytest.raises(ValueError, match="sidecar pair_id mismatch"):
        trainer._p2_resolve_canonical_lineage_for_split(records=records, sidecar_by_id=sidecar, split="train")


def test_p2_canonical_lineage_rejects_multiple_canonical_ids_in_one_pair() -> None:
    records, sidecar = _p2_lineage_records("pair_x", "train")
    sidecar["pair_x__entity_swap"]["canonical_row_id"] = "pair_x__entity_swap"
    with pytest.raises(ValueError, match="multiple canonical_row_id"):
        trainer._p2_resolve_canonical_lineage_for_split(records=records, sidecar_by_id=sidecar, split="train")


def test_p2_canonical_lineage_rejects_missing_or_cross_pair_target() -> None:
    records, sidecar = _p2_lineage_records("pair_x", "train")
    sidecar["pair_x__none"]["canonical_row_id"] = "missing"
    for row in records[1:]:
        sidecar[row["id"]]["canonical_row_id"] = "missing"
    with pytest.raises(ValueError, match="canonical target missing"):
        trainer._p2_resolve_canonical_lineage_for_split(records=records, sidecar_by_id=sidecar, split="train")
    records, sidecar = _p2_lineage_records("pair_x", "train")
    other = _p2_lineage_record("other_pair__none", "other_pair", "none")
    records.append(other)
    sidecar[other["id"]] = _p2_lineage_sidecar(other, "train", other["id"])
    for row in records[:-1]:
        sidecar[row["id"]]["canonical_row_id"] = other["id"]
    with pytest.raises(ValueError, match="canonical target pair mismatch"):
        trainer._p2_resolve_canonical_lineage_for_split(records=records, sidecar_by_id=sidecar, split="train")


def test_p2_canonical_lineage_rejects_non_self_anchored_canonical_target() -> None:
    records, sidecar = _p2_lineage_records("pair_x", "train")
    sidecar["pair_x__none"]["canonical_row_id"] = "pair_x__entity_swap"
    with pytest.raises(ValueError, match="canonical target is not self-anchored"):
        trainer._p2_resolve_canonical_lineage_for_split(records=records, sidecar_by_id=sidecar, split="train")

def test_metadata_eligibility_and_checkpoint_contract_source_names() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    assert trainer.P2_GENERATOR_COMPONENT_STATUS_FIELDS == (
        "schema_status",
        "dataset_source_status",
        "grammar_status",
        "canonical_status",
        "intervention_contract_status",
        "polarity_contamination_status",
        "time_swap_status",
    )
    assert trainer._p2_normalized_generator_status({field: "PASS" for field in trainer.P2_GENERATOR_COMPONENT_STATUS_FIELDS}) == "CLEAN"
    assert trainer._p2_checkpoint_metadata_from_args(SimpleNamespace(reason_router_arm="none")) == {}
    assert trainer._p2_checkpoint_metadata_from_args(SimpleNamespace(reason_router_arm="A3")).get("reason_router_p2_schema_version") == "reason_router_p2_v1"


def test_a0_reference_population_join_and_recovery_harm_definitions(tmp_path) -> None:
    import json
    import scripts.train_controlled_v6b_minimal as trainer

    reference_path = tmp_path / "a0_reference.jsonl"
    reference_rows = [
        {"stable_id": "r_ne_refute", "pair_id": "p1", "gold_label": "NOT_ENTITLED", "pred_label": "REFUTE"},
        {"stable_id": "r_ne_support", "pair_id": "p2", "gold_label": "NOT_ENTITLED", "pred_label": "SUPPORT"},
        {"stable_id": "r_support", "pair_id": "p3", "gold_label": "SUPPORT", "pred_label": "SUPPORT"},
    ]
    reference_path.write_text("".join(json.dumps(row) + "\n" for row in reference_rows), encoding="utf-8")
    args = SimpleNamespace(reason_router_arm="A3", reason_router_a0_reference_predictions=reference_path)

    def exported(stable_id: str, pair_id: str, gold: str, pred: str) -> dict:
        item = {
            "stable_id": stable_id,
            "pair_id": pair_id,
            "gold_label": gold,
            "gold_label_id": 1 if gold == "NOT_ENTITLED" else 2,
            "pred_label": pred,
            "final_logits": [0.0, 2.0, 0.0] if pred == "NOT_ENTITLED" else [2.0, 0.0, 0.0],
            "final_probs": [0.1, 0.8, 0.1] if pred == "NOT_ENTITLED" else [0.8, 0.1, 0.1],
        }
        record = {"id": stable_id, "pair_id": pair_id, "p2_secondary_reasons_3": [0, 0, 0]}
        output = {
            "reason_logits_4": torch.zeros(1, 4),
            "original_product_logits_3": torch.zeros(1, 3),
            "collapsed_logits_3": torch.zeros(1, 3),
            "collapsed_probs_3": torch.tensor([[0.1, 0.8, 0.1]]),
            "q_masses_4": torch.tensor([[0.2, 0.3, 0.1, 0.4]]),
            "primary_reason_posterior": torch.tensor([[0.7, 0.2, 0.1]]),
        }
        trainer._add_reason_router_p2_prediction_exports(item, record, output, 0, args)
        return item

    recovered_refute = exported("r_ne_refute", "p1", "NOT_ENTITLED", "NOT_ENTITLED")
    recovered_support = exported("r_ne_support", "p2", "NOT_ENTITLED", "NOT_ENTITLED")
    harmed_ne = exported("r_support", "p3", "SUPPORT", "NOT_ENTITLED")
    harmed_refute = exported("r_support", "p3", "SUPPORT", "REFUTE")
    preserved = exported("r_support", "p3", "SUPPORT", "SUPPORT")

    assert recovered_refute["a0_fixed_false_entitlement_population"] is True
    assert recovered_support["a0_fixed_false_entitlement_population"] is True
    assert recovered_refute["recovery_from_a0_false_entitlement"] is True
    assert recovered_support["recovery_from_a0_false_entitlement"] is True
    assert harmed_ne["harm_support_to_not_entitled"] is True
    assert harmed_refute["harm_support_to_refute"] is True
    assert preserved["support_preserved_from_a0"] is True


def test_a1_a3_export_requires_a0_reference_not_shadow_prediction() -> None:
    import pytest
    import scripts.train_controlled_v6b_minimal as trainer

    item = {"stable_id": "r1", "pair_id": "p1", "gold_label": "NOT_ENTITLED", "gold_label_id": 1, "pred_label": "NOT_ENTITLED", "final_logits": [0, 1, 0], "final_probs": [0.2, 0.6, 0.2]}
    record = {"id": "r1", "pair_id": "p1", "p2_secondary_reasons_3": [0, 0, 0]}
    output = {"reason_logits_4": torch.zeros(1, 4), "original_product_logits_3": torch.zeros(1, 3)}
    args = SimpleNamespace(reason_router_arm="A3", reason_router_a0_reference_predictions=None)
    with pytest.raises(ValueError, match="P2_A0_REFERENCE_REQUIRED"):
        trainer._add_reason_router_p2_prediction_exports(item, record, output, 0, args)


def test_product_arm_q_export_and_original_active_alias() -> None:
    import pytest
    import scripts.train_controlled_v6b_minimal as trainer

    item = {"stable_id": "r1", "pair_id": "p1", "gold_label": "NOT_ENTITLED", "gold_label_id": 1, "pred_label": "NOT_ENTITLED", "final_logits": [0.0, 1.0, 0.0], "final_probs": [0.2, 0.6, 0.2]}
    record = {"id": "r1", "pair_id": "p1", "p2_secondary_reasons_3": [0, 0, 0]}
    output = {
        "reason_logits_4": None,
        "q_frame": torch.tensor([0.1]),
        "q_predicate": torch.tensor([0.2]),
        "q_sufficiency": torch.tensor([0.3]),
        "q_authorized": torch.tensor([0.4]),
    }
    args = SimpleNamespace(reason_router_arm="A0", reason_router_a0_reference_predictions=None)
    trainer._add_reason_router_p2_prediction_exports(item, record, output, 0, args)
    assert item["original_product_logits_3"] == item["active_collapsed_logits_3"]
    assert item["original_product_probs_3"] == item["active_collapsed_probs_3"]
    assert item["revised_collapsed_logits_3"] is None
    assert item["q_frame"] == pytest.approx(0.1)
    assert item["q_predicate"] == pytest.approx(0.2)
    assert item["q_sufficiency"] == pytest.approx(0.3)
    assert item["q_authorized"] == pytest.approx(0.4)


def test_predicted_reason_uses_posterior_and_external_validity() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    def make_item(pred: str) -> dict:
        return {"stable_id": f"r_{pred}", "pair_id": f"p_{pred}", "gold_label": "NOT_ENTITLED", "gold_label_id": 1, "pred_label": pred, "final_logits": [0.0, 1.0, 0.0], "final_probs": [0.2, 0.6, 0.2]}

    record = {"id": "r_NOT_ENTITLED", "pair_id": "p_NOT_ENTITLED", "p2_secondary_reasons_3": [0, 0, 0]}
    output = {
        "reason_logits_4": torch.tensor([[100.0, 0.0, 0.0, 200.0]]),
        "primary_reason_posterior": torch.tensor([[0.1, 0.8, 0.1]]),
        "q_masses_4": torch.tensor([[0.1, 0.8, 0.1, 0.0]]),
    }
    args = SimpleNamespace(reason_router_arm="A0", reason_router_a0_reference_predictions=None)
    ne_item = make_item("NOT_ENTITLED")
    trainer._add_reason_router_p2_prediction_exports(ne_item, record, output, 0, args)
    assert ne_item["predicted_primary_reason"] == "PREDICATE"
    assert ne_item["predicted_primary_reason_id"] == 1
    assert ne_item["primary_reason_posterior_valid"] is True

    support_item = make_item("SUPPORT")
    trainer._add_reason_router_p2_prediction_exports(support_item, {**record, "id": "r_SUPPORT", "pair_id": "p_SUPPORT"}, output, 0, args)
    assert support_item["predicted_primary_reason"] is None
    assert support_item["predicted_primary_reason_id"] is None
    assert support_item["primary_reason_posterior_valid"] is False


def test_legacy_metadata_validation_continues_during_p2_migration(tmp_path) -> None:
    import pytest
    import scripts.train_controlled_v6b_minimal as trainer

    args = SimpleNamespace(
        reason_router_arm="A3",
        architecture="v6b_minimal",
        backbone="mamba",
        model_name="current",
        vnext_router_mode=None,
        vnext_use_slot_mismatch_head=None,
        vnext_slot_mismatch_detach_input=None,
        vnext_slot_mismatch_input_mode=None,
        vnext_slot_mismatch_head_type=None,
    )
    metadata = {"architecture": "other", "backbone": "mamba", "model_name": "current"}
    with pytest.raises(ValueError, match="not compatible"):
        trainer._validate_model_checkpoint_metadata(metadata, args, tmp_path / "legacy.pt")


def test_exact_resume_p2_metadata_mismatch_rejected(tmp_path) -> None:
    import pytest
    import scripts.train_controlled_v6b_minimal as trainer

    args = SimpleNamespace(
        reason_router_arm="A3",
        resolved_reason_router_mode="conditional_first_blocker",
        resolved_gradient_ownership_mode="explicit_local",
        resolved_reason_loss_weight=1.0,
        reason_router_epsilon=1e-8,
        reason_min_train_count=1,
        reason_min_dev_count=1,
        expected_integrity_sidecar_semantic_sha256="sha",
        resolved_split_seed=174,
        resolved_split_policy="fixed_explicit_split_seed",
        resolved_split_seed_explicit=True,
        dev_ratio=0.2,
        class_weighting="none",
        weighted_label_loss=False,
        balanced_sampler=False,
        lambda_frame_preserve=None,
        lambda_frame_anchor=None,
        ranking_weight=0.0,
        boundary_loss_weight=0.0,
        frame_violation_loss_weight=0.0,
        predicate_isolation_loss_weight=0.0,
        preservation_entitlement_loss_weight=0.0,
        lr=1e-4,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        freeze_encoder=True,
        architecture="v6b_minimal",
        backbone="mamba",
        model_name="same",
        vnext_router_mode=None,
        vnext_use_slot_mismatch_head=None,
        vnext_slot_mismatch_detach_input=None,
        vnext_slot_mismatch_input_mode=None,
        vnext_slot_mismatch_head_type=None,
    )
    metadata = trainer._p2_checkpoint_metadata_from_args(args)
    metadata.update({"architecture": "v6b_minimal", "backbone": "mamba", "model_name": "same"})
    metadata["reason_router_epsilon"] = 1e-6
    with pytest.raises(ValueError, match="P2_CHECKPOINT_RESUME_FORBIDDEN"):
        trainer._validate_model_checkpoint_metadata(metadata, args, tmp_path / "p2.pt")


def test_pretraining_a0_reference_validation_gold_pair_hash(tmp_path) -> None:
    import json
    import pytest
    import scripts.train_controlled_v6b_minimal as trainer

    reference_path = tmp_path / "a0_reference.jsonl"
    reference_path.write_text(
        json.dumps({"stable_id": "r1", "pair_id": "p1", "gold_label": "NOT_ENTITLED", "pred_label": "SUPPORT"}) + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(reason_router_arm="A3", reason_router_a0_reference_predictions=reference_path)
    records = [{"id": "r1", "pair_id": "p1", "final_label": "NOT_ENTITLED"}]
    audit = trainer._p2_validate_a0_reference_for_universe(args, records)
    assert audit["joined_row_count"] == 1
    assert audit["reference_row_count"] == 1
    assert len(audit["sha256"]) == 64

    with pytest.raises(ValueError, match="P2_A0_REFERENCE_PAIR_MISMATCH"):
        trainer._p2_validate_a0_reference_for_universe(args, [{"id": "r1", "pair_id": "other", "final_label": "NOT_ENTITLED"}])
    with pytest.raises(ValueError, match="P2_A0_REFERENCE_GOLD_MISMATCH"):
        trainer._p2_validate_a0_reference_for_universe(args, [{"id": "r1", "pair_id": "p1", "final_label": "SUPPORT"}])
    missing_args = SimpleNamespace(reason_router_arm="A3", reason_router_a0_reference_predictions=None)
    with pytest.raises(ValueError, match="P2_A0_REFERENCE_REQUIRED"):
        trainer._p2_validate_a0_reference_for_universe(missing_args, records)


def _p2_args_for_checkpoint(arm: str, mode: str, ownership: str, epsilon: float = 1e-8, reason_weight: float = 1.0):
    return SimpleNamespace(
        reason_router_arm=arm,
        resolved_reason_router_mode=mode,
        resolved_gradient_ownership_mode=ownership,
        resolved_reason_loss_weight=reason_weight,
        reason_router_epsilon=epsilon,
        reason_min_train_count=1,
        reason_min_dev_count=1,
        expected_integrity_sidecar_semantic_sha256="sha",
        resolved_split_seed=174,
        resolved_split_policy="fixed_explicit_split_seed",
        resolved_split_seed_explicit=True,
        p2_train_row_identity_hash="train_hash",
        p2_dev_row_identity_hash="dev_hash",
        dev_ratio=0.2,
        class_weighting="none",
        weighted_label_loss=False,
        balanced_sampler=False,
        lambda_frame_preserve=None,
        lambda_frame_anchor=None,
        ranking_weight=0.0,
        boundary_loss_weight=0.0,
        frame_violation_loss_weight=0.0,
        predicate_isolation_loss_weight=0.0,
        preservation_entitlement_loss_weight=0.0,
        lr=1e-4,
        head_lr=None,
        encoder_lr=None,
        weight_decay=None,
        freeze_encoder=True,
        architecture="v6b_minimal",
        backbone="mamba",
        model_name="same",
        vnext_router_mode=None,
        vnext_use_slot_mismatch_head=None,
        vnext_slot_mismatch_detach_input=None,
        vnext_slot_mismatch_input_mode=None,
        vnext_slot_mismatch_head_type=None,
    )


def test_p2_load_state_dict_contract_cases() -> None:
    import pytest
    import scripts.train_controlled_v6b_minimal as trainer

    a0 = _production_model("A0")
    a2 = _production_model("A2")
    assert trainer._p2_load_state_dict_with_contract(_production_model("A0"), a0.state_dict(), SimpleNamespace(reason_router_arm="A0")) == "strict_legacy_or_product"
    assert trainer._p2_load_state_dict_with_contract(_production_model("A2"), a2.state_dict(), SimpleNamespace(reason_router_arm="A2")) == "strict_legacy_or_product"

    legacy_state = a0.state_dict()
    assert trainer._p2_load_state_dict_with_contract(_production_model("A1"), legacy_state, SimpleNamespace(reason_router_arm="A1")) == "p2_common_initialization_migration"
    assert trainer._p2_load_state_dict_with_contract(_production_model("A3"), legacy_state, SimpleNamespace(reason_router_arm="A3")) == "p2_common_initialization_migration"

    missing_extra = dict(legacy_state)
    missing_extra.pop("decision_head.raw_alpha")
    with pytest.raises(RuntimeError, match="P2_CHECKPOINT_COMPATIBILITY_FAILED"):
        trainer._p2_load_state_dict_with_contract(_production_model("A3"), missing_extra, SimpleNamespace(reason_router_arm="A3"))
    unexpected = dict(legacy_state)
    unexpected["unexpected.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="P2_CHECKPOINT_COMPATIBILITY_FAILED"):
        trainer._p2_load_state_dict_with_contract(_production_model("A3"), unexpected, SimpleNamespace(reason_router_arm="A3"))

    exact = _production_model("A3")
    assert trainer._p2_load_state_dict_with_contract(_production_model("A3"), exact.state_dict(), SimpleNamespace(reason_router_arm="A3")) == "p2_exact_resume"


def test_p2_exact_resume_metadata_cross_arm_and_config_rejection(tmp_path) -> None:
    import pytest
    import scripts.train_controlled_v6b_minimal as trainer

    args = _p2_args_for_checkpoint("A3", "conditional_first_blocker", "explicit_local")
    metadata = trainer._p2_checkpoint_metadata_from_args(args)
    metadata.update({"architecture": "v6b_minimal", "backbone": "mamba", "model_name": "same"})
    trainer._validate_model_checkpoint_metadata(dict(metadata), args, tmp_path / "ok.pt")

    for field, value in (
        ("reason_router_arm", "A1"),
        ("reason_router_composer", "explicit_product"),
        ("gradient_ownership_mode", "joint"),
        ("reason_router_epsilon", 1e-6),
        ("reason_loss_weight", 2.0),
    ):
        bad = dict(metadata)
        bad[field] = value
        with pytest.raises(ValueError, match="P2_CHECKPOINT_RESUME_FORBIDDEN"):
            trainer._validate_model_checkpoint_metadata(bad, args, tmp_path / f"{field}.pt")


def test_a3_raw_owner_local_polarity_ce_gradients() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    model = _production_model("A3")
    output = _forward_for_arm(model, "A3")
    trainer._p2_reason_router_losses(output, _p2_loss_inputs(), torch.arange(4), False, 1.0)["polarity"].backward()
    assert _has_grad(model.polarity_energy_head)
    assert not _has_grad(model.frame_gate)
    assert not _has_grad(model.predicate_coverage_head)
    assert not _has_grad(model.sufficiency_gate)
    assert model.decision_head.reason_bias_3.grad is None
    assert model.decision_head.not_entitled_bias.grad is None
    assert model.decision_head.raw_alpha.grad is None
    _assert_backbone_grad_none(model)


def test_p2_loss_export_after_class_weight_replacement_identity_and_history() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    model = _production_model("A3")
    output = _forward_for_arm(model, "A3")
    inputs = _p2_loss_inputs()
    indices = torch.arange(4)
    losses = trainer._p2_reason_router_losses(output, inputs, indices, False, 1.0)
    selected_labels = inputs["final_labels"].index_select(0, indices)
    weighted_label = F.cross_entropy(
        output["logits"].index_select(0, indices),
        selected_labels,
        weight=torch.tensor([1.0, 3.0, 5.0]),
    )
    replaced = dict(losses)
    replaced["label"] = weighted_label
    replaced["total"] = losses["total"] - losses["label"] + weighted_label
    export = trainer._p2_reason_arm_loss_export(replaced, inputs, indices, 1.0)
    history = []
    first = trainer._p2_record_epoch_loss_snapshot(history, epoch=1, loss_export=export)
    original_first_total = first["loss_summary"]["loss_total"]["value"]
    export["loss_total"]["value"] = -999.0
    second_export = trainer._p2_reason_arm_loss_export(replaced, inputs, indices, 1.0)
    second = trainer._p2_record_epoch_loss_snapshot(history, epoch=2, loss_export=second_export)

    assert len(history) == 2
    assert first["epoch"] == 1
    assert second["epoch"] == 2
    assert "loss_summary" in first
    assert "loss_summary" in second
    assert first is not second
    assert first["loss_summary"] is not export
    assert second["loss_summary"] is not second_export
    assert first["loss_summary"]["loss_total"]["value"] == original_first_total
    assert history[-1]["loss_summary"] == second_export
    assert first["loss_summary"]["loss_final_3way_ce"]["value"] == float(weighted_label.detach().cpu().item())
    assert first["loss_summary"]["loss_total"]["value"] == float(replaced["total"].detach().cpu().item())



def test_p2_polarity_applicable_diagnostic_uses_actual_target_mask() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    model = _production_model("A3")
    output = _forward_for_arm(model, "A3")
    inputs = _p2_loss_inputs()
    inputs["p2_polarity_applicability_mask"] = torch.tensor([True, True, True, False])
    inputs["p2_polarity_targets_2"] = torch.tensor([0, -100, 1, -100])
    losses = trainer._p2_reason_router_losses(output, inputs, torch.arange(4), False, 1.0)
    export = trainer._p2_reason_arm_loss_export(losses, inputs, torch.arange(4), 1.0)
    assert int(losses["p2_polarity_applicable_count"].item()) == 2
    assert export["loss_authorized_polarity_ce"]["applicable_count"] == 2
    assert export["loss_authorized_polarity_ce"]["ignored_count"] == 2

def test_p2_product_loss_export_uses_legacy_polarity_mask_and_null_reason() -> None:
    import scripts.train_controlled_v6b_minimal as trainer

    inputs = {
        "final_labels": torch.tensor([0, 1, 2, 1]),
        "frame_compatible_labels": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "predicate_covered_labels": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "sufficiency_labels": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "polarity_labels": torch.tensor([1, 0, 2, 0]),
    }
    losses = {name: torch.tensor(0.25) for name in ("total", "label", "frame", "predicate", "sufficiency", "polarity")}
    export = trainer._p2_product_arm_loss_export(losses, inputs, torch.arange(4))
    assert export["loss_authorized_polarity_ce"]["applicable_count"] == 2
    assert export["loss_authorized_polarity_ce"]["ignored_count"] == 2
    assert export["loss_primary_reason_ce"]["value"] is None
    assert export["loss_primary_reason_ce"]["weighted_value"] is None


def test_p2_disallowed_training_objectives_fail_fast() -> None:
    import pytest
    import scripts.train_controlled_v6b_minimal as trainer

    class Parser:
        def error(self, message: str) -> None:
            raise ValueError(message)

    def args_with(**updates):
        base = SimpleNamespace(
            reason_router_arm="A3",
            architecture="v6b_minimal",
            reason_router_mode="auto",
            gradient_ownership_mode="auto",
            reason_loss_weight=1.0,
            freeze_encoder=True,
            frame_downstream_gradient_mode="joint",
            reason_router_epsilon=1e-8,
            reason_min_train_count=1,
            reason_min_dev_count=1,
            ranking_weight=0.0,
            use_intervention_loss=False,
            compatible_positive_margin_weight=0.0,
            boundary_loss_weight=0.0,
            frame_violation_loss_weight=0.0,
            predicate_isolation_loss_weight=0.0,
            preservation_entitlement_loss_weight=0.0,
            stage175b_support_anchor_mode="off",
            stage175b_support_anchor_weight=0.0,
            stage177c_frame_pairwise_mode="off",
            stage177c_frame_pairwise_weight=0.0,
            use_pair_contrastive_frame_loss=False,
            pair_contrastive_frame_loss_weight=0.0,
            pair_contrastive_frame_data=None,
            use_temporal_diagnostic_loss=False,
            use_temporal_residual_adapter=False,
            use_temporal_adapter_loss=False,
            use_temporal_channel=False,
            use_temporal_channel_loss=False,
            teacher_observer_mode="off",
        )
        for key, value in updates.items():
            setattr(base, key, value)
        return base

    for updates in (
        {"ranking_weight": 0.1},
        {"use_intervention_loss": True},
        {"compatible_positive_margin_weight": 0.05},
        {"boundary_loss_weight": 0.1},
        {"frame_violation_loss_weight": 0.1},
        {"predicate_isolation_loss_weight": 0.1},
        {"preservation_entitlement_loss_weight": 0.1},
        {"stage175b_support_anchor_mode": "paraphrase_margin", "stage175b_support_anchor_weight": 0.1},
        {"stage177c_frame_pairwise_mode": "pair_softplus", "stage177c_frame_pairwise_weight": 0.1},
        {"use_pair_contrastive_frame_loss": True},
        {"pair_contrastive_frame_loss_weight": 0.1},
        {"pair_contrastive_frame_data": Path("dummy.jsonl")},
        {"use_temporal_diagnostic_loss": True},
        {"use_temporal_residual_adapter": True},
        {"use_temporal_adapter_loss": True},
        {"use_temporal_channel": True},
        {"use_temporal_channel_loss": True},
    ):
        with pytest.raises(ValueError, match="P2_INCOMPATIBLE_OPTION"):
            trainer._p2_resolve_arm_contract(args_with(**updates), ["--reason-loss-weight", "1.0"], Parser())


def test_p2_router_negative_nonfinite_rejection_and_normalization_ok_contract() -> None:
    import pytest

    head = FinalEntitlementDecisionHead(decision_mode="conditional_first_blocker")
    valid = _sample_inputs(torch.float32)
    output = head(**valid)
    assert output["normalization_ok"].shape == torch.Size([])
    assert output["normalization_ok"].dtype == torch.bool
    assert output["normalization_ok"]
    for key in ("q_masses_4", "reason_probs_4", "internal_probs_5", "collapsed_probs_3"):
        assert torch.isfinite(output[key]).all()
        assert (output[key] >= 0.0).all()

    negative_q = dict(valid)
    negative_q["frame_prob"] = torch.tensor([1.0 + 5e-7, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="P2_ROUTER_NUMERICAL_CONTRACT_FAILED"):
        head(**negative_q)

    with pytest.raises(ValueError, match="P2_ROUTER_NUMERICAL_CONTRACT_FAILED: q_masses_4"):
        _p2_validate_finite_nonnegative("q_masses_4", torch.tensor([0.2, -0.1, 0.9]))
    with pytest.raises(ValueError, match="P2_ROUTER_NUMERICAL_CONTRACT_FAILED: q_masses_4"):
        _p2_validate_finite_nonnegative("q_masses_4", torch.tensor([0.2, float("nan"), 0.8]))
    with pytest.raises(ValueError, match="P2_ROUTER_NUMERICAL_CONTRACT_FAILED: reason_log_input_4"):
        _p2_validate_finite_nonnegative("reason_log_input_4", torch.tensor([0.2, -0.01, 0.81]))
    with pytest.raises(ValueError, match="P2_ROUTER_NUMERICAL_CONTRACT_FAILED: reason_log_input_4"):
        _p2_validate_finite_nonnegative("reason_log_input_4", torch.tensor([0.2, float("inf"), 0.8]))
    with pytest.raises(ValueError, match="P2_ROUTER_NUMERICAL_CONTRACT_FAILED: collapsed_log_input_3"):
        _p2_validate_finite_nonnegative("collapsed_log_input_3", torch.tensor([0.2, -0.01, 0.81]))
    with pytest.raises(ValueError, match="P2_ROUTER_NUMERICAL_CONTRACT_FAILED: collapsed_log_input_3"):
        _p2_validate_finite_nonnegative("collapsed_log_input_3", torch.tensor([0.2, float("nan"), 0.8]))
    assert bool(_p2_finite_nonnegative(torch.tensor([0.0, 1.0])))
    assert not bool(_p2_finite_nonnegative(torch.tensor([-0.1, 1.0])))
    assert not bool(_p2_finite_nonnegative(torch.tensor([float("nan"), 1.0])))
    assert not bool(_p2_finite_nonnegative(torch.tensor([float("inf"), 1.0])))
