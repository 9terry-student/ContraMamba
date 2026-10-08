from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from scripts import (
    audit_native_recurrent_constructive_interference_pair_gap_localization as pairgap,
)
from scripts import (
    audit_native_recurrent_constructive_interference_serialization_region_pair_localization as mod,
)


def _shape(intermediate: int = 2, state_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        intermediate_size=intermediate,
        state_size=state_size,
        state_width=intermediate * state_size,
    )


def _role_masks(
    *,
    batch: int = 1,
    claim_len: int = 2,
    evidence_len: int = 2,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    seq_len = mod.p3a.MAX_LENGTH
    claim = torch.zeros(batch, seq_len, dtype=torch.bool)
    eos = torch.zeros_like(claim)
    evidence = torch.zeros_like(claim)
    attention = torch.zeros_like(claim)
    for row in range(batch):
        claim[row, :claim_len] = True
        eos[row, claim_len] = True
        evidence[row, claim_len + 1 : claim_len + 1 + evidence_len] = True
        attention[row, : claim_len + 1 + evidence_len] = True
    return {
        "CLAIM": claim,
        "EOS": eos,
        "EVIDENCE": evidence,
    }, attention


def _features(
    *,
    batch: int = 2,
    claim_len: int = 3,
    evidence_len: int = 4,
) -> dict[str, torch.Tensor]:
    roles, attention = _role_masks(
        batch=batch,
        claim_len=claim_len,
        evidence_len=evidence_len,
    )
    input_ids = torch.zeros(
        batch,
        mod.p3a.MAX_LENGTH,
        dtype=torch.long,
    )
    input_ids[roles["CLAIM"]] = 7
    input_ids[roles["EVIDENCE"]] = 11
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "claim_mask": roles["CLAIM"],
        "evidence_mask": roles["EVIDENCE"],
    }


def _brute_region_gap(
    *,
    component: torch.Tensor,
    log_a: torch.Tensor,
    survival: torch.Tensor,
    role_masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    k_count, batch, seq_len, width = component.shape
    p = torch.cumsum(
        log_a.reshape(batch, seq_len, width).to(torch.float64),
        dim=1,
    )
    w = component.to(torch.float64)
    output = {
        name: torch.zeros(k_count, seq_len, dtype=torch.float64)
        for name in mod.REGION_PAIR_NAMES
    }
    for pair_name, (left, right) in zip(
        mod.REGION_PAIR_NAMES,
        mod.REGION_PAIRS,
    ):
        for tau in range(seq_len - 1):
            for sigma in range(tau + 1, seq_len):
                active = (
                    role_masks[left][:, tau]
                    & role_masks[right][:, sigma]
                )
                if not bool(torch.any(active).item()):
                    continue
                weight = (
                    torch.exp(p[:, sigma] - p[:, tau])
                    * survival[:, sigma].to(torch.float64)
                    * active[:, None].to(torch.float64)
                )
                pair = 2.0 * torch.sum(
                    w[:, :, tau, :]
                    * w[:, :, sigma, :]
                    * weight.unsqueeze(0),
                    dim=(1, 2),
                    dtype=torch.float64,
                )
                output[pair_name][:, sigma - tau] += pair
    return output


def test_serialization_role_masks_distinguish_eos_from_pad_zero() -> None:
    features = _features(batch=2, claim_len=3, evidence_len=4)
    masks = mod._serialization_role_masks(features)
    assert set(masks) == {"CLAIM", "EOS", "EVIDENCE"}
    assert torch.all(masks["EOS"].sum(dim=1) == 1)
    assert torch.all(masks["EOS"][:, 3])
    assert not torch.any(masks["EOS"][:, 8:])
    assert torch.all(features["input_ids"][masks["EOS"]] == 0)
    assert torch.all(features["input_ids"][~features["attention_mask"]] == 0)


def test_serialization_role_masks_support_max_budgets() -> None:
    features = _features(
        batch=1,
        claim_len=mod.p3a.CLAIM_BUDGET,
        evidence_len=mod.p3a.EVIDENCE_BUDGET,
    )
    masks = mod._serialization_role_masks(features)
    assert int(masks["CLAIM"].sum().item()) == 63
    assert int(masks["EOS"].sum().item()) == 1
    assert int(masks["EVIDENCE"].sum().item()) == 64
    assert int(features["attention_mask"].sum().item()) == 128


def test_serialization_role_masks_reject_extra_active_gap() -> None:
    features = _features(batch=1, claim_len=2, evidence_len=2)
    features["attention_mask"][0, 10] = True
    with pytest.raises(
        mod.SerializationRegionPairLocalizationError,
        match="ROLE_EOS_COUNT",
    ):
        mod._serialization_role_masks(features)


def test_pair_opportunity_counts_cover_each_strict_pair_once() -> None:
    roles, _attention = _role_masks(
        batch=1,
        claim_len=2,
        evidence_len=2,
    )
    counts = mod._pair_opportunity_counts(roles)
    assert counts["CLAIM->CLAIM"]["all"] == 1
    assert counts["CLAIM->EOS"]["all"] == 2
    assert counts["CLAIM->EVIDENCE"]["all"] == 4
    assert counts["EOS->EVIDENCE"]["all"] == 2
    assert counts["EVIDENCE->EVIDENCE"]["all"] == 1
    assert sum(row["all"] for row in counts.values()) == 10


def test_window_sums_are_exact_fixed_prefixes() -> None:
    vector = list(range(1, 128))
    out = mod._window_sums(vector)
    assert out["gap1_8"] == sum(range(1, 9))
    assert out["gap9_16"] == sum(range(9, 17))
    assert out["gap17_32"] == sum(range(17, 33))
    assert out["gap33_64"] == sum(range(33, 65))
    assert out["gap65_127"] == sum(range(65, 128))
    assert out["gap1_32"] == sum(range(1, 33))
    assert out["all"] == sum(range(1, 128))


def test_region_pair_fft_matches_bruteforce_fast_path() -> None:
    torch.manual_seed(101)
    shape = _shape(2, 2)
    roles, attention = _role_masks(
        batch=2,
        claim_len=3,
        evidence_len=4,
    )
    component = 0.2 * torch.randn(
        3,
        2,
        mod.p3a.MAX_LENGTH,
        shape.state_width,
        dtype=torch.float32,
    )
    component = torch.where(
        attention[None, :, :, None],
        component,
        torch.zeros_like(component),
    )
    log_a = torch.full(
        (
            2,
            mod.p3a.MAX_LENGTH,
            shape.intermediate_size,
            shape.state_size,
        ),
        -0.03,
        dtype=torch.float32,
    )
    survival = pairgap._backward_survival_factor(
        log_a_flat=log_a.reshape(2, mod.p3a.MAX_LENGTH, shape.state_width),
        attention_mask=attention,
    )
    observed, span = mod._region_pair_gap_interference_fft(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention,
        role_masks=roles,
        shape=shape,
    )
    expected = _brute_region_gap(
        component=component,
        log_a=log_a,
        survival=survival,
        role_masks=roles,
    )
    assert span < mod.FAST_FLOAT32_HALF_SPAN_LIMIT
    for name in mod.REGION_PAIR_NAMES:
        torch.testing.assert_close(
            observed[name],
            expected[name],
            rtol=2e-5,
            atol=2e-6,
        )


def test_region_pair_fft_large_span_dyadic_matches_bruteforce() -> None:
    torch.manual_seed(102)
    shape = _shape(1, 2)
    roles, attention = _role_masks(
        batch=1,
        claim_len=4,
        evidence_len=4,
    )
    component = 0.15 * torch.randn(
        2,
        1,
        mod.p3a.MAX_LENGTH,
        shape.state_width,
        dtype=torch.float32,
    )
    component = torch.where(
        attention[None, :, :, None],
        component,
        torch.zeros_like(component),
    )
    log_a = torch.full(
        (
            1,
            mod.p3a.MAX_LENGTH,
            shape.intermediate_size,
            shape.state_size,
        ),
        -0.05,
        dtype=torch.float32,
    )
    log_a[:, 2] = -1000.0
    log_a[:, 5] = -1000.0
    survival = pairgap._backward_survival_factor(
        log_a_flat=log_a.reshape(1, mod.p3a.MAX_LENGTH, shape.state_width),
        attention_mask=attention,
    )
    observed, span = mod._region_pair_gap_interference_fft(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention,
        role_masks=roles,
        shape=shape,
    )
    expected = _brute_region_gap(
        component=component,
        log_a=log_a,
        survival=survival,
        role_masks=roles,
    )
    assert span > mod.FAST_FLOAT32_HALF_SPAN_LIMIT
    for name in mod.REGION_PAIR_NAMES:
        assert torch.isfinite(observed[name]).all()
        torch.testing.assert_close(
            observed[name],
            expected[name],
            rtol=2e-5,
            atol=2e-6,
        )


def test_region_pair_sum_reconstructs_validated_pair_gap_engine() -> None:
    torch.manual_seed(103)
    shape = _shape(2, 2)
    roles, attention = _role_masks(
        batch=2,
        claim_len=5,
        evidence_len=6,
    )
    component = 0.25 * torch.randn(
        2,
        2,
        mod.p3a.MAX_LENGTH,
        shape.state_width,
        dtype=torch.float32,
    )
    component = torch.where(
        attention[None, :, :, None],
        component,
        torch.zeros_like(component),
    )
    log_a = (
        -0.02
        - 0.04
        * torch.rand(
            2,
            mod.p3a.MAX_LENGTH,
            shape.intermediate_size,
            shape.state_size,
        )
    )
    survival = pairgap._backward_survival_factor(
        log_a_flat=log_a.reshape(2, mod.p3a.MAX_LENGTH, shape.state_width),
        attention_mask=attention,
    )
    regions, _span = mod._region_pair_gap_interference_fft(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention,
        role_masks=roles,
        shape=shape,
    )
    base, _base_span = pairgap._pair_gap_interference_fft(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention,
        shape=shape,
    )
    reconstructed = sum(regions.values())
    torch.testing.assert_close(
        reconstructed,
        base,
        rtol=2e-5,
        atol=2e-5,
    )


def test_component_region_pair_stats_preserve_base_stats() -> None:
    torch.manual_seed(104)
    shape = _shape(1, 2)
    roles, attention = _role_masks(
        batch=1,
        claim_len=3,
        evidence_len=3,
    )
    component = 0.2 * torch.randn(
        2,
        1,
        mod.p3a.MAX_LENGTH,
        shape.state_width,
        dtype=torch.float32,
    )
    component = torch.where(
        attention[None, :, :, None],
        component,
        torch.zeros_like(component),
    )
    log_a = torch.full(
        (
            1,
            mod.p3a.MAX_LENGTH,
            shape.intermediate_size,
            shape.state_size,
        ),
        -0.04,
        dtype=torch.float32,
    )
    survival = pairgap._backward_survival_factor(
        log_a_flat=log_a.reshape(1, mod.p3a.MAX_LENGTH, shape.state_width),
        attention_mask=attention,
    )
    rows = mod._component_region_pair_stats(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention,
        role_masks=roles,
        shape=shape,
    )
    assert len(rows) == 2
    for row in rows:
        assert set(row["region_pair_C_by_gap"]) == set(mod.REGION_PAIR_NAMES)
        assert row["region_pair_gap_reconstruction_abs_max"] <= 2e-5


def _synthetic_orientation_rows() -> list[dict[str, object]]:
    rows = []
    for group, source, target in mod.all_orientations():
        delta = 2.0 if group == "PRIMARY_A" else 1.0
        region = {}
        for pair_index, pair in enumerate(mod.REGION_PAIR_NAMES, 1):
            region[pair] = {
                "H_windows": {
                    window: delta * pair_index
                    for window in mod.WINDOWS
                }
            }
        rows.append(
            {
                "group": group,
                "source": mod.cell_name(source),
                "target": mod.cell_name(target),
                "log_Q_visible": 0.1 * delta,
                "log_Q_complement": 0.2 * delta,
                "L_interference": 0.1 * delta,
                "region_pair_visible": region,
                "region_pair_complement": region,
            }
        )
    return rows


def test_source_matched_preserves_nine_source_cells() -> None:
    matched = mod._source_matched(_synthetic_orientation_rows())
    assert len(matched) == 9
    for row in matched:
        assert math.isclose(
            row["metrics"]["log_Q_complement"]["primary_minus_control"],
            0.2,
        )
        assert math.isclose(
            row["region_pair_metrics"]["complement"]["CLAIM->EVIDENCE"]
            ["gap1_32"]["primary_minus_control"],
            3.0,
        )


def test_expected_counts_reuse_frozen_forward_budget() -> None:
    counts = mod._expected_counts()
    assert counts["total_source_gradient_forwards"] == 243
    assert counts["total_base_pair_gap_group_decompositions"] == 486
    assert counts["total_serialization_region_group_decompositions"] == 486
    assert counts["total_downstream_stage_transport_calls"] == 0
    assert counts["total_final_head_transport_calls"] == 0


def test_implementation_source_has_no_training_or_downstream_transport() -> None:
    source = inspect.getsource(mod)
    assert ".backward(" not in source
    assert "torch.optim." not in source
    assert "_transport_c_readout" not in source
    assert "_transport_gated_scan" not in source
    assert "_transport_layer22_out_proj" not in source
    assert "_transport_final_head" not in source


def test_frozen_design_and_reference_identities() -> None:
    assert mod.DESIGN_COMMIT == "699b76d66b1204c33b4e01f04302cad52b47aa3c"
    assert mod.DESIGN_BLOB == "7755a7b5ff3f62010a63f5237ce233f5fce154c4"
    assert (
        mod.VALIDATED_EVIDENCE_COMMIT
        == "9f60dacc98240eb9a087974e0996e5763f4d2b9a"
    )
    assert (
        mod.VALIDATED_PAIR_GAP_SCRIPT_BLOB
        == "a37970570ddd659a81523cf2e59e324a854c576e"
    )
    assert mod.REGION_PAIR_NAMES == (
        "CLAIM->CLAIM",
        "CLAIM->EOS",
        "CLAIM->EVIDENCE",
        "EOS->EVIDENCE",
        "EVIDENCE->EVIDENCE",
    )
