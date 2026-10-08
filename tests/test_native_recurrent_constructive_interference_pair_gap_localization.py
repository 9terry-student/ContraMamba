from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import audit_native_recurrent_constructive_interference_pair_gap_localization as mod


def _manual_pair_gap(
    component: torch.Tensor,
    log_a: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    k_count, batch, seq_len, width = component.shape
    p = torch.cumsum(log_a.to(dtype=torch.float64), dim=1)

    survival = torch.zeros(
        (batch, seq_len, width),
        dtype=torch.float64,
        device=component.device,
    )
    carry = torch.zeros(
        (batch, width),
        dtype=torch.float64,
        device=component.device,
    )
    for sigma in range(seq_len - 1, -1, -1):
        if sigma + 1 < seq_len:
            carry = (
                attention_mask[:, sigma, None].to(dtype=torch.float64)
                + torch.exp(2.0 * log_a[:, sigma + 1].to(dtype=torch.float64))
                * carry
            )
        else:
            carry = attention_mask[:, sigma, None].to(dtype=torch.float64)
        survival[:, sigma] = carry

    w = component.to(dtype=torch.float64)
    gap = torch.zeros(
        (k_count, seq_len),
        dtype=torch.float64,
        device=component.device,
    )
    for tau in range(seq_len):
        for sigma in range(tau + 1, seq_len):
            value = (
                2.0
                * w[:, :, tau, :]
                * w[:, :, sigma, :]
                * torch.exp(p[:, sigma, :] - p[:, tau, :]).unsqueeze(0)
                * survival[:, sigma, :].unsqueeze(0)
            )
            gap[:, sigma - tau] += torch.sum(
                value,
                dim=(1, 2),
                dtype=torch.float64,
            )
    return gap



def _observed_pair_gap(
    component: torch.Tensor,
    log_a: torch.Tensor,
    attention_mask: torch.Tensor,
    shape: SimpleNamespace,
):
    batch, seq_len = attention_mask.shape
    survival = mod._backward_survival_factor(
        log_a_flat=log_a.reshape(batch, seq_len, -1),
        attention_mask=attention_mask,
    )
    return mod._pair_gap_interference_fft(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention_mask,
        shape=shape,
    )

def _shape(intermediate: int, state_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        intermediate_size=intermediate,
        state_size=state_size,
        state_width=intermediate * state_size,
    )


def _context_from_log_a(
    log_a_value: float,
    intermediate: int,
    state_size: int,
    batch: int,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    return {
        "a_continuous": torch.full(
            (intermediate, state_size),
            float(log_a_value),
            dtype=torch.float32,
        ),
        "discrete_time_step": torch.ones(
            (batch, intermediate, seq_len),
            dtype=torch.float32,
        ),
    }


def _args(**overrides):
    values = dict(
        static_contract_check=False,
        preflight=False,
        run_worker=False,
        merge_only=False,
        run=False,
        expected_head=None,
        implementation_freeze_commit=None,
        model_snapshot=None,
        tokenizer_snapshot=None,
        checkpoint=None,
        scratch_root=None,
        output_root=None,
        worker_id=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pair_gap_fft_matches_bruteforce():
    torch.manual_seed(7)
    k_count, batch, seq_len = 3, 2, 6
    intermediate, state_size = 2, 2
    width = intermediate * state_size

    component = torch.randn(k_count, batch, seq_len, width) * 0.2
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]],
        dtype=torch.long,
    )
    component[:, 1, 4:] = 0.0
    log_a = -0.05 - 0.25 * torch.rand(
        batch,
        seq_len,
        intermediate,
        state_size,
    )

    observed, _span = _observed_pair_gap(
        component,
        log_a,
        attention_mask,
        _shape(intermediate, state_size),
    )
    expected = _manual_pair_gap(
        component,
        log_a.reshape(batch, seq_len, width),
        attention_mask,
    )

    assert observed[:, 0].tolist() == pytest.approx([0.0] * k_count, abs=0.0)
    torch.testing.assert_close(
        observed,
        expected,
        rtol=2e-5,
        atol=2e-6,
    )


def test_pair_gap_preserves_mixed_positive_and_negative_interference():
    component = torch.tensor(
        [[[[1.0], [-2.0], [3.0], [1.5]]]],
        dtype=torch.float32,
    )
    attention_mask = torch.ones((1, 4), dtype=torch.long)
    log_a = torch.zeros((1, 4, 1, 1), dtype=torch.float32)

    observed, _span = _observed_pair_gap(
        component,
        log_a,
        attention_mask,
        _shape(1, 1),
    )
    expected = _manual_pair_gap(
        component,
        log_a.reshape(1, 4, 1),
        attention_mask,
    )

    torch.testing.assert_close(
        observed,
        expected,
        rtol=2e-5,
        atol=2e-6,
    )
    signed = observed[0, 1:].tolist()
    assert any(value > 0.0 for value in signed)
    assert any(value < 0.0 for value in signed)


def test_pair_gap_indexing_and_pair_accounting_are_exact_for_all_ones():
    seq_len = 4
    component = torch.ones((1, 1, seq_len, 1), dtype=torch.float32)
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    log_a = torch.zeros((1, seq_len, 1, 1), dtype=torch.float32)

    observed, _span = _observed_pair_gap(
        component,
        log_a,
        attention_mask,
        _shape(1, 1),
    )

    # For all-one writes and A=1:
    # C(d) = 2 * sum_{sigma=d..L-1} (L-sigma)
    #      = (L-d) * (L-d+1).
    expected = [0.0, 12.0, 6.0, 2.0]
    assert observed[0].tolist() == pytest.approx(expected, abs=2e-6)
    assert sum(observed[0, 1:].tolist()) == pytest.approx(20.0, abs=2e-6)


def test_padding_tail_does_not_change_valid_pair_gap_result():
    base_component = torch.tensor(
        [[[[1.0], [2.0], [-1.0], [3.0]]]],
        dtype=torch.float32,
    )
    base_mask = torch.ones((1, 4), dtype=torch.long)
    base_log_a = torch.full((1, 4, 1, 1), -0.1, dtype=torch.float32)

    padded_component = torch.cat(
        (
            base_component,
            torch.tensor(
                [[[[100.0], [-200.0], [300.0], [-400.0]]]],
                dtype=torch.float32,
            ),
        ),
        dim=2,
    )
    padded_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    padded_log_a = torch.cat(
        (
            base_log_a,
            torch.full((1, 4, 1, 1), -1000.0, dtype=torch.float32),
        ),
        dim=1,
    )

    base, _ = _observed_pair_gap(
        base_component,
        base_log_a,
        base_mask,
        _shape(1, 1),
    )
    padded, _ = _observed_pair_gap(
        padded_component,
        padded_log_a,
        padded_mask,
        _shape(1, 1),
    )

    assert padded[0, :4].tolist() == pytest.approx(
        base[0].tolist(),
        rel=2e-5,
        abs=2e-6,
    )
    assert padded[0, 4:].tolist() == pytest.approx([0.0] * 4, abs=5e-6)


def test_large_active_prefix_span_uses_stable_path_and_matches_bruteforce():
    torch.manual_seed(13)
    k_count, batch, seq_len = 2, 2, 8
    intermediate, state_size = 2, 2
    width = intermediate * state_size
    component = torch.randn(k_count, batch, seq_len, width) * 0.2
    attention_mask = torch.tensor(
        [[1] * 8, [1, 1, 1, 1, 1, 0, 0, 0]],
        dtype=torch.long,
    )
    component[:, 1, 5:] = 0.0

    log_a = torch.full(
        (batch, seq_len, intermediate, state_size),
        -0.1,
        dtype=torch.float32,
    )
    log_a[:, 2] = -1000.0
    log_a[:, 5] = -1000.0

    observed, span = _observed_pair_gap(
        component,
        log_a,
        attention_mask,
        _shape(intermediate, state_size),
    )
    expected = _manual_pair_gap(
        component,
        log_a.reshape(batch, seq_len, width),
        attention_mask,
    )

    assert span > math.log(torch.finfo(torch.float64).max)
    assert torch.isfinite(observed).all()
    torch.testing.assert_close(
        observed,
        expected,
        rtol=2e-5,
        atol=2e-6,
    )


def test_component_stats_reconstruct_exact_cross_term_and_q():
    torch.manual_seed(17)
    k_count, batch, seq_len = 2, 2, 5
    intermediate, state_size = 2, 2
    width = intermediate * state_size
    component = torch.randn(k_count, batch, seq_len, width) * 0.1
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
        dtype=torch.long,
    )
    component[:, 1, 4] = 0.0
    context = _context_from_log_a(
        -0.2,
        intermediate,
        state_size,
        batch,
        seq_len,
    )
    shape = _shape(intermediate, state_size)
    log_a = mod.kernel._discrete_log_a(context, shape=shape)
    survival = mod._backward_survival_factor(
        log_a_flat=log_a.reshape(batch, seq_len, width),
        attention_mask=attention_mask,
    )

    rows = mod._component_pair_gap_stats(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention_mask,
        shape=shape,
    )

    assert len(rows) == k_count
    for row in rows:
        c_gap = sum(row["pair_gap_C"])
        assert c_gap == pytest.approx(
            row["C"],
            rel=2e-5,
            abs=2e-5,
        )
        assert 1.0 + c_gap / row["S"] == pytest.approx(
            row["Q"],
            rel=2e-5,
            abs=2e-5,
        )


def test_gap_band_summary_uses_fixed_descriptive_bins_without_abs():
    row = mod._gap_band_summary(
        [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0]
    )
    assert row == pytest.approx(
        {
            "gap1": 1.0,
            "gap2_4": -3.0,
            "gap5_8": -2.0,
            "gap9plus": -1.0,
            "total": -5.0,
        }
    )


def test_validated_kernel_reference_is_exactly_36_orientations():
    reference = mod._load_validated_kernel_reference()
    assert len(reference) == 36
    assert all("S_visible" in row for row in reference.values())
    assert all("S_complement" in row for row in reference.values())
    assert all("Q_visible" in row for row in reference.values())
    assert all("Q_complement" in row for row in reference.values())
    assert all("L_interference" in row for row in reference.values())


def test_static_contract_preserves_population_and_shards():
    assert mod.DESIGN_COMMIT == "6e2cbeec8c6a46bb7b3a9b11d8774bea6b30c7a8"
    assert mod.VALIDATED_EVIDENCE_COMMIT == (
        "d64a4f6f60419de529eb239516b82ae62e613b17"
    )
    assert mod.DEV_ROWS == 840
    assert mod.VALID_TOKEN_COUNT == 60094
    assert mod.worker_row_range(0) == (0, 416)
    assert mod.worker_row_range(1) == (416, 840)
    rows = mod.all_orientations()
    assert len(rows) == 36
    assert sum(group == "PRIMARY_A" for group, _, _ in rows) == 18
    assert sum(group == "CONTROL_R" for group, _, _ in rows) == 18


def test_expected_counts_are_recurrence_only():
    counts = mod._expected_counts()
    assert counts["total_source_gradient_forwards"] == 243
    assert counts["total_pair_gap_group_decompositions"] == 486
    assert counts["total_downstream_stage_transport_calls"] == 0
    assert counts["total_final_head_transport_calls"] == 0
    assert counts["workers"]["0"]["source_gradient_forwards"] == 117
    assert counts["workers"]["1"]["source_gradient_forwards"] == 126
    assert counts["workers"]["0"]["pair_gap_group_decompositions"] == 234
    assert counts["workers"]["1"]["pair_gap_group_decompositions"] == 252


def test_pair_gap_scientific_path_is_fft_not_cpu_pair_loop():
    source = inspect.getsource(mod._pair_gap_interference_fft)
    dyadic = inspect.getsource(mod._pair_gap_strict_dyadic_fft)
    assert "torch.fft.rfft" in source
    assert "torch.fft.rfft" in dyadic
    assert "for gap" not in source
    assert "for tau" not in source
    assert "for sigma" not in source
    assert "for tau" not in dyadic
    assert "for sigma" not in dyadic


def test_worker_does_not_execute_downstream_transport_stages():
    source = inspect.getsource(mod.run_worker)
    assert "_stream_transport_group(" not in source
    assert "_stage_final_logits(" not in source
    assert "c_readout" not in source
    assert "gated_scan" not in source
    assert "layer22_out_proj" not in source
    assert "_component_pair_gap_stats(" in source


def test_training_and_backward_method_are_forbidden():
    for fn in (mod.run_preflight, mod.run_worker, mod.run_full):
        source = inspect.getsource(fn)
        assert ".backward(" not in source
        assert "torch.optim" not in source


def test_parser_and_argument_contract():
    parser = mod.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    for option in (
        "--static-contract-check",
        "--preflight",
        "--run-worker",
        "--merge-only",
        "--run",
    ):
        assert option in options

    mod.validate_args(_args(static_contract_check=True))
    mod.validate_args(
        _args(
            preflight=True,
            expected_head="abc",
            implementation_freeze_commit="abc",
            model_snapshot=Path("model"),
            tokenizer_snapshot=Path("tokenizer"),
            checkpoint=Path("checkpoint"),
        )
    )


def test_output_collision_is_fail_closed():
    source = inspect.getsource(mod.run_full)
    assert "SCRATCH_COLLISION" in source
    assert "OUTPUT_COLLISION" in source


def test_worker_sha_sidecar_uses_real_newline():
    digest = "c" * 64
    payload = mod._worker_sha_sidecar_bytes(digest)
    assert payload == (digest + "\n").encode("utf-8")
    assert not payload.endswith(b"\\n")

def test_finalize_orientation_preserves_raw_reconstruction_max_abs():
    accumulator = {
        "group": "PRIMARY_A",
        "source": "synthetic_source",
        "target": "synthetic_target",
        "example_count": mod.DEV_ROWS,
        "raw_reconstruction_max_abs": 1.25e-7,
        "R_visible": 10.0,
        "R_complement": 20.0,
        "S_visible": 4.0,
        "S_complement": 5.0,
        "E_visible": 6.0,
        "E_complement": 10.0,
        "pair_gap_C_visible": [2.0],
        "pair_gap_C_complement": [5.0],
        "pair_gap_C_reconstruction_abs_max": 0.0,
        "pair_gap_Q_reconstruction_abs_max": 0.0,
        "pair_gap_fft_half_log_span_max": 0.5,
    }
    reference = {
        "R_visible": 10.0,
        "R_complement": 20.0,
        "S_visible": 4.0,
        "S_complement": 5.0,
        "E_visible": 6.0,
        "E_complement": 10.0,
        "C_visible": 2.0,
        "C_complement": 5.0,
        "Q_visible": 1.5,
        "Q_complement": 2.0,
        "L_interference": math.log(2.0 / 1.5),
    }

    row = mod._finalize_orientation(accumulator, reference)

    assert row["raw_reconstruction_max_abs"] == pytest.approx(1.25e-7, abs=0.0)
