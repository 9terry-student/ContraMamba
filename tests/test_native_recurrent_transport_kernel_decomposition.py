from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import audit_native_recurrent_transport_kernel_decomposition as mod


def _args(**overrides):
    base = dict(
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
    base.update(overrides)
    return argparse.Namespace(**base)


def _manual_component(component, log_a, mask):
    k_count, batch, seq_len, _, _ = component.shape
    raw = [0.0] * k_count
    e = [0.0] * k_count
    s = [0.0] * k_count
    lag = [[0.0 for _ in range(seq_len)] for _ in range(k_count)]

    for k in range(k_count):
        for b in range(batch):
            state = torch.zeros_like(component[k, b, 0])
            contributions = []
            for t in range(seq_len):
                a = torch.exp(log_a[b, t])
                contributions = [a * old for old in contributions]
                contributions.append(component[k, b, t])
                state = a * state + component[k, b, t]
                if int(mask[b, t]) != 0:
                    raw[k] += float(torch.sum(component[k, b, t] ** 2))
                    e[k] += float(torch.sum(state ** 2))
                    s[k] += sum(float(torch.sum(x ** 2)) for x in contributions)
                    for tau, value in enumerate(contributions):
                        lag_value = t - tau
                        lag[k][lag_value] += float(torch.sum(value ** 2))
    return raw, e, s, lag


def test_exact_recurrence_self_and_lag_decomposition_matches_bruteforce():
    torch.manual_seed(4)
    k_count, batch, seq_len, intermediate, state_size = 2, 2, 4, 2, 2
    width = intermediate * state_size
    component = torch.randn(k_count, batch, seq_len, width) * 0.2
    mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0]],
        dtype=torch.long,
    )
    component[:, 1, 3] = 0.0

    a_cont = torch.tensor(
        [[-0.25, -0.15], [-0.35, -0.20]],
        dtype=torch.float32,
    )
    dt = torch.tensor(
        [
            [[0.8, 1.0, 0.7, 1.1], [0.9, 0.6, 1.2, 0.8]],
            [[1.1, 0.7, 0.9, 1.0], [0.5, 1.3, 0.8, 1.1]],
        ],
        dtype=torch.float32,
    )
    context = {
        "a_continuous": a_cont,
        "discrete_time_step": dt,
    }
    shape = SimpleNamespace(
        intermediate_size=intermediate,
        state_size=state_size,
        state_width=width,
    )

    observed = mod._component_recurrence_stats(
        component=component,
        context=context,
        attention_mask=mask,
        shape=shape,
    )
    log_a = mod._discrete_log_a(context, shape=shape)
    manual_raw, manual_e, manual_s, manual_lag = _manual_component(
        component.reshape(k_count, batch, seq_len, intermediate, state_size),
        log_a,
        mask,
    )

    for k in range(k_count):
        assert observed[k]["R"] == pytest.approx(manual_raw[k], rel=2e-6, abs=1e-7)
        assert observed[k]["E"] == pytest.approx(manual_e[k], rel=2e-6, abs=1e-7)
        assert observed[k]["S"] == pytest.approx(manual_s[k], rel=2e-6, abs=1e-7)
        assert observed[k]["C"] == pytest.approx(
            manual_e[k] - manual_s[k],
            rel=2e-6,
            abs=1e-7,
        )
        assert observed[k]["lag"] == pytest.approx(
            manual_lag[k],
            rel=2e-5,
            abs=2e-6,
        )
        assert sum(observed[k]["lag"]) == pytest.approx(
            observed[k]["S"],
            rel=2e-5,
            abs=2e-6,
        )


def test_log_selective_retention_splits_exactly_into_kernel_and_interference():
    acc = mod._empty_orientation(
        "PRIMARY_A",
        (6201, 6201),
        (6202, 6201),
    )
    acc["example_count"] = mod.DEV_ROWS
    acc["R_visible"] = 2.0
    acc["R_complement"] = 8.0
    acc["S_visible"] = 6.0
    acc["S_complement"] = 12.0
    acc["E_visible"] = 3.0
    acc["E_complement"] = 3.0
    acc["J1_visible_numer"] = 2.0
    acc["J1_visible_denom"] = 2.0
    acc["J1_complement_numer"] = 4.0
    acc["J1_complement_denom"] = 8.0
    acc["lag_visible"] = [2.0, 2.0, 2.0]
    acc["lag_complement"] = [8.0, 3.0, 1.0]

    expected_log = math.log((3.0 / 8.0) / (3.0 / 2.0))
    reference = {
        "R_visible": 2.0,
        "R_complement": 8.0,
        "E_visible": 3.0,
        "E_complement": 3.0,
        "LOG_SELECTIVE_RETENTION": expected_log,
    }
    row = mod._finalize_orientation(acc, reference)
    assert row["LOG_SELECTIVE_RETENTION"] == pytest.approx(
        row["L_kernel"] + row["L_interference"],
        abs=1e-12,
    )
    assert row["L_kernel"] == pytest.approx(math.log(0.5))
    assert row["L_interference"] == pytest.approx(math.log(0.5))
    assert row["LOG_SELECTIVE_RETENTION"] == pytest.approx(math.log(0.25))


def test_lag_summary_reports_exact_bins_mean_and_quantiles():
    row = mod._lag_summary([4.0, 2.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    assert row["lag0_fraction"] == pytest.approx(0.2)
    assert row["lag1_fraction"] == pytest.approx(0.1)
    assert row["lag2_4_fraction"] == pytest.approx(0.2)
    assert row["lag5_8_fraction"] == pytest.approx(0.0)
    assert row["lag9plus_fraction"] == pytest.approx(0.5)
    assert row["lag50"] == 4
    assert row["lag90"] == 9
    assert row["mean_lag"] == pytest.approx(
        (1 * 2 + 2 * 1 + 3 * 1 + 4 * 2 + 9 * 10) / 20
    )


def test_validated_reference_is_exactly_36_orientations_and_two_stages():
    ref = mod._load_validated_reference()
    assert len(ref) == 36
    expected = {
        "R_visible",
        "R_complement",
        "E_visible",
        "E_complement",
        "LOG_SELECTIVE_RETENTION",
    }
    assert all(set(row) == expected for row in ref.values())


def test_static_contract_keeps_original_population_and_shards():
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
    assert counts["total_recurrence_group_decompositions"] == 486
    assert counts["total_downstream_stage_transport_calls"] == 0
    assert counts["total_final_head_transport_calls"] == 0
    assert counts["workers"]["0"]["source_gradient_forwards"] == 117
    assert counts["workers"]["1"]["source_gradient_forwards"] == 126
    assert counts["workers"]["0"]["recurrence_group_decompositions"] == 234
    assert counts["workers"]["1"]["recurrence_group_decompositions"] == 252


def test_worker_does_not_execute_downstream_transport_stages():
    source = inspect.getsource(mod.run_worker)
    assert "_stream_transport_group(" not in source
    assert "_stage_final_logits(" not in source
    assert "c_readout" not in source
    assert "gated_scan" not in source
    assert "layer22_out_proj" not in source
    assert "_component_recurrence_stats(" in source


def test_training_and_backward_method_are_forbidden():
    for fn in (mod.run_preflight, mod.run_worker, mod.run_full):
        source = inspect.getsource(fn)
        assert ".backward(" not in source
        assert "torch.optim" not in source


def test_historical_projector_scalar_is_not_reintroduced_as_gate():
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "RAW_WRITE_PROJECTOR_TARGET" not in source
    assert "HISTORICAL_OUTCOME_CONTEXT_ONLY" not in source
    assert "validated_transport_replay" in source


def test_worker_sha_sidecar_uses_real_newline():
    digest = "b" * 64
    payload = mod._worker_sha_sidecar_bytes(digest)
    assert payload == (digest + "\n").encode("utf-8")
    assert not payload.endswith(b"\\n")


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
