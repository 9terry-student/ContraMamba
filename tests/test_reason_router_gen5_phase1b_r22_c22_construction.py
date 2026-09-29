from __future__ import annotations

import inspect
from pathlib import Path

import torch

from scripts import reason_router_gen5_phase1b_r22_c22_construction as runner


def _synthetic_lines():
    class Mixer:
        def slow(self, discrete_A, deltaB_u):
            ssm_state = torch.zeros((1, 1536, 16), dtype=torch.float32)
            outputs = []
            for i in range(discrete_A.shape[2]):
                ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
                scan_output = ssm_state.sum()
                outputs.append(scan_output)
            return ssm_state, outputs

    source, first = inspect.getsourcelines(Mixer.slow)
    update = None
    readout = None
    for offset, line in enumerate(source):
        if "ssm_state = discrete_A" in line:
            update = first + offset
        if "scan_output = ssm_state.sum()" in line:
            readout = first + offset
    assert update is not None and readout is not None
    return Mixer, int(update), int(readout)


def test_construction_scope_and_budget_are_frozen() -> None:
    assert runner.DATA_ROOT.as_posix().endswith(
        "reason_router_gen5_phase1b_xg1_construction_v1"
    )
    assert runner.expected_pairs()[0] == "xg1_fact_7801"
    assert runner.expected_pairs()[-1] == "xg1_fact_8100"
    assert len(runner.expected_pairs()) == 300
    assert runner.CONDITIONS == (
        "native",
        "pp3_neutralized",
        "pp5_coefficient_control",
    )
    assert runner.BRANCHES == ("tp", "tm")
    assert runner.FULL_FORWARD_BUDGET == 1800


def test_source_has_no_confirmation_data_roots() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "reason_router_gen5_phase1b_xg1_necessity_confirmation_v1" not in source
    assert "reason_router_gen5_phase1b_xg1_restoration_confirmation_v1" not in source


def test_layer22_observer_captures_write_and_post_state() -> None:
    Mixer, update, readout = _synthetic_lines()
    mixer = Mixer()
    steps = 3
    g = torch.full((1, 1536, steps, 16), 0.5, dtype=torch.float32)
    w = torch.zeros((1, 1536, steps, 16), dtype=torch.float32)
    w[:, :, 1, :] = 2.0

    collector = runner.Layer22NativeWriteCollector(
        code=Mixer.slow.__code__,
        update_line=update,
        readout_line=readout,
        mixer22=mixer,
        target_indices=(1,),
    )
    with collector.capture():
        final, _ = mixer.slow(g, w)
    assert final.shape == (1, 1536, 16)
    assert collector.records is not None
    assert set(collector.records) == {1}
    rec = collector.records[1]
    assert torch.equal(rec.w, w[:, :, 1, :])
    assert torch.equal(rec.s_post, torch.full_like(rec.s_post, 2.0))
    assert rec.reconstruction_relative_residual == 0.0


def test_condition_hook_zero_probe_and_matched_norm() -> None:
    output = torch.randn((1, 4, 3072), dtype=torch.float32)
    mask = torch.zeros(1536, dtype=torch.bool)
    mask[:395] = True

    eye = torch.eye(395, dtype=torch.float64)
    planes = {
        "pp3_plus": eye[:, 0],
        "pp3_minus": eye[:, 1],
        "pp5_plus": eye[:, 2],
        "pp5_minus": eye[:, 3],
    }

    a3 = {}
    a5 = {}
    out3 = runner.condition_hook(
        output,
        token_index=2,
        strong_mask=mask,
        condition="pp3_neutralized",
        planes=planes,
        audit=a3,
    )
    out5 = runner.condition_hook(
        output,
        token_index=2,
        strong_mask=mask,
        condition="pp5_coefficient_control",
        planes=planes,
        audit=a5,
    )

    assert a3["native_pp3_a"] == a5["native_pp3_a"]
    assert a3["native_pp3_b"] == a5["native_pp3_b"]
    assert abs(a3["condition_correction_l2"] - a5["condition_correction_l2"]) <= 1e-12
    assert torch.equal(out3[:, :2, :], output[:, :2, :])
    assert torch.equal(out5[:, 3:, :], output[:, 3:, :])
    assert torch.equal(out3[:, :, 1536:], output[:, :, 1536:])
    assert torch.equal(out5[:, :, 1536:], output[:, :, 1536:])


def test_rank2_basis_sign_and_gap_contract(monkeypatch) -> None:
    monkeypatch.setattr(runner, "PAIR_COUNT", 5)
    matrix = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
        ],
        dtype=torch.float64,
    )
    basis, meta = runner.rank2_basis_from_centered(matrix, label="TEST")
    assert basis.shape == (4, 2)
    assert meta["sigma2_minus_sigma3"] > meta["gap_tolerance"]
    for col in range(2):
        vec = basis[:, col]
        j = int(torch.argmax(torch.abs(vec)).item())
        assert float(vec[j]) >= 0.0
    assert torch.max(torch.abs(basis.T @ basis - torch.eye(2, dtype=torch.float64))) <= 1e-10


def test_basis_serialization_is_vector1_then_vector2(monkeypatch) -> None:
    monkeypatch.setattr(runner, "STATE_WIDTH", 4)
    basis = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    raw = runner.basis_bytes(basis)
    values = torch.frombuffer(bytearray(raw), dtype=torch.float64)
    assert values.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
