from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "longterm_k0_rvg_raw_recurrence_observer.py"

spec = importlib.util.spec_from_file_location("k0_rvg_observer", SCRIPT)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_authority_and_scope_constants():
    assert m.AUTHORITY_COMMIT == "d8d717a09516b8562f64f7c503d4bbdcc9c34c5d"
    assert m.OBSERVER_REL == "scripts/longterm_k0_rvg_raw_recurrence_observer.py"
    assert m.TEST_REL == "tests/test_longterm_k0_rvg_raw_recurrence_observer.py"
    assert m.VELOCITY_ATOL == 1e-6
    assert m.VELOCITY_RTOL == 1e-5
    assert m.A0_COMMIT == "55debe94f0d19d16a334395e8561901fed6b52fa"


def test_repo_contract_uses_authority_ancestry_not_exact_head_lock():
    import inspect

    source = inspect.getsource(m._repo_contract)
    assert 'merge-base' in source
    assert '--is-ancestor' in source
    assert 'head == AUTHORITY_COMMIT' not in source
    assert 'IMPLEMENTATION_AUTHORITY_NOT_ANCESTOR' in source


def test_real_frozen_source_binding():
    binding = m.resolve_source_binding()
    assert binding.source_sha256 == m.EXPECTED_MAMBA_SOURCE_SHA256
    assert binding.slow_forward_line == 270
    assert binding.discrete_a_line == 322
    assert binding.deltab_u_line == 324
    assert binding.loop_line == 349
    assert binding.update_line == 350
    assert binding.readout_line == 351
    assert binding.forward_line == 366




def _valid_source(two_loops: bool = False) -> bytes:
    second = ""
    if two_loops:
        second = """
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state, C[:, i, :].unsqueeze(-1))
"""
    source = f"""
class MambaMixer:
    def slow_forward(self, input_states):
        discrete_A = torch.exp(A * discrete_time_step)
        deltaB_u = discrete_B * hidden_states
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state, C[:, i, :].unsqueeze(-1))
{second}
        return scan_output

    def forward(self, hidden_states):
        return self.slow_forward(hidden_states)
"""
    return source.encode("utf-8")


def test_analyzer_accepts_unique_structural_fixture():
    result = m.analyze_mamba_source(_valid_source())
    assert result["update_line"] + 1 == result["readout_line"]
    assert result["discrete_a_line"] < result["update_line"]
    assert result["deltab_u_line"] < result["update_line"]


def test_analyzer_rejects_ambiguous_recurrence():
    with pytest.raises(m.ContractError, match="MAMBA_SEQUENTIAL_RECURRENCE_AMBIGUOUS"):
        m.analyze_mamba_source(_valid_source(two_loops=True))


def test_analyzer_rejects_non_python():
    with pytest.raises(m.ContractError, match="MAMBA_SOURCE_PARSE_FAILURE"):
        m.analyze_mamba_source(b"\xff")


def test_analyzer_rejects_missing_mixer():
    with pytest.raises(m.ContractError, match="MAMBA_MIXER_CLASS_AMBIGUOUS"):
        m.analyze_mamba_source(b"def x():\n    pass\n")


def test_analyzer_rejects_ambiguous_mixer():
    raw = b"class MambaMixer:\n    pass\n\nclass MambaMixer:\n    pass\n"
    with pytest.raises(m.ContractError, match="MAMBA_MIXER_CLASS_AMBIGUOUS"):
        m.analyze_mamba_source(raw)


def test_analyzer_rejects_ambiguous_recurrence():
    raw = b"""
class MambaMixer:
    def slow_forward(self):
        discrete_A = A + discrete_time_step
        deltaB_u = discrete_B * hidden_states
        for i in range(2):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state, C[:, i, :])
        for i in range(2):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state, C[:, i, :])
    def forward(self):
        return self.slow_forward()
"""
    with pytest.raises(m.ContractError, match="MAMBA_SEQUENTIAL_RECURRENCE_AMBIGUOUS"):
        m.analyze_mamba_source(raw)


def _record(s_prev, g, w, s_post):
    meta = m.TensorMeta(tuple(s_prev.shape), str(s_prev.dtype), str(s_prev.device))
    return m.RecurrenceRecord(
        layer_index=0,
        token_index=0,
        s_prev=s_prev,
        g=g,
        w=w,
        s_post=s_post,
        s_prev_meta=meta,
        g_meta=m.TensorMeta(tuple(g.shape), str(g.dtype), str(g.device)),
        w_meta=m.TensorMeta(tuple(w.shape), str(w.dtype), str(w.device)),
        s_post_meta=m.TensorMeta(tuple(s_post.shape), str(s_post.dtype), str(s_post.device)),
    )


def test_exact_recurrence_and_velocity_validation():
    s_prev = torch.tensor([[[1.0, -2.0], [0.25, 4.0]]], dtype=torch.float32)
    g = torch.tensor([[[0.5, 0.75], [1.0, 0.25]]], dtype=torch.float32)
    w = torch.tensor([[[0.25, 0.5], [-0.5, 1.0]]], dtype=torch.float32)
    s_post = g * s_prev + w
    result = m.validate_recurrence_record(_record(s_prev, g, w, s_post))
    assert result["recurrence_exact"] == "PASS_EXACT"
    assert result["velocity_rearrangement"] == "PASS_TOLERANCE"
    assert result["velocity_rearrangement_allclose"] is True


def test_recurrence_mismatch_fails_closed():
    s_prev = torch.ones((1, 2, 2), dtype=torch.float32)
    g = torch.full((1, 2, 2), 0.5, dtype=torch.float32)
    w = torch.zeros((1, 2, 2), dtype=torch.float32)
    s_post = g * s_prev + w
    bad = s_post.clone()
    bad[0, 0, 0] += 1e-3
    with pytest.raises(m.ContractError, match="RECURRENCE_EXACT_RECONSTRUCTION_FAILURE"):
        m.validate_recurrence_record(_record(s_prev, g, w, bad))


def test_snapshot_is_clone_and_cpu():
    source = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    snapshot, meta = m._snapshot_tensor(source, "TEST")
    assert snapshot.device.type == "cpu"
    assert meta.shape == (1, 2, 3)
    assert snapshot.data_ptr() != source.data_ptr()
    assert torch.equal(snapshot, source)


def test_snapshot_rejects_nonfinite():
    source = torch.tensor([[[float("nan")]]], dtype=torch.float32)
    with pytest.raises(m.ContractError, match="TEST_NONFINITE"):
        m._snapshot_tensor(source, "TEST")


def test_metadata_requires_float32_cpu_and_equal_shapes():
    x = torch.ones((1, 2, 2), dtype=torch.float32)
    rec = _record(x, x, x, x)
    m._validate_record_metadata(rec)

    bad_meta = m.TensorMeta((1, 2, 2), "torch.float64", "cpu")
    bad = m.RecurrenceRecord(
        layer_index=0,
        token_index=0,
        s_prev=x,
        g=x,
        w=x,
        s_post=x,
        s_prev_meta=bad_meta,
        g_meta=rec.g_meta,
        w_meta=rec.w_meta,
        s_post_meta=rec.s_post_meta,
    )
    with pytest.raises(m.ContractError, match="RECURRENCE_DTYPE_MISMATCH"):
        m._validate_record_metadata(bad)


def test_metadata_rejects_wrong_expected_mixer_shape():
    x = torch.ones((1, 2, 2), dtype=torch.float32)
    rec = _record(x, x, x, x)
    with pytest.raises(m.ContractError, match="RECURRENCE_EXPECTED_SHAPE_MISMATCH"):
        m._validate_record_metadata(rec, expected_shape=(1, 3, 2))


def test_collector_reuse_is_rejected_without_model_forward():
    def dummy():
        return None

    binding = m.SourceBinding(
        code=dummy.__code__,
        source_path=Path(__file__),
        source_sha256="0" * 64,
        source_bytes=0,
        slow_forward_line=1,
        discrete_a_line=1,
        deltab_u_line=1,
        loop_line=1,
        update_line=10_000,
        readout_line=10_001,
        forward_line=1,
    )
    layers = {i: i for i in range(m.N_LAYERS)}
    collector = m.RawRecurrenceCollector(binding, layers, [0])
    with collector.capture():
        pass
    with pytest.raises(m.ContractError, match="TRACE_COLLECTOR_REUSE"):
        with collector.capture():
            pass


def test_parser_exposes_no_scientific_execution_flag():
    parser = m.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--synthetic-preflight" in options
    assert "--seed180-handoff" in options
    assert "--scientific" not in options
    assert "--population" not in options
    assert "--evaluate" not in options
    assert "--intervene" not in options


def test_observer_source_has_no_scientific_population_loader_call():
    tree = ast.parse(SCRIPT.read_text("utf-8"))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert "load_frozen_population" not in called
    assert "build_input_contracts" not in called
    assert "execute_scientific" not in called


def test_historical_k2s_helper_identity():
    helper = ROOT / m.K2S_REL
    assert m.file_sha256(helper) == m.K2S_SHA256


def test_record_hashes_are_role_complete():
    x = torch.ones((1, 2, 2), dtype=torch.float32)
    rec = _record(x, x, x, x)
    hashes = m.record_hashes(rec)
    assert set(hashes) == {"S_prev", "G", "W", "S_post"}
    assert all(len(v) == 64 for v in hashes.values())


def test_exact_recurrence_rearrangement_exceedance_is_diagnostic_only():
    s_prev = torch.zeros((1, 2, 2), dtype=torch.float32)
    g = torch.ones((1, 2, 2), dtype=torch.float32)
    w = torch.zeros((1, 2, 2), dtype=torch.float32)
    s_prev[0, 0, 0] = 1540996.125
    g[0, 0, 0] = 0.9999
    w[0, 0, 0] = -71.9258
    s_post = g * s_prev + w

    result = m.validate_recurrence_record(_record(s_prev, g, w, s_post))

    assert result["recurrence_exact"] == "PASS_EXACT"
    assert result["velocity_rearrangement"] == "DIAGNOSTIC_TOLERANCE_EXCEEDED"
    assert result["velocity_rearrangement_allclose"] is False
    assert result["velocity_atol"] == 1e-6
    assert result["velocity_rtol"] == 1e-5
    assert result["max_scaled_tolerance_residual"] > 1.0


def test_r2_scope_includes_exact_four_tracked_files():
    assert m.R2_IMPLEMENTATION_FILES == {
        "scripts/longterm_k0_rvg_raw_recurrence_observer.py",
        "tests/test_longterm_k0_rvg_raw_recurrence_observer.py",
        "scripts/longterm_k0_rvg_p1_raw_vector_execution.py",
        "tests/test_longterm_k0_rvg_p1_raw_vector_execution.py",
    }
