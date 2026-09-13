import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

P = (
    Path(__file__).parents[1]
    / "scripts"
    / "longterm_k0_rvg_strong_alignment_causal_falsification.py"
)
spec = importlib.util.spec_from_file_location("fals", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_alignment_exact_target_norms_and_span():
    torch.manual_seed(3)
    x = torch.randn(240, dtype=torch.float64)
    y = torch.randn(240, dtype=torch.float64)
    delta, check = m.alignment_delta(x, y, 0.2)
    y_star = y + delta

    assert abs(check["realized_cosine"] - 0.2) < 1e-12
    assert check["a_residual"] < 1e-12
    assert check["b_residual"] < 1e-12
    assert abs(torch.linalg.vector_norm(y_star).item() - torch.linalg.vector_norm(y).item()) < 1e-12

    # y* remains in span{x,y}; solve least-squares and require tiny residual.
    basis = torch.stack((x, y), dim=1)
    coeff = torch.linalg.lstsq(basis, y_star).solution
    residual = torch.linalg.vector_norm(basis @ coeff - y_star).item()
    assert residual < 1e-10


def test_magnitude_control_preserves_cosine_and_targets_ctrl_norms():
    torch.manual_seed(4)
    x = torch.randn(240, dtype=torch.float64)
    y = torch.randn(240, dtype=torch.float64)
    delta, check = m.magnitude_delta(x, y, 2.0, 3.0)
    assert delta.shape == (240,)
    assert abs(check["realized_A"] - 2.0) < 1e-12
    assert abs(check["realized_B"] - 3.0) < 1e-12
    assert check["cosine_residual"] < 1e-12


def test_intervention_absolute_token_only_strong_x_and_symmetric_midpoint():
    torch.manual_seed(5)
    base_m = torch.randn(1, 9, 3072, dtype=torch.float32)
    base_s = torch.randn(1, 9, 3072, dtype=torch.float32)
    mask = torch.zeros(1536, dtype=torch.bool)
    mask[100:340] = True
    delta = torch.randn(240, dtype=torch.float64)

    ma = {}
    sa = {}
    out_m = m._apply_intervention(
        base_m,
        token_index=6,
        strong_mask=mask,
        delta_h=delta,
        matched=True,
        audit=ma,
    )
    out_s = m._apply_intervention(
        base_s,
        token_index=6,
        strong_mask=mask,
        delta_h=delta,
        matched=False,
        audit=sa,
    )

    assert torch.equal(out_m[:, :, 1536:], base_m[:, :, 1536:])
    assert torch.equal(out_s[:, :, 1536:], base_s[:, :, 1536:])
    assert torch.equal(out_m[:, :6, :], base_m[:, :6, :])
    assert torch.equal(out_s[:, :6, :], base_s[:, :6, :])
    assert torch.equal(out_m[:, 7:, :], base_m[:, 7:, :])
    assert torch.equal(out_s[:, 7:, :], base_s[:, 7:, :])
    assert m.midpoint_residual(ma, sa) <= m.MIDPOINT_TOL


def test_relative_k_is_not_literal_token_two():
    row = {"anchor": 11}
    assert int(row["anchor"]) + m.TARGET_K == 13
    assert int(row["anchor"]) + m.TARGET_K != m.TARGET_K


def test_degenerate_and_out_of_range_alignment_fail_closed():
    x = torch.ones(240, dtype=torch.float64)
    y = torch.ones(240, dtype=torch.float64)
    with pytest.raises(m.FalsificationError):
        m.alignment_delta(x, y, 0.2)

    x = torch.randn(240, dtype=torch.float64)
    y = torch.randn(240, dtype=torch.float64)
    with pytest.raises(m.FalsificationError):
        m.alignment_delta(x, y, 1.01)


def test_forward_budget_consumes_actual_forwards_only():
    b = m.ForwardBudget(16)
    for _ in range(16):
        b.consume()
    assert b.count == 16
    with pytest.raises(m.FalsificationError):
        b.consume()

    b = m.ForwardBudget(2640)
    b.consume(2639)
    b.consume()
    assert b.count == 2640
    with pytest.raises(m.FalsificationError):
        b.consume()


def test_frozen_artifact_parsing_common_330_and_mask_identity():
    root = P.parents[1]
    ctx = m.load_authenticated_parent_stack(root, runtime=False)
    assert len(ctx["common_indices"]) == 330

    mask = m._expected_strong_mask(ctx["geometry_parent_channels"])
    assert mask.numel() == 1536
    assert int(mask.sum().item()) == 240


def test_static_scaffold_blockers_removed_and_output_set_exact():
    source = P.read_text(encoding="utf-8")
    assert "RUNTIME_PREFLIGHT_REQUIRES_INDEPENDENT_RUNTIME_VERIFICATION" not in source
    assert "EXECUTION_REQUIRES_INDEPENDENT_RUNTIME_VERIFICATION" not in source
    assert "budget.consume(8*330)" not in source
    assert {m.ITEM_FILE, m.SUMMARY_FILE, m.MANIFEST_FILE} == {
        "strong_alignment_causal_falsification_item_metrics.jsonl",
        "summary.json",
        "execution_manifest.json",
    }


def test_public_artifact_rejects_raw_tensor_and_logits_field():
    with pytest.raises(m.FalsificationError):
        m._json_bytes({"x": torch.ones(2)})
    with pytest.raises(m.FalsificationError):
        m._json_bytes({"logits": [1.0, 2.0]})
    assert b'"value":1.0' in m._json_bytes({"value": 1.0})


def _fake_row():
    return {
        "S_c0_minus_ctrl": 5.0,
        "S_cA_minus_ctrl": 3.0,
        "S_cM_minus_ctrl": 4.5,
        "W_c0_minus_ctrl": 4.0,
        "W_cA_minus_ctrl": 2.0,
        "W_cM_minus_ctrl": 3.5,
        "C_c0_minus_cA": 0.1,
        "C_c0_minus_cM": 0.05,
        "S_c0_minus_cA": 2.0,
        "S_c0_minus_cM": 0.5,
        "W_c0_minus_cA": 2.0,
        "W_c0_minus_cM": 0.5,
        "C_corr": 0.4,
        "C_ctrl": 0.2,
        "geometry_bridge_max_abs_residual": 0.0,
        "native_bridge_max_abs_residual": 0.0,
        "alignment_A_preservation_abs_residual": 0.0,
        "alignment_B_preservation_abs_residual": 0.0,
        "alignment_realized_cosine": 0.2,
        "alignment_target_cosine": 0.2,
        "alignment_midpoint_max_abs_residual": 0.0,
        "alignment_applied_correction_max_abs_residual": 0.0,
        "magnitude_realized_A": 1.0,
        "magnitude_target_A": 1.0,
        "magnitude_realized_B": 1.0,
        "magnitude_target_B": 1.0,
        "magnitude_cosine_preservation_abs_residual": 0.0,
        "magnitude_midpoint_max_abs_residual": 0.0,
        "magnitude_applied_correction_max_abs_residual": 0.0,
        "alignment_runtime_correction_l2": 1.2,
        "magnitude_runtime_correction_l2": 0.2,
    }


def test_summary_classification_A_and_forward_count():
    rows = [_fake_row() for _ in range(330)]
    summary = m.build_summary(rows, forward_count=2640)
    assert summary["outcome"] == "A"
    assert summary["R_SA"] > summary["R_SM"] > 0
    assert summary["R_WA"] > summary["R_WM"] > 0
    assert summary["all_mandatory_manipulation_checks_pass"] is True


def test_runtime_provenance_is_crlf_safe_and_blob_canonical():
    source = P.read_text(encoding="utf-8")
    assert "HEAD_BYTE_MISMATCH" not in source
    assert "runner_blob_bytes = _git_bytes(root, f\"{head}:{RUNNER_REL}\")" in source
    assert "test_blob_bytes = _git_bytes(root, f\"{head}:{TEST_REL}\")" in source
    assert "info[\"runner_sha256\"] = sha256_bytes(runner_blob_bytes)" in source
    assert "info[\"test_sha256\"] = sha256_bytes(test_blob_bytes)" in source

def test_runtime_capture_uses_direct_backbone_forward_without_nested_trace_parent():
    source = P.read_text(encoding="utf-8")
    start = source.index("def capture_branch(")
    end = source.index("\ndef endpoint_metrics", start)
    capture = source[start:end]
    assert 'runtime["base"].direct_backbone_forward(' in capture
    assert 'ctx["geometry_parent"].run_branch(' not in capture
    assert "with collector.capture():" in capture
    assert 'runtime["layer20"]' in capture
    assert 'runtime["norm22"]' in capture

    start = source.index("def resolve_runtime(")
    end = source.index("\ndef common_plan_pairs", start)
    resolve = source[start:end]
    assert 'deep_ctx = stack["stack"]["ctx"]' in resolve
    assert 'base = deep_ctx["base"]' in resolve
    assert '"DIRECT_BACKBONE_FORWARD_MISSING"' in resolve

def test_direct_capture_smoke_without_nested_parent_trace():
    from contextlib import contextmanager
    from types import SimpleNamespace

    class FakeCollector:
        def __init__(self, _binding, _layer_map, targets):
            self.target = tuple(targets)[0]
            self.records = None

        @contextmanager
        def capture(self):
            z = torch.zeros(1, m.INTERMEDIATE, m.STATE_SIZE, dtype=torch.float32)
            o = torch.ones_like(z)
            self.records = {
                (m.INTERVENTION_LAYER, self.target): SimpleNamespace(
                    s_prev=z.clone(),
                    g=o,
                    w=z.clone(),
                    s_post=z.clone(),
                )
            }
            yield self

    class FakeObserver:
        RawRecurrenceCollector = FakeCollector

    class Mixer20(torch.nn.Module):
        def forward(self, x):
            return 0.25 * x

    class Block20(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = Mixer20()

        def forward(self, x):
            return x + self.mixer(x)

    class Norm22(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(m.HIDDEN))
            self.variance_epsilon = 1e-5

        def forward(self, x):
            scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
            return self.weight * (x * scale)

    class Mixer22(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.intermediate_size = m.INTERMEDIATE
            self.in_proj = torch.nn.Linear(m.HIDDEN, 2 * m.INTERMEDIATE, bias=False)

    class Block22(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = Norm22()
            self.mixer = Mixer22()

    class DummyBlock(torch.nn.Module):
        def forward(self, x):
            return x

    layers = [DummyBlock() for _ in range(m.LAYER_COUNT)]
    layers[m.SOURCE_BLOCK] = Block20()
    layers[m.INTERVENTION_LAYER] = Block22()
    model = SimpleNamespace(mamba=SimpleNamespace(layers=layers))

    class Base:
        @staticmethod
        def direct_backbone_forward(model, token_ids):
            length = len(token_ids)
            x = torch.arange(
                length * m.HIDDEN, dtype=torch.float32
            ).reshape(1, length, m.HIDDEN) / 1000.0 + 0.5
            x = model.mamba.layers[m.SOURCE_BLOCK](x)
            b22 = model.mamba.layers[m.INTERVENTION_LAYER]
            x = b22.norm(x)
            _ = b22.mixer.in_proj(x)
            return x

    strong_mask = torch.zeros(m.INTERMEDIATE, dtype=torch.bool)
    strong_mask[: m.STRONG_COUNT] = True
    runtime = {
        "observer": FakeObserver,
        "observer_binding": object(),
        "layer_map": {},
        "layer20": layers[m.SOURCE_BLOCK],
        "norm22": layers[m.INTERVENTION_LAYER].norm,
        "mixer22": layers[m.INTERVENTION_LAYER].mixer,
        "base": Base,
        "model": model,
        "strong_mask": strong_mask,
    }
    row = {"anchor": 2}
    budget = m.ForwardBudget(1)
    outputs, endpoint, audit = m.capture_branch(
        {}, runtime, row, tuple(range(7)), budget
    )

    assert budget.count == 1
    assert audit is None
    assert outputs[1]["X32"].shape == (m.HIDDEN,)
    assert outputs[3]["R20"].shape == (m.HIDDEN,)
    assert outputs[3]["Y20"].shape == (m.HIDDEN,)
    assert endpoint["S_POST32"].shape == (1, m.INTERMEDIATE, m.STATE_SIZE)

def test_public_negative_provenance_flags_are_allowed_only_when_false():
    payload = {
        "forbidden_action_flags": {
            "logits_read": False,
            "raw_vectors_persisted": False,
        }
    }
    encoded = m._json_bytes(payload)
    assert b'"logits_read":false' in encoded
    assert b'"raw_vectors_persisted":false' in encoded

    with pytest.raises(m.FalsificationError, match="PUBLIC_NEGATIVE_FLAG_NOT_FALSE"):
        m._json_bytes({"logits_read": True})
    with pytest.raises(m.FalsificationError, match="PUBLIC_NEGATIVE_FLAG_NOT_FALSE"):
        m._json_bytes({"raw_vectors_persisted": True})

    with pytest.raises(m.FalsificationError, match="FORBIDDEN_PUBLIC_FIELD"):
        m._json_bytes({"logits_vector": [0.1]})
    with pytest.raises(m.FalsificationError, match="FORBIDDEN_PUBLIC_FIELD"):
        m._json_bytes({"raw_vector_payload": [0.1]})


def test_publish_serializes_before_creating_partial(tmp_path, monkeypatch):
    out = tmp_path / "out"
    rows = [{"x": 1}] * 330
    summary = {"logits_vector": [0.1]}
    manifest = {"schema_version": "x"}

    with pytest.raises(m.FalsificationError, match="FORBIDDEN_PUBLIC_FIELD"):
        m.publish(out, rows, summary, manifest)

    assert not out.exists()
    assert not Path(str(out) + ".partial").exists()
