from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "longterm_k0_rvg_p1_raw_vector_execution.py"

spec = importlib.util.spec_from_file_location("k0_rvg_p1", SCRIPT)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_authority_and_scope_constants():
    assert m.AUTHORITY_COMMIT == "c2b1990b649701cbf5ec71a360f03b4b7ff27465"
    assert m.RUNNER_REL == "scripts/longterm_k0_rvg_p1_raw_vector_execution.py"
    assert m.TEST_REL == "tests/test_longterm_k0_rvg_p1_raw_vector_execution.py"
    assert m.PRIMARY_LAYER == 23
    assert m.WINDOW == 8
    assert m.TARGET_COUNT == 9
    assert m.SUPPORT_MIN_BLOCKS == 160


def test_p0_six_file_sha_binding():
    assert m.P0_ARTIFACT_SHA256 == {
        "candidate_pool.jsonl": "743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246",
        "generated_source.jsonl": "8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf",
        "phase_pair_mapping.json": "c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda",
        "token_contracts.jsonl": "6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998",
        "provisioning_manifest.json": "feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52",
        "validation_report_candidate.md": "ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a",
    }


def test_observer_and_k2s_binding_constants():
    assert m.OBSERVER_SHA256 == "12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25"
    assert m.OBSERVER_GIT_BLOB == "f2dbdfe52661eca384897578ab272e602e36deac"
    assert m.K2S_SHA256 == "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
    assert m.K2S_GIT_BLOB == "3a651fb508669bdcf72441a4869b863d6eee6c1f"
    assert m.A0_COMMIT == "55debe94f0d19d16a334395e8561901fed6b52fa"
    assert m.A0_MODEL_BLOB_SHA == "f0ddc0eda64937de6fcd27943e30a296082c01d5"
    assert m.A0_HEADS_TREE_SHA == "68d26855aa511fcd41d6f395ae5f87177a162678"


def test_real_code_dependency_provenance_includes_a0():
    deps = m.authenticate_code_dependencies(ROOT)
    assert deps["a0_commit"] == m.A0_COMMIT
    assert deps["a0_model_blob_sha"] == m.A0_MODEL_BLOB_SHA
    assert deps["a0_heads_tree_sha"] == m.A0_HEADS_TREE_SHA


def test_handoff_checkpoint_encoder_constants():
    assert m.EXPECTED_ZIP_SHA256 == "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
    assert m.EXPECTED_CHECKPOINT_SHA256 == "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
    assert m.COMMON_ENCODER_CANONICAL_SHA256 == "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
    assert m.COMMON_ENCODER_RAW_CONCAT_SHA256 == "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"


def test_authenticate_p0_archive_real():
    p0 = m.authenticate_p0_archive(ROOT)
    assert len(p0.candidates) == 336
    assert len(p0.mapping) == 168
    assert len(p0.token_contracts) == 336
    assert p0.candidates[0]["pair_id"] == "generated_fact_1237"
    assert p0.candidates[-1]["pair_id"] == "generated_fact_1572"


def test_branch_reconstruction_exact():
    candidates = [
        {"prefix_text": "P0|", "correction_text": "C0", "control_text": "K0"},
        *[
            {"prefix_text": f"P{i}|", "correction_text": f"C{i}", "control_text": f"K{i}"}
            for i in range(1, 168)
        ],
        {"prefix_text": "P168|", "correction_text": "C168", "control_text": "K168"},
    ]
    # Fill to 336.
    candidates.extend(
        {"prefix_text": f"P{i}|", "correction_text": f"C{i}", "control_text": f"K{i}"}
        for i in range(169, 336)
    )
    b = m.reconstruct_branch_texts(candidates, 0)
    assert b.matched_corr == "P0|C0"
    assert b.matched_ctrl == "P0|K0"
    assert b.swapped_corr == "P0|C168"
    assert b.swapped_ctrl == "P0|K168"


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        return {"input_ids": [ord(ch) for ch in text]}


def test_token_contract_revalidation_fixture():
    tok = FakeTokenizer()
    texts = m.BranchTexts(
        prefix="ABC",
        matched_corr="ABCx123456789",
        matched_ctrl="ABCy123456789",
        swapped_corr="ABCm123456789",
        swapped_ctrl="ABCn123456789",
    )
    prefix = [ord(c) for c in "ABC"]
    archived = {
        "prefix_token_count": 3,
        "prefix_token_sha256": m.token_id_sha256(prefix),
        "matched_correction_token_count": len(texts.matched_corr),
        "matched_control_token_count": len(texts.matched_ctrl),
        "matched_divergence_anchor": 3,
        "matched_divergence_offset_from_prefix": 0,
        "matched_w8_available": True,
        "swapped_correction_token_count": len(texts.swapped_corr),
        "swapped_control_token_count": len(texts.swapped_ctrl),
        "swapped_divergence_anchor": 3,
        "swapped_divergence_offset_from_prefix": 0,
        "swapped_w8_available": True,
    }
    got = m.revalidate_token_contract(tok, texts, archived)
    assert got.matched_te == 3
    assert got.swapped_te == 3


def test_token_contract_mismatch_fails():
    tok = FakeTokenizer()
    texts = m.BranchTexts(
        prefix="ABC",
        matched_corr="ABCx123456789",
        matched_ctrl="ABCy123456789",
        swapped_corr="ABCm123456789",
        swapped_ctrl="ABCn123456789",
    )
    archived = {
        "prefix_token_count": 3,
        "prefix_token_sha256": "0" * 64,
        "matched_correction_token_count": len(texts.matched_corr),
        "matched_control_token_count": len(texts.matched_ctrl),
        "matched_divergence_anchor": 3,
        "matched_divergence_offset_from_prefix": 0,
        "matched_w8_available": True,
        "swapped_correction_token_count": len(texts.swapped_corr),
        "swapped_control_token_count": len(texts.swapped_ctrl),
        "swapped_divergence_anchor": 3,
        "swapped_divergence_offset_from_prefix": 0,
        "swapped_w8_available": True,
    }
    with pytest.raises(m.ContractError, match="PREFIX_TOKEN_SHA_MISMATCH"):
        m.revalidate_token_contract(tok, texts, archived)


def test_target_indices_exact_nine():
    assert m.target_indices(10) == (9, 10, 11, 12, 13, 14, 15, 16, 17)
    with pytest.raises(m.ContractError):
        m.target_indices(0)


def test_frobenius_dot_norm_cosine_float64():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    y = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    assert m.frobenius_dot(x, y) == 11.0
    assert math.isclose(m.frobenius_norm(x), math.sqrt(5.0))
    expected = 11.0 / math.sqrt(5.0 * 25.0)
    assert math.isclose(m.frobenius_cosine(x, y), expected)


def test_zero_vector_cosine_undefined():
    torch = pytest.importorskip("torch")
    z = torch.zeros((1, 2), dtype=torch.float32)
    x = torch.ones((1, 2), dtype=torch.float32)
    assert m.frobenius_cosine(z, x) is None
    assert m.frobenius_cosine(x, z) is None


class Rec:
    def __init__(self, s_prev, s_post, g=None, w=None):
        self.s_prev = s_prev
        self.s_post = s_post
        self.g = g if g is not None else s_prev * 0 + 1
        self.w = w if w is not None else s_post - s_prev


def _pair_records(turn_scale_corr=1.0, turn_scale_ctrl=1.0):
    torch = pytest.importorskip("torch")
    records_corr = {}
    records_ctrl = {}
    # incoming velocity [1,0]
    prev = torch.tensor([[[0.0, 0.0]]])
    incoming_post = torch.tensor([[[1.0, 0.0]]])
    records_corr[0] = Rec(prev, incoming_post)
    records_ctrl[0] = Rec(prev.clone(), incoming_post.clone())
    for t in range(1, 9):
        sp = torch.tensor([[[float(t), 0.0]]])
        corr_v = torch.tensor([[[turn_scale_corr, 1.0]]])
        ctrl_v = torch.tensor([[[turn_scale_ctrl, 0.0]]])
        records_corr[t] = Rec(sp, sp + corr_v)
        records_ctrl[t] = Rec(sp.clone(), sp + ctrl_v)
    return records_corr, records_ctrl


def test_common_incoming_state_exact():
    corr, ctrl = _pair_records()
    assert m.common_incoming_state(corr[0], ctrl[0]) == 4
    ctrl[0].w = ctrl[0].w + 1
    with pytest.raises(m.ContractError, match="INCOMING_COMMON_STATE_MISMATCH"):
        m.common_incoming_state(corr[0], ctrl[0])


def test_turning_and_coherence_formula():
    corr, ctrl = _pair_records()
    metrics = m.compute_pair_metrics(corr, ctrl, 1)
    assert metrics.valid_turning
    assert metrics.valid_coherence
    assert metrics.turning is not None
    assert metrics.turning > 0.0
    assert metrics.coherence is not None
    assert math.isclose(metrics.coherence, 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_item_endpoint_formula():
    a = m.PairMetrics(True, True, 0.0, 1.0, 1.0, 0.8, {})
    b = m.PairMetrics(True, True, 0.0, 0.5, 0.5, 0.3, {})
    row = m.item_endpoint_metrics(a, b)
    assert row["X_turn"] == 0.5
    assert row["X_coh"] == 0.5


def test_carry_write_diagnostics_algebraic_fixture():
    corr, ctrl = _pair_records()
    metrics = m.compute_pair_metrics(corr, ctrl, 1)
    diag = metrics.carry_write_diagnostics
    assert math.isclose(diag["response_norm_mean"], 1.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(diag["carry_response_norm_mean"], 0.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(diag["write_response_norm_mean"], 1.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(diag["write_total_cosine_mean"], 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert diag["carry_total_cosine_mean"] is None
    assert diag["write_carry_cosine_mean"] is None


def test_sign_classification_exact_zero():
    assert m.sign_classification(1e-300) == "positive"
    assert m.sign_classification(-1e-300) == "negative"
    assert m.sign_classification(0.0) == "zero"
    assert m.sign_classification(None) is None


def test_exact_sign_test_known_cases():
    assert m.exact_two_sided_sign_test(10, 0) == 2 / 1024
    assert m.exact_two_sided_sign_test(5, 5) == 1.0
    assert m.exact_two_sided_sign_test(0, 0) is None


def test_holm_m2():
    got = m.holm_m2({"turning": 0.01, "response_coherence": 0.04})
    assert got == {"turning": 0.02, "response_coherence": 0.04}
    got2 = m.holm_m2({"turning": 0.03, "response_coherence": 0.01})
    assert got2 == {"response_coherence": 0.02, "turning": 0.03}


def test_holm_m2_preserves_other_endpoint_when_one_missing():
    got = m.holm_m2({"turning": 0.01, "response_coherence": None})
    assert got["turning"] == 0.02
    assert got["response_coherence"] is None


def test_support_threshold():
    values = [1.0] * 159 + [None] * 9
    summary = m.endpoint_summary_one("turning", values, None)
    assert summary["endpoint_verdict"] == "NOT_EVALUABLE_DUE_TO_VECTOR_NORM_SUPPORT_FAILURE"
    assert summary["valid_block_count"] == 159


def test_endpoint_verdict_positive():
    values = [1.0] * 168
    raw = m.exact_two_sided_sign_test(168, 0)
    adjusted = m.holm_m2({"turning": raw, "response_coherence": raw})
    summary = m.endpoint_summary_one("turning", values, adjusted["turning"])
    assert summary["endpoint_verdict"] == "TURNING_POSITIVE_DIRECTIONAL_SIGNAL"
    assert summary["sign_effect"] == 1.0


def test_overall_verdict_mapping():
    def e(v): return {"endpoint_verdict": v}
    assert m.overall_verdict(
        e("TURNING_POSITIVE_DIRECTIONAL_SIGNAL"),
        e("RESPONSE_COHERENCE_POSITIVE_DIRECTIONAL_SIGNAL"),
    ) == "RAW_NATIVE_VECTOR_ORGANIZATION_CONVERGENT"
    assert m.overall_verdict(
        e("TURNING_POSITIVE_DIRECTIONAL_SIGNAL"),
        e("RESPONSE_COHERENCE_DIRECTIONAL_SIGNAL_NOT_ESTABLISHED"),
    ) == "RAW_NATIVE_VECTOR_ORGANIZATION_ENDPOINT_SPECIFIC"
    assert m.overall_verdict(
        e("TURNING_DIRECTIONAL_SIGNAL_NOT_ESTABLISHED"),
        e("RESPONSE_COHERENCE_DIRECTIONAL_SIGNAL_NOT_ESTABLISHED"),
    ) == "RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED"
    assert m.overall_verdict(
        e("TURNING_POSITIVE_DIRECTIONAL_SIGNAL"),
        e("RESPONSE_COHERENCE_REVERSED_DIRECTIONAL_SIGNAL"),
    ) == "RAW_NATIVE_VECTOR_ORGANIZATION_MIXED_NOT_PROMOTABLE"


def test_block_aggregation():
    items = []
    for i in range(336):
        items.append({
            "local_template_index": i,
            "stable_item_id": f"s{i}",
            "turning_valid": True,
            "coherence_valid": True,
            "X_turn": float(i),
            "X_coh": float(-i),
        })
    mapping = [
        {
            "block_index": p,
            "phase_class": p,
            "item_a_stable_id": f"s{p}",
            "item_b_stable_id": f"s{p+168}",
        }
        for p in range(168)
    ]
    blocks = m.aggregate_blocks(items, mapping)
    assert len(blocks) == 168
    assert blocks[0]["B_turn"] == 84.0
    assert blocks[0]["B_coh"] == -84.0


def test_authority_marker_parser():
    text = """
`SCIENTIFIC_EXECUTION_AUTHORIZED = YES`
P1_IMPLEMENTATION_COMMIT = 0123456789012345678901234567890123456789
"""
    markers = m.authority_markers(text)
    assert markers["SCIENTIFIC_EXECUTION_AUTHORIZED"] == "YES"
    assert markers["P1_IMPLEMENTATION_COMMIT"] == "0123456789012345678901234567890123456789"


def test_validate_p1_implementation_commit_exact_parent_and_scope(monkeypatch):
    implementation = "1" * 40

    def fake_git(root, *args):
        if args == ("rev-parse", f"{implementation}^"):
            return m.AUTHORITY_COMMIT
        if args == (
            "diff-tree", "--no-commit-id", "--name-only", "-r", implementation
        ):
            return m.RUNNER_REL + "\n" + m.TEST_REL
        if args == ("rev-parse", f"{implementation}:{m.RUNNER_REL}"):
            return "2" * 40
        if args == ("rev-parse", f"{implementation}:{m.TEST_REL}"):
            return "3" * 40
        raise AssertionError(args)

    monkeypatch.setattr(m, "_git", fake_git)
    got = m.validate_p1_implementation_commit(ROOT, implementation)
    assert got == {"runner_blob": "2" * 40, "test_blob": "3" * 40}


def test_validate_p1_implementation_commit_rejects_wrong_parent(monkeypatch):
    implementation = "1" * 40

    def fake_git(root, *args):
        if args == ("rev-parse", f"{implementation}^"):
            return "0" * 40
        raise AssertionError(args)

    monkeypatch.setattr(m, "_git", fake_git)
    with pytest.raises(m.ContractError, match="P1_IMPLEMENTATION_PARENT_MISMATCH"):
        m.validate_p1_implementation_commit(ROOT, implementation)


def test_execution_authority_absent_rejected(tmp_path):
    with pytest.raises(m.ContractError, match="EXECUTION_AUTHORITY_NOT_TRACKED"):
        m.authenticate_execution_authority(ROOT, "reports/does_not_exist.md")


def test_execution_authority_absolute_path_rejected():
    with pytest.raises(m.ContractError, match="EXECUTION_AUTHORITY_PATH_ABSOLUTE"):
        m.authenticate_execution_authority(ROOT, str((ROOT / "reports/x.md").resolve()))


def test_execution_authority_tracked_marker_mismatch_rejected(monkeypatch, tmp_path):
    report = tmp_path / "reports" / "authority.md"
    report.parent.mkdir(parents=True)
    raw = (
        "SCIENTIFIC_EXECUTION_AUTHORIZED = NO\n"
        "SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = YES\n"
        "SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = YES\n"
        "LOGITS_READ_AUTHORIZED = NO\n"
        "CAUSAL_INTERVENTION_AUTHORIZED = NO\n"
    ).encode("utf-8")
    report.write_bytes(raw)

    monkeypatch.setattr(m.subprocess, "check_output", lambda *args, **kwargs: b"reports/authority.md\n")
    monkeypatch.setattr(m, "_git_bytes", lambda root, object_name: raw)
    monkeypatch.setattr(m, "_git", lambda root, *args: "a" * 40)

    with pytest.raises(
        m.ContractError,
        match="EXECUTION_AUTHORITY_MARKER_MISMATCH:SCIENTIFIC_EXECUTION_AUTHORIZED",
    ):
        m.authenticate_execution_authority(tmp_path, "reports/authority.md")


def test_cli_has_no_frozen_quantity_overrides():
    parser = m.build_parser()
    options = {opt for action in parser._actions for opt in action.option_strings}
    assert "--synthetic-preflight" in options
    assert "--execute-scientific" in options
    assert "--execution-authority" in options
    assert "--seed180-handoff" in options
    assert "--output-dir" in options
    for forbidden in (
        "--layer",
        "--window",
        "--metric",
        "--alpha",
        "--population",
        "--pair-mapping",
        "--zero-threshold",
    ):
        assert forbidden not in options


def test_synthetic_mode_rejects_execution_authority(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "run_synthetic_preflight", lambda root, handoff: {"status": "x"})
    with pytest.raises(m.ContractError, match="SYNTHETIC_EXECUTION_AUTHORITY_FORBIDDEN"):
        m.main([
            "--synthetic-preflight",
            "--seed180-handoff", str(tmp_path / "h.zip"),
            "--execution-authority", "reports/a.md",
        ])


def test_json_rejects_nan():
    with pytest.raises(ValueError):
        m.canonical_json_bytes({"x": float("nan")})


def test_json_none_becomes_null():
    raw = m.canonical_json_bytes({"x": None})
    assert raw == b'{"x":null}'


def test_execution_manifest_binds_authority_blob_and_a0(monkeypatch):
    monkeypatch.setattr(m, "_git", lambda root, *args: "e" * 40)
    monkeypatch.setattr(m, "file_sha256", lambda path: "a" * 64)
    authority = m.ExecutionAuthority(
        path="reports/authority.md",
        commit="b" * 40,
        implementation_commit="c" * 40,
        git_blob="d" * 40,
    )
    archive = m.P0Archive(
        candidates=(),
        mapping=(),
        token_contracts=(),
        manifest={},
        hashes=dict(m.P0_ARTIFACT_SHA256),
    )
    manifest = m.build_execution_manifest(
        ROOT,
        authority,
        archive,
        {"zip_sha256": "z", "checkpoint_sha256": "c"},
        {"canonical_digest": "e", "raw_concat_digest": "r"},
        {},
    )
    assert manifest["scientific_execution_authority_git_blob"] == "d" * 40
    assert manifest["a0_runtime_contract"] == {
        "commit": m.A0_COMMIT,
        "model_blob_sha": m.A0_MODEL_BLOB_SHA,
        "heads_tree_sha": m.A0_HEADS_TREE_SHA,
    }


def test_result_artifact_names_exact():
    assert m.RESULT_ARTIFACTS == (
        "item_metrics.jsonl",
        "block_metrics.jsonl",
        "endpoint_summary.json",
        "recurrence_audit.json",
        "state_hash_audit.jsonl",
        "execution_manifest.json",
    )


def test_result_row_schema_and_block_order_contract():
    items = []
    for i in range(m.P0_ITEM_COUNT):
        items.append({
            "local_template_index": i,
            "stable_item_id": f"s{i}",
            "turning_valid": True,
            "coherence_valid": True,
            "X_turn": 1.0,
            "X_coh": -1.0,
        })
    mapping = [
        {
            "block_index": p,
            "phase_class": p % 2,
            "item_a_stable_id": f"s{p}",
            "item_b_stable_id": f"s{p + m.P0_BLOCK_COUNT}",
        }
        for p in range(m.P0_BLOCK_COUNT)
    ]
    blocks = m.aggregate_blocks(list(reversed(items)), mapping)
    assert [row["block_index"] for row in blocks] == list(range(m.P0_BLOCK_COUNT))
    required_block_keys = {
        "schema_version",
        "block_index",
        "phase_class",
        "item_a_stable_id",
        "item_b_stable_id",
        "turning_valid",
        "B_turn",
        "turning_sign",
        "coherence_valid",
        "B_coh",
        "coherence_sign",
    }
    assert all(required_block_keys <= set(row) for row in blocks)

    summary = m.summarize_endpoints(blocks)
    assert summary["schema_version"] == "k0-rvg-p1-endpoint-summary-v1"
    assert summary["alpha"] == 0.05
    assert summary["multiplicity_method"] == "HOLM_M2"
    assert summary["primary_endpoint_count"] == 2
    for name in ("turning", "response_coherence"):
        assert {
            "endpoint",
            "valid_block_count",
            "invalid_block_count",
            "positive_count",
            "negative_count",
            "zero_count",
            "effective_n",
            "raw_sign_test_p",
            "holm_adjusted_p",
            "sign_effect",
            "endpoint_verdict",
        } <= set(summary[name])


def test_atomic_writer_cleans_partial_on_failure(tmp_path):
    out = tmp_path / "final"
    artifacts = {
        "item_metrics.jsonl": b"",
        "block_metrics.jsonl": b"",
        "endpoint_summary.json": b"{}\n",
        "recurrence_audit.json": b"{}\n",
        "state_hash_audit.jsonl": b"",
    }
    with pytest.raises(RuntimeError):
        m.write_scientific_artifacts_atomic(
            out,
            artifacts,
            lambda hashes: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_no_import_time_model_execution():
    source = SCRIPT.read_text("utf-8")
    assert "if __name__ == \"__main__\":" in source
    # Model construction appears only inside functions and must not be invoked at import.
    assert "build_a0_model" in source


def test_scientific_gate_called_before_model_builder():
    import inspect
    source = inspect.getsource(m.execute_scientific)
    assert source.index("authenticate_execution_authority") < source.index("validate_runtime_and_build_model")


def test_synthetic_candidates_are_not_p0_ids():
    candidates, _ = m._synthetic_candidates()
    assert all(not str(x["pair_id"]).startswith("generated_fact_") for x in candidates)


def test_state_hash_deterministic():
    torch = pytest.importorskip("torch")
    x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    assert m.tensor_sha256(x) == m.tensor_sha256(x.clone())
