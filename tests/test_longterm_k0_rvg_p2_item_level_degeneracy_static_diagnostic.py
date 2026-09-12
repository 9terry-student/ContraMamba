from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "longterm_k0_rvg_p2_item_level_degeneracy_static_diagnostic.py"

spec = importlib.util.spec_from_file_location("k0_rvg_p2", SCRIPT)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def _diag(value=1.0):
    return {field: value for field in m.DIAGNOSTIC_FIELDS}


def _item(
    *,
    a_corr_m=0.2,
    a_ctrl_m=0.8,
    t_m=0.6,
    c_m=0.25,
    a_corr_s=0.2,
    a_ctrl_s=0.8,
    t_s=0.6,
    c_s=0.25,
    x_turn=0.0,
    x_coh=0.0,
):
    return {
        "turning_valid": True,
        "coherence_valid": True,
        "matched_divergence_anchor": 10,
        "swapped_divergence_anchor": 12,
        "matched": {
            "A_corr": a_corr_m,
            "A_ctrl": a_ctrl_m,
            "T_M": t_m,
            "C_M": c_m,
            "diagnostics": _diag(),
        },
        "swapped": {
            "A_corr": a_corr_s,
            "A_ctrl": a_ctrl_s,
            "T_S": t_s,
            "C_S": c_s,
            "diagnostics": _diag(),
        },
        "X_turn": x_turn,
        "X_coh": x_coh,
    }


def _hash_rows(local=0, m_te=10, s_te=12, differing=None):
    rows = []
    differing = differing or set()
    for m_role, s_role, _ in m.BRANCH_PAIRS:
        for rel in m.EVENT_RELATIVE_COORDS:
            for role, te, side in ((m_role, m_te, "m"), (s_role, s_te, "s")):
                base = f"{local}:{side}:{role}:{rel}"
                row = {
                    "schema_version": "k0-rvg-p1-state-hash-v1",
                    "local_template_index": local,
                    "branch_role": role,
                    "token_index": te + rel,
                }
                for field in m.HASH_FIELDS:
                    pair_key = (m_role, s_role, rel, field)
                    same = pair_key not in differing
                    canonical = f"{local}:{m_role}:{s_role}:{rel}:{field}"
                    source = canonical if same else base + ":" + field
                    row[field] = m.sha256_bytes(source.encode())
                rows.append(row)
    return rows


def test_authority_and_scope_constants():
    assert m.AUTHORITY_COMMIT == "95709b318bb4b4454d57edf52266860ded9e9da3"
    assert m.RUNNER_REL == "scripts/longterm_k0_rvg_p2_item_level_degeneracy_static_diagnostic.py"
    assert m.TEST_REL == "tests/test_longterm_k0_rvg_p2_item_level_degeneracy_static_diagnostic.py"
    assert m.ITEM_COUNT == 336
    assert m.BLOCK_COUNT == 168
    assert m.STATE_HASH_ROW_COUNT == 12096


def test_componentwise_turning_identity():
    assert m.classify_turning(_item()) == m.TURN_COMPONENTWISE


def test_turning_equality_by_nonzero_offset_cancellation():
    item = _item(
        a_corr_m=0.1,
        a_ctrl_m=0.7,
        t_m=0.6,
        a_corr_s=0.2,
        a_ctrl_s=0.8,
        t_s=0.6,
    )
    assert m.classify_turning(item) == m.TURN_CANCELLATION


def test_turning_difference():
    assert m.classify_turning(_item(t_s=0.5, x_turn=0.1)) == m.TURN_DIFFERENT


def test_coherence_equality_and_difference():
    assert m.classify_coherence(_item()) == m.COH_IDENTITY
    assert m.classify_coherence(_item(c_s=0.2, x_coh=0.05)) == m.COH_DIFFERENT


def test_full_state_hash_identity():
    index = m.index_state_hash_rows(_hash_rows())
    got = m.compare_item_state_hashes(0, 10, 12, index)
    assert got["classification"] == m.STATE_IDENTITY
    assert got["total_aligned_hash_comparisons"] == 72
    assert got["exact_equal_count"] == 72
    assert got["different_count"] == 0
    assert got["first_difference"] is None


def test_single_state_hash_difference_first_difference_localization():
    diff = {("matched_corr", "swapped_corr", -1, "G_sha256")}
    index = m.index_state_hash_rows(_hash_rows(differing=diff))
    got = m.compare_item_state_hashes(0, 10, 12, index)
    assert got["classification"] == m.STATE_DIFFERENCE
    assert got["different_count"] == 1
    assert got["first_difference"] == {
        "branch_role": "corr",
        "relative_coordinate": -1,
        "tensor_field": "G_sha256",
        "matched_token_index": 9,
        "swapped_token_index": 11,
    }


def test_different_divergence_anchors_are_event_relative_aligned():
    index = m.index_state_hash_rows(_hash_rows(m_te=100, s_te=207))
    got = m.compare_item_state_hashes(0, 100, 207, index)
    assert got["classification"] == m.STATE_IDENTITY
    assert got["exact_equal_count"] == 72


def test_diagnostic_summary_equality_difference_and_null():
    item = _item()
    item["matched"]["diagnostics"]["response_norm_mean"] = None
    item["swapped"]["diagnostics"]["response_norm_mean"] = None
    item["swapped"]["diagnostics"]["write_response_norm_mean"] = 2.0
    got = m.compare_diagnostic_summaries(item)
    assert got["response_norm_mean"] == "NULL_NULL_EQUAL"
    assert got["carry_response_norm_mean"] == "EXACT_EQUAL"
    assert got["write_response_norm_mean"] == "DIFFERENT"


def test_construction_metadata_and_static_text_reconstruction():
    candidate = {
        "stable_item_id": "a",
        "correction_text": "corr-a",
        "control_text": "ctrl-a",
    }
    mate = {
        "stable_item_id": "b",
        "correction_text": "corr-b",
        "control_text": "ctrl-b",
    }
    contract = {
        "matched_divergence_anchor": 10,
        "swapped_divergence_anchor": 10,
        "matched_correction_token_count": 20,
        "swapped_correction_token_count": 20,
        "matched_control_token_count": 21,
        "swapped_control_token_count": 21,
        "matched_w8_available": True,
        "swapped_w8_available": True,
    }
    got = m.compare_construction_metadata(candidate, mate, contract)
    assert got["divergence_anchor_exact_equal"] is True
    assert got["correction_token_count_exact_equal"] is True
    assert got["control_token_count_exact_equal"] is True
    assert got["w8_availability_exact_equal"] is True
    assert got["correction_continuation_text_relation"] == "DIFFERENT"
    assert got["control_continuation_text_relation"] == "DIFFERENT"


def test_text_reconstruction_unavailable_without_authorized_static_fields():
    got = m.compare_construction_metadata(
        None,
        None,
        {
            "matched_divergence_anchor": 1,
            "swapped_divergence_anchor": 2,
            "matched_correction_token_count": 9,
            "swapped_correction_token_count": 10,
            "matched_control_token_count": 9,
            "swapped_control_token_count": 10,
            "matched_w8_available": True,
            "swapped_w8_available": True,
        },
    )
    assert got["correction_continuation_text_relation"] == m.TEXT_UNAVAILABLE
    assert got["control_continuation_text_relation"] == m.TEXT_UNAVAILABLE


def test_zero_endpoint_localization_state_identity():
    item = _item()
    assert (
        m.localize_zero_endpoint(
            "turning",
            item,
            m.TURN_COMPONENTWISE,
            m.COH_IDENTITY,
            m.STATE_IDENTITY,
        )
        == m.LOC_STATE_IDENTITY
    )


def test_zero_turning_localization_distinct_components_cancellation():
    item = _item(
        a_corr_m=0.1,
        a_ctrl_m=0.7,
        a_corr_s=0.2,
        a_ctrl_s=0.8,
    )
    assert (
        m.localize_zero_endpoint(
            "turning",
            item,
            m.TURN_CANCELLATION,
            m.COH_IDENTITY,
            m.STATE_DIFFERENCE,
        )
        == m.LOC_TURN_CANCELLATION
    )


def test_zero_coherence_localization_distinct_state_functional_collapse():
    item = _item()
    assert (
        m.localize_zero_endpoint(
            "response_coherence",
            item,
            m.TURN_COMPONENTWISE,
            m.COH_IDENTITY,
            m.STATE_DIFFERENCE,
        )
        == m.LOC_FUNCTIONAL_COLLAPSE
    )


def test_artifact_hash_mismatch_fails_closed(tmp_path):
    p = tmp_path / "only.json"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(m.ContractError, match="ARTIFACT_SHA256_MISMATCH"):
        m.validate_exact_artifact_set(
            tmp_path,
            {"only.json": "0" * 64},
        )


def test_unauthorized_real_execution_fails_before_input_read(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    p1 = tmp_path / "does-not-exist"
    out = tmp_path / "out"

    monkeypatch.setattr(
        m,
        "authenticate_real_execution_authority",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            m.ContractError("P2_REAL_ARTIFACT_DIAGNOSTIC_EXECUTION_NOT_AUTHORIZED")
        ),
    )
    called = {"inputs": False}

    def forbidden_inputs(*args, **kwargs):
        called["inputs"] = True
        raise AssertionError("real inputs must not be read")

    monkeypatch.setattr(m, "validate_real_inputs", forbidden_inputs)
    with pytest.raises(
        m.ContractError,
        match="P2_REAL_ARTIFACT_DIAGNOSTIC_EXECUTION_NOT_AUTHORIZED",
    ):
        m.execute_real(root, p1, "reports/fake.md", out)
    assert called["inputs"] is False


def test_no_model_checkpoint_or_tensor_runtime_dependencies_in_module_source():
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "import torch",
        "from torch",
        "transformers",
        "AutoModel",
        "MambaForCausalLM",
        "load_state_dict",
        "checkpoint_path",
        "RawRecurrenceCollector",
    ]
    for token in forbidden:
        assert token not in source


def test_synthetic_self_check():
    got = m.synthetic_self_check()
    assert got == {
        "schema_version": "k0-rvg-p2-synthetic-self-check-v1",
        "status": "PASS_SYNTHETIC_P2_STATIC_DIAGNOSTIC_CORE",
        "real_p1_artifact_read": False,
        "model_forward": False,
        "recurrent_state_read": False,
    }
