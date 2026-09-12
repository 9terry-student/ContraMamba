from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "longterm_k0_rvg_post_p2_token_window_static_audit.py"
SPEC = importlib.util.spec_from_file_location("post_p2_token_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def seq_with_diff(te: int, relative: int, *, length: int = 20):
    a = list(range(length))
    b = list(a)
    index = te + relative
    b[index] += 1000
    return a, b


def role(a, b, te=5):
    return m.analyze_role(a, b, te, te)


def test_full_branch_identity():
    a = list(range(20))
    out = role(a, list(a))
    assert out["classification"] == m.C_IDENTICAL
    assert out["window_exact_equal"] is True
    assert out["first_difference_absolute_index"] is None


def test_first_difference_before_k_minus_1():
    a, b = seq_with_diff(5, -2)
    assert role(a, b)["classification"] == m.C_PREWINDOW


def test_first_difference_at_k_minus_1():
    a, b = seq_with_diff(5, -1)
    assert role(a, b)["classification"] == m.C_K_MINUS_1


def test_first_difference_at_k_zero():
    a, b = seq_with_diff(5, 0)
    assert role(a, b)["classification"] == m.C_INWINDOW


def test_first_difference_at_k_plus_7():
    a, b = seq_with_diff(5, 7)
    assert role(a, b)["classification"] == m.C_INWINDOW


def test_first_difference_after_window():
    a, b = seq_with_diff(5, 8)
    assert role(a, b)["classification"] == m.C_AFTER


def test_strict_prefix_matched_sequence():
    a = list(range(14))
    b = list(range(16))
    out = role(a, b, te=5)
    assert out["first_difference_absolute_index"] == 14
    assert out["classification"] == m.C_AFTER


def test_strict_prefix_swapped_sequence():
    a = list(range(16))
    b = list(range(14))
    out = role(a, b, te=5)
    assert out["first_difference_absolute_index"] == 14
    assert out["classification"] == m.C_AFTER


def test_unequal_event_anchors_fail_closed():
    a = list(range(20))
    with pytest.raises(m.ContractError, match="MATCHED_SWAPPED_EVENT_ANCHOR_MISMATCH"):
        m.analyze_role(a, a, 5, 6)


def test_missing_window_coordinate_fail_closed():
    a = list(range(10))
    with pytest.raises(m.ContractError, match="WINDOW_COORDINATE_MISSING"):
        m.analyze_role(a, a, 5, 5)


def test_exact_nine_coordinate_window_equality():
    a = list(range(20))
    out = m.compare_window(a, list(a), 5, 5)
    assert len(out) == 9
    assert all(row["exact_equal"] for row in out)


def test_one_coordinate_window_difference():
    a, b = seq_with_diff(5, 3)
    out = m.compare_window(a, b, 5, 5)
    assert sum(not row["exact_equal"] for row in out) == 1


def test_deterministic_phase_mate_reconstruction():
    mapping = {
        "blocks": [
            {"item_a_local_index": 0, "item_b_local_index": 1},
            {"item_a_local_index": 2, "item_b_local_index": 3},
        ]
    }
    mate = m.build_phase_mate_map(mapping, expected_count=4)
    assert mate == {0: 1, 1: 0, 2: 3, 3: 2}


def test_branch_text_concatenation_no_normalization():
    candidates = {
        0: {
            "prefix_text": "P \n",
            "correction_text": "  C",
            "control_text": "\tD",
        },
        1: {
            "prefix_text": "Q",
            "correction_text": "  X",
            "control_text": "\tY",
        },
    }
    texts = m.reconstruct_branch_texts(candidates, {0: 1, 1: 0}, 0)
    assert texts["matched_corr"] == "P \n  C"
    assert texts["matched_ctrl"] == "P \n\tD"
    assert texts["swapped_corr"] == "P \n  X"
    assert texts["swapped_ctrl"] == "P \n\tY"


def test_cross_level_window_identity():
    identical = {"classification": m.C_IDENTICAL}
    after = {"classification": m.C_AFTER}
    assert m.cross_level_classification(m.STATE_IDENTITY, identical, after) == m.X_WINDOW_IDENTITY


def test_cross_level_prewindow():
    pre = {"classification": m.C_PREWINDOW}
    ident = {"classification": m.C_IDENTICAL}
    assert m.cross_level_classification(m.STATE_IDENTITY, pre, ident) == m.X_PREWINDOW


def test_cross_level_inwindow():
    win = {"classification": m.C_INWINDOW}
    pre = {"classification": m.C_PREWINDOW}
    assert m.cross_level_classification(m.STATE_IDENTITY, win, pre) == m.X_INWINDOW


def test_cross_level_unresolved_for_missing_evidence():
    unknown = {"classification": m.C_UNKNOWN}
    ident = {"classification": m.C_IDENTICAL}
    assert m.cross_level_classification(m.STATE_IDENTITY, unknown, ident) == m.X_UNRESOLVED


def test_cross_level_unresolved_for_state_difference():
    ident = {"classification": m.C_IDENTICAL}
    assert m.cross_level_classification(m.STATE_DIFFERENCE, ident, ident) == m.X_UNRESOLVED


def test_duplicate_authority_marker_rejected():
    text = (
        "`POST_P2_TOKEN_WINDOW_REAL_ARTIFACT_EXECUTION_AUTHORIZED = YES`\n"
        "`POST_P2_TOKEN_WINDOW_REAL_ARTIFACT_EXECUTION_AUTHORIZED = NO`\n"
    )
    with pytest.raises(m.ContractError, match="DUPLICATE_AUTHORITY_MARKER"):
        m.parse_authority_markers(text)


def valid_future_markers(commit: str):
    out = dict(m.REQUIRED_FUTURE_MARKERS)
    out["TOKENIZER_SNAPSHOT_MANIFEST_SHA256"] = "a" * 64
    out["POST_P2_TOKEN_WINDOW_IMPLEMENTATION_COMMIT"] = commit
    return out


def test_wrong_implementation_commit_marker_rejected():
    actual = "a" * 40
    markers = valid_future_markers("b" * 40)
    with pytest.raises(m.ContractError, match="IMPLEMENTATION_COMMIT_MISMATCH"):
        m.validate_execution_markers(markers, actual)


def test_missing_tokenizer_snapshot_manifest_marker_rejected():
    actual = "a" * 40
    markers = valid_future_markers(actual)
    del markers["TOKENIZER_SNAPSHOT_MANIFEST_SHA256"]
    with pytest.raises(m.ContractError, match="TOKENIZER_SNAPSHOT_SHA256_INVALID"):
        m.validate_execution_markers(markers, actual)


def test_tokenizer_snapshot_manifest_hashes_only_frozen_file_family(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pytorch_model.bin").write_bytes(b"ignored-model-weight")
    hashes, digest = m.tokenizer_snapshot_manifest(tmp_path)
    assert sorted(hashes) == [
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert len(digest) == 64
    assert "pytorch_model.bin" not in hashes


def test_tokenizer_snapshot_manifest_changes_when_authenticated_file_changes(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text('{"version":"1"}', encoding="utf-8")
    _, digest_a = m.tokenizer_snapshot_manifest(tmp_path)
    path.write_text('{"version":"2"}', encoding="utf-8")
    _, digest_b = m.tokenizer_snapshot_manifest(tmp_path)
    assert digest_a != digest_b


def test_local_tokenizer_rejects_snapshot_mismatch_before_transformers_import(tmp_path):
    (tmp_path / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
    markers = {"TOKENIZER_SNAPSHOT_MANIFEST_SHA256": "0" * 64}
    with pytest.raises(m.ContractError, match="TOKENIZER_SNAPSHOT_MANIFEST_SHA256_MISMATCH"):
        m.load_local_tokenizer(tmp_path, markers)


def test_execution_authority_missing_before_real_read(tmp_path):
    touched = {"input": False, "tokenizer": False}

    def deny(_root, _authority):
        raise m.ContractError("DENIED")

    def input_loader(_root, _p2):
        touched["input"] = True
        raise AssertionError("must not run")

    def tokenizer_loader(_snapshot, _markers):
        touched["tokenizer"] = True
        raise AssertionError("must not run")

    with pytest.raises(m.ContractError, match="DENIED"):
        m.execute_real(
            tmp_path,
            "reports/future.md",
            tmp_path / "p2.json",
            tmp_path / "snapshot",
            tmp_path / "out",
            authority_authenticator=deny,
            real_input_loader=input_loader,
            tokenizer_loader=tokenizer_loader,
        )
    assert touched == {"input": False, "tokenizer": False}


def test_output_directory_already_exists_fail_closed(tmp_path):
    output = tmp_path / "out"
    output.mkdir()

    def allow(_root, _authority):
        return {"ok": "yes"}

    touched = {"input": False}

    def input_loader(_root, _p2):
        touched["input"] = True
        raise AssertionError("must not read after output collision")

    with pytest.raises(m.ContractError, match="OUTPUT_ALREADY_EXISTS"):
        m.execute_real(
            tmp_path,
            "reports/future.md",
            tmp_path / "p2.json",
            tmp_path / "snapshot",
            output,
            authority_authenticator=allow,
            real_input_loader=input_loader,
        )
    assert touched["input"] is False


def test_canonical_serialization_repeat_identity():
    value = {"z": [3, 2, 1], "a": {"x": True}}
    assert m.canonical_json_bytes(value) == m.canonical_json_bytes(value)


def test_synthetic_self_check():
    result = m.synthetic_self_check()
    assert result["status"] == "PASS_SYNTHETIC_POST_P2_TOKEN_WINDOW_AUDIT_CORE"
    assert result["real_p0_artifact_read"] is False
    assert result["real_p2_artifact_read"] is False
    assert result["real_hf_tokenizer_loaded"] is False
    assert result["network_access"] is False
    assert result["model_constructed"] is False
    assert result["checkpoint_loaded"] is False
    assert result["scientific_model_forward_executed"] is False
    assert result["scientific_recurrent_state_read"] is False


def test_fake_tokenizer_retokenized_item_contract():
    candidates = {
        0: {
            "local_template_index": 0,
            "stable_item_id": "a",
            "prefix_text": "P:",
            "correction_text": " ca",
            "control_text": " da",
        },
        1: {
            "local_template_index": 1,
            "stable_item_id": "b",
            "prefix_text": "Q:",
            "correction_text": " cb",
            "control_text": " db",
        },
    }
    texts = m.reconstruct_branch_texts(candidates, {0: 1, 1: 0}, 0)
    arrays = {
        texts["prefix"]: [10, 11],
        texts["matched_corr"]: [10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        texts["matched_ctrl"]: [10, 11, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        texts["swapped_corr"]: [10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 99],
        texts["swapped_ctrl"]: [10, 11, 20, 31, 32, 33, 34, 35, 36, 37, 38, 98],
    }
    contract = {
        "prefix_token_count": 2,
        "prefix_token_sha256": m.token_id_sha256([10, 11]),
        "matched_correction_token_count": 12,
        "matched_control_token_count": 12,
        "swapped_correction_token_count": 12,
        "swapped_control_token_count": 12,
        "matched_divergence_anchor": 3,
        "swapped_divergence_anchor": 3,
    }
    out = m.validate_retokenized_item(m.FakeTokenizer(arrays), texts, contract)
    assert out["correction_role"]["classification"] == m.C_AFTER
    assert out["control_role"]["classification"] == m.C_AFTER
