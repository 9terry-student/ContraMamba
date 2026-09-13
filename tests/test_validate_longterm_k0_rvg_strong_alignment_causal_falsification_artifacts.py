from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[1] / "scripts" / "validate_longterm_k0_rvg_strong_alignment_causal_falsification_artifacts.py"
spec = importlib.util.spec_from_file_location("validator", P)
assert spec is not None and spec.loader is not None
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def test_validator_is_independent_of_causal_runner_import():
    source = P.read_text(encoding="utf-8")
    assert "import longterm_k0_rvg_strong_alignment_causal_falsification" not in source
    assert "from scripts import longterm_k0_rvg_strong_alignment_causal_falsification" not in source
    assert "recompute_summary(" in source
    assert "def classify(" in source


def test_outcome_classification_contract():
    assert v.classify(0.2, 0.3, 0.1, 0.2, True) == "A"
    assert v.classify(0.2, 0.3, 0.2, 0.2, True) == "B"
    assert v.classify(0.2, -0.1, 0.0, 0.0, True) == "C"
    assert v.classify(-0.2, -0.1, 0.0, 0.0, True) == "D"
    assert v.classify(1.0, 1.0, 0.0, 0.0, False) == "Invalid"


def test_negative_provenance_flags_are_false_only():
    v.reject_private_payload({"logits_read": False, "raw_vectors_persisted": False})
    with pytest.raises(v.ValidationError, match="NEGATIVE_PUBLIC_FLAG_NOT_FALSE"):
        v.reject_private_payload({"logits_read": True})
    with pytest.raises(v.ValidationError, match="NEGATIVE_PUBLIC_FLAG_NOT_FALSE"):
        v.reject_private_payload({"raw_vectors_persisted": True})
    with pytest.raises(v.ValidationError, match="FORBIDDEN_PUBLIC_FIELD"):
        v.reject_private_payload({"logits_payload": [1.0]})


def test_summary_recomputation_contract_on_synthetic_rows():
    base = {
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
        "magnitude_realized_B": 2.0,
        "magnitude_target_B": 2.0,
        "magnitude_cosine_preservation_abs_residual": 0.0,
        "magnitude_midpoint_max_abs_residual": 0.0,
        "magnitude_applied_correction_max_abs_residual": 0.0,
        "alignment_runtime_correction_l2": 1.0,
        "magnitude_runtime_correction_l2": 0.5,
        "C_corr": 0.4,
        "C_ctrl": 0.2,
    }
    rows = []
    for i in range(v.EXPECTED_COUNT):
        r = dict(base)
        r.update({
            "S_c0_minus_ctrl": 1.0,
            "S_cA_minus_ctrl": 0.7,
            "S_cM_minus_ctrl": 0.8,
            "W_c0_minus_ctrl": 1.0,
            "W_cA_minus_ctrl": 0.6,
            "W_cM_minus_ctrl": 0.9,
            "C_c0_minus_cA": 0.01,
            "C_c0_minus_cM": 0.02,
            "S_c0_minus_cA": 0.3,
            "S_c0_minus_cM": 0.2,
            "W_c0_minus_cA": 0.4,
            "W_c0_minus_cM": 0.1,
        })
        rows.append(r)
    s = v.recompute_summary(rows)
    assert s["outcome"] == "A"
    assert s["R_SA"] == pytest.approx(0.3)
    assert s["R_WA"] == pytest.approx(0.4)
    assert s["all_mandatory_manipulation_checks_pass"] is True


def test_output_hashes_are_frozen_to_reported_execution_identity():
    assert v.EXPECTED_OUTPUT_SHA256 == {
        "execution_manifest.json": "bf35236c29d6bf359cb6051a3843c7b94756cee03cb75149063bd1d9a05358ac",
        "strong_alignment_causal_falsification_item_metrics.jsonl": "45660a82a177c38dd8724bfe0402acd8d5213d39432978e65d0d717d76558518",
        "summary.json": "0663d354ba55912e22b6428600362837667cf8240445564c239c12a3b873d5c8",
    }
