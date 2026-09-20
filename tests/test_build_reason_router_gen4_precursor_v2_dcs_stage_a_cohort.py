from __future__ import annotations

import inspect

from scripts import build_reason_router_gen4_precursor_v2_dcs_stage_a_cohort as m


def _row(i: int, label: str) -> dict:
    correct = "REFUTE" if label == "Refuted" else "SUPPORT"
    correct_id = 0 if label == "Refuted" else 2
    return {
        "schema_version": "GEN4_AVERITEC_TRAIN_FRESH_STEERING_COHORT_V1",
        "averitec_train_index": i,
        "example_id": f"averitec_train_{i:04d}",
        "claim": f"claim {i}",
        "normalized_claim": f"claim {i}",
        "original_claim_url": f"https://example.test/{i}",
        "evidence": f"evidence {i}",
        "source_label": label,
        "correct_label": correct,
        "correct_label_id": correct_id,
        "question_count": 1,
        "answer_count": 1,
        "claim_raw_token_count": 3,
        "claim_consumed_token_count": 3,
        "evidence_raw_token_count": 4,
        "evidence_consumed_token_count": 4,
        "claim_truncated": False,
        "evidence_truncated": False,
        "anchor_name": "A_CLAIM_EVIDENCE_BOUNDARY",
        "absolute_anchor_token_index": 3,
        "target_intervention_token_index": 5,
        "serialized_attended_length": 8,
        "target_offset": 2,
        "token_gate_pass": True,
        "token_gate_reasons": [],
    }


def test_design_and_source_pins():
    assert m.DESIGN_GIT_BLOB == "d24011d698de1a53436a6bb997b2bb400881a718"
    assert m.DESIGN_SHA256 == "fa820362e3ad7da014ef22c838b30189fb12ec775296dee5b1ff68c5b3461946"
    assert m.SOURCE_COHORT_SHA256 == "ce271c1b57b33e12399bc180f06d417ceef6e4e92ad9e5383e2d8187829a2810"
    assert m.SOURCE_MANIFEST_SHA256 == "7fef7f8c9544f46fbdeb1184bee313259bd52154181c346c4ae6d367d08da738"


def test_rank_key_exact_namespace_contract():
    ex = "averitec_train_0001"
    expected = __import__("hashlib").sha256(
        ("CONTRAMAMBA_PRECURSOR_V2_DCS_STAGE_A_V1|" + ex).encode("utf-8")
    ).hexdigest()
    assert m.selection_rank_key(ex) == expected


def test_selection_is_balanced_and_deterministic(monkeypatch):
    monkeypatch.setattr(m, "TARGET_N", 4)
    monkeypatch.setattr(m, "TARGET_PER_SOURCE_LABEL", {"Refuted": 2, "Supported": 2})
    monkeypatch.setattr(m, "TARGET_PER_CORRECT_LABEL", {"REFUTE": 2, "SUPPORT": 2})

    rows = [_row(i, "Refuted") for i in range(5)]
    rows += [_row(i + 100, "Supported") for i in range(5)]

    a = m.select_rows(rows)
    b = m.select_rows(rows)
    assert a == b
    assert len(a) == 4
    assert sum(r["source_label"] == "Refuted" for r in a) == 2
    assert sum(r["source_label"] == "Supported" for r in a) == 2
    assert all(r["fresh_for_precursor_v2_generation_response"] for r in a)
    assert all(not r["experiment3_response_accessed_for_selection"] for r in a)


def test_no_response_fields_are_introduced(monkeypatch):
    monkeypatch.setattr(m, "TARGET_N", 2)
    monkeypatch.setattr(m, "TARGET_PER_SOURCE_LABEL", {"Refuted": 1, "Supported": 1})
    monkeypatch.setattr(m, "TARGET_PER_CORRECT_LABEL", {"REFUTE": 1, "SUPPORT": 1})
    rows = [_row(1, "Refuted"), _row(2, "Supported")]
    out = m.select_rows(rows)
    for row in out:
        assert not (m.FORBIDDEN_RESPONSE_FIELDS & set(row))


def test_source_validation_rejects_response_field():
    rows = [_row(i, "Refuted") for i in range(m.SOURCE_N)]
    # Make IDs/claims unique and source ordering valid; inject one forbidden field.
    rows[0]["native_prediction"] = "REFUTE"
    try:
        m.validate_source_rows(rows)
    except m.CohortError as exc:
        assert "FORBIDDEN_RESPONSE_FIELD" in str(exc) or "ROW_LABEL_COUNTS" in str(exc)
    else:
        raise AssertionError("expected CohortError")


def test_script_contains_no_model_or_inference_path():
    src = inspect.getsource(m).lower()
    for token in (
        "import torch",
        "cuda.",
        "scipy",
        "ttest",
        "wilcoxon",
        "mannwhitney",
        "permutation_test",
        "model(",
        ".generate(",
    ):
        assert token not in src
    assert '"model_forward_count": 0' in src
    assert '"p_value_count_added": 0' in src
    assert '"scientific_execution_authorized": false' in src
    assert '"runner_implemented": false' in src
    assert '"equivalence_executed": false' in src
