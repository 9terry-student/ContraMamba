from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts import (
    build_reason_router_gen4_averitec_gold_evidence_130m370m_token_gate
    as gate,
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = list(ids)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        # Deterministic whitespace tokenizer; IDs do not matter for the gate.
        pieces = [p for p in str(text).split() if p]
        return FakeEncoding(range(1, len(pieces) + 1))


def fixture_example(label: str, *, multi: bool = False):
    answers = [
        {"answer": "first answer", "answer_type": "Extractive"},
    ]
    if multi:
        answers.append(
            {"answer": "second answer", "answer_type": "Abstractive"}
        )
    return {
        "claim": "A natural language claim",
        "label": label,
        "questions": [
            {
                "question": "What happened?",
                "answers": answers,
            }
        ],
    }


def fake_source():
    rows = []
    rows.extend(fixture_example("Refuted") for _ in range(305))
    rows.extend(fixture_example("Supported") for _ in range(122))
    rows.extend(
        fixture_example("Not Enough Evidence")
        for _ in range(35)
    )
    rows.extend(
        fixture_example("Conflicting Evidence/Cherrypicking")
        for _ in range(38)
    )
    return rows


def test_pinned_external_identity_constants() -> None:
    assert gate.PRIMARY_SCALES == ("mamba130m", "mamba370m")
    assert gate.PRIMARY_P_VALUE_COUNT_PLANNED == 2
    assert gate.PRIMARY_MULTIPLICITY == "holm"
    assert gate.SHARED_TOKENIZER_BYTES_ACROSS_PRIMARY_SCALES is True
    assert gate.FROZEN_CAUSAL_OBJECTS["mamba130m"]["selected_plane"] == "P3"
    assert gate.FROZEN_CAUSAL_OBJECTS["mamba130m"]["control_plane"] == "P5"
    assert gate.FROZEN_CAUSAL_OBJECTS["mamba130m"]["intervention_layer"] == 17
    assert gate.FROZEN_CAUSAL_OBJECTS["mamba370m"]["selected_plane"] == "P3"
    assert gate.FROZEN_CAUSAL_OBJECTS["mamba370m"]["control_plane"] == "P5"
    assert gate.FROZEN_CAUSAL_OBJECTS["mamba370m"]["intervention_layer"] == 35
    assert gate.UPSTREAM_COMMIT == (
        "7c62d1ec8df3fb560d6efe2b85fa191135636f81"
    )
    assert gate.UPSTREAM_BLOB_SHA1 == (
        "40974243267f395dc583d805d10f043812419249"
    )
    assert gate.UPSTREAM_BYTES == 1785475
    assert gate.EXPECTED_SOURCE_EXAMPLES == 500
    assert gate.EXPECTED_COMPATIBLE_COUNT == 462


def test_label_mapping_and_fourth_class_exclusion() -> None:
    source = fake_source()
    compatible = gate.compatible_examples(source)
    assert len(compatible) == 462
    labels = [row["label"] for _, row in compatible]
    assert labels.count("Refuted") == 305
    assert labels.count("Supported") == 122
    assert labels.count("Not Enough Evidence") == 35
    assert gate.EXCLUDED_LABEL not in labels
    assert gate.LABEL_MAP == {
        "Refuted": {"label": "REFUTE", "label_id": 0},
        "Not Enough Evidence": {
            "label": "NOT_ENTITLED",
            "label_id": 1,
        },
        "Supported": {"label": "SUPPORT", "label_id": 2},
    }


def test_gold_evidence_serialization_repeats_question_per_answer() -> None:
    example = fixture_example("Supported", multi=True)
    evidence, q_count, a_count = gate.serialize_gold_evidence(example)
    assert q_count == 1
    assert a_count == 2
    assert evidence == (
        "Question: What happened?\nAnswer: first answer\n\n"
        "Question: What happened?\nAnswer: second answer"
    )


def test_boundary_gate_uses_consumed_claim_eos_plus_two() -> None:
    row = gate.gate_row(
        tokenizer=FakeTokenizer(),
        source_index=7,
        example=fixture_example("Refuted"),
    )
    assert row["token_gate_pass"] is True
    assert row["absolute_anchor_token_index"] == 4
    assert row["target_intervention_token_index"] == 6
    assert row["target_offset"] == 2
    assert row["correct_label"] == "REFUTE"
    assert row["correct_label_id"] == 0
    assert not (gate.FORBIDDEN_RESPONSE_FIELDS & set(row))


def test_gate_fails_closed_when_evidence_has_fewer_than_two_tokens() -> None:
    example = {
        "claim": "claim tokens",
        "label": "Supported",
        "questions": [
            {
                "question": "",
                "answers": [{"answer": "", "answer_type": "Abstractive"}],
            }
        ],
    }
    # Bypass serializer input validation by a minimal tokenizer-specific
    # monkeypatch: this tests the fail-closed token rule itself.
    original = gate.serialize_gold_evidence
    try:
        gate.serialize_gold_evidence = lambda _x: ("one", 1, 1)
        row = gate.gate_row(
            tokenizer=FakeTokenizer(),
            source_index=0,
            example=example,
        )
    finally:
        gate.serialize_gold_evidence = original

    assert row["token_gate_pass"] is False
    assert (
        "FEWER_THAN_TWO_CONSUMED_EVIDENCE_TOKENS"
        in row["token_gate_reasons"]
    )


def test_git_blob_sha1_matches_git_object_contract() -> None:
    raw = b"hello\n"
    expected = hashlib.sha1(
        b"blob 6\0" + raw
    ).hexdigest()
    assert gate.git_blob_sha1(raw) == expected


def test_build_rows_is_deterministic_and_has_no_response_fields() -> None:
    source = fake_source()
    tokenizer = FakeTokenizer()
    a = gate.build_rows(source=source, tokenizer=tokenizer)
    b = gate.build_rows(source=source, tokenizer=tokenizer)
    assert gate.jsonl_bytes(a) == gate.jsonl_bytes(b)
    assert len(a) == 462
    assert all(row["token_gate_pass"] for row in a)
    assert all(
        not (gate.FORBIDDEN_RESPONSE_FIELDS & set(row))
        for row in a
    )


def test_manifest_is_response_blind_and_cpu_only() -> None:
    source = fake_source()
    rows = gate.build_rows(
        source=source,
        tokenizer=FakeTokenizer(),
    )
    raw = json.dumps(source).encode()
    # build_manifest does not validate upstream bytes: source identity
    # is separately enforced by parse_source/materialize.
    cohort_raw = gate.jsonl_bytes(rows)
    manifest = gate.build_manifest(
        source_raw=raw,
        rows=rows,
        tokenizer_provenance={"fake": True},
        cohort_sha256=gate.sha256_bytes(cohort_raw),
    )
    assert manifest["result"] == gate.PASS_RESULT
    assert manifest["primary_external_transfer_family"] == {
        "scale_order": ["mamba130m", "mamba370m"],
        "primary_p_value_count_planned": 2,
        "multiplicity": "holm",
        "mamba14b_in_family": False,
        "historical_behavioral_p_values_in_family": 0,
    }
    assert manifest["shared_tokenizer_bytes_across_primary_scales"] is True
    assert manifest["token_gate"]["pass_count"] == 462
    assert manifest["token_gate"]["fail_count"] == 0
    assert manifest["model_checkpoint_loaded"] is False
    assert manifest["model_forward_count"] == 0
    assert manifest["cuda_executed"] is False
    assert manifest["scientific_inference_executed"] is False
    assert manifest["p_value_count_added"] == 0
    assert manifest["response_fields_present"] is False


def test_production_builder_contains_no_model_execution_path() -> None:
    source = inspect.getsource(gate).lower()
    forbidden = (
        "historical_forward",
        "reconstruct_model",
        "automodelforcausallm",
        "selected_downstream_checkpoint.pt",
        "torch.cuda",
        "ttest",
        "scipy",
    )
    for token in forbidden:
        assert token not in source


def test_direct_cli_has_fail_closed_pass_exit_contract() -> None:
    source = inspect.getsource(gate.main)
    assert "manifest[\"result\"] == PASS_RESULT" in source
    assert "return 0" in source
    assert "else 2" in source
