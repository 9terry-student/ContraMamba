from __future__ import annotations

import hashlib
import inspect
import json

from scripts import (
    build_reason_router_gen4_averitec_gold_evidence_mamba14b_token_gate
    as gate,
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = list(ids)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        pieces = [p for p in str(text).split() if p]
        return FakeEncoding(range(1, len(pieces) + 1))


def fixture_example(label: str, *, multi: bool = False):
    answers = [{"answer": "first answer", "answer_type": "Extractive"}]
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


def test_frozen_plan_and_mamba14b_identity_constants() -> None:
    assert gate.REQUIRED_PLAN_COMMIT == (
        "f174b092a06b1d0973fa90c90856b2aa23b3526d"
    )
    assert gate.PLAN_SHA256 == (
        "11cae5cce793f3f27a3d8ec449f3dde1deab6aa162e37c97d6deaf9e04d6c845"
    )
    assert gate.SCALE == "mamba14b"
    assert gate.HF_REPO == "state-spaces/mamba-1.4b-hf"
    assert gate.HF_REVISION == (
        "6e46eae61c27280517feef46f536d16b91076f08"
    )
    assert gate.FROZEN_CAUSAL_OBJECT["checkpoint_sha256"] == (
        "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a"
    )
    assert gate.FROZEN_CAUSAL_OBJECT["selected_plane"] == "P5"
    assert gate.FROZEN_CAUSAL_OBJECT["control_plane"] == "P4"
    assert gate.FROZEN_CAUSAL_OBJECT["intervention_layer"] == 35


def test_mamba14b_tokenizer_identity_is_distinct_and_exact() -> None:
    assert gate.TOKENIZER_FILE_SHA256 == {
        "tokenizer.json":
            "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
        "tokenizer_config.json":
            "3ba257483d22a5a84aab5465aa427e59bdaeb55f09fb14349e2d571ff67e8020",
    }
    assert gate.TOKENIZERS_VERSION == "0.22.2"


def test_primary_extension_is_one_sided_less_and_independent() -> None:
    assert gate.PRIMARY_P_VALUE_COUNT_PLANNED == 1
    assert gate.PRIMARY_ALTERNATIVE == "less"
    assert gate.PRIMARY_MULTIPLICITY == "none"
    assert gate.HISTORICAL_AVERITEC_FAMILY_REOPENED is False


def test_pinned_source_and_label_contract() -> None:
    assert gate.UPSTREAM_COMMIT == (
        "7c62d1ec8df3fb560d6efe2b85fa191135636f81"
    )
    assert gate.UPSTREAM_BLOB_SHA1 == (
        "40974243267f395dc583d805d10f043812419249"
    )
    assert gate.UPSTREAM_BYTES == 1785475
    assert gate.EXPECTED_SOURCE_EXAMPLES == 500
    assert gate.EXPECTED_COMPATIBLE_COUNT == 462
    source = fake_source()
    compatible = gate.compatible_examples(source)
    assert len(compatible) == 462
    assert all(
        row["label"] != gate.EXCLUDED_LABEL
        for _, row in compatible
    )


def test_gold_evidence_serialization_is_unchanged() -> None:
    evidence, q_count, a_count = gate.serialize_gold_evidence(
        fixture_example("Supported", multi=True)
    )
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
    assert not (gate.FORBIDDEN_RESPONSE_FIELDS & set(row))


def test_build_rows_is_deterministic_and_response_blind() -> None:
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


def test_manifest_is_cpu_only_zero_inference() -> None:
    rows = gate.build_rows(
        source=fake_source(),
        tokenizer=FakeTokenizer(),
    )
    source_raw = json.dumps(fake_source()).encode()
    cohort_raw = gate.jsonl_bytes(rows)
    manifest = gate.build_manifest(
        source_raw=source_raw,
        rows=rows,
        tokenizer_provenance={"fake": True},
        cohort_sha256=gate.sha256_bytes(cohort_raw),
    )
    assert manifest["result"] == gate.PASS_RESULT
    ext = manifest["primary_external_transfer_extension"]
    assert ext["scale"] == "mamba14b"
    assert ext["alternative"] == "less"
    assert ext["primary_p_value_count_planned"] == 1
    assert ext["multiplicity"] == "none"
    assert ext["historical_130m370m_averitec_family_reopened"] is False
    assert ext["historical_p_values_in_new_family"] == 0
    assert manifest["token_gate"]["pass_count"] == 462
    assert manifest["model_checkpoint_loaded"] is False
    assert manifest["model_forward_count"] == 0
    assert manifest["cuda_executed"] is False
    assert manifest["scientific_inference_executed"] is False
    assert manifest["p_value_count_added"] == 0
    assert manifest["response_fields_present"] is False


def test_git_blob_sha1_matches_git_object_contract() -> None:
    raw = b"hello\n"
    assert gate.git_blob_sha1(raw) == hashlib.sha1(
        b"blob 6\0" + raw
    ).hexdigest()


def test_production_builder_contains_no_model_execution_or_statistics() -> None:
    source = inspect.getsource(gate).lower()
    forbidden = (
        "historical_forward",
        "reconstruct_model",
        "automodelforcausallm",
        "torch.cuda",
        "ttest",
        "scipy",
        "external_transfer_rows.jsonl",
    )
    for token in forbidden:
        assert token not in source


def test_cli_has_fail_closed_pass_exit_contract() -> None:
    source = inspect.getsource(gate.main)
    assert 'manifest["result"] == PASS_RESULT' in source
    assert "return 0" in source
    assert "else 2" in source

def test_cli_requires_explicit_execution_head() -> None:
    parser = gate.parse_args([
        "--expected-head",
        "0123456789abcdef0123456789abcdef01234567",
        "--output-dir",
        "tmp-out",
    ])
    assert parser.expected_head == (
        "0123456789abcdef0123456789abcdef01234567"
    )

