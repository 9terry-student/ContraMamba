from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import (
    build_reason_router_gen4_averitec_train_fresh_steering_token_gate
    as subject,
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = list(ids)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        pieces = [p for p in str(text).split() if p]
        return FakeEncoding(range(1, len(pieces) + 1))


def example(
    label: str,
    claim: str,
    *,
    url: str | None = None,
):
    return {
        "claim": claim,
        "label": label,
        "original_claim_url": url,
        "questions": [
            {
                "question": "What happened?",
                "answers": [
                    {"answer": "A sufficiently long evidence answer"}
                ],
            }
        ],
    }


def test_frozen_constants():
    assert subject.REQUIRED_ANCESTOR == (
        "ead82711af027589af8ede257519391a6283dcf7"
    )
    assert subject.DESIGN_ARTIFACT_GIT_BLOB == (
        "0d35f9aebb37e6203bea40304d3a55ab2361b684"
    )
    assert subject.SOURCE_ELIGIBILITY_CORRECTION_GIT_BLOB == (
        "3a55da349d7c9780f8e51de76e3fd9abd67531aa"
    )
    assert subject.BASE_GATE_GIT_BLOB == (
        "0f480164695b6bc4c4206575e1767926fbb71cf0"
    )

    assert subject.UPSTREAM_COMMIT == (
        "7c62d1ec8df3fb560d6efe2b85fa191135636f81"
    )
    assert subject.TRAIN_BLOB_SHA1 == (
        "0f190e115cf2ee23416e8a539c8d6ac043d7cc83"
    )
    assert subject.TRAIN_BYTES == 10184813
    assert subject.DEV_BLOB_SHA1 == (
        "40974243267f395dc583d805d10f043812419249"
    )
    assert subject.DEV_BYTES == 1785475

    assert subject.EXPECTED_TRAIN_ROWS == 3068
    assert subject.EXPECTED_COMPATIBLE_BEFORE_FRESHNESS == 2873
    assert subject.EXPECTED_FINAL_COUNT == 2799
    assert subject.EXPECTED_EXCLUSION_COUNTS == {
        "incompatible_label": 195,
        "empty_normalized_claim": 1,
        "dev_claim_overlap": 6,
        "dev_url_overlap": 1,
        "later_within_train_duplicate_claim": 66,
    }
    assert subject.EXPECTED_FINAL_LABEL_COUNTS == {
        "Refuted": 1727,
        "Supported": 805,
        "Not Enough Evidence": 267,
    }


def test_normalize_claim_exact_contract():
    assert subject.normalize_claim(
        "  A\tCLAIM\nWith   Spaces  "
    ) == "a claim with spaces"

    assert subject.normalize_claim(
        "Straße"
    ) == "straße"

    assert subject.normalize_claim(
        "  \t\n  "
    ) == ""


def test_exact_nonempty_url_does_not_strip_or_normalize():
    assert subject.exact_nonempty_url(
        {"original_claim_url": ""}
    ) is None
    assert subject.exact_nonempty_url(
        {"original_claim_url": None}
    ) is None
    assert subject.exact_nonempty_url(
        {"original_claim_url": " https://x "}
    ) == " https://x "


def test_train_source_allows_empty_question_string_without_row_exclusion():
    row = example(
        "Not Enough Evidence",
        "fresh claim with pinned empty question",
        url="https://example.test/438",
    )
    row["questions"][0]["question"] = ""

    # Source validation must preserve the pinned train row rather than
    # invent a post-design exclusion.
    subject._validate_example_structure(
        row,
        index=438,
        label_prefix="TRAIN",
    )

    gated = subject.gate_train_row(
        tokenizer=FakeTokenizer(),
        source_index=438,
        example=row,
    )

    assert gated["averitec_train_index"] == 438
    assert gated["source_label"] == "Not Enough Evidence"
    assert gated["correct_label"] == "NOT_ENTITLED"
    assert gated["token_gate_pass"] is True
    assert gated["token_gate_reasons"] == []
    assert gated["evidence"].startswith(
        "Question: \nAnswer: "
    )

    # The correction relaxes only emptiness, not field type.
    bad = example(
        "Not Enough Evidence",
        "bad non-string question",
    )
    bad["questions"][0]["question"] = None

    with pytest.raises(
        subject.SteeringCohortError,
        match="TRAIN_QUESTION_TEXT:438:0",
    ):
        subject._validate_example_structure(
            bad,
            index=438,
            label_prefix="TRAIN",
        )


def test_empty_normalized_claim_is_prospectively_excluded(monkeypatch):
    # The pinned train row 1948 is structurally representable as source
    # data but is not eligible for the frozen claim/evidence boundary.
    empty = example(
        "Supported",
        "   \t ",
        url="https://empty-claim.test",
    )

    subject._validate_example_structure(
        empty,
        index=1948,
        label_prefix="TRAIN",
    )

    dev = [
        example(
            "Refuted",
            f"dev filler {i}",
        )
        for i in range(500)
    ]
    train = [
        empty,
        example("Refuted", "kept refuted"),
        example("Supported", "kept supported"),
    ]

    monkeypatch.setattr(subject, "EXPECTED_TRAIN_ROWS", 3)
    monkeypatch.setattr(
        subject,
        "EXPECTED_COMPATIBLE_BEFORE_FRESHNESS",
        3,
    )
    monkeypatch.setattr(subject, "EXPECTED_FINAL_COUNT", 2)
    monkeypatch.setattr(
        subject,
        "EXPECTED_EXCLUSION_COUNTS",
        {"empty_normalized_claim": 1},
    )
    monkeypatch.setattr(
        subject,
        "EXPECTED_FINAL_LABEL_COUNTS",
        {
            "Refuted": 1,
            "Supported": 1,
        },
    )

    kept, audit = subject.derive_fresh_examples(
        train=train,
        dev=dev,
    )

    assert [index for index, _ in kept] == [1, 2]
    assert audit["exclusion_counts"] == {
        "empty_normalized_claim": 1,
    }
    assert audit["final_count"] == 2


def test_git_blob_sha1_contract():
    raw = b"hello\n"
    expected = hashlib.sha1(
        b"blob 6\0" + raw
    ).hexdigest()
    assert subject.git_blob_sha1(raw) == expected


def test_freshness_order_claim_then_url_then_within_train_duplicate(monkeypatch):
    # Patch expected counts only for this focused synthetic derivation.
    monkeypatch.setattr(subject, "EXPECTED_TRAIN_ROWS", 7)
    monkeypatch.setattr(subject, "EXPECTED_COMPATIBLE_BEFORE_FRESHNESS", 6)
    monkeypatch.setattr(subject, "EXPECTED_FINAL_COUNT", 2)
    monkeypatch.setattr(
        subject,
        "EXPECTED_EXCLUSION_COUNTS",
        {
            "incompatible_label": 1,
            "dev_claim_overlap": 1,
            "dev_url_overlap": 1,
            "later_within_train_duplicate_claim": 2,
        },
    )
    monkeypatch.setattr(
        subject,
        "EXPECTED_FINAL_LABEL_COUNTS",
        {
            "Refuted": 1,
            "Supported": 1,
        },
    )

    dev = [
        example(
            "Refuted",
            "same claim",
            url="https://dev-claim",
        ),
        example(
            "Supported",
            "different dev claim",
            url="https://dev-url",
        ),
    ]
    # derive_fresh_examples only needs len(dev)==500 by production contract.
    dev.extend(
        example(
            "Refuted",
            f"filler dev {i}",
            url=None,
        )
        for i in range(498)
    )

    train = [
        example(
            "Conflicting Evidence/Cherrypicking",
            "incompatible",
        ),
        # Claim overlap wins even though URL also overlaps.
        example(
            "Refuted",
            "  SAME   CLAIM ",
            url="https://dev-url",
        ),
        # URL overlap is counted only after claim-overlap removal.
        example(
            "Supported",
            "unique url overlap",
            url="https://dev-url",
        ),
        # Lowest source index retained.
        example("Refuted", "duplicate train"),
        example("Supported", " DUPLICATE\tTRAIN "),
        example("Supported", "fresh keep"),
        # Another later duplicate of fresh keep.
        example("Supported", "fresh   keep"),
    ]

    kept, audit = subject.derive_fresh_examples(
        train=train,
        dev=dev,
    )

    assert [index for index, _ in kept] == [3, 5]
    assert audit["exclusion_counts"] == {
        "incompatible_label": 1,
        "dev_claim_overlap": 1,
        "dev_url_overlap": 1,
        "later_within_train_duplicate_claim": 2,
    }


def test_train_row_reuses_frozen_serializer_and_anchor_contract():
    row = subject.gate_train_row(
        tokenizer=FakeTokenizer(),
        source_index=42,
        example=example(
            "Refuted",
            "A natural language claim",
            url="https://example.test",
        ),
    )

    assert row["schema_version"] == subject.COHORT_SCHEMA
    assert row["averitec_train_index"] == 42
    assert row["example_id"] == "averitec_train_0042"
    assert row["correct_label"] == "REFUTE"
    assert row["correct_label_id"] == 0
    assert row["anchor_name"] == "A_CLAIM_EVIDENCE_BOUNDARY"
    assert row["target_offset"] == 2
    assert row["target_intervention_token_index"] == (
        row["absolute_anchor_token_index"] + 2
    )
    assert row["token_gate_pass"] is True
    assert not (subject.FORBIDDEN_RESPONSE_FIELDS & set(row))


def test_build_rows_deterministic_source_order_and_response_blind():
    fresh = [
        (
            2,
            example(
                "Refuted",
                "claim two",
                url="https://2",
            ),
        ),
        (
            7,
            example(
                "Supported",
                "claim seven",
                url="https://7",
            ),
        ),
    ]

    # Focused unit test patches only final-size expectation.
    original = subject.EXPECTED_FINAL_COUNT
    try:
        subject.EXPECTED_FINAL_COUNT = 2
        a = subject.build_rows(
            fresh_examples=fresh,
            tokenizer=FakeTokenizer(),
        )
        b = subject.build_rows(
            fresh_examples=fresh,
            tokenizer=FakeTokenizer(),
        )
    finally:
        subject.EXPECTED_FINAL_COUNT = original

    assert subject.jsonl_bytes(a) == subject.jsonl_bytes(b)
    assert [row["averitec_train_index"] for row in a] == [2, 7]
    assert all(row["token_gate_pass"] for row in a)
    assert all(
        not (subject.FORBIDDEN_RESPONSE_FIELDS & set(row))
        for row in a
    )


def test_manifest_is_cpu_static_and_does_not_authorize_scientific_execution(monkeypatch):
    rows = [
        subject.gate_train_row(
            tokenizer=FakeTokenizer(),
            source_index=i,
            example=example(
                "Refuted" if i == 0 else "Supported",
                f"claim {i}",
            ),
        )
        for i in range(2)
    ]

    monkeypatch.setattr(subject, "EXPECTED_FINAL_COUNT", 2)
    monkeypatch.setattr(
        subject,
        "EXPECTED_FINAL_LABEL_COUNTS",
        {"Refuted": 1, "Supported": 1},
    )

    manifest = subject.build_manifest(
        train_raw=b"train",
        dev_raw=b"dev",
        rows=rows,
        freshness_audit={"synthetic": True},
        tokenizer_provenance={"fake": True},
        cohort_sha256=subject.sha256_bytes(
            subject.jsonl_bytes(rows)
        ),
    )

    assert manifest["result"] == subject.PASS_RESULT
    assert manifest["model_checkpoint_loaded"] is False
    assert manifest["model_forward_count"] == 0
    assert manifest["cuda_executed"] is False
    assert manifest["training_executed"] is False
    assert manifest["backward_executed"] is False
    assert manifest["scientific_inference_executed"] is False
    assert manifest["p_value_count_added"] == 0
    assert manifest["response_fields_present"] is False
    assert manifest["planned_steering_execution"] == {
        "model_scale": "mamba370m",
        "conditions": [
            "native",
            "p3_mirror_steer",
            "p5_matched_control",
        ],
        "expected_row_count": 2,
        "expected_full_model_forward_count": 8397,
        "scientific_execution_authorized_by_this_artifact": False,
    }


def test_production_source_has_no_model_or_statistical_execution_path():
    source = inspect.getsource(subject).lower()

    forbidden = (
        "automodelfor",
        "selected_downstream_checkpoint.pt",
        "torch.cuda",
        "optimizer",
        ".backward(",
        "scipy",
        "ttest",
        "mcnemar",
        "binomtest",
    )
    for token in forbidden:
        assert token not in source


def test_direct_cli_help():
    repo_root = Path(subject.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / (
            "build_reason_router_gen4_averitec_train_fresh_"
            "steering_token_gate.py"
        )
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output-dir" in completed.stdout
    assert "--expected-head" in completed.stdout
    assert "--tokenizer-snapshot" in completed.stdout
