import json
import random

import pytest

from scripts import build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar as builder


def row(row_id, pair_id, intervention_type="none", frame=1, predicate=1, sufficiency=1):
    return {
        "id": row_id,
        "pair_id": pair_id,
        "claim": "claim",
        "evidence": "evidence",
        "final_label": "SUPPORT" if frame and predicate and sufficiency else "NOT_ENTITLED",
        "frame_compatible_label": frame,
        "predicate_covered_label": predicate,
        "sufficiency_label": sufficiency,
        "polarity_label": "SUPPORT" if frame and predicate and sufficiency else "NONE",
        "primary_failure_type": (
            "frame" if not frame else "predicate" if not predicate else "sufficiency" if not sufficiency else "none"
        ),
        "intervention_type": intervention_type,
    }


def test_deterministic_split_is_pair_level_and_seeded():
    rows = [row(f"p{i}__none", f"p{i}") for i in range(10)]
    observed = builder.deterministic_pair_split(rows, seed=174, dev_ratio=0.2)
    pair_ids = sorted({item["pair_id"] for item in rows})
    shuffled = list(pair_ids)
    random.Random(174).shuffle(shuffled)
    expected_dev = set(shuffled[:2])
    assert observed == {pair_id: "dev" if pair_id in expected_dev else "train" for pair_id in pair_ids}


def test_canonical_mapping_requires_unique_same_pair_none_self_anchor():
    rows = [
        row("p1__none", "p1", "none"),
        row("p1__paraphrase", "p1", "paraphrase"),
        row("p2__none", "p2", "none"),
    ]
    split = {"p1": "train", "p2": "dev"}
    canonical = builder.canonical_row_ids(rows, split)
    assert canonical["p1__none"] == "p1__none"
    assert canonical["p1__paraphrase"] == "p1__none"
    builder.validate_canonical_lineage(rows, split, canonical)

    duplicate_none = rows + [row("p1__none2", "p1", "none")]
    with pytest.raises(builder.BuildBlocked):
        builder.canonical_row_ids(duplicate_none, split)


def test_semantic_sidecar_hash_excludes_created_at_only():
    rows_a = [{"row_id": "a", "created_at": "one", "status": "PASS", "hash_field": "kept"}]
    rows_b = [{"row_id": "a", "created_at": "two", "status": "PASS", "hash_field": "kept"}]
    rows_c = [{"row_id": "a", "created_at": "one", "status": "FAIL", "hash_field": "kept"}]
    assert builder.semantic_sidecar_sha256(rows_a) == builder.semantic_sidecar_sha256(rows_b)
    assert builder.semantic_sidecar_sha256(rows_a) != builder.semantic_sidecar_sha256(rows_c)


def test_exact_binary_rejects_bool_and_non_binary_integer():
    assert builder.exact_binary({"x": 1}, "x", "r1") == 1
    with pytest.raises(builder.BuildBlocked):
        builder.exact_binary({"x": True}, "x", "r1")
    with pytest.raises(builder.BuildBlocked):
        builder.exact_binary({"x": 2}, "x", "r1")


def test_duplicate_row_id_rejected():
    rows = [row("dup", "p1"), row("dup", "p2")]
    with pytest.raises(builder.BuildBlocked):
        builder.validate_source_rows(rows)


def test_fail_closed_invalid_source_identity(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row("p1__none", "p1")) + "\n", encoding="utf-8")
    monkeypatch.setattr(builder, "SOURCE_DATASET_PATH", source.name)
    monkeypatch.setattr(builder, "SOURCE_DATASET_SHA256", "wrong")
    with pytest.raises(builder.BuildBlocked):
        builder.validate_source_dataset(tmp_path, source)


def test_reason_derivation_order_and_expected_primary():
    assert builder.primary_reason_from_axes(0, 0, 0) == "FRAME"
    assert builder.primary_reason_from_axes(1, 0, 0) == "PREDICATE"
    assert builder.primary_reason_from_axes(1, 1, 0) == "SUFFICIENCY"
    assert builder.primary_reason_from_axes(1, 1, 1) == "AUTHORIZED"
    assert builder.expected_primary_from_record({"primary_failure_type": "polarity"}) == "AUTHORIZED"


def test_positive_margin_eligibility_contract():
    sidecar = {
        "integrity_status": "ELIGIBLE",
        "split": "train",
        "frame_compatible_label": 1,
        "time_swap_status": "PASS",
        "dataset_source_status": "PASS",
    }
    assert builder.positive_margin_eligible(sidecar) is True
    assert builder.positive_margin_eligible({**sidecar, "split": "dev"}) is False
    assert builder.positive_margin_eligible({**sidecar, "frame_compatible_label": 0}) is False


def test_compact_jsonl_serialization_lf_no_bom_and_final_newline():
    payload = builder.compact_jsonl_bytes([{"b": 2, "a": 1}])
    assert payload == b'{"a":1,"b":2}\n'
    assert payload.endswith(b"\n")
    assert not payload.startswith(b"\xef\xbb\xbf")


def test_finalize_fails_closed_when_output_dir_preexists_with_unrelated_contents(tmp_path):
    output_dir = tmp_path / "canonical_output"
    unrelated_dir = output_dir / "unrelated_dir"
    unrelated_file = unrelated_dir / "user_data.txt"
    unrelated_dir.mkdir(parents=True)
    unrelated_file.write_text("do not touch\n", encoding="utf-8")
    payloads = {
        builder.SIDECAR_NAME: b'{"row_id":"attempt"}\n',
        builder.PROVENANCE_NAME: b'{"attempt":true}\n',
    }

    with pytest.raises(builder.BuildBlocked):
        builder.finalize_payloads_atomic(output_dir, payloads)

    assert unrelated_dir.is_dir()
    assert unrelated_file.read_text(encoding="utf-8") == "do not touch\n"
    assert not (output_dir / builder.SIDECAR_NAME).exists()
    assert not (output_dir / builder.PROVENANCE_NAME).exists()
    assert not list(tmp_path.glob(".canonical_output.p4l-backup-*"))


def test_finalize_fails_closed_when_output_dir_preexists_empty(tmp_path):
    output_dir = tmp_path / "canonical_output"
    output_dir.mkdir()
    payloads = {
        builder.SIDECAR_NAME: b'{"row_id":"attempt"}\n',
        builder.PROVENANCE_NAME: b'{"attempt":true}\n',
    }

    with pytest.raises(builder.BuildBlocked):
        builder.finalize_payloads_atomic(output_dir, payloads)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_finalize_fails_closed_when_output_path_preexists_as_file(tmp_path):
    output_path = tmp_path / "canonical_output"
    output_path.write_text("existing file\n", encoding="utf-8")
    payloads = {
        builder.SIDECAR_NAME: b'{"row_id":"attempt"}\n',
        builder.PROVENANCE_NAME: b'{"attempt":true}\n',
    }

    with pytest.raises(builder.BuildBlocked):
        builder.finalize_payloads_atomic(output_path, payloads)

    assert output_path.read_text(encoding="utf-8") == "existing file\n"
