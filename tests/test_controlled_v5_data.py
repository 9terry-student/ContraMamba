from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.build_controlled_v5 import (
    FINAL_LABEL_TO_ID,
    INTERVENTION_TYPES,
    REQUIRED_FIELDS,
    build_seed_records,
    build_controlled_records,
    build_v1_records,
    load_jsonl,
    split_by_pair_id,
    validate_record,
)


def test_controlled_seed_schema_validation() -> None:
    records = build_seed_records()
    assert len(records) == 130
    assert all(REQUIRED_FIELDS <= set(record) for record in records)

    invalid = deepcopy(records[0])
    invalid["evidence"] = ""
    with pytest.raises(ValueError, match="evidence must be a non-empty string"):
        validate_record(invalid)

    invalid = deepcopy(records[0])
    invalid["frame_compatible_label"] = 2
    with pytest.raises(ValueError, match="must be integer 0 or 1"):
        validate_record(invalid)


def test_final_label_mapping_is_stable() -> None:
    assert FINAL_LABEL_TO_ID == {
        "REFUTE": 0,
        "NOT_ENTITLED": 1,
        "SUPPORT": 2,
    }


def test_pair_id_split_has_no_leakage() -> None:
    records = build_v1_records()
    train, dev = split_by_pair_id(records, dev_ratio=0.2, seed=17)
    train_pairs = {record["pair_id"] for record in train}
    dev_pairs = {record["pair_id"] for record in dev}

    assert train_pairs.isdisjoint(dev_pairs)
    assert len(train_pairs) == 24
    assert len(dev_pairs) == 6
    assert len(train) == 312
    assert len(dev) == 78
    assert {record["intervention_type"] for record in train} == INTERVENTION_TYPES
    assert {record["intervention_type"] for record in dev} == INTERVENTION_TYPES


def test_intervention_vocabulary_is_complete() -> None:
    expected = {
        "none",
        "paraphrase",
        "entity_swap",
        "event_swap",
        "time_swap",
        "location_swap",
        "role_swap",
        "title_name_swap",
        "predicate_swap",
        "evidence_deletion",
        "evidence_truncation",
        "irrelevant_evidence",
        "polarity_flip",
    }
    records = build_v1_records()
    assert INTERVENTION_TYPES == expected
    assert {record["intervention_type"] for record in records} == expected


def test_controlled_v1_file_has_complete_groups_and_valid_labels() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "controlled_v5_v1.jsonl"
    records = load_jsonl(path)
    assert len(records) == 390
    pair_ids = {record["pair_id"] for record in records}
    assert len(pair_ids) == 30
    for pair_id in pair_ids:
        interventions = {
            record["intervention_type"]
            for record in records
            if record["pair_id"] == pair_id
        }
        assert interventions == INTERVENTION_TYPES
    assert {record["final_label"] for record in records} <= set(FINAL_LABEL_TO_ID)
    assert {record["polarity_label"] for record in records} <= {
        "NONE", "REFUTE", "SUPPORT"
    }
    assert {record["primary_failure_type"] for record in records} <= {
        "none", "frame", "predicate", "sufficiency", "polarity"
    }


def test_v1_has_no_epistemicbert_human_set_content() -> None:
    records = build_v1_records()
    serialized = json.dumps(records).lower()
    assert "epistemicbert" not in serialized
    assert "219 human" not in serialized


def test_controlled_v2_has_100_complete_pair_groups_and_safe_split() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "controlled_v5_v2.jsonl"
    records = load_jsonl(path)
    generated = build_controlled_records(100)
    assert records == generated
    assert len(records) == 1300

    by_pair: dict[str, list[dict]] = {}
    for record in records:
        by_pair.setdefault(record["pair_id"], []).append(record)
        assert record["final_label"] in {"REFUTE", "NOT_ENTITLED", "SUPPORT"}
        for field in (
            "frame_compatible_label",
            "predicate_covered_label",
            "sufficiency_label",
        ):
            assert type(record[field]) is int
            assert record[field] in (0, 1)
    assert len(by_pair) == 100
    assert all(len(group) == 13 for group in by_pair.values())
    assert all(
        {record["intervention_type"] for record in group} == INTERVENTION_TYPES
        for group in by_pair.values()
    )

    train, dev = split_by_pair_id(records, dev_ratio=0.2, seed=17)
    train_pairs = {record["pair_id"] for record in train}
    dev_pairs = {record["pair_id"] for record in dev}
    assert train_pairs.isdisjoint(dev_pairs)
    assert len(train_pairs) == 80
    assert len(dev_pairs) == 20
    assert {record["intervention_type"] for record in train} == INTERVENTION_TYPES
    assert {record["intervention_type"] for record in dev} == INTERVENTION_TYPES

    serialized = json.dumps(records).lower()
    assert "epistemicbert" not in serialized
    assert "219 human" not in serialized
