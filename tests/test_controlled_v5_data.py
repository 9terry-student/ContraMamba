from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.build_controlled_v5 import (
    FINAL_LABEL_TO_ID,
    INTERVENTION_TYPES,
    REQUIRED_FIELDS,
    build_seed_records,
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
    records = build_seed_records()
    train, dev = split_by_pair_id(records, dev_ratio=0.2, seed=17)
    train_pairs = {record["pair_id"] for record in train}
    dev_pairs = {record["pair_id"] for record in dev}

    assert train_pairs.isdisjoint(dev_pairs)
    assert len(train_pairs) == 8
    assert len(dev_pairs) == 2
    assert len(train) == 104
    assert len(dev) == 26


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
    records = build_seed_records()
    assert INTERVENTION_TYPES == expected
    assert {record["intervention_type"] for record in records} == expected

