from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_controlled_v5 import (
    FINAL_LABEL_TO_ID,
    INTERVENTION_TYPES,
    REQUIRED_FIELDS,
    build_controlled_records,
    load_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "data" / "controlled_v5_v3.jsonl"
BINARY_FIELDS = (
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
)


def test_controlled_v5_v3_has_300_complete_pair_groups() -> None:
    assert V3_PATH.exists(), f"missing Stage 7A dataset: {V3_PATH}"
    records = load_jsonl(V3_PATH)
    assert len(records) == 3900, "v3 must contain exactly 300 x 13 records"

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["pair_id"]].append(record)
    assert len(groups) == 300, "v3 must contain exactly 300 unique pair_id groups"

    for pair_id, group in groups.items():
        assert len(group) == 13, f"pair_id {pair_id!r} does not contain 13 rows"
        interventions = {record["intervention_type"] for record in group}
        assert interventions == INTERVENTION_TYPES, (
            f"pair_id {pair_id!r} has malformed intervention coverage: "
            f"{sorted(interventions)}"
        )


def test_controlled_v5_v3_schema_ids_and_labels_are_valid() -> None:
    records = load_jsonl(V3_PATH)
    ids = [record["id"] for record in records]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    assert not duplicates, f"duplicate id values found: {duplicates[:5]}"

    for row_number, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - set(record)
        assert not missing, f"row {row_number} is missing fields: {sorted(missing)}"
        assert record["final_label"] in FINAL_LABEL_TO_ID, (
            f"row {row_number} has invalid final_label: {record['final_label']!r}"
        )
        assert record["polarity_label"] in {"NONE", "REFUTE", "SUPPORT"}
        assert record["primary_failure_type"] in {
            "none", "frame", "predicate", "sufficiency", "polarity"
        }
        for field in BINARY_FIELDS:
            assert type(record[field]) is int and record[field] in (0, 1), (
                f"row {row_number} field {field!r} must be binary"
            )

    assert FINAL_LABEL_TO_ID == {
        "REFUTE": 0,
        "NOT_ENTITLED": 1,
        "SUPPORT": 2,
    }
    assert {record["final_label"] for record in records} == set(FINAL_LABEL_TO_ID)
    assert records == build_controlled_records(300), (
        "v3 does not match the deterministic controlled intervention generator"
    )
