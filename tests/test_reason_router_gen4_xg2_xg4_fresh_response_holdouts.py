from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts import build_reason_router_gen4_generator_family_prevalence_cohorts as calibration
from scripts import build_reason_router_gen4_xg2_xg4_fresh_response_holdouts as holdouts


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_ROOT = Path(
    "data/reason_router_gen4_generator_family_prevalence_v1"
)

EXPECTED_CALIBRATION = {
    "xg2": {
        "source": (
            "0d24fe924a9e30ba7341b515e6cd2bf1164eff441462451317ca5ec3beb33c5e"
        ),
        "rows": (
            "c96ce74bb89dbd374386c27f3a7daa10b5c6630c22711074e0b9a44d59ab5cfa"
        ),
    },
    "xg4": {
        "source": (
            "23f81cb93a5deb7a18c98a6a2f4718b34f4bfd1713bd1fb294e037a714f71fb0"
        ),
        "rows": (
            "9146404eefdbcf26e25b81c65496cbcc459d2eb85c90b5da44db35307de9a347"
        ),
    },
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_frozen_authority_and_ranges():
    assert holdouts.DESIGN_FREEZE_COMMIT == (
        "2074f52d39bf0fca6d63248016ce47c54bff5e06"
    )
    assert holdouts.DESIGN_BLOB == (
        "809884f817a8ff1314d9237aa28d3ad4627943de"
    )
    assert holdouts.CALIBRATION_BUILDER_BLOB == (
        "4505acc99db0627733592694da290a007d281421"
    )
    assert holdouts.FAMILY_KEYS == ("xg2", "xg4")
    assert holdouts.HOLDOUT_FIRST_PAIR == 301
    assert holdouts.HOLDOUT_LAST_PAIR == 600
    assert holdouts.HOLDOUT_PAIR_COUNT == 300
    assert holdouts.EXPECTED_ROW_COUNT == 1800


def test_builder_is_static_only():
    source = inspect.getsource(holdouts)

    for token in (
        "torch.",
        "torch.cuda",
        "AutoTokenizer",
        "load_representative_model",
        "capture_branch(",
        "R_ALIGN",
        "backward(",
        ".train(",
    ):
        assert token not in source


def test_only_frozen_xg2_xg4_family_specs_are_used():
    assert holdouts.family_spec("xg2") is calibration.family_spec("xg2")
    assert holdouts.family_spec("xg4") is calibration.family_spec("xg4")

    with pytest.raises(
        holdouts.FreshResponseHoldoutBuildError,
        match="UNSUPPORTED_FAMILY:xg3",
    ):
        holdouts.family_spec("xg3")


def test_exact_fresh_pair_ids_and_absolute_object_suffixes():
    for key in holdouts.FAMILY_KEYS:
        spec = holdouts.family_spec(key)
        facts = holdouts.build_source_facts(spec)

        assert len(facts) == 300
        assert [fact["pair_id"] for fact in facts] == [
            f"{key}_fact_{i:03d}"
            for i in range(301, 601)
        ]

        assert facts[0]["object"].endswith(
            f"packet {spec.object_tag}301"
        )
        assert facts[-1]["object"].endswith(
            f"packet {spec.object_tag}600"
        )


def test_first_fresh_pair_uses_absolute_schedule_index_300():
    for key in holdouts.FAMILY_KEYS:
        spec = holdouts.family_spec(key)
        fact = holdouts.build_source_facts(spec)[0]

        i = 300
        s = spec.schedule

        assert fact["name"] == calibration._select(
            spec.names,
            i,
            s.name_stride,
            s.name_offset,
        )
        assert fact["alternate_name"] == calibration._select(
            spec.names,
            i,
            s.name_stride,
            s.name_alt_offset,
        )
        assert fact["title"] == calibration._select(
            spec.titles,
            i,
            s.title_stride,
            s.title_offset,
        )
        assert fact["role"] == calibration._select(
            spec.roles,
            i,
            s.role_stride,
            s.role_offset,
        )
        assert fact["location"] == calibration._select(
            spec.locations,
            i,
            s.location_stride,
            s.location_offset,
        )
        assert fact["time"] == calibration._select(
            spec.times,
            i,
            s.time_stride,
            s.time_offset,
        )


def test_fresh_materialization_has_exact_shape_and_no_outcomes():
    for key in holdouts.FAMILY_KEYS:
        spec = holdouts.family_spec(key)
        facts = holdouts.build_source_facts(spec)
        rows = holdouts.materialize_facts(spec, facts)

        assert len(rows) == 1800
        assert len({row["row_id"] for row in rows}) == 1800

        assert [row["contrast_cell_id"] for row in rows[:6]] == list(
            calibration.CELL_IDS
        )

        assert not any(
            calibration.FORBIDDEN_FIELDS & set(fact)
            for fact in facts
        )
        assert not any(
            calibration.FORBIDDEN_FIELDS & set(row)
            for row in rows
        )


def test_original_calibration_cohorts_regenerate_byte_identically():
    for key in holdouts.FAMILY_KEYS:
        spec = calibration.family_spec(key)

        facts = calibration.build_source_facts(spec)
        rows = calibration.materialize_facts(spec, facts)

        source_raw = calibration.jsonl_bytes(facts)
        row_raw = calibration.jsonl_bytes(rows)

        source_path = (
            ROOT
            / CALIBRATION_ROOT
            / key
            / calibration.SOURCE_FILE
        )
        row_path = (
            ROOT
            / CALIBRATION_ROOT
            / key
            / calibration.ROW_FILE
        )

        assert source_path.read_bytes() == source_raw
        assert row_path.read_bytes() == row_raw

        assert sha256(source_raw) == EXPECTED_CALIBRATION[key]["source"]
        assert sha256(row_raw) == EXPECTED_CALIBRATION[key]["rows"]


def test_fresh_holdouts_are_disjoint_from_calibration_pairs_and_strings():
    for key in holdouts.FAMILY_KEYS:
        spec = holdouts.family_spec(key)

        fresh_facts = holdouts.build_source_facts(spec)
        fresh_rows = holdouts.materialize_facts(
            spec,
            fresh_facts,
        )

        calibration_facts = read_jsonl(
            ROOT
            / CALIBRATION_ROOT
            / key
            / calibration.SOURCE_FILE
        )
        calibration_rows = read_jsonl(
            ROOT
            / CALIBRATION_ROOT
            / key
            / calibration.ROW_FILE
        )

        fresh_ids = {
            str(fact["pair_id"])
            for fact in fresh_facts
        }
        calibration_ids = {
            str(fact["pair_id"])
            for fact in calibration_facts
        }

        assert fresh_ids.isdisjoint(calibration_ids)

        fresh_claims = {
            str(row["claim"])
            for row in fresh_rows
        }
        calibration_claims = {
            str(row["claim"])
            for row in calibration_rows
        }

        fresh_evidence = {
            str(row["evidence"])
            for row in fresh_rows
        }
        calibration_evidence = {
            str(row["evidence"])
            for row in calibration_rows
        }

        assert fresh_claims.isdisjoint(calibration_claims)
        assert fresh_evidence.isdisjoint(calibration_evidence)


def test_fresh_regeneration_is_byte_deterministic():
    for key in holdouts.FAMILY_KEYS:
        spec = holdouts.family_spec(key)

        facts_a = holdouts.build_source_facts(spec)
        rows_a = holdouts.materialize_facts(spec, facts_a)

        facts_b = holdouts.build_source_facts(spec)
        rows_b = holdouts.materialize_facts(spec, facts_b)

        assert (
            calibration.jsonl_bytes(facts_a)
            == calibration.jsonl_bytes(facts_b)
        )
        assert (
            calibration.jsonl_bytes(rows_a)
            == calibration.jsonl_bytes(rows_b)
        )


def test_output_collision_fails_closed(tmp_path):
    output = tmp_path / "out"
    output.mkdir()

    with pytest.raises(
        holdouts.FreshResponseHoldoutBuildError,
        match="OUTPUT_DIR_COLLISION",
    ):
        holdouts.write_holdouts(output)


def test_builder_writes_only_prevalidation_source_and_rows(tmp_path):
    output = tmp_path / "out"

    hashes = holdouts.write_holdouts(output)

    assert set(hashes) == {"xg2", "xg4"}
    assert {path.name for path in output.iterdir()} == {
        "xg2",
        "xg4",
    }

    for key in holdouts.FAMILY_KEYS:
        family_dir = output / key

        assert {path.name for path in family_dir.iterdir()} == {
            calibration.SOURCE_FILE,
            calibration.ROW_FILE,
        }

        facts = read_jsonl(
            family_dir / calibration.SOURCE_FILE
        )
        rows = read_jsonl(
            family_dir / calibration.ROW_FILE
        )

        assert len(facts) == 300
        assert len(rows) == 1800


def test_structural_validator_is_static_only():
    from scripts import validate_reason_router_gen4_xg2_xg4_fresh_response_holdouts as validator

    source = inspect.getsource(validator)

    for token in (
        "import torch",
        "torch.cuda",
        "AutoTokenizer",
        "load_representative_model",
        "capture_branch(",
        ".backward(",
        ".train(",
    ):
        assert token not in source


def test_structural_validator_frozen_identities():
    from scripts import validate_reason_router_gen4_xg2_xg4_fresh_response_holdouts as validator

    assert validator.DESIGN_FREEZE_COMMIT == (
        "2074f52d39bf0fca6d63248016ce47c54bff5e06"
    )
    assert validator.DESIGN_BLOB == (
        "809884f817a8ff1314d9237aa28d3ad4627943de"
    )
    assert validator.CALIBRATION_BUILDER_BLOB == (
        "4505acc99db0627733592694da290a007d281421"
    )


def test_structural_validator_manifest_schemas():
    from scripts import validate_reason_router_gen4_xg2_xg4_fresh_response_holdouts as validator

    assert validator.FAMILY_MANIFEST_SCHEMA == (
        "GEN4_XG2_XG4_FRESH_RESPONSE_HOLDOUT_STRUCTURAL_MANIFEST_V1"
    )
    assert validator.CROSS_MANIFEST_SCHEMA == (
        "GEN4_XG2_XG4_FRESH_RESPONSE_HOLDOUT_CROSS_MANIFEST_V1"
    )


def test_structural_validator_calibration_hashes_are_frozen():
    from scripts import validate_reason_router_gen4_xg2_xg4_fresh_response_holdouts as validator

    assert validator.EXPECTED_CALIBRATION["xg2"]["source"] == (
        "0d24fe924a9e30ba7341b515e6cd2bf1164eff441462451317ca5ec3beb33c5e"
    )
    assert validator.EXPECTED_CALIBRATION["xg2"]["rows"] == (
        "c96ce74bb89dbd374386c27f3a7daa10b5c6630c22711074e0b9a44d59ab5cfa"
    )
    assert validator.EXPECTED_CALIBRATION["xg4"]["source"] == (
        "23f81cb93a5deb7a18c98a6a2f4718b34f4bfd1713bd1fb294e037a714f71fb0"
    )
    assert validator.EXPECTED_CALIBRATION["xg4"]["rows"] == (
        "9146404eefdbcf26e25b81c65496cbcc459d2eb85c90b5da44db35307de9a347"
    )


def test_structural_validator_canonical_json_is_deterministic():
    from scripts import validate_reason_router_gen4_xg2_xg4_fresh_response_holdouts as validator

    left = validator.canonical_json_bytes(
        {"b": 2, "a": 1}
    )
    right = validator.canonical_json_bytes(
        {"a": 1, "b": 2}
    )

    assert left == right
    assert left == b'{"a":1,"b":2}\n'