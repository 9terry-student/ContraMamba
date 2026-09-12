from __future__ import annotations

import copy
import json
from collections import OrderedDict

import pytest

from scripts.materialize_reason_router_gen4_six_cell_contrast import (
    CELL_IDS,
    CONTRAST_SPECIFICATION_COMMIT,
    GENERATOR_AUTHORITY_COMMIT,
    GENERATOR_SOURCE_BLOB,
    MECHANISM_ID,
    OUTPUT_FIELDS,
    SCHEMA_VERSION,
    MaterializationError,
    cell_spec,
    materialize_fact,
    materialize_facts,
    materialize_jsonl,
    serialize_materialized_rows,
    validate_materialized_rows,
)


def _fact(pair_id: str = "pair_001", **overrides):
    fact = {
        "pair_id": pair_id,
        "title": "Dr",
        "name": "Mira Chen",
        "alternate_title": "Mr",
        "alternate_name": "Jon Bell",
        "role": "director",
        "alternate_role": "auditor",
        "predicate": "approved",
        "alternate_predicate": "reviewed",
        "object": "the Orion project",
        "alternate_object": "the Vega project",
        "time": "Monday",
        "alternate_time": "Tuesday",
        "location": "Seoul",
        "alternate_location": "Busan",
    }
    fact.update(overrides)
    return fact


def test_exact_frozen_constants_and_output_schema():
    assert SCHEMA_VERSION == "GEN4_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1"
    assert MECHANISM_ID == "masked_slot_substitution_v1"
    assert GENERATOR_AUTHORITY_COMMIT == "91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea"
    assert GENERATOR_SOURCE_BLOB == "baee23a9f71333125f4a8735c2c92d20cab7eb4f"
    assert CONTRAST_SPECIFICATION_COMMIT == "0a0da5354782e542520fb5bba146ab1a599d17ef"
    assert OUTPUT_FIELDS == (
        "schema_version",
        "generator_authority_commit",
        "generator_source_blob",
        "contrast_specification_commit",
        "mechanism_id",
        "source_pair_id",
        "row_id",
        "contrast_cell_id",
        "axis_mask",
        "intended_changed_axes",
        "generator_source_fields",
        "claim",
        "evidence",
    )


def test_exact_six_cell_ids_order_masks_axes_and_source_fields():
    assert CELL_IDS == (
        "C0_SHAM",
        "C1_TITLE",
        "C2_NAME",
        "C3_ROLE",
        "C4_PREDICATE",
        "C5_TITLE_NAME",
    )
    expected = {
        "C0_SHAM": ([0, 0, 0, 0], [], {}),
        "C1_TITLE": ([1, 0, 0, 0], ["title"], {"title": "alternate_title"}),
        "C2_NAME": ([0, 1, 0, 0], ["name"], {"name": "alternate_name"}),
        "C3_ROLE": ([0, 0, 1, 0], ["role"], {"role": "alternate_role"}),
        "C4_PREDICATE": (
            [0, 0, 0, 1],
            ["predicate"],
            {"predicate": "alternate_predicate"},
        ),
        "C5_TITLE_NAME": (
            [1, 1, 0, 0],
            ["title", "name"],
            {"title": "alternate_title", "name": "alternate_name"},
        ),
    }
    for cell_id, (mask, axes, fields) in expected.items():
        spec = cell_spec(cell_id)
        assert spec["axis_mask"] == mask
        assert spec["intended_changed_axes"] == axes
        assert spec["generator_source_fields"] == fields


def test_materialized_block_has_exact_schema_order_and_deterministic_row_ids():
    rows = materialize_fact(_fact())
    assert len(rows) == 6
    for row, cell_id in zip(rows, CELL_IDS, strict=True):
        assert tuple(row.keys()) == OUTPUT_FIELDS
        assert row["row_id"] == (
            f"pair_001__masked_slot_substitution_v1__{cell_id.lower()}"
        )
        assert row["contrast_cell_id"] == cell_id


def test_claim_is_invariant_across_all_six_cells():
    rows = materialize_fact(_fact())
    assert len({row["claim"] for row in rows}) == 1
    assert rows[0]["claim"] == (
        "Dr Mira Chen, the director, approved the Orion project in Seoul during Monday."
    )


def test_same_renderer_empty_mask_sham_and_exact_substitutions():
    rows = {row["contrast_cell_id"]: row for row in materialize_fact(_fact())}
    assert rows["C0_SHAM"]["evidence"] == rows["C0_SHAM"]["claim"]
    assert rows["C1_TITLE"]["evidence"] == (
        "Mr Mira Chen, the director, approved the Orion project in Seoul during Monday."
    )
    assert rows["C2_NAME"]["evidence"] == (
        "Dr Jon Bell, the director, approved the Orion project in Seoul during Monday."
    )
    assert rows["C3_ROLE"]["evidence"] == (
        "Dr Mira Chen, the auditor, approved the Orion project in Seoul during Monday."
    )
    assert rows["C4_PREDICATE"]["evidence"] == (
        "Dr Mira Chen, the director, reviewed the Orion project in Seoul during Monday."
    )
    assert rows["C5_TITLE_NAME"]["evidence"] == (
        "Mr Jon Bell, the director, approved the Orion project in Seoul during Monday."
    )


def test_same_renderer_is_called_once_for_claim_and_once_per_cell():
    calls = []

    def renderer(fact, **overrides):
        calls.append(dict(overrides))
        values = {**fact, **overrides}
        return f"{values['title']}|{values['name']}|{values['role']}|{values['predicate']}"

    rows = materialize_fact(_fact(), statement_renderer=renderer)
    assert len(rows) == 6
    assert calls == [
        {},
        {},
        {"title": "Mr"},
        {"name": "Jon Bell"},
        {"role": "auditor"},
        {"predicate": "reviewed"},
        {"title": "Mr", "name": "Jon Bell"},
    ]


def test_object_time_location_and_alternate_values_are_fixed_within_pair():
    rows = materialize_fact(_fact())
    for row in rows:
        assert "the Orion project" in row["evidence"]
        assert "Seoul" in row["evidence"]
        assert "Monday" in row["evidence"]
        assert "the Vega project" not in row["evidence"]
        assert "Busan" not in row["evidence"]
        assert "Tuesday" not in row["evidence"]


def test_source_order_is_preserved_and_each_pair_is_complete():
    rows = materialize_facts([_fact("pair_b"), _fact("pair_a")])
    assert [row["source_pair_id"] for row in rows[:6]] == ["pair_b"] * 6
    assert [row["source_pair_id"] for row in rows[6:]] == ["pair_a"] * 6
    assert [row["contrast_cell_id"] for row in rows[:6]] == list(CELL_IDS)
    assert [row["contrast_cell_id"] for row in rows[6:]] == list(CELL_IDS)


def test_duplicate_source_pair_is_rejected():
    with pytest.raises(MaterializationError, match="duplicate source pair id"):
        materialize_facts([_fact("same"), _fact("same")])


def test_duplicate_generated_row_is_rejected():
    rows = materialize_fact(_fact())
    duplicate = copy.deepcopy(rows)
    duplicate[1]["row_id"] = duplicate[0]["row_id"]
    with pytest.raises(MaterializationError, match="duplicate generated row id"):
        validate_materialized_rows(duplicate)


@pytest.mark.parametrize(
    "field",
    [
        "pair_id",
        "title",
        "name",
        "role",
        "predicate",
        "alternate_title",
        "alternate_name",
        "alternate_role",
        "alternate_predicate",
        "object",
        "time",
        "location",
    ],
)
def test_missing_required_source_field_is_rejected(field):
    fact = _fact()
    del fact[field]
    with pytest.raises(MaterializationError, match="missing required source field"):
        materialize_fact(fact)


@pytest.mark.parametrize("value", [123, [], object()])
def test_non_string_required_source_field_is_rejected(value):
    with pytest.raises(MaterializationError, match="must be a string"):
        materialize_fact(_fact(title=value))


def test_empty_required_source_field_is_rejected():
    with pytest.raises(MaterializationError, match="must be non-empty"):
        materialize_fact(_fact(alternate_name=""))


@pytest.mark.parametrize(
    "field,alternate_field",
    [
        ("title", "alternate_title"),
        ("name", "alternate_name"),
        ("role", "alternate_role"),
        ("predicate", "alternate_predicate"),
    ],
)
def test_original_alternate_equality_is_rejected(field, alternate_field):
    fact = _fact()
    fact[alternate_field] = fact[field]
    with pytest.raises(MaterializationError, match="original and alternate values must differ"):
        materialize_fact(fact)


def test_structural_output_excludes_all_label_and_outcome_fields():
    fact = _fact(
        final_label="SUPPORT",
        primary_failure_type="none",
        predictions=[1.0],
        logits=[99.0],
    )
    rows = materialize_fact(fact)
    forbidden = {
        "final_label",
        "frame_compatible_label",
        "predicate_covered_label",
        "sufficiency_label",
        "polarity_label",
        "primary_failure_type",
        "predictions",
        "logits",
        "probabilities",
    }
    for row in rows:
        assert not (forbidden & set(row))


def test_repeated_serialization_is_byte_identical_compact_utf8_and_trailing_lf():
    facts = [_fact("pair_α"), _fact("pair_β", name="José")]
    first = materialize_jsonl(facts)
    second = materialize_jsonl(facts)
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")
    assert first.endswith("\n")
    assert "José" in first
    assert "\\u00e9" not in first
    parsed = [json.loads(line) for line in first.splitlines()]
    assert len(parsed) == 12
    assert first.splitlines()[0].startswith('{"schema_version":')
    assert ": " not in first.splitlines()[0]


def test_serializer_rejects_extra_or_reordered_schema():
    valid = materialize_fact(_fact())
    reordered = copy.deepcopy(valid)
    first = reordered[0]
    bad = OrderedDict()
    bad["mechanism_id"] = first["mechanism_id"]
    for key in OUTPUT_FIELDS:
        if key != "mechanism_id":
            bad[key] = first[key]
    reordered[0] = bad
    with pytest.raises(MaterializationError, match="unauthorized schema or field order"):
        serialize_materialized_rows(reordered)

    extra = copy.deepcopy(valid)
    extra[0]["forbidden"] = "x"
    with pytest.raises(MaterializationError, match="unauthorized schema or field order"):
        serialize_materialized_rows(extra)


def test_incomplete_block_is_rejected():
    rows = materialize_fact(_fact())[:-1]
    with pytest.raises(MaterializationError, match="incomplete six-cell block"):
        validate_materialized_rows(rows)


def test_cell_order_violation_is_rejected():
    rows = materialize_fact(_fact())
    rows[1], rows[2] = rows[2], rows[1]
    with pytest.raises(MaterializationError, match="cell ordering or membership violation"):
        validate_materialized_rows(rows)


def test_rendered_text_does_not_define_structural_identity():
    first = materialize_fact(_fact(object="alternate_name C4_PREDICATE"))
    second = materialize_fact(_fact(object="unrelated rendered phrase"))
    structural_fields = OUTPUT_FIELDS[:11]
    assert [tuple(row[field] if not isinstance(row[field], (list, dict)) else repr(row[field]) for field in structural_fields) for row in first] == [
        tuple(row[field] if not isinstance(row[field], (list, dict)) else repr(row[field]) for field in structural_fields) for row in second
    ]


def test_unknown_cell_has_no_structural_spec():
    with pytest.raises(MaterializationError, match="unknown contrast cell"):
        cell_spec("C6_UNKNOWN")
