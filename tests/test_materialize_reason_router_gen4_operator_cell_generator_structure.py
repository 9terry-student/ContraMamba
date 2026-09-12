from __future__ import annotations

from collections import OrderedDict

import pytest

from scripts.materialize_reason_router_gen4_operator_cell_generator_structure import (
    AUTHORITY_COMMIT,
    OUTPUT_FIELDS,
    SCHEMA_VERSION,
    MaterializationError,
    materialize_jsonl,
    materialize_row,
    materialize_rows,
    operator_spec,
    serialize_materialized_rows,
    validate_authority_commit,
)


def _row(
    intervention_type: str,
    row_id: str = "pair_001__target",
    pair_id: str = "pair_001",
    **extra,
):
    row = {
        "id": row_id,
        "pair_id": pair_id,
        "intervention_type": intervention_type,
    }
    row.update(extra)
    return row


def test_entity_swap_exact_mapping():
    result = materialize_row(_row("entity_swap"))

    assert result["intended_changed_axes"] == ["name"]
    assert result["generator_source_fields"] == {
        "name": "alternate_name",
    }
    assert result["operator_cells"] == [
        "entity_swap:name",
    ]


def test_role_swap_exact_mapping():
    result = materialize_row(_row("role_swap"))

    assert result["intended_changed_axes"] == ["role"]
    assert result["generator_source_fields"] == {
        "role": "alternate_role",
    }
    assert result["operator_cells"] == [
        "role_swap:role",
    ]


def test_title_name_swap_exact_mapping_and_canonical_order():
    result = materialize_row(_row("title_name_swap"))

    assert result["intended_changed_axes"] == [
        "title",
        "name",
    ]
    assert list(result["generator_source_fields"].items()) == [
        ("title", "alternate_title"),
        ("name", "alternate_name"),
    ]
    assert result["operator_cells"] == [
        "title_name_swap:title",
        "title_name_swap:name",
    ]


def test_predicate_swap_exact_mapping():
    result = materialize_row(_row("predicate_swap"))

    assert result["intended_changed_axes"] == ["predicate"]
    assert result["generator_source_fields"] == {
        "predicate": "alternate_predicate",
    }
    assert result["operator_cells"] == [
        "predicate_swap:predicate",
    ]


def test_authority_and_schema_constants_are_exact():
    assert (
        AUTHORITY_COMMIT
        == "91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea"
    )
    assert (
        SCHEMA_VERSION
        == "GEN4_OPERATOR_CELL_GENERATOR_STRUCTURE_V1"
    )


def test_output_field_set_and_order_are_exact():
    result = materialize_row(_row("entity_swap"))

    assert tuple(result.keys()) == OUTPUT_FIELDS
    assert OUTPUT_FIELDS == (
        "schema_version",
        "authority_commit",
        "row_id",
        "pair_id",
        "intervention_type",
        "intended_changed_axes",
        "generator_source_fields",
        "operator_cells",
    )


def test_target_source_order_is_preserved():
    rows = [
        _row(
            "predicate_swap",
            row_id="p3",
            pair_id="pair_3",
        ),
        _row(
            "entity_swap",
            row_id="p1",
            pair_id="pair_1",
        ),
        _row(
            "role_swap",
            row_id="p2",
            pair_id="pair_2",
        ),
    ]

    result = materialize_rows(rows)

    assert [row["row_id"] for row in result] == [
        "p3",
        "p1",
        "p2",
    ]


def test_non_target_interventions_are_excluded_without_identity_access():
    rows = [
        {
            "intervention_type": "none",
            "claim": object(),
            "evidence": object(),
        },
        _row(
            "entity_swap",
            row_id="target",
            pair_id="pair",
        ),
        {
            "intervention_type": "polarity_flip",
            "final_label": object(),
        },
    ]

    result = materialize_rows(rows)

    assert len(result) == 1
    assert result[0]["row_id"] == "target"


def test_duplicate_target_row_id_is_rejected():
    rows = [
        _row(
            "entity_swap",
            row_id="duplicate",
            pair_id="pair_a",
        ),
        _row(
            "role_swap",
            row_id="duplicate",
            pair_id="pair_b",
        ),
    ]

    with pytest.raises(
        MaterializationError,
        match="duplicate target row id",
    ):
        materialize_rows(rows)


@pytest.mark.parametrize(
    "row, message",
    [
        (
            {
                "pair_id": "pair",
                "intervention_type": "entity_swap",
            },
            "missing required field: id",
        ),
        (
            {
                "id": "",
                "pair_id": "pair",
                "intervention_type": "entity_swap",
            },
            "required field must be a non-empty string: id",
        ),
        (
            {
                "id": "row",
                "intervention_type": "entity_swap",
            },
            "missing required field: pair_id",
        ),
        (
            {
                "id": "row",
                "pair_id": "",
                "intervention_type": "entity_swap",
            },
            "required field must be a non-empty string: pair_id",
        ),
        (
            {
                "id": "row",
                "pair_id": "pair",
            },
            "missing required field: intervention_type",
        ),
        (
            {
                "id": "row",
                "pair_id": "pair",
                "intervention_type": 123,
            },
            "required field must be a non-empty string: intervention_type",
        ),
    ],
)
def test_malformed_target_identity_fails_closed(row, message):
    with pytest.raises(MaterializationError, match=message):
        materialize_row(row)


def test_repeated_serialization_is_byte_identical():
    rows = [
        _row(
            "title_name_swap",
            row_id="a",
            pair_id="pa",
        ),
        _row(
            "predicate_swap",
            row_id="b",
            pair_id="pb",
        ),
    ]

    first = materialize_jsonl(rows)
    second = materialize_jsonl(rows)

    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_forbidden_semantic_fields_cannot_change_materialized_output():
    identity = {
        "id": "same_id",
        "pair_id": "same_pair",
        "intervention_type": "entity_swap",
    }

    first = {
        **identity,
        "claim": "completely unrelated claim A",
        "evidence": "alternate_name appears here",
        "final_label": "SUPPORT",
        "frame_compatible_label": 1,
        "predicate_covered_label": 1,
        "sufficiency_label": 1,
        "polarity_label": "SUPPORT",
        "primary_failure_type": "none",
        "observed_changed_axes": ["predicate"],
        "entity_changed": False,
    }

    second = {
        **identity,
        "claim": "contradictory claim B",
        "evidence": "no generator values whatsoever",
        "final_label": "REFUTE",
        "frame_compatible_label": 0,
        "predicate_covered_label": 0,
        "sufficiency_label": 0,
        "polarity_label": "REFUTE",
        "primary_failure_type": "predicate",
        "observed_changed_axes": ["name", "polarity"],
        "entity_changed": True,
    }

    first_output = materialize_jsonl([first])
    second_output = materialize_jsonl([second])

    assert first_output == second_output


def test_different_authority_commit_is_rejected():
    validate_authority_commit(AUTHORITY_COMMIT)

    with pytest.raises(
        MaterializationError,
        match="authority commit is fixed",
    ):
        validate_authority_commit(
            "0000000000000000000000000000000000000000"
        )


def test_unauthorized_operator_has_no_structural_spec():
    with pytest.raises(
        MaterializationError,
        match="unauthorized Gen4 operator",
    ):
        operator_spec("location_swap")


def test_serializer_rejects_extra_or_reordered_output_schema():
    valid = materialize_row(_row("predicate_swap"))

    reordered = OrderedDict()
    reordered["authority_commit"] = valid["authority_commit"]
    reordered["schema_version"] = valid["schema_version"]
    for key in OUTPUT_FIELDS[2:]:
        reordered[key] = valid[key]

    with pytest.raises(
        MaterializationError,
        match="unauthorized schema or field order",
    ):
        serialize_materialized_rows([reordered])

    extra = dict(valid)
    extra["forbidden"] = "value"

    with pytest.raises(
        MaterializationError,
        match="unauthorized schema or field order",
    ):
        serialize_materialized_rows([extra])