from __future__ import annotations

from collections import Counter

from scripts import (
    materialize_reason_router_gen4_large_correction_validation_cohort as cohort,
)
from scripts import materialize_reason_router_gen4_six_cell_contrast as materializer


def test_frozen_design_constants():
    assert cohort.DISCOVERY_FREEZE_COMMIT == (
        "41f4678d09f7c779a8edc2eec6cc08a9effd9f41"
    )
    assert cohort.ALIGNMENT_SHIFT_ABS_THRESHOLD == 0.11228626366380845
    assert cohort.VALIDATION_PAIR_START == 300
    assert cohort.VALIDATION_PAIR_STOP == 600
    assert cohort.VALIDATION_PAIR_COUNT == 300


def test_expected_validation_pair_ids():
    ids = cohort.expected_validation_pair_ids()
    assert len(ids) == 300
    assert ids[0] == "generated_fact_301"
    assert ids[-1] == "generated_fact_600"
    assert len(set(ids)) == 300


def test_source_cohorts_are_pair_disjoint():
    discovery, validation = cohort.build_source_cohorts()
    d = {row["pair_id"] for row in discovery}
    v = {row["pair_id"] for row in validation}
    assert len(d) == 300
    assert len(v) == 300
    assert d.isdisjoint(v)


def test_source_cohorts_are_claim_disjoint():
    discovery, validation = cohort.build_source_cohorts()
    d = {
        materializer._frozen_statement_renderer(dict(row))
        for row in discovery
    }
    v = {
        materializer._frozen_statement_renderer(dict(row))
        for row in validation
    }
    assert len(v) == 300
    assert d.isdisjoint(v)


def test_materialized_validation_shape():
    _discovery, validation = cohort.build_source_cohorts()
    rows = cohort.materialize_validation_rows(validation)
    assert len(rows) == 1800
    assert len({row["source_pair_id"] for row in rows}) == 300
    assert Counter(row["contrast_cell_id"] for row in rows) == Counter(
        {cell: 300 for cell in materializer.CELL_IDS}
    )


def test_materialized_rows_have_no_outcome_fields():
    _discovery, validation = cohort.build_source_cohorts()
    rows = cohort.materialize_validation_rows(validation)
    for row in rows:
        assert not (materializer.FORBIDDEN_OUTPUT_FIELDS & set(row))


def test_serialization_is_deterministic():
    _discovery, validation = cohort.build_source_cohorts()
    rows1 = cohort.materialize_validation_rows(validation)
    rows2 = cohort.materialize_validation_rows(validation)
    assert cohort.serialize_rows(rows1) == cohort.serialize_rows(rows2)


def test_manifest_freezes_prospective_rule():
    _discovery, validation = cohort.build_source_cohorts()
    rows = cohort.materialize_validation_rows(validation)
    raw = cohort.serialize_rows(rows)
    manifest = cohort.build_manifest(
        provenance={
            "branch": cohort.EXPECTED_BRANCH,
            "head": "x",
            "discovery_freeze_commit": cohort.DISCOVERY_FREEZE_COMMIT,
            "generator_authority_commit": cohort.GENERATOR_AUTHORITY_COMMIT,
            "generator_source_blob": cohort.GENERATOR_SOURCE_BLOB,
            "materializer_source_blob": cohort.MATERIALIZER_SOURCE_BLOB,
        },
        rows_bytes=raw,
        validation_facts=validation,
    )
    assert manifest["scientific_outcomes_observed"] is False
    assert manifest["model_forward_count"] == 0
    assert manifest["tokenizer_invoked"] is False
    assert manifest["alignment_shift_abs_threshold"] == 0.11228626366380845
    assert manifest["prospective_regime_rule"].startswith("LARGE iff")
