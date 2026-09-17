from __future__ import annotations

import inspect

from scripts import (
    build_reason_router_gen4_xg1_cross_generator_cohort
    as base
)
from scripts import (
    build_reason_router_gen4_xg1_fresh_specificity_cohort
    as fresh
)
from scripts import (
    prepare_reason_router_gen4_pp3_pp5_fresh_xg1_specificity
    as prep
)
from scripts import (
    validate_reason_router_gen4_xg1_fresh_specificity_cohort
    as validator
)


def test_design_and_population_are_frozen():
    assert fresh.DESIGN_FREEZE_COMMIT == (
        "0bc49ab95cbb2c8735b4bc79422d660fa64e3e01"
    )
    assert fresh.FRESH_START == 301
    assert fresh.FRESH_END == 600
    assert fresh.SOURCE_PAIR_COUNT == 300
    assert fresh.EXPECTED_ROW_COUNT == 1800


def test_original_001_300_regenerates_byte_exact():
    result = fresh.prove_original_byte_identity()
    assert result["source_sha256"] == (
        "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0"
    )
    assert result["row_sha256"] == (
        "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f"
    )


def test_fresh_population_is_exact_301_600():
    facts = fresh.build_source_facts()

    assert len(facts) == 300
    assert [fact["pair_id"] for fact in facts] == [
        f"xg1_fact_{index:03d}"
        for index in range(301, 601)
    ]

    rows = fresh.materialize_facts(facts)
    assert len(rows) == 1800
    assert len({row["row_id"] for row in rows}) == 1800

    assert [
        rows[offset]["source_pair_id"]
        for offset in range(0, 1800, 6)
    ] == fresh.expected_pair_ids()


def test_fresh_has_zero_original_claim_and_evidence_overlap():
    old_facts = fresh.build_source_facts(1, 300)
    old_rows = fresh.materialize_facts(old_facts)

    new_facts = fresh.build_source_facts(301, 600)
    new_rows = fresh.materialize_facts(new_facts)

    old_claims = {row["claim"] for row in old_rows}
    old_evidence = {row["evidence"] for row in old_rows}
    new_claims = {row["claim"] for row in new_rows}
    new_evidence = {row["evidence"] for row in new_rows}

    assert not (new_claims & old_claims)
    assert not (new_evidence & old_evidence)


def test_fresh_materialization_preserves_six_cell_contract():
    facts = fresh.build_source_facts()
    rows = fresh.materialize_facts(facts)

    for offset in range(0, len(rows), 6):
        block = rows[offset : offset + 6]
        assert [
            row["contrast_cell_id"]
            for row in block
        ] == list(base.CELL_IDS)
        assert len({row["claim"] for row in block}) == 1


def test_fresh_regeneration_is_byte_deterministic():
    facts_a = fresh.build_source_facts()
    rows_a = fresh.materialize_facts(facts_a)

    facts_b = fresh.build_source_facts()
    rows_b = fresh.materialize_facts(facts_b)

    assert base.jsonl_bytes(facts_a) == base.jsonl_bytes(facts_b)
    assert base.jsonl_bytes(rows_a) == base.jsonl_bytes(rows_b)


def test_static_scripts_have_no_model_execution_surface():
    for module in (fresh, validator, prep):
        source = inspect.getsource(module)
        for token in (
            "AutoTokenizer",
            "AutoModel",
            "load_representative_model",
            "capture_branch(",
            "run_full(",
            "torch.cuda",
            ".cuda(",
        ):
            assert token not in source


def test_pp3_reconstruction_is_exact_and_pp5_is_frozen_geometry():
    geometry = prep.reconstruct_geometry()

    assert geometry["pp3_plus_sha256"] == prep.PP3_PLUS_SHA
    assert geometry["pp3_minus_sha256"] == prep.PP3_MINUS_SHA

    assert geometry["c3"] == prep.C3
    assert geometry["s3"] == prep.S3
    assert geometry["c5"] == prep.C5
    assert geometry["s5"] == prep.S5

    metrics = geometry["pp5_metrics"]

    assert abs(metrics["plus_norm"] - 1.0) <= 1.0e-12
    assert abs(metrics["minus_norm"] - 1.0) <= 1.0e-12
    assert abs(metrics["plus_minus_dot"]) <= 1.0e-12

    assert len(geometry["pp5_plus_raw"]) == 395 * 8
    assert len(geometry["pp5_minus_raw"]) == 395 * 8


def test_validator_schema_is_structural_only():
    assert validator.MANIFEST_SCHEMA == (
        "GEN4_XG1_FRESH_SPECIFICITY_STRUCTURAL_MANIFEST_V1"
    )
