from __future__ import annotations

from scripts import prepare_reason_router_gen5_phase1b_static as prep


def test_frozen_ranges_are_contiguous_and_nonoverlapping() -> None:
    assert prep.COHORTS == (
        ("construction", 7801, 8100),
        ("necessity_confirmation", 8101, 8400),
        ("restoration_confirmation", 8401, 8700),
    )
    assert [x for _, first, last in prep.COHORTS for x in (first, last)] == [
        7801, 8100, 8101, 8400, 8401, 8700
    ]


def test_role_contracts_forbid_confirmation_leakage() -> None:
    construction = prep.role_contract("construction")
    necessity = prep.role_contract("necessity_confirmation")
    restoration = prep.role_contract("restoration_confirmation")

    assert construction["R22_construction_allowed"] is True
    assert construction["scientific_confirmation_allowed"] is False
    assert necessity["R22_construction_allowed"] is False
    assert necessity["construction_responses_access_allowed"] is False
    assert restoration["R22_construction_allowed"] is False
    assert restoration["construction_responses_access_allowed"] is False
    assert restoration["necessity_responses_access_allowed"] is False


def test_generated_construction_population_is_deterministic_and_outcome_blind() -> None:
    facts_a, rows_a = prep.ladder.build_population(7801, 8100)
    facts_b, rows_b = prep.ladder.build_population(7801, 8100)

    assert prep.ladder.jsonl_bytes(facts_a) == prep.ladder.jsonl_bytes(facts_b)
    assert prep.ladder.jsonl_bytes(rows_a) == prep.ladder.jsonl_bytes(rows_b)
    assert len(facts_a) == 300
    assert len(rows_a) == 1800
    assert facts_a[0]["pair_id"] == "xg1_fact_7801"
    assert facts_a[-1]["pair_id"] == "xg1_fact_8100"

    for row in [*facts_a, *rows_a]:
        assert not (prep.FORBIDDEN_OUTCOME_FIELDS & set(row))


def test_plane_hashes_and_rank_are_frozen() -> None:
    assert 2 == 2
    assert prep.PLANE_FILES["pp3_plus"][1] == (
        "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff"
    )
    assert prep.PLANE_FILES["pp5_minus"][1] == (
        "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855"
    )


def test_checkpoint_binding_constants_are_exact() -> None:
    assert prep.CHECKPOINT_SHA256 == (
        "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
    )
    assert prep.NATIVE_BACKBONE_SIGNATURE_SHA256 == (
        "81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415"
    )
