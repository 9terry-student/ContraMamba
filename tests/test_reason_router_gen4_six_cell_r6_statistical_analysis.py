from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_six_cell_r6_statistical_analysis as R,
)


CELL_EFFECTS = {
    "C0_SHAM": 0.00,
    "C1_TITLE": 0.10,
    "C2_NAME": 0.20,
    "C3_ROLE": -0.05,
    "C4_PREDICATE": 0.30,
    "C5_TITLE_NAME": 0.37,
}


def synthetic_records(
    pair_count: int = R.PRIMARY_PAIR_COUNT,
):
    records = []

    evaluator_offsets = [
        (index - 8.5) * 0.001
        for index in range(R.EVALUATOR_COUNT)
    ]

    assert math.isclose(
        sum(evaluator_offsets),
        0.0,
        abs_tol=1.0e-15,
    )

    evaluators = list(R.EXPECTED_EVALUATORS)

    for pair_index in range(pair_count):
        pair_id = f"pair-{pair_index:03d}"
        base = pair_index * 0.0001

        # Add deterministic pair-level variation while keeping the
        # exact algebraic contrast definitions transparent.
        pair_variation = (
            (pair_index - (pair_count - 1) / 2.0)
            * 0.00001
        )

        for cell in R.CELLS:
            row_id = f"row-{pair_index:03d}-{cell}"

            cell_effect = CELL_EFFECTS[cell]

            if cell == "C1_TITLE":
                cell_effect += pair_variation
            elif cell == "C2_NAME":
                cell_effect -= 0.5 * pair_variation
            elif cell == "C3_ROLE":
                cell_effect += 0.25 * pair_variation
            elif cell == "C4_PREDICATE":
                cell_effect -= 0.75 * pair_variation
            elif cell == "C5_TITLE_NAME":
                cell_effect += 1.5 * pair_variation

            for evaluator_index, (seed, arm) in enumerate(evaluators):
                q = (
                    base
                    + cell_effect
                    + evaluator_offsets[evaluator_index]
                )

                records.append({
                    "evaluator_seed": seed,
                    "evaluator_arm": arm,
                    "row_id": row_id,
                    "source_pair_id": pair_id,
                    "contrast_cell_id": cell,
                    "q_authorized": q,
                    "entitlement_prob": 0.5 + 0.1 * q,
                    "support_vs_best_nonsupport_logit_margin":
                        q - 0.2,
                    "prediction": "SUPPORT",
                })

    return records


@pytest.fixture(scope="module")
def canonical_synthetic_records():
    records = synthetic_records()

    assert len(records) == R.EXPECTED_ROW_COUNT

    return records


def mutated_copy(records, index=0):
    result = list(records)
    result[index] = dict(result[index])
    return result


def test_exact_evaluator_averaging_and_all_six_formulas(
    canonical_synthetic_records,
):
    averaged = R.evaluator_averaged_cells(
        canonical_synthetic_records
    )

    pair0 = averaged["pair-000"]

    assert pair0["C0_SHAM"]["q_authorized"] == pytest.approx(
        0.0,
        abs=1e-15,
    )

    contrasts = R.pair_contrasts(averaged)
    c = contrasts["pair-000"]

    pair_variation = (
        (0 - (R.PRIMARY_PAIR_COUNT - 1) / 2.0)
        * 0.00001
    )

    expected_title = 0.10 + pair_variation
    expected_name = 0.20 - 0.5 * pair_variation
    expected_role = -0.05 + 0.25 * pair_variation
    expected_predicate = 0.30 - 0.75 * pair_variation
    expected_c5 = 0.37 + 1.5 * pair_variation

    assert c["delta_title"] == pytest.approx(expected_title)
    assert c["delta_name"] == pytest.approx(expected_name)
    assert c["delta_role"] == pytest.approx(expected_role)
    assert c["delta_predicate"] == pytest.approx(
        expected_predicate
    )

    assert c["interaction_title_name"] == pytest.approx(
        expected_c5
        - expected_title
        - expected_name
    )

    assert c["title_minus_name"] == pytest.approx(
        expected_title - expected_name
    )


def test_primary_n_300_is_enforced(
    canonical_synthetic_records,
):
    R.validate_records(
        canonical_synthetic_records
    )

    records_299 = synthetic_records(
        R.PRIMARY_PAIR_COUNT - 1
    )

    with pytest.raises(RuntimeError):
        R.validate_records(records_299)


def test_student_t_df_299_and_critical_value():
    critical = R.student_t_critical(
        299,
        confidence=0.95,
    )

    assert critical == pytest.approx(
        1.96792966906536,
        rel=0.0,
        abs=2e-12,
    )

    p = R.student_t_two_sided_p(
        critical,
        299,
    )

    assert p == pytest.approx(
        0.05,
        rel=0.0,
        abs=2e-12,
    )


def test_one_sample_t_and_ci_use_df_299():
    values = [
        math.sin(index / 17.0) + 0.25
        for index in range(R.PRIMARY_PAIR_COUNT)
    ]

    result = R.one_sample_t_statistics(values)

    assert result["n"] == 300
    assert result["df"] == 299
    assert result["sample_sd"] > 0
    assert result["ci95_low"] < result["mean"]
    assert result["ci95_high"] > result["mean"]

    expected_dz = (
        result["mean"]
        / result["sample_sd"]
    )

    assert result["d_z"] == pytest.approx(
        expected_dz,
        rel=0.0,
        abs=1e-15,
    )


def test_zero_variance_dz_sentinel():
    values = [0.25] * R.PRIMARY_PAIR_COUNT

    result = R.one_sample_t_statistics(values)

    assert result["sample_sd"] == 0.0
    assert (
        result["d_z"]
        == R.DZ_ZERO_VARIANCE_SENTINEL
    )
    assert (
        result["t_statistic"]
        == R.POSITIVE_INFINITY_SENTINEL
    )
    assert result["raw_p_value"] == 0.0
    assert result["ci95_low"] == 0.25
    assert result["ci95_high"] == 0.25


def test_zero_mean_zero_variance_is_non_rejection():
    values = [0.0] * R.PRIMARY_PAIR_COUNT

    result = R.one_sample_t_statistics(values)

    assert result["t_statistic"] == 0.0
    assert result["raw_p_value"] == 1.0
    assert (
        result["d_z"]
        == R.DZ_ZERO_VARIANCE_SENTINEL
    )


def test_holm_exact_adjusted_values_and_stepdown_semantics():
    result = R.holm_bonferroni(
        [0.001, 0.01, 0.02, 0.04, 0.2, 0.9]
    )

    adjusted = [
        row["holm_adjusted_p_value"]
        for row in result
    ]

    assert adjusted == pytest.approx(
        [0.006, 0.05, 0.08, 0.12, 0.4, 0.9],
        rel=0.0,
        abs=1e-15,
    )

    # Frozen decision rule is adjusted p < 0.05,
    # not <= 0.05.
    assert [
        row["reject_holm_alpha_0_05"]
        for row in result
    ] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_duplicate_key_rejected(
    canonical_synthetic_records,
):
    records = list(canonical_synthetic_records)
    records.append(dict(records[0]))

    with pytest.raises(RuntimeError):
        R.validate_records(records)


def test_missing_evaluator_rejected(
    canonical_synthetic_records,
):
    records = list(canonical_synthetic_records)
    records.pop(0)

    with pytest.raises(RuntimeError):
        R.validate_records(records)


def test_missing_cell_rejected(
    canonical_synthetic_records,
):
    records = [
        row
        for row in canonical_synthetic_records
        if not (
            row["source_pair_id"] == "pair-000"
            and row["contrast_cell_id"] == "C4_PREDICATE"
        )
    ]

    with pytest.raises(RuntimeError):
        R.validate_records(records)


def test_wrong_row_multiplicity_rejected(
    canonical_synthetic_records,
):
    records = mutated_copy(
        canonical_synthetic_records,
        0,
    )

    records[0]["row_id"] = "wrong-row-id"

    with pytest.raises(RuntimeError):
        R.validate_records(records)


def test_wrong_source_pair_multiplicity_rejected(
    canonical_synthetic_records,
):
    records = mutated_copy(
        canonical_synthetic_records,
        0,
    )

    records[0]["source_pair_id"] = "unexpected-extra-pair"

    with pytest.raises(RuntimeError):
        R.validate_records(records)


def test_nonfinite_primary_outcome_rejected(
    canonical_synthetic_records,
):
    records = mutated_copy(
        canonical_synthetic_records,
        0,
    )

    records[0]["q_authorized"] = float("nan")

    with pytest.raises(RuntimeError):
        R.validate_records(records)


def test_nonfinite_secondary_outcome_rejected(
    canonical_synthetic_records,
):
    records = mutated_copy(
        canonical_synthetic_records,
        0,
    )

    records[0]["entitlement_prob"] = float("inf")

    with pytest.raises(RuntimeError):
        R.validate_records(records)


def test_incomplete_matrix_forbidden(
    canonical_synthetic_records,
):
    records = list(
        canonical_synthetic_records[
            :-R.EVALUATOR_COUNT
        ]
    )

    with pytest.raises(RuntimeError):
        R.analyze_records(records)


def test_analysis_exact_family_and_no_evaluator_pseudoreplication(
    canonical_synthetic_records,
):
    primary, secondary = R.analyze_records(
        canonical_synthetic_records
    )

    assert [
        row["estimand"]
        for row in primary
    ] == list(R.ESTIMANDS)

    assert len(primary) == 6

    for row in primary:
        assert row["n"] == 300
        assert row["df"] == 299

    assert len(secondary) == (
        len(R.SECONDARY_METRICS)
        * len(R.CELLS)
    )

    for row in secondary:
        assert row["n"] == 300


def test_deterministic_serialization(
    canonical_synthetic_records,
    tmp_path: Path,
):
    primary, secondary = R.analyze_records(
        canonical_synthetic_records
    )

    summary = {
        "schema_version": R.SUMMARY_SCHEMA,
        "input_jsonl_sha256": "synthetic",
        "row_count": len(canonical_synthetic_records),
        "pair_count": 300,
        "evaluator_count": 18,
    }

    first = tmp_path / "first"
    second = tmp_path / "second"

    hashes_first = R.write_analysis_outputs(
        first,
        primary=primary,
        secondary=secondary,
        summary=summary,
    )

    hashes_second = R.write_analysis_outputs(
        second,
        primary=primary,
        secondary=secondary,
        summary=summary,
    )

    assert hashes_first == hashes_second

    for filename in hashes_first:
        assert (
            (first / filename).read_bytes()
            == (second / filename).read_bytes()
        )


def test_canonical_identity_constants_are_frozen_only():
    # This test deliberately checks constants only.
    # It must never open/read the canonical R5 outcome JSONL.
    assert R.CANONICAL_R5_ROWS_SHA256 == (
        "e3157cb5e4e878e4fbe99914a689e576"
        "48568d18b4ae71162e3ba1278093204e"
    )

    assert R.CANONICAL_R5_ROWS_BYTES == 53_510_707

    assert R.CANONICAL_R5_SUMMARY_SHA256 == (
        "e6887d86c1ac2242ce5c2d74c379912"
        "fd81ac441d71b23a13d0db37412d2905b"
    )


def test_canonical_json_bytes_is_deterministic():
    left = R.canonical_json_bytes({
        "b": 2,
        "a": 1,
    })

    right = R.canonical_json_bytes({
        "a": 1,
        "b": 2,
    })

    assert left == right
    assert left == b'{"a":1,"b":2}\n'
