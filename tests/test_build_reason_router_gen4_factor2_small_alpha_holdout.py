from __future__ import annotations

import inspect

from scripts import (
    build_reason_router_gen4_factor2_small_alpha_holdout
    as builder,
)


def test_protocol_constants_are_exact() -> None:
    assert builder.PROSPECTIVE_FREEZE_COMMIT == (
        "8614dd5a4ed77cb56cc6a485f4ab379434a284fa"
    )
    assert builder.PROSPECTIVE_REPORT_BLOB == (
        "90ec869dde6cad2131411ef74beef3328c42c4e7"
    )
    assert builder.FIRST_PAIR == 5701
    assert builder.LAST_PAIR == 6000
    assert builder.PAIR_COUNT == 300
    assert builder.ROW_COUNT == 1800
    assert builder.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert builder.SCALES == ("mamba370m", "mamba14b")
    assert builder.BEHAVIORAL_ALPHAS == (
        0.25,
        0.125,
        0.0625,
        0.03125,
    )


def test_frozen_geometry_is_unchanged() -> None:
    assert builder.SELECTED_PLANES == {
        "mamba370m": "P3",
        "mamba14b": "P5",
    }
    assert builder.CONTROL_PLANES == {
        "mamba370m": "P5",
        "mamba14b": "P4",
    }
    assert builder.SOURCE_BLOCK == 33
    assert builder.TARGET_RESIDUAL_LAYER == 34
    assert builder.INTERVENTION_LAYER == 35
    assert builder.ANCHOR_NAME == "A_IDENTITY"
    assert builder.TARGET_OFFSET == 2


def test_previous_structural_artifact_is_hash_pinned() -> None:
    assert builder.PREVIOUS_SOURCE_SHA256 == (
        "eddd6a264130e6451c72aeab010758dac43de82d9521898dfd1489c86717d11a"
    )
    assert builder.PREVIOUS_ROWS_SHA256 == (
        "1d0f21ba44b4a282f42edb7e56626bb7d522ce09e25e1611bf1424ead4da2b28"
    )
    assert builder.PREVIOUS_MANIFEST_SHA256 == (
        "684821b2b8d5d3e84ee9cec38b55cab848a50ebec75daf7892935a417183b28d"
    )


def test_population_generation_is_exact_and_deterministic() -> None:
    facts_a, rows_a = builder.build_population()
    facts_b, rows_b = builder.build_population()

    assert len(facts_a) == 300
    assert len(rows_a) == 1800
    assert [row["pair_id"] for row in facts_a] == [
        f"xg1_fact_{i}" for i in range(5701, 6001)
    ]
    assert builder.jsonl_bytes(facts_a) == builder.jsonl_bytes(facts_b)
    assert builder.jsonl_bytes(rows_a) == builder.jsonl_bytes(rows_b)


def test_builder_contains_no_outcome_or_execution_path() -> None:
    source = inspect.getsource(builder).lower()

    forbidden = (
        "torch.",
        "transformers",
        "cuda(",
        "historical_forward",
        "correct_class_logit_margin",
        "d_beh",
        "pearson",
        "spearman",
        "rmse",
        "ttest",
    )
    for token in forbidden:
        assert token not in source

    assert '"tokenizer_executed": false' in source
    assert '"model_executed": false' in source
    assert '"cuda_executed": false' in source


def test_manifest_freezes_single_grid_and_no_negative_arm() -> None:
    source = inspect.getsource(builder.build_payload)
    assert '"behavioral_alphas": list(BEHAVIORAL_ALPHAS)' in source
    assert '"primary_curve": "K(alpha)"' in source
    assert '"negative_alpha_arm_allowed": False' in source
    assert '"outcome_conditioned_grid_extension_allowed": False' in source
    assert '"cohort_replacement_allowed": False' in source
    assert '"row_filtering_allowed": False' in source


def test_output_contract_is_four_static_files() -> None:
    assert builder.SOURCE_FILE == "structured_source_facts.jsonl"
    assert builder.ROW_FILE == "synthetic_reason_router_six_cell.jsonl"
    assert builder.MANIFEST_FILE == "structural_manifest.json"
    assert builder.CHECKSUM_FILE == "SHA256SUMS.txt"
