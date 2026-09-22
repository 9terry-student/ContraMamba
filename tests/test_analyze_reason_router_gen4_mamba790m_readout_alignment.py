from __future__ import annotations

import inspect

import numpy as np

from scripts import (
    analyze_reason_router_gen4_mamba790m_readout_alignment
    as subject,
)


def _synthetic_rows(
    pair_values: np.ndarray,
) -> list[dict]:
    assert pair_values.shape == (300,)

    rows: list[dict] = []

    for index, pair in enumerate(
        subject.PAIR_IDS
    ):
        for cell, offset in (
            ("C0_SHAM", -0.25),
            ("C2_NAME", 0.25),
        ):
            owned = float(
                pair_values[index]
                + offset
            )

            rows.append({
                "scale":
                    "mamba790m",
                "source_pair_id":
                    pair,
                "contrast_cell_id":
                    cell,
                "selected_plane":
                    "P2",
                "control_plane":
                    "P5",
                "Delta_L_owned":
                    owned,
                "Delta_L_forward_equivalent":
                    2.0 * owned,
                "p_value_count_executed":
                    0,
                "inferential_test_performed":
                    False,
                "selection_reopened":
                    False,
                "rescue_performed":
                    False,
                "sweep_performed":
                    False,
            })

    return rows


def test_protocol_constants_are_exact() -> None:
    assert subject.RAW_FREEZE_COMMIT == (
        "4c3d8be94ec1b31e4cfe6fd8a6ef47b49669e571"
    )
    assert subject.RAW_EXECUTION_HEAD == (
        "6e3f0dc8a47da849cc83076147f744b4b93109d1"
    )
    assert subject.PAIR_IDS[0] == "xg1_fact_7501"
    assert subject.PAIR_IDS[-1] == "xg1_fact_7800"
    assert len(subject.PAIR_IDS) == 300
    assert subject.CELLS == (
        "C0_SHAM",
        "C2_NAME",
    )
    assert subject.SELECTED_PLANE == "P2"
    assert subject.CONTROL_PLANE == "P5"
    assert subject.OWNERSHIP_LAMBDA == 0.5
    assert subject.FORWARD_SCALE == 2.0


def test_frozen_readout_plan_has_no_prespecified_test() -> None:
    manifest = (
        subject.validate_readout_plan()
    )

    assert manifest["role"] == "readout"
    assert manifest["scale"] == "mamba790m"
    assert (
        manifest["planned_quantity"]
        == (
            "Delta_L_owned_with_forward_"
            "equivalent_recorded_if_applicable"
        )
    )
    assert "planned_primary_test" not in manifest
    assert "success_rule" not in manifest
    assert manifest["selection_allowed"] is False
    assert (
        manifest[
            "response_guided_selection_allowed"
        ]
        is False
    )
    assert manifest["rescue_policy"] == "none"


def test_frozen_raw_bundle_is_exact() -> None:
    rows, summary, manifest = (
        subject.read_raw_bundle(
            subject.RAW_DIR
        )
    )

    assert len(rows) == 600
    assert summary["pair_count"] == 300
    assert summary["item_count"] == 600
    assert summary["selected_plane"] == "P2"
    assert summary["control_plane"] == "P5"
    assert (
        summary["inferential_test_performed"]
        is False
    )
    assert summary["p_value_count_executed"] == 0
    assert summary["scientific_conclusion"] is None

    assert manifest["item_count"] == 600
    assert (
        manifest["inferential_test_performed"]
        is False
    )
    assert manifest["p_value_count_executed"] == 0
    assert manifest["scientific_conclusion"] is None


def test_pair_aggregation_matches_historical_two_cell_semantics() -> None:
    target = np.linspace(
        -1.5,
        2.0,
        300,
        dtype=np.float64,
    )

    rows = _synthetic_rows(
        target
    )

    pairs = subject.pair_values(
        rows
    )

    assert len(pairs) == 300

    observed = np.asarray(
        [
            row["Delta_L_owned"]
            for row in pairs
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        observed,
        target,
        rtol=0.0,
        atol=1e-15,
    )

    for row in pairs:
        assert (
            row[
                "Delta_L_forward_equivalent"
            ]
            == 2.0
            * row[
                "Delta_L_owned"
            ]
        )


def test_descriptive_analysis_has_no_inferential_gate() -> None:
    target = np.linspace(
        0.1,
        1.0,
        300,
        dtype=np.float64,
    )

    result, pairs = (
        subject.analyze_rows(
            _synthetic_rows(
                target
            )
        )
    )

    assert len(pairs) == 300

    assert (
        result["descriptive_mean_sign"]
        == "positive"
    )
    assert (
        result["formal_inference_performed"]
        is False
    )
    assert result["p_value_count"] == 0
    assert result["scientific_conclusion"] is None
    assert (
        result["cross_scale_synthesis_performed"]
        is False
    )
    assert (
        result["threshold_estimation_performed"]
        is False
    )
    assert (
        result["monotonicity_test_performed"]
        is False
    )

    owned = result[
        "pair_Delta_L_owned"
    ]
    forward = result[
        "pair_Delta_L_forward_equivalent"
    ]

    assert (
        forward["mean"]
        == 2.0
        * owned["mean"]
    )
    assert (
        forward["fraction_positive"]
        == owned["fraction_positive"]
    )


def test_mean_sign_is_descriptive_only() -> None:
    assert subject.mean_sign(1.0) == "positive"
    assert subject.mean_sign(-1.0) == "negative"
    assert subject.mean_sign(0.0) == "zero"


def test_source_contains_no_statistical_inference() -> None:
    src = inspect.getsource(
        subject
    ).lower()

    assert "scipy" not in src
    assert "ttest" not in src
    assert "alternative=" not in src
    assert "alpha_gate" not in src
    assert "positive_alignment_supported" not in src

    assert (
        '"formal_inference_performed":\n            false'
        in src
        or
        '"formal_inference_performed": false'
        in " ".join(src.split())
    )

    assert (
        '"p_value_count":\n            0'
        in src
        or
        '"p_value_count": 0'
        in " ".join(src.split())
    )


def test_exact_raw_hashes_are_frozen() -> None:
    assert subject.EXPECTED_RAW_SHA256 == {
        "artifact_manifest.json":
            (
                "01c99f196eff8ba8f3d2fb5853e4e967"
                "d26203e0d4d868e66b6648245bbcc135"
            ),
        "raw_readout_alignment_summary.json":
            (
                "159704b7f791e2d14a95d9e76937c8c9"
                "3a835977c19093d4aa94bfb1b3908510"
            ),
        "readout_alignment_items.jsonl":
            (
                "8138bb9dccee658f2478b8ffad2f8ab8"
                "db51fa415c9ae3114b4d6d48e59b2833"
            ),
    }
