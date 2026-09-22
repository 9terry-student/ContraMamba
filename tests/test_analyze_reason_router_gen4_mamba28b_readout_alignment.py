from __future__ import annotations

import inspect

import numpy as np

from scripts import (
    analyze_reason_router_gen4_mamba28b_readout_alignment
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
                    "mamba28b",
                "source_pair_id":
                    pair,
                "contrast_cell_id":
                    cell,
                "selected_plane":
                    "P3",
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
        "b245f3dc198eadf343bdbef490441819b0b75b8a"
    )
    assert subject.RAW_EXECUTION_HEAD == (
        "1c3671ead1e08ba80651802a37147390806ed0e4"
    )
    assert subject.PAIR_IDS[0] == "xg1_fact_6601"
    assert subject.PAIR_IDS[-1] == "xg1_fact_6900"
    assert len(subject.PAIR_IDS) == 300
    assert subject.CELLS == (
        "C0_SHAM",
        "C2_NAME",
    )
    assert subject.SELECTED_PLANE == "P3"
    assert subject.CONTROL_PLANE == "P5"
    assert subject.OWNERSHIP_LAMBDA == 0.5
    assert subject.FORWARD_SCALE == 2.0


def test_frozen_readout_plan_has_no_prespecified_test() -> None:
    manifest = (
        subject.validate_readout_plan()
    )

    assert manifest["role"] == "readout"
    assert manifest["scale"] == "mamba28b"
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
    assert summary["selected_plane"] == "P3"
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
                "583b72d27096862c87cb19b14ba6f5dc"
                "e9f6183f5203aec6aa88832978c4f198"
            ),
        "raw_readout_alignment_summary.json":
            (
                "d0e2444667f66b61cb15a861b6b6b1a9"
                "0f7f7dfb76419b266f1616a3dcb24826"
            ),
        "readout_alignment_items.jsonl":
            (
                "6b66749e53816a2f0b9f4b35305b5312"
                "50fce8e7b04fa5f2a98ce213bed6f883"
            ),
    }
