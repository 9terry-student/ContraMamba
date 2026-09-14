import hashlib
import json
import zipfile

import pytest

from scripts import (
    reason_router_gen4_k_transport_failure_tail_influence
    as tail,
)


def test_frozen_authority_constants():
    assert (
        tail.STATIC_FREEZE_COMMIT
        == "4d68f06d5bfaf9bc680e625000062e36b29c0b30"
    )
    assert (
        tail.STATIC_REPORT_SHA256
        == "10adb86cf8b86b14230d657a19420c02a9bc0d19bebfff9ff3374fee879fd459"
    )
    assert tail.REMOVAL_COUNTS == (
        1, 5, 10, 20, 30, 40, 50, 60, 75
    )
    assert tail.TAIL_COUNTS == (
        10, 20, 30, 60
    )


def test_contribution_mass():
    out = tail.contribution_mass(
        [-4.0, -1.0, 2.0, 1.0]
    )
    assert out["positive_sum"] == 3.0
    assert out["negative_sum"] == -5.0
    assert out[
        "negative_absolute_mass_fraction"
    ] == pytest.approx(5 / 8)
    assert out[
        "negative_to_positive_mass_ratio"
    ] == pytest.approx(5 / 3)


def test_removal_curve_is_worst_first():
    values = [
        -10.0,
        -2.0,
        1.0,
        2.0,
        3.0,
    ] * 60
    pair_ids = [
        f"p{i:03d}"
        for i in range(300)
    ]
    curve = tail.removal_curve(
        values,
        pair_ids,
    )
    one = next(
        row
        for row in curve
        if row["removed_worst_count"] == 1
    )
    assert one["removed_sum"] == -10.0
    assert one["remaining_mean"] > statistics_mean(values)


def statistics_mean(values):
    return sum(values) / len(values)


def test_symmetric_trimmed_mean():
    values = [
        -100.0,
        *([1.0] * 298),
        100.0,
    ]
    rows = tail.symmetric_trimmed_means(
        values
    )
    ten = next(
        row
        for row in rows
        if row[
            "trim_fraction_each_tail"
        ] == 0.10
    )
    assert ten["trim_count_each_tail"] == 30
    assert ten["trimmed_mean"] == 1.0


def test_quartile_labels_equal_count():
    pair_ids = [
        f"p{i:03d}"
        for i in range(300)
    ]
    values = [
        float(i // 5)
        for i in range(300)
    ]
    labels = tail.stable_quartile_labels(
        pair_ids,
        values,
    )
    assert [
        labels.count(q)
        for q in range(1, 5)
    ] == [75, 75, 75, 75]


def test_tail_overlap_detects_q4_enrichment():
    pair_ids = [
        f"p{i:03d}"
        for i in range(300)
    ]
    feature = [
        float(i)
        for i in range(300)
    ]
    response = [
        -float(i)
        for i in range(300)
    ]
    features = {
        name: list(feature)
        for name in tail.QUARTILE_FEATURES
    }
    out = tail.tail_overlap(
        response=response,
        pair_ids=pair_ids,
        features=features,
    )
    assert (
        out["bottom_30"][
            "quartile_enrichment"
        ]["alignment_shift_abs"][
            "counts"
        ]["Q4"]
        == 30
    )
    assert (
        out["bottom_30"][
            "quartile_enrichment"
        ]["alignment_shift_abs"][
            "Q4_enrichment_ratio_vs_population"
        ]
        == 4.0
    )


def test_specificity_identity():
    pair_ids = [
        f"p{i:03d}"
        for i in range(300)
    ]
    align = [
        i / 1000
        for i in range(300)
    ]
    mag = [
        i / 2000
        for i in range(300)
    ]
    spec = [
        a - b
        for a, b in zip(align, mag)
    ]
    out = tail.specificity_tail_decomposition(
        pair_ids=pair_ids,
        responses={
            "R_ALIGN": align,
            "R_MAG": mag,
            "ALIGNMENT_SPECIFICITY": spec,
        },
    )
    assert (
        out["bottom_10"]["mean_specificity"]
        == pytest.approx(
            out["bottom_10"]["mean_R_ALIGN"]
            - out["bottom_10"]["mean_R_MAG"]
        )
    )


def test_feature_derivation():
    row = {
        "target_A": 1.0,
        "target_B": 2.0,
        "target_C": 0.2,
        "reference_A": 2.0,
        "reference_B": 1.0,
        "reference_C": 0.5,
        "delta_baseline": -0.1,
    }
    f = tail.derive_features(row)
    assert f["alignment_shift"] == pytest.approx(0.3)
    assert f["alignment_shift_abs"] == pytest.approx(0.3)
    assert f["baseline_severity_abs"] == 0.1
    assert f["magnitude_shift_norm"] > 0.0
