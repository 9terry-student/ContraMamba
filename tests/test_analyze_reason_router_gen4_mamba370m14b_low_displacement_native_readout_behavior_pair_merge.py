from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import (
    analyze_reason_router_gen4_mamba370m14b_low_displacement_native_readout_behavior_pair_merge
    as subject,
)


def _readout_rows():
    rows = []
    for pair_index, pair in enumerate(subject.EXPECTED_PAIRS):
        for scale in subject.SCALES:
            base = (pair_index - 120.0) / 100000.0
            if scale == "mamba14b":
                base = -base - 0.0002
            for cell_index, cell in enumerate(subject.CELLS):
                rows.append(
                    {
                        "scale": scale,
                        "source_pair_id": pair,
                        "contrast_cell_id": cell,
                        "selected_plane": subject.EXPECTED_SELECTED[scale],
                        "control_plane": subject.EXPECTED_CONTROL[scale],
                        "checkpoint_sha256": subject.EXPECTED_CHECKPOINT[scale],
                        "Delta_L_row": base + cell_index * 0.00002,
                        "behavioral_response_accessed": False,
                        "p_value_count_executed": 0,
                    }
                )
    return rows


def _behavior(readout):
    rows = []
    for index, pair in enumerate(subject.EXPECTED_PAIRS):
        d370 = float(readout["mamba370m"][index]["Delta_L"])
        d14 = float(readout["mamba14b"][index]["Delta_L"])
        rows.append(
            {
                "source_pair_id": pair,
                "mamba370m_D_BEH_alpha_0_25": 0.5 * d370 + 0.00001,
                "mamba14b_D_BEH_alpha_0_25": 0.5 * d14 - 0.00001,
            }
        )
    return rows


def test_pair_aggregation_is_mean_of_exact_two_cells() -> None:
    rows = _readout_rows()
    out = subject.pair_readout_from_rows(rows)

    pair = subject.EXPECTED_PAIRS[0]
    source = {
        (row["scale"], row["source_pair_id"], row["contrast_cell_id"]):
            float(row["Delta_L_row"])
        for row in rows
    }

    for scale in subject.SCALES:
        expected = sum(
            source[(scale, pair, cell)]
            for cell in subject.CELLS
        ) / 2.0
        assert out[scale][0]["source_pair_id"] == pair
        assert out[scale][0]["Delta_L"] == pytest.approx(expected)


def test_metric_family_is_exactly_prior_descriptive_family() -> None:
    readout = subject.pair_readout_from_rows(_readout_rows())
    behavior = subject.behavior_values_from_rows(_behavior(readout))
    analysis, merged = subject.merge_and_analyze_values(readout, behavior)

    assert len(merged) == 300
    assert analysis["primary_p_value_count_added"] == 0
    assert analysis["secondary_p_value_count_added"] == 0
    assert analysis["inference_performed"] is False
    assert analysis["row_filter_performed"] is False
    assert analysis["rescue_performed"] is False
    assert analysis["scientific_conclusion"] is None

    assert analysis["descriptive_metric_family"] == [
        "pearson_Delta_L_vs_D_BEH",
        "spearman_Delta_L_vs_D_BEH",
        "sign_agreement",
        "residual_D_BEH_minus_Delta_L",
    ]

    for scale in subject.SCALES:
        result = analysis["scale_results"][scale]
        assert set(result) == {
            "pearson_Delta_L_vs_D_BEH",
            "spearman_Delta_L_vs_D_BEH",
            "Delta_L",
            "D_BEH",
            "residual_D_BEH_minus_Delta_L",
            "sign_agreement",
        }
        assert result["pearson_Delta_L_vs_D_BEH"] == pytest.approx(1.0)
        assert result["spearman_Delta_L_vs_D_BEH"] == pytest.approx(1.0)


def test_alignment_mismatch_blocks() -> None:
    readout = subject.pair_readout_from_rows(_readout_rows())
    behavior = subject.behavior_values_from_rows(_behavior(readout))
    behavior[5] = dict(behavior[5])
    behavior[5]["source_pair_id"] = "wrong_pair"

    with pytest.raises(subject.LowDispPairMergeError, match="BEHAVIOR_ALIGN"):
        subject.merge_and_analyze_values(readout, behavior)


def test_duplicate_readout_tuple_blocks() -> None:
    rows = _readout_rows()
    rows[-1] = dict(rows[-2])

    with pytest.raises(subject.LowDispPairMergeError, match="READOUT_DUPLICATE"):
        subject.pair_readout_from_rows(rows)

def test_direct_cli_import_surface_works_from_repo_root() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "scripts"
        / "analyze_reason_router_gen4_mamba370m14b_"
          "low_displacement_native_readout_behavior_pair_merge.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--readout-raw-dir" in completed.stdout
    assert "--behavior-analysis-dir" in completed.stdout
    assert "--output-dir" in completed.stdout
