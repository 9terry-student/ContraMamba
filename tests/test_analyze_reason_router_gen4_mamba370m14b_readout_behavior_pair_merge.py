from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import (
    analyze_reason_router_gen4_mamba370m14b_readout_behavior_pair_merge
    as subject,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sums(root: Path, names: list[str]) -> None:
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(root / name)}  {name}\n" for name in sorted(names)),
        encoding="utf-8",
        newline="\n",
    )


def _readout_fixture(root: Path) -> Path:
    d = root / "readout"
    d.mkdir()
    rows = []
    for index, pair in enumerate(subject.EXPECTED_PAIRS):
        d370 = (index - 140.0) / 100000.0
        d14 = (120.0 - index) / 80000.0
        rows.append(
            {
                "source_pair_id": pair,
                "Delta_L_370M": d370,
                "Delta_L_1.4B": d14,
                "R": d370 - d14,
            }
        )
    (d / subject.READOUT_PAIR_FILE).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    (d / subject.READOUT_ANALYSIS_FILE).write_text(
        json.dumps({"result": "fixture"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (d / subject.READOUT_MANIFEST_FILE).write_text(
        json.dumps(
            {
                "primary_p_value_count": 1,
                "rescue_performed": False,
                "row_filter_performed": False,
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_sums(
        d,
        [
            subject.READOUT_PAIR_FILE,
            subject.READOUT_ANALYSIS_FILE,
            subject.READOUT_MANIFEST_FILE,
        ],
    )
    return d


def _behavior_fixture(root: Path) -> Path:
    run = root / "behavior"
    for scale in subject.SCALES:
        for shard_id in (0, 1):
            d = run / scale / f"shard{shard_id}"
            d.mkdir(parents=True)
            rows = []
            for pair_index, pair in enumerate(subject.shard_expected_pairs(shard_id)):
                global_index = int(pair.rsplit("_", 1)[1]) - 4801
                if scale == "mamba370m":
                    base_effect = (global_index - 110.0) / 70000.0
                else:
                    base_effect = (80.0 - global_index) / 65000.0
                for cell_index, cell in enumerate(subject.CELLS):
                    for condition in subject.CONDITIONS:
                        margin = 1.0 + 0.01 * cell_index
                        if condition == "dominant_restored":
                            margin += 0.5 * base_effect
                        elif condition == "dominant_control":
                            margin -= 0.5 * base_effect
                        rows.append(
                            {
                                "source_pair_id": pair,
                                "contrast_cell_id": cell,
                                "condition": condition,
                                "scale": scale,
                                "shard_index": shard_id,
                                "selected_dominant_candidate":
                                    subject.EXPECTED_SELECTED[scale],
                                "response_blind_control_plane":
                                    subject.EXPECTED_CONTROL[scale],
                                "checkpoint_sha256":
                                    subject.EXPECTED_CHECKPOINT[scale],
                                "correct_class_logit_margin": margin,
                            }
                        )
            (d / subject.BEHAVIOR_ROW_FILE).write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
                newline="\n",
            )
            summary = {
                "result": "PASS_RAW_CROSS_SCALE_BEHAVIORAL_SHARD",
                "scale": scale,
                "shard_index": shard_id,
                "source_pair_count": 150,
                "target_cells": list(subject.CELLS),
                "condition_order": list(subject.CONDITIONS),
                "primary_inference_executed": False,
                "scientific_conclusion": None,
                "selection_reopened": False,
                "rescue_performed": False,
                "training_executed": False,
                "backward_executed": False,
            }
            (d / subject.BEHAVIOR_SUMMARY_FILE).write_text(
                json.dumps(summary, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            _write_sums(
                d,
                [subject.BEHAVIOR_ROW_FILE, subject.BEHAVIOR_SUMMARY_FILE],
            )
    return run


def test_sign_categories_keep_exact_zero_rows() -> None:
    assert subject.sign_category(1.0, 2.0) == "both_positive"
    assert subject.sign_category(-1.0, -2.0) == "both_negative"
    assert subject.sign_category(1.0, -2.0) == "delta_positive_beh_negative"
    assert subject.sign_category(-1.0, 2.0) == "delta_negative_beh_positive"
    assert subject.sign_category(0.0, 0.0) == "both_zero"
    assert subject.sign_category(0.0, 2.0) == "delta_zero_beh_positive"
    assert subject.sign_category(-1.0, 0.0) == "delta_negative_beh_zero"


def test_spearman_is_descriptive_coefficient_only() -> None:
    x = np.linspace(-2.0, 2.0, subject.N)
    y = x**3
    value = subject.spearman_without_p(x, y)
    assert value == pytest.approx(1.0)
    out = subject.summarize_scale(x, y)
    assert "spearman_Delta_L_vs_D_BEH" in out
    assert not any("p_value" in key.lower() for key in out)


def test_end_to_end_merge_has_zero_inferential_p_values(tmp_path: Path) -> None:
    readout = _readout_fixture(tmp_path)
    behavior = _behavior_fixture(tmp_path)
    out_dir = tmp_path / "out"

    analysis = subject.write_outputs(
        readout_dir=readout,
        behavior_run_dir=behavior,
        output_dir=out_dir,
    )

    assert analysis["result"] == subject.RESULT
    assert analysis["primary_p_value_count"] == 0
    assert analysis["secondary_p_value_count"] == 0
    assert analysis["inference_performed"] is False
    assert analysis["source_pair_count"] == 300

    merged = [
        json.loads(line)
        for line in (out_dir / "pair_level_merge.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(merged) == 300
    assert merged[0]["source_pair_id"] == "xg1_fact_4801"
    assert merged[-1]["source_pair_id"] == "xg1_fact_5100"

    manifest = json.loads((out_dir / "artifact_manifest.json").read_text())
    assert manifest["primary_p_value_count"] == 0
    assert manifest["secondary_p_value_count"] == 0
    assert manifest["inference_performed"] is False
    assert manifest["forward_executed"] is False
    assert manifest["backward_executed"] is False
    assert manifest["rescue_performed"] is False


def test_behavior_checksum_corruption_blocks(tmp_path: Path) -> None:
    readout = _readout_fixture(tmp_path)
    behavior = _behavior_fixture(tmp_path)
    target = (
        behavior
        / "mamba370m"
        / "shard0"
        / subject.BEHAVIOR_ROW_FILE
    )
    target.write_text(target.read_text() + "{}\n", encoding="utf-8", newline="\n")

    with pytest.raises(subject.PairMergeError, match="SUM_MISMATCH"):
        subject.merge_and_analyze(readout, behavior)


def test_duplicate_behavior_tuple_blocks_even_with_valid_checksum(
    tmp_path: Path,
) -> None:
    readout = _readout_fixture(tmp_path)
    behavior = _behavior_fixture(tmp_path)
    shard = behavior / "mamba14b" / "shard1"
    rows = (shard / subject.BEHAVIOR_ROW_FILE).read_text().splitlines()
    rows[-1] = rows[-2]
    (shard / subject.BEHAVIOR_ROW_FILE).write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_sums(
        shard,
        [subject.BEHAVIOR_ROW_FILE, subject.BEHAVIOR_SUMMARY_FILE],
    )

    with pytest.raises(subject.PairMergeError, match="BEH_DUPLICATE"):
        subject.merge_and_analyze(readout, behavior)
