from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import (
    analyze_reason_router_gen4_mamba370m14b_behavioral_bridge
    as analysis,
)
from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as runner,
)


def write_scale(
    root: Path,
    *,
    scale: str,
    base_d: float,
) -> Path:
    run_dir = root / scale
    spec = runner.scale_spec(scale)

    for shard in runner.SHARDS:
        shard_id = int(shard["shard_id"])
        shard_dir = run_dir / f"shard{shard_id}"
        shard_dir.mkdir(parents=True)

        rows = []
        for global_index in range(
            int(shard["start_index"]),
            int(shard["end_index"]),
        ):
            pair = runner.PAIR_IDS[global_index]
            d_value = base_d + 0.0001 * (global_index % 10)

            for cell, label_id in (
                ("C0_SHAM", 2),
                ("C2_NAME", 1),
            ):
                margins = {
                    "native": 1.5,
                    "dominant_neutralized": 1.0,
                    "dominant_restored": 1.5,
                    "dominant_control": 1.5 - d_value,
                }
                for condition in runner.CONDITIONS:
                    margin = margins[condition]
                    if label_id == 2:
                        logits = [0.0, 0.0, margin]
                        prediction = 2 if margin >= 0 else 0
                    else:
                        logits = [0.0, margin, 0.0]
                        prediction = 1 if margin >= 0 else 0

                    rows.append({
                        "scale": scale,
                        "checkpoint_sha256":
                            spec["checkpoint_sha256"],
                        "source_pair_id": pair,
                        "contrast_cell_id": cell,
                        "condition": condition,
                        "correct_class_logit_margin": margin,
                        "final_logits": logits,
                        "prediction_id": prediction,
                        "is_correct": prediction == label_id,
                        "q_authorized": margin / 10.0,
                        "shard_index": shard_id,
                        "physical_device":
                            int(shard["physical_device"]),
                        "scientific_full_model_forward_count": 1,
                    })

        row_raw = b"".join(
            (
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            for row in rows
        )
        summary = {
            "result": runner.RESULT_PASS,
            "execution_head":
                "0123456789abcdef0123456789abcdef01234567",
            "scale": scale,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "selected_dominant_candidate":
                spec["selected_plane"],
            "response_blind_control_plane":
                spec["control_plane"],
            "selection_uses_behavioral_response": False,
            "shard_index": shard_id,
            "physical_device": int(shard["physical_device"]),
            "population_first": shard["pair_first"],
            "population_last": shard["pair_last"],
            "source_pair_count": 150,
            "target_cells": list(runner.TARGET_CELLS),
            "condition_order": list(runner.CONDITIONS),
            "label_contract": {
                "C0_SHAM": "SUPPORT",
                "C2_NAME": "NOT_ENTITLED",
            },
            "scientific_full_model_forward_count_this_run":
                runner.FORWARDS_PER_SHARD,
            "primary_inference_executed": False,
            "scientific_conclusion": None,
            "confirmation_inference_accessed": False,
            "selection_reopened": False,
            "rescue_performed": False,
            "training_executed": False,
            "backward_executed": False,
            "structural_source_sha256": "a" * 64,
            "structural_rows_sha256": "b" * 64,
            "tokenizer": {"fake": True},
        }
        summary_raw = (
            json.dumps(
                summary,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode()

        (shard_dir / analysis.ROW_FILE).write_bytes(row_raw)
        (shard_dir / analysis.SUMMARY_FILE).write_bytes(summary_raw)

        hashes = {
            analysis.ROW_FILE:
                hashlib.sha256(row_raw).hexdigest(),
            analysis.SUMMARY_FILE:
                hashlib.sha256(summary_raw).hexdigest(),
        }
        (shard_dir / analysis.CHECKSUM_FILE).write_text(
            "".join(
                f"{digest}  {name}\n"
                for name, digest in sorted(hashes.items())
            ),
            encoding="utf-8",
        )

    return run_dir


def test_holm_two_test_adjustment() -> None:
    adjusted = analysis.holm_adjust({
        "mamba370m": 0.01,
        "mamba14b": 0.04,
    })
    assert adjusted["mamba370m"] == 0.02
    assert adjusted["mamba14b"] == 0.04


def test_both_positive_scales_support_cross_scale_bridge(
    tmp_path: Path,
) -> None:
    m370 = write_scale(
        tmp_path,
        scale="mamba370m",
        base_d=0.20,
    )
    m14 = write_scale(
        tmp_path,
        scale="mamba14b",
        base_d=0.10,
    )
    result = analysis.analyze(m370, m14)

    assert result["primary_family"]["primary_p_value_count"] == 2
    assert result["primary_family"]["multiplicity"] == "holm"
    assert result["support_by_scale"] == {
        "mamba370m": True,
        "mamba14b": True,
    }
    assert result["cross_scale_behavioral_bridge_supported"] is True
    assert result["scientific_conclusion"] == analysis.RESULT_CROSS_SCALE
    assert (
        result["scientific_full_model_forward_count_total"]
        == 4800
    )


def test_one_failed_scale_yields_scale_specific_only(
    tmp_path: Path,
) -> None:
    m370 = write_scale(
        tmp_path,
        scale="mamba370m",
        base_d=0.20,
    )
    m14 = write_scale(
        tmp_path,
        scale="mamba14b",
        base_d=-0.10,
    )
    result = analysis.analyze(m370, m14)

    assert result["support_by_scale"]["mamba370m"] is True
    assert result["support_by_scale"]["mamba14b"] is False
    assert result["cross_scale_behavioral_bridge_supported"] is False
    assert result["scientific_conclusion"] == analysis.RESULT_PARTIAL
    assert result["scale_specific_rescue_cohort_used"] is False
    assert result["additional_primary_p_values_executed"] is False
