from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import analyze_reason_router_gen4_seed181_behavioral_restoration_bridge as a


def write_shard(root: Path, shard: int, d_value: float) -> None:
    shard_dir = root / f"shard{shard}"
    shard_dir.mkdir(parents=True)
    first, last = a.SHARD_RANGES[shard]
    rows = []
    margins = {
        "native": 2.0,
        "pp3_neutralized": 1.0,
        "pp3_restored": 2.0,
        "pp5_replacement": 2.0 - d_value,
    }
    for i in range(first, last + 1):
        pair = f"xg1_fact_{i:03d}"
        for cell, label_id in (("C0_SHAM", 2), ("C2_NAME", 1)):
            for condition in a.EXPECTED_CONDITIONS:
                margin = margins[condition]
                if label_id == 2:
                    logits = [0.0, 0.0, margin]
                    prediction = 2 if margin >= 0 else 0
                else:
                    logits = [0.0, margin, 0.0]
                    prediction = 1 if margin >= 0 else 0
                rows.append({
                    "checkpoint_sha256": a.EXPECTED_CHECKPOINT_SHA256,
                    "source_pair_id": pair,
                    "contrast_cell_id": cell,
                    "condition": condition,
                    "correct_class_logit_margin": margin,
                    "final_logits": logits,
                    "prediction_id": prediction,
                    "is_correct": prediction == label_id,
                    "q_authorized": margin / 10.0,
                    "shard_index": shard,
                    "physical_device": shard,
                    "scientific_full_model_forward_count": 1,
                })
    row_raw = b"".join(
        (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for x in rows
    )
    summary = {
        "result": "PASS_RAW_BEHAVIORAL_SHARD",
        "execution_head": "0123456789abcdef0123456789abcdef01234567",
        "design_commit": a.EXPECTED_DESIGN_COMMIT,
        "checkpoint_sha256": a.EXPECTED_CHECKPOINT_SHA256,
        "homolog_plane_index": 3,
        "control_plane_index": 5,
        "selection_uses_response": False,
        "shard_index": shard,
        "physical_device": shard,
        "population_first": f"xg1_fact_{first:03d}",
        "population_last": f"xg1_fact_{last:03d}",
        "source_pair_count": 150,
        "target_cells": list(a.EXPECTED_CELLS),
        "condition_order": list(a.EXPECTED_CONDITIONS),
        "scientific_full_model_forward_count_this_run": a.FORWARDS_PER_SHARD,
        "physical_gpu_inventory_count": 2,
        "model_name": "state-spaces/mamba-130m-hf",
        "model_contract": "historical_v6b_minimal_strict_checkpoint_load",
        "label_contract": {"C0_SHAM": "SUPPORT", "C2_NAME": "NOT_ENTITLED"},
        "training_executed": False,
        "backward_executed": False,
        "primary_inference_executed": False,
        "scientific_conclusion": None,
        "structural_source_sha256": "a" * 64,
        "structural_rows_sha256": "b" * 64,
        "geometry_json_sha256": "c" * 64,
        "geometry_pt_sha256": "d" * 64,
        "tokenizer": {"analysis_reference": "fake", "tokenizers_version": "fake"},
    }
    summary_raw = (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode()
    (shard_dir / a.ROW_FILE).write_bytes(row_raw)
    (shard_dir / a.SUMMARY_FILE).write_bytes(summary_raw)
    hashes = {
        a.ROW_FILE: hashlib.sha256(row_raw).hexdigest(),
        a.SUMMARY_FILE: hashlib.sha256(summary_raw).hexdigest(),
    }
    (shard_dir / a.CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )


def test_positive_synthetic_bridge_passes(tmp_path: Path) -> None:
    write_shard(tmp_path, 0, 0.4)
    write_shard(tmp_path, 1, 0.6)
    result = a.analyze(tmp_path)
    assert result["N"] == 300
    assert result["mean_D_BEH"] == 0.5
    assert result["behavioral_bridge_supported"] is True
    assert result["behavioral_result_label"] == "SEED181_BEHAVIORAL_RESTORATION_BRIDGE_SUPPORTED"
    assert result["seed180_behavioral_rescue_executed"] is False


def test_zero_synthetic_bridge_not_established(tmp_path: Path) -> None:
    write_shard(tmp_path, 0, 0.0)
    write_shard(tmp_path, 1, 0.0)
    result = a.analyze(tmp_path)
    assert result["mean_D_BEH"] == 0.0
    assert result["behavioral_bridge_supported"] is False
    assert result["behavioral_result_label"] == "SEED181_BEHAVIORAL_RESTORATION_BRIDGE_NOT_ESTABLISHED"
