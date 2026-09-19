from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import reason_router_gen4_mamba14b_training_launcher as launcher


def test_training_contract_is_frozen_and_microbatch_only_changes_forward_partition(
    tmp_path: Path,
) -> None:
    argv = launcher.trainer_argv(Path("/exact/snapshot"), tmp_path / "run")
    joined = " ".join(argv)

    assert "--reason-router-arm G3-GROUP-D-HALF" in joined
    assert "--seed 181" in joined
    assert "--split-seed 8192" in joined
    assert "--epochs 20" in joined
    assert "--lr 0.001" in joined
    assert "--class-weighting none" in joined
    assert "--select-metric final_macro_f1" in joined
    assert "--ranking-weight 0.0" in joined
    assert "--train-batch-size 32" in joined
    assert "--eval-batch-size 32" in joined
    assert "--gradient-accumulation-steps 1" in joined
    assert "--freeze-encoder true" in joined
    assert "--reason-loss-weight 0.0" in joined

    edge_arg = argv[argv.index("--edge-gradient-lambdas") + 1]
    edge = json.loads(edge_arg)
    assert edge == launcher.EDGE_LAMBDAS
    assert {k for k, v in edge.items() if v == 0.5} == {
        "F_TO_D", "P_TO_D", "S_TO_D", "Q_TO_D"
    }


def test_tensor_state_hash_is_key_order_invariant() -> None:
    a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    b = torch.tensor([3, 4], dtype=torch.int64)
    one = launcher.tensor_state_sha256([("b", b), ("a", a)])
    two = launcher.tensor_state_sha256([("a", a), ("b", b)])
    assert one == two


def test_compaction_rejects_backbone_mismatch(monkeypatch, tmp_path: Path) -> None:
    full = tmp_path / "selected_checkpoint.pt"
    compact = tmp_path / "selected_downstream_checkpoint.pt"
    manifest = tmp_path / "compact_checkpoint_manifest.json"

    torch.save(
        {
            "schema_version": "stage176a0_selected_checkpoint_v1",
            "model_state_dict": {
                "mamba.weight": torch.tensor([1.0]),
                "head.weight": torch.tensor([2.0]),
            },
            "metadata": {"selected_epoch": 19},
        },
        full,
    )

    monkeypatch.setattr(
        launcher,
        "iter_pinned_backbone_tensors",
        lambda snapshot: iter([("weight", torch.tensor([9.0]))]),
    )

    with pytest.raises(launcher.TrainingLauncherError, match="BACKBONE_TENSOR_MISMATCH"):
        launcher.compact_selected_checkpoint(
            full_checkpoint=full,
            snapshot=tmp_path,
            compact_checkpoint=compact,
            manifest_path=manifest,
            expected_head="a" * 40,
        )

    assert full.exists()
    assert not compact.exists()


def test_compaction_keeps_only_downstream_and_removes_verified_full(
    monkeypatch,
    tmp_path: Path,
) -> None:
    full = tmp_path / "selected_checkpoint.pt"
    compact = tmp_path / "selected_downstream_checkpoint.pt"
    manifest = tmp_path / "compact_checkpoint_manifest.json"

    backbone = torch.tensor([1.0, 2.0])
    downstream = torch.tensor([[3.0, 4.0]])

    torch.save(
        {
            "schema_version": "stage176a0_selected_checkpoint_v1",
            "model_state_dict": {
                "mamba.weight": backbone.clone(),
                "head.weight": downstream.clone(),
            },
            "metadata": {"selected_epoch": 19},
        },
        full,
    )

    monkeypatch.setattr(
        launcher,
        "iter_pinned_backbone_tensors",
        lambda snapshot: iter([("weight", backbone.clone())]),
    )

    observed = launcher.compact_selected_checkpoint(
        full_checkpoint=full,
        snapshot=tmp_path,
        compact_checkpoint=compact,
        manifest_path=manifest,
        expected_head="b" * 40,
    )

    assert not full.exists()
    assert compact.is_file()
    assert manifest.is_file()
    payload = torch.load(compact, map_location="cpu", weights_only=False)
    assert set(payload["downstream_state_dict"]) == {"head.weight"}
    assert torch.equal(payload["downstream_state_dict"]["head.weight"], downstream)
    assert observed["reconstruction_verification"]["key_set_exact"] is True
    assert observed["full_checkpoint_removed_after_verified_compaction"] is True


def test_snapshot_identity_is_exactly_pinned() -> None:
    assert launcher.MODEL_REPO == "state-spaces/mamba-1.4b-hf"
    assert launcher.MODEL_REVISION == "6e46eae61c27280517feef46f536d16b91076f08"
    assert set(launcher.SNAPSHOT_FILES) == {
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert launcher.TRAIN_ROW_COUNT == 2880
    assert launcher.DEV_ROW_COUNT == 720

def test_dual_gpu_cache_partition_uses_both_devices_evenly() -> None:
    train = launcher.dual_gpu_cache_slices(
        launcher.TRAIN_ROW_COUNT,
        launcher.DUAL_GPU_CACHE_BATCH_SIZE,
    )
    dev = launcher.dual_gpu_cache_slices(
        launcher.DEV_ROW_COUNT,
        launcher.DUAL_GPU_CACHE_BATCH_SIZE,
    )

    left_rows = sum(item[0].stop - item[0].start for item in [*train, *dev])
    right_rows = sum(item[1].stop - item[1].start for item in [*train, *dev])

    assert launcher.DUAL_GPU_CACHE_DEVICE_IDS == (0, 1)
    assert launcher.DUAL_GPU_CACHE_BATCH_SIZE == 64
    assert left_rows == 1800
    assert right_rows == 1800
    assert left_rows + right_rows == (
        launcher.TRAIN_ROW_COUNT + launcher.DEV_ROW_COUNT
    )


def test_dual_gpu_cache_runtime_contract_is_not_gradient_data_parallel() -> None:
    assert launcher.GRADIENT_ACCUMULATION_STEPS == 1
    assert launcher.FORWARD_MICROBATCH_SIZE == 32
    assert launcher.DUAL_GPU_CACHE_DEVICE_IDS == (0, 1)
    assert launcher.TRAIN_ROW_COUNT == 2880
    assert launcher.DEV_ROW_COUNT == 720
