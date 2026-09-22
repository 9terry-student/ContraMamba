from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import reason_router_gen4_mamba790m_training_launcher as launcher


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
    assert launcher.MODEL_REPO == "state-spaces/mamba-790m-hf"
    assert launcher.MODEL_REVISION == "9822dd4b76af2bd9099b6ce2f19efd8329189a7e"
    assert set(launcher.SNAPSHOT_FILES) == {
        "config.json",
        "generation_config.json",
        "model.safetensors",
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


def test_authenticate_repo_allows_detached_exact_head(monkeypatch) -> None:
    expected_head = "a" * 40

    def fake_check_output(argv, *, cwd, text):
        assert cwd == launcher.ROOT
        assert text is True
        if argv == ["git", "branch", "--show-current"]:
            return "\n"
        if argv == ["git", "rev-parse", "HEAD"]:
            return expected_head + "\n"
        if argv == ["git", "status", "--porcelain"]:
            return ""
        raise AssertionError(argv)

    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        fake_check_output,
    )

    launcher.authenticate_repo(expected_head)


def test_authenticate_repo_rejects_wrong_named_branch(monkeypatch) -> None:
    expected_head = "b" * 40

    def fake_check_output(argv, *, cwd, text):
        assert cwd == launcher.ROOT
        assert text is True
        if argv == ["git", "branch", "--show-current"]:
            return "main\n"
        if argv == ["git", "rev-parse", "HEAD"]:
            return expected_head + "\n"
        if argv == ["git", "status", "--porcelain"]:
            return ""
        raise AssertionError(argv)

    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        fake_check_output,
    )

    with pytest.raises(
        launcher.TrainingLauncherError,
        match="BRANCH:main",
    ):
        launcher.authenticate_repo(expected_head)


def test_exact_gen3_grouped_snapshot_binding(
    tmp_path: Path,
) -> None:
    trainer = launcher.load_exact_gen3_grouped_trainer()

    from contramamba import (
        modeling_v6b_minimal_gen3_grouped_snapshot
        as model_snapshot,
    )

    assert (
        launcher.git_blob_identity(
            launcher.GEN3_GROUPED_TRAINER_SNAPSHOT
        )
        == launcher.GEN3_GROUPED_TRAINER_GIT_BLOB
    )
    assert (
        launcher.git_blob_identity(
            launcher.GEN3_GROUPED_MODEL_SNAPSHOT
        )
        == launcher.GEN3_GROUPED_MODEL_GIT_BLOB
    )

    assert (
        trainer.ContraMambaV6BMinimal
        is model_snapshot.ContraMambaV6BMinimal
    )

    assert trainer.P2_ARM_CONTRACTS[launcher.ARM] == (
        "explicit_product",
        "edge_specific",
    )
    assert tuple(
        trainer.G3_GROUPED_ARM_EDGE_SETS[launcher.ARM]
    ) == (
        "F_TO_D",
        "P_TO_D",
        "S_TO_D",
        "Q_TO_D",
    )

    argv = launcher.trainer_argv(
        Path("/exact/snapshot"),
        tmp_path / "run",
    )
    parser = trainer.build_parser()
    args = parser.parse_args(argv)

    contract = trainer._p2_resolve_arm_contract(
        args,
        argv,
        parser,
    )

    assert contract["arm"] == launcher.ARM
    assert contract["router_mode"] == "explicit_product"
    assert contract["gradient_ownership_mode"] == "edge_specific"
    assert (
        contract["edge_gradient_lambdas"]
        == launcher.EDGE_LAMBDAS
    )


def test_exact_gen3_grouped_trainer_uses_cache_hook() -> None:
    trainer = launcher.load_exact_gen3_grouped_trainer()

    assert callable(
        trainer.v5.cache_frozen_encoder_states
    )


def test_tensor_state_hash_accepts_scalar_tensor() -> None:
    scalar = torch.tensor(3.5, dtype=torch.float32)
    same = torch.tensor(3.5, dtype=torch.float32)
    other = torch.tensor(4.5, dtype=torch.float32)

    one = launcher.tensor_state_sha256([("scalar", scalar)])
    two = launcher.tensor_state_sha256([("scalar", same)])
    three = launcher.tensor_state_sha256([("scalar", other)])

    assert one == two
    assert one != three


def test_compaction_maps_hf_backbone_namespace_to_trainer_namespace(
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
            "metadata": {"selected_epoch": 20},
        },
        full,
    )

    monkeypatch.setattr(
        launcher,
        "iter_pinned_backbone_tensors",
        lambda snapshot: iter(
            [("backbone.weight", backbone.clone())]
        ),
    )

    observed = launcher.compact_selected_checkpoint(
        full_checkpoint=full,
        snapshot=tmp_path,
        compact_checkpoint=compact,
        manifest_path=manifest,
        expected_head="c" * 40,
    )

    assert not full.exists()
    assert compact.is_file()
    assert observed["pretrained_backbone"]["state_key_count"] == 1
    assert (
        observed["reconstruction_verification"]
        ["tensor_equal_all_backbone_keys"]
        is True
    )


def test_validate_training_outputs_accepts_historical_single_run_schema(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = {
        "best_epoch": 20,
        "best_dev_metrics": {
            "final_macro_f1": 0.8137084393485566,
        },
        "runs": {
            "single": {
                "select_metric": "final_macro_f1",
                "reason_router_p2": {
                    "contract": {
                        "arm": launcher.ARM,
                        "router_mode": "explicit_product",
                        "gradient_ownership_mode": "edge_specific",
                    }
                },
            }
        },
    }

    (run_dir / "training_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    (run_dir / "clean_dev_predictions.json").write_text(
        "[]",
        encoding="utf-8",
    )
    (run_dir / "run_provenance.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (run_dir / "selected_checkpoint.pt").write_bytes(b"stub")

    summary = launcher.validate_training_outputs(run_dir)

    assert summary["best_epoch"] == 20
    assert (
        summary["best_dev_metrics"]["final_macro_f1"]
        == 0.8137084393485566
    )
