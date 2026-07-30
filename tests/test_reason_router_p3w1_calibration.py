from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import aggregate_reason_router_p3w1_calibration as agg
from scripts import train_controlled_v6b_minimal as trainer

EXEC = "a" * 40
DATA = "b" * 64
SIDE = "c" * 64
ROW_HASH = "d" * 64


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def _args(**overrides):
    values = dict(
        reason_router_arm="A3",
        architecture="v6b_minimal",
        backbone="mamba",
        model_name="state-spaces/mamba-130m-hf",
        max_length=128,
        device="cuda",
        flag_source="controlled_heuristic",
        reason_router_mode="auto",
        gradient_ownership_mode="auto",
        reason_loss_weight=0.0,
        freeze_encoder=True,
        frame_downstream_gradient_mode="joint",
        reason_router_epsilon=1e-8,
        reason_min_train_count=50,
        reason_min_dev_count=1,
        reason_router_weight_calibration_export=Path("unit.json"),
        reason_router_weight_calibration_execution_commit=EXEC,
        reason_router_weight_calibration_forward_batch_size=8,
        expected_integrity_sidecar_semantic_sha256=SIDE,
        seed=180,
        resolved_split_seed=174,
        dev_ratio=0.2,
        resolved_reason_router_mode="conditional_first_blocker",
        resolved_gradient_ownership_mode="explicit_local",
        resolved_reason_loss_weight=0.0,
        use_temporal_comparator=True,
        use_predicate_comparator=True,
        resolved_use_temporal_comparator=False,
        resolved_use_predicate_comparator=False,
        train_batch_size=None,
        balanced_sampler=False,
        weighted_label_loss=False,
        class_weighting="none",
        save_selected_checkpoint=False,
        output_json=None,
        output_predictions_json=None,
        output_ood_json=None,
        output_ood_predictions_json=None,
        ranking_weight=0.0,
        reason_router_a0_reference_predictions=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolve(args, raw):
    return trainer._p2_resolve_arm_contract(args, raw, _parser())


def _patch_tensor(monkeypatch) -> None:
    monkeypatch.setattr(
        trainer.torch,
        "tensor",
        lambda value, **kwargs: {"value": value, "kwargs": kwargs},
    )


def _reason_record(row_id: str, pair_id: str, split: str, kind: str) -> dict:
    configs = {
        "none": (1, 1, 1, "SUPPORT", "SUPPORT", "none", "none"),
        "frame": (0, 1, 1, "NOT_ENTITLED", "NONE", "frame", "entity_swap"),
        "predicate": (1, 0, 1, "NOT_ENTITLED", "NONE", "predicate", "predicate_swap"),
        "sufficiency": (1, 1, 0, "NOT_ENTITLED", "NONE", "sufficiency", "evidence_deletion"),
        "support": (1, 1, 1, "SUPPORT", "SUPPORT", "none", "none"),
        "refute": (1, 1, 1, "REFUTE", "REFUTE", "none", "polarity_flip"),
    }
    frame, predicate, sufficiency, final_label, polarity, primary, intervention_type = configs[kind]
    return {
        "id": row_id,
        "pair_id": pair_id,
        "intervention_type": intervention_type,
        "final_label": final_label,
        "frame_compatible_label": frame,
        "predicate_covered_label": predicate,
        "sufficiency_label": sufficiency,
        "polarity_label": polarity,
        "primary_failure_type": primary,
        "split": split,
    }


def _reason_sidecar(record: dict, split: str, canonical_row_id: str) -> dict:
    sidecar = {
        "row_id": record["id"],
        "pair_id": record["pair_id"],
        "split": split,
        "canonical_row_id": canonical_row_id,
        "canonical_status": "PASS",
        "intervention_contract_status": "PASS",
        "frame_compatible_label": record["frame_compatible_label"],
    }
    for field in trainer.P2_GENERATOR_COMPONENT_STATUS_FIELDS:
        sidecar[field] = "PASS"
    return sidecar


def _reason_records(split: str, kinds=("frame", "predicate", "sufficiency", "support", "refute")) -> list[dict]:
    pair_id = f"{split}_pair"
    ordered_kinds = ["none", *[kind for kind in kinds if kind != "none"]]
    suffix_by_kind = {
        "none": "none",
        "frame": "entity_swap",
        "predicate": "predicate_swap",
        "sufficiency": "evidence_deletion",
        "support": "support_control",
        "refute": "polarity_flip",
    }
    return [
        _reason_record(f"{pair_id}__{suffix_by_kind[kind]}", pair_id, split, kind)
        for kind in ordered_kinds
    ]


def _sidecars(records: list[dict], split: str) -> dict[str, dict]:
    canonical_by_pair: dict[str, str] = {}
    for record in records:
        canonical_by_pair.setdefault(record["pair_id"], record["id"])
    return {
        record["id"]: _reason_sidecar(record, split, canonical_by_pair[record["pair_id"]])
        for record in records
    }


def _many_reason_records(split: str, pairs: int = 50, kinds=("frame", "predicate", "sufficiency", "support")) -> list[dict]:
    records: list[dict] = []
    for index in range(pairs):
        pair_records = _reason_records(f"{split}_{index}", kinds=kinds)
        records.extend(pair_records)
    return records

def test_normal_a3_weight_zero_rejected() -> None:
    args = _args(reason_router_weight_calibration_export=None)
    with pytest.raises(SystemExit):
        _resolve(args, ["--reason-loss-weight", "0.0"])


def test_calibration_a3_weight_zero_allowed() -> None:
    args = _args()
    contract = _resolve(args, ["--reason-loss-weight", "0.0", "--reason-router-weight-calibration-export", "unit.json"])
    assert contract["reason_loss_weight"] == 0.0


def test_calibration_bypasses_positive_weight_gate_only() -> None:
    args = _args(use_intervention_loss=True)
    with pytest.raises(SystemExit):
        _resolve(
            args,
            [
                "--reason-loss-weight",
                "0.0",
                "--reason-router-weight-calibration-export",
                "unit.json",
            ],
        )


def test_calibration_accepts_legacy_raw_comparator_defaults_when_resolved_false() -> None:
    args = _args(
        use_temporal_comparator=True,
        use_predicate_comparator=True,
        resolved_use_temporal_comparator=False,
        resolved_use_predicate_comparator=False,
    )
    assert trainer._p3w1_validate_calibration_only_args(args) == Path("unit.json")


def test_calibration_rejects_resolved_temporal_comparator_true() -> None:
    with pytest.raises(ValueError, match="resolved temporal comparator"):
        trainer._p3w1_validate_calibration_only_args(
            _args(
                use_temporal_comparator=True,
                resolved_use_temporal_comparator=True,
            )
        )


def test_calibration_rejects_resolved_predicate_comparator_true() -> None:
    with pytest.raises(ValueError, match="resolved predicate comparator"):
        trainer._p3w1_validate_calibration_only_args(
            _args(
                use_predicate_comparator=True,
                resolved_use_predicate_comparator=True,
            )
        )


def test_explicit_temporal_comparator_cli_still_rejected_by_p2_resolver() -> None:
    with pytest.raises(SystemExit):
        _resolve(
            _args(),
            [
                "--reason-loss-weight",
                "0.0",
                "--reason-router-weight-calibration-export",
                "unit.json",
                "--use-temporal-comparator",
            ],
        )


def test_explicit_predicate_comparator_cli_still_rejected_by_p2_resolver() -> None:
    with pytest.raises(SystemExit):
        _resolve(
            _args(),
            [
                "--reason-loss-weight",
                "0.0",
                "--reason-router-weight-calibration-export",
                "unit.json",
                "--use-predicate-comparator",
            ],
        )


def test_wrong_arm_rejected() -> None:
    args = _args(reason_router_arm="A2")
    with pytest.raises(ValueError, match="reason_router_arm"):
        trainer._p3w1_validate_calibration_only_args(args)


def test_wrong_ownership_rejected() -> None:
    args = _args(resolved_gradient_ownership_mode="joint")
    with pytest.raises(ValueError, match="gradient_ownership"):
        trainer._p3w1_validate_calibration_only_args(args)


def test_non_null_train_batch_size_rejected() -> None:
    with pytest.raises(ValueError, match="train_batch_size"):
        trainer._p3w1_validate_calibration_only_args(_args(train_batch_size=8))


def test_balanced_sampler_rejected() -> None:
    with pytest.raises(ValueError, match="balanced_sampler"):
        trainer._p3w1_validate_calibration_only_args(_args(balanced_sampler=True))


def test_normal_output_checkpoint_option_rejected() -> None:
    with pytest.raises(ValueError, match="output_json"):
        trainer._p3w1_validate_calibration_only_args(_args(output_json=Path("report.json")))
    with pytest.raises(ValueError, match="save_checkpoint_path"):
        trainer._p3w1_validate_calibration_only_args(_args(save_checkpoint_path=Path("model.pt")))


def test_existing_export_path_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unit.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        trainer._p3w1_validate_calibration_only_args(_args(reason_router_weight_calibration_export=path))


def test_calibration_rejects_non_null_a0_reference_prediction_path() -> None:
    with pytest.raises(ValueError, match="A0 reference predictions are forbidden"):
        trainer._p3w1_validate_calibration_only_args(
            _args(reason_router_a0_reference_predictions=Path("a0.json"))
        )


def test_calibration_accepts_a0_reference_path_unset() -> None:
    assert trainer._p3w1_validate_calibration_only_args(
        _args(reason_router_a0_reference_predictions=None)
    ) == Path("unit.json")
    assert trainer._p3w1_validate_calibration_only_args(
        _args(reason_router_a0_reference_predictions="")
    ) == Path("unit.json")


def test_execution_commit_without_export_rejected() -> None:
    with pytest.raises(ValueError, match="requires calibration export"):
        trainer._p3w1_validate_calibration_only_args(
            _args(
                reason_router_weight_calibration_export=None,
                reason_router_weight_calibration_execution_commit=EXEC,
            )
        )


def test_export_without_execution_commit_rejected() -> None:
    with pytest.raises(ValueError, match="requires execution commit"):
        trainer._p3w1_validate_calibration_only_args(
            _args(reason_router_weight_calibration_execution_commit=None)
        )


def test_neither_calibration_flag_preserves_normal_flow() -> None:
    assert trainer._p3w1_validate_calibration_only_args(
        _args(
            reason_router_weight_calibration_export=None,
            reason_router_weight_calibration_execution_commit=None,
            reason_router_weight_calibration_forward_batch_size=None,
        )
    ) is None

def test_invalid_calibration_seed_rejected() -> None:
    for seed in (179, 183):
        with pytest.raises(ValueError, match="seed"):
            trainer._p3w1_validate_calibration_only_args(_args(seed=seed))


def test_invalid_resolved_split_seed_rejected() -> None:
    with pytest.raises(ValueError, match="split seed"):
        trainer._p3w1_validate_calibration_only_args(_args(resolved_split_seed=175))


def test_invalid_dev_ratio_rejected() -> None:
    with pytest.raises(ValueError, match="dev_ratio"):
        trainer._p3w1_validate_calibration_only_args(_args(dev_ratio=0.25))


def test_calibration_forward_batch_size_required() -> None:
    with pytest.raises(ValueError, match="forward batch size"):
        trainer._p3w1_validate_calibration_only_args(
            _args(reason_router_weight_calibration_forward_batch_size=None)
        )


def test_only_forward_batch_size_8_accepted() -> None:
    for size in (1, 16):
        with pytest.raises(ValueError, match="exactly 8"):
            trainer._p3w1_validate_calibration_only_args(
                _args(reason_router_weight_calibration_forward_batch_size=size)
            )
    assert trainer._p3w1_validate_calibration_only_args(
        _args(reason_router_weight_calibration_forward_batch_size=8)
    ) == Path("unit.json")


def test_normal_a1_a2_a3_still_require_a0_reference_predictions() -> None:
    for arm in ("A1", "A2", "A3"):
        with pytest.raises(ValueError, match="P2_A0_REFERENCE_REQUIRED"):
            trainer._p2_validate_a0_reference_for_universe(
                _args(
                    reason_router_arm=arm,
                    reason_router_weight_calibration_export=None,
                    reason_router_a0_reference_predictions=None,
                ),
                [],
            )


def test_normal_p2_path_still_builds_dev_supervision(monkeypatch) -> None:
    _patch_tensor(monkeypatch)
    train_records = _reason_records("train")
    dev_records = _reason_records("dev")
    sidecar_by_id = {**_sidecars(train_records, "train"), **_sidecars(dev_records, "dev")}
    train_inputs: dict = {}
    dev_inputs: dict = {}
    audit = trainer._p2_prepare_reason_supervision(
        train_records=train_records,
        dev_records=dev_records,
        train_inputs=train_inputs,
        dev_inputs=dev_inputs,
        train_source_labels=["clean_main"] * len(train_records),
        sidecar_by_id=sidecar_by_id,
        require_min_counts=True,
        min_train_count=1,
        min_dev_count=1,
        device="cpu",
    )
    assert audit["dev_reason_counts"]["FRAME"] == 1
    assert "p2_primary_reason_targets_4" in dev_inputs


def test_normal_a1_a3_path_still_applies_dev_minimum_count_gate(monkeypatch) -> None:
    _patch_tensor(monkeypatch)
    train_records = _reason_records("train")
    dev_records = _reason_records("dev", kinds=("predicate", "sufficiency", "support", "refute"))
    sidecar_by_id = {**_sidecars(train_records, "train"), **_sidecars(dev_records, "dev")}
    with pytest.raises(ValueError, match="P2_REASON_MIN_CLASS_COUNT_FAILED"):
        trainer._p2_prepare_reason_supervision(
            train_records=train_records,
            dev_records=dev_records,
            train_inputs={},
            dev_inputs={},
            train_source_labels=["clean_main"] * len(train_records),
            sidecar_by_id=sidecar_by_id,
            require_min_counts=True,
            min_train_count=1,
            min_dev_count=1,
            device="cpu",
        )


def test_normal_a1_a3_path_still_applies_dev_cohort_degeneracy_gate(monkeypatch) -> None:
    _patch_tensor(monkeypatch)
    train_records = _reason_records("train")
    dev_records = _reason_records("dev", kinds=("frame", "predicate", "sufficiency", "support"))
    sidecar_by_id = {**_sidecars(train_records, "train"), **_sidecars(dev_records, "dev")}
    with pytest.raises(ValueError, match="P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE"):
        trainer._p2_prepare_reason_supervision(
            train_records=train_records,
            dev_records=dev_records,
            train_inputs={},
            dev_inputs={},
            train_source_labels=["clean_main"] * len(train_records),
            sidecar_by_id=sidecar_by_id,
            require_min_counts=True,
            min_train_count=1,
            min_dev_count=1,
            device="cpu",
        )


def test_calibration_train_only_supervision_isolation(monkeypatch) -> None:
    _patch_tensor(monkeypatch)
    train_records = _many_reason_records("train")
    train_inputs: dict = {}
    audit, a0_audit = trainer._p3w1_prepare_train_only_reason_supervision_for_calibration(
        train_records=train_records,
        train_inputs=train_inputs,
        train_source_labels=["clean_main"] * len(train_records),
        sidecar_by_id=_sidecars(train_records, "train"),
        require_min_counts=True,
        min_train_count=50,
        device="cpu",
    )
    assert audit["calibration_data_scope"] == "TRAIN_ONLY"
    assert audit["train_reason_supervision_built"] is True
    assert audit["dev_reason_supervision_built"] is False
    assert audit["dev_counts_used_for_gate"] is False
    assert a0_audit == {"required": False, "accessed": False, "reason": "P3W1_TRAIN_ONLY_CALIBRATION"}
    assert "p2_primary_reason_targets_4" in train_inputs


def test_calibration_train_only_helper_has_no_dev_or_a0_inputs() -> None:
    names = trainer._p3w1_prepare_train_only_reason_supervision_for_calibration.__code__.co_varnames
    assert "dev_records" not in names
    assert "dev_inputs" not in names
    assert "reason_router_a0_reference_predictions" not in names


def test_calibration_train_only_supervision_does_not_invoke_a0_loader(monkeypatch) -> None:
    _patch_tensor(monkeypatch)

    def fail_a0_loader(*args, **kwargs):
        raise AssertionError("A0 reference loader must not be called")

    monkeypatch.setattr(trainer, "_p2_validate_a0_reference_for_universe", fail_a0_loader)
    train_records = _many_reason_records("train")
    _, a0_audit = trainer._p3w1_prepare_train_only_reason_supervision_for_calibration(
        train_records=train_records,
        train_inputs={},
        train_source_labels=["clean_main"] * len(train_records),
        sidecar_by_id=_sidecars(train_records, "train"),
        require_min_counts=True,
        min_train_count=50,
        device="cpu",
    )
    assert a0_audit["accessed"] is False


def test_calibration_train_only_does_not_apply_dev_minimum_or_degeneracy_gate(monkeypatch) -> None:
    _patch_tensor(monkeypatch)
    train_records = _many_reason_records("train")
    audit, _ = trainer._p3w1_prepare_train_only_reason_supervision_for_calibration(
        train_records=train_records,
        train_inputs={},
        train_source_labels=["clean_main"] * len(train_records),
        sidecar_by_id=_sidecars(train_records, "train"),
        require_min_counts=True,
        min_train_count=50,
        device="cpu",
    )
    assert audit["dev_reason_counts"] is None
    assert audit["dev_counts_used_for_gate"] is False


def test_calibration_pair_level_lineage_counts_nonzero_and_match_full_helper(monkeypatch) -> None:
    _patch_tensor(monkeypatch)
    train_records = _many_reason_records("train")
    dev_records = _many_reason_records("dev")
    train_sidecars = _sidecars(train_records, "train")
    dev_sidecars = _sidecars(dev_records, "dev")
    train_only_audit, _ = trainer._p3w1_prepare_train_only_reason_supervision_for_calibration(
        train_records=[dict(record) for record in train_records],
        train_inputs={},
        train_source_labels=["clean_main"] * len(train_records),
        sidecar_by_id=train_sidecars,
        require_min_counts=True,
        min_train_count=50,
        device="cpu",
    )
    full_audit = trainer._p2_prepare_reason_supervision(
        train_records=[dict(record) for record in train_records],
        dev_records=[dict(record) for record in dev_records],
        train_inputs={},
        dev_inputs={},
        train_source_labels=["clean_main"] * len(train_records),
        sidecar_by_id={**train_sidecars, **dev_sidecars},
        require_min_counts=True,
        min_train_count=50,
        min_dev_count=50,
        device="cpu",
    )
    assert all(count > 0 for count in train_only_audit["train_reason_counts"].values())
    assert train_only_audit["train_reason_counts"] == full_audit["train_reason_counts"]
    assert train_only_audit["train_exclusion_counts"].get("P2_CANONICAL_ROW_ID_MISMATCH", 0) == 0
    assert full_audit["train_exclusion_counts"].get("P2_CANONICAL_ROW_ID_MISMATCH", 0) == 0


def test_calibration_reason_authority_passes_with_polarity_degenerate(monkeypatch) -> None:
    _patch_tensor(monkeypatch)
    train_records = _many_reason_records("train", kinds=("frame", "predicate", "sufficiency", "support"))
    audit, _ = trainer._p3w1_prepare_train_only_reason_supervision_for_calibration(
        train_records=train_records,
        train_inputs={},
        train_source_labels=["clean_main"] * len(train_records),
        sidecar_by_id=_sidecars(train_records, "train"),
        require_min_counts=True,
        min_train_count=50,
        device="cpu",
    )
    assert audit["primary_reason_min_count_gate_pass"] is True
    assert audit["polarity_local_training_ready"] is False
    assert audit["weight_resolution_measurement_valid"] is True
    assert audit["normal_a1_a3_training_ready"] is False


def test_calibration_reason_authority_rejects_primary_count_below_50(monkeypatch) -> None:
    _patch_tensor(monkeypatch)
    train_records = _many_reason_records("train", pairs=49, kinds=("frame", "predicate", "sufficiency", "support"))
    with pytest.raises(ValueError, match="train reason count"):
        trainer._p3w1_prepare_train_only_reason_supervision_for_calibration(
            train_records=train_records,
            train_inputs={},
            train_source_labels=["clean_main"] * len(train_records),
            sidecar_by_id=_sidecars(train_records, "train"),
            require_min_counts=True,
            min_train_count=50,
            device="cpu",
        )


def test_reason_min_train_count_must_be_50() -> None:
    with pytest.raises(ValueError, match="reason_min_train_count"):
        trainer._p3w1_validate_calibration_only_args(_args(reason_min_train_count=49))


def test_calibration_rejects_intervention_self_reference_against_pair_authority() -> None:
    train_records = _reason_records("train")
    sidecars = _sidecars(train_records, "train")
    sidecars["train_pair__entity_swap"]["canonical_row_id"] = "train_pair__entity_swap"
    with pytest.raises(ValueError, match="multiple canonical_row_id"):
        trainer._p2_resolve_canonical_lineage_for_split(
            records=train_records,
            sidecar_by_id=sidecars,
            split="train",
        )


def test_calibration_rejects_canonical_row_self_anchor_corruption() -> None:
    train_records = _reason_records("train")
    sidecars = _sidecars(train_records, "train")
    sidecars["train_pair__none"]["canonical_row_id"] = "train_pair__entity_swap"
    with pytest.raises(ValueError, match="canonical target is not self-anchored"):
        trainer._p2_resolve_canonical_lineage_for_split(
            records=train_records,
            sidecar_by_id=sidecars,
            split="train",
        )


def test_train_only_and_full_helper_canonical_lineage_resolution_identical() -> None:
    train_records = _reason_records("train")
    assert trainer._p2_resolve_canonical_lineage_for_split(
        records=train_records,
        sidecar_by_id=_sidecars(train_records, "train"),
        split="train",
    ) == {"train_pair": "train_pair__none"}


def test_wrong_backbone_rejected() -> None:
    with pytest.raises(ValueError, match="backbone"):
        trainer._p3w1_validate_calibration_only_args(_args(backbone="bert"))


def test_wrong_model_name_rejected() -> None:
    with pytest.raises(ValueError, match="model_name"):
        trainer._p3w1_validate_calibration_only_args(_args(model_name="state-spaces/mamba-370m-hf"))


def test_wrong_max_length_rejected() -> None:
    for value in (64, "128"):
        with pytest.raises(ValueError, match="max_length"):
            trainer._p3w1_validate_calibration_only_args(_args(max_length=value))


def test_wrong_device_rejected() -> None:
    with pytest.raises(ValueError, match="device"):
        trainer._p3w1_validate_calibration_only_args(_args(device="cpu"))


def test_wrong_flag_source_rejected() -> None:
    with pytest.raises(ValueError, match="flag_source"):
        trainer._p3w1_validate_calibration_only_args(_args(flag_source="sidecar"))


def test_wrong_reason_router_epsilon_rejected() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        trainer._p3w1_validate_calibration_only_args(_args(reason_router_epsilon=1e-7))


def test_ordered_train_hash_deterministic() -> None:
    rows = [
        {"id": "r1", "pair_id": "p1", "final_label": "SUPPORT"},
        {"row_id": "r2", "pair_id": "p2", "final_label": "REFUTES"},
    ]
    assert trainer._p3w1_ordered_train_identity(rows) == trainer._p3w1_ordered_train_identity(list(rows))
    assert trainer._p3w1_ordered_train_identity(list(reversed(rows)))["ordered_train_row_identity_hash"] != trainer._p3w1_ordered_train_identity(rows)["ordered_train_row_identity_hash"]


def _local_counts_for_primary(primary_counts: dict[str, int]) -> dict[str, dict[int, int]]:
    frame = primary_counts["FRAME"]
    predicate = primary_counts["PREDICATE"]
    sufficiency = primary_counts["SUFFICIENCY"]
    authorized = primary_counts["AUTHORIZED"]
    return {
        "frame": {0: frame, 1: predicate + sufficiency + authorized},
        "predicate": {0: predicate, 1: sufficiency + authorized},
        "sufficiency": {0: sufficiency, 1: authorized},
        "polarity": {0: 0, 1: authorized},
    }


def _readiness_for_local_counts(local_counts: dict[str, dict[int, int]]) -> dict[str, bool]:
    return {cohort: counts[0] >= 1 and counts[1] >= 1 for cohort, counts in local_counts.items()}


def _unit(
    seed: int = 180,
    *,
    final_mean: float = 2.0,
    reason_mean: float = 4.0,
    row_hash: str = ROW_HASH,
    primary_counts: dict[str, int] | None = None,
):
    if primary_counts is None:
        primary_counts = {"FRAME": 50, "PREDICATE": 50, "SUFFICIENCY": 50, "AUTHORIZED": 50}
    local_counts = _local_counts_for_primary(primary_counts)
    local_readiness = _readiness_for_local_counts(local_counts)
    all_local_ready = all(local_readiness.values())
    row_count = sum(primary_counts.values())
    return {
        "schema_version": agg.UNIT_SCHEMA,
        "status": "PASS",
        "seed": seed,
        "unit_index": 0,
        "unit_scope": "COMPLETE_AUTHORITATIVE_TRAIN_SPLIT",
        "ordered_train_row_count": row_count,
        "ordered_train_row_identity_hash": row_hash,
        "model_mode": "train",
        "measurement_arm": "conditional_first_blocker",
        "measurement_gradient_ownership": "explicit_local",
        "reason_loss_weight_placeholder": 0.0,
        "calibration_gate_scope": "PRIMARY_REASON_CLASS_COUNTS_ONLY",
        "primary_reason_min_train_count": 50,
        "primary_reason_class_counts": primary_counts,
        "primary_reason_min_count_gate_pass": True,
        "local_binary_cohort_counts": local_counts,
        "local_binary_training_readiness": local_readiness,
        "all_local_binary_cohorts_training_ready": all_local_ready,
        "polarity_local_training_ready": local_readiness["polarity"],
        "weight_resolution_measurement_valid": True,
        "normal_a1_a3_training_ready": all_local_ready,
        "training_readiness_separate_from_weight_resolution": True,
        "architecture": "v6b_minimal",
        "backbone": "mamba",
        "model_name": "state-spaces/mamba-130m-hf",
        "max_length": 128,
        "device": "cuda",
        "flag_source": "controlled_heuristic",
        "freeze_encoder": True,
        "reason_router_epsilon": 1e-8,
        "train_batch_size": None,
        "balanced_sampler": False,
        "weighted_label_loss": False,
        "class_weighting": "none",
        "calibration_forward_batch_size": 8,
        "logical_units_per_seed": 1,
        "logical_unit_scope": "COMPLETE_AUTHORITATIVE_TRAIN_SPLIT",
        "fresh_initialization": True,
        "checkpoint_loaded": False,
        "gradient_tracking_enabled": False,
        "before_backward": True,
        "before_optimizer_step": True,
        "before_scheduler_step": True,
        "parameter_update_count": 0,
        "optimizer_step_executed": False,
        "scheduler_step_executed": False,
        "dev_forward_executed": False,
        "calibration_data_scope": "TRAIN_ONLY",
        "train_reason_supervision_built": True,
        "dev_reason_supervision_built": False,
        "dev_inputs_accessed_for_calibration": False,
        "dev_labels_used_for_calibration": False,
        "dev_counts_used_for_gate": False,
        "dev_metrics_used_for_calibration": False,
        "a0_reference_predictions_required": False,
        "a0_reference_predictions_accessed": False,
        "a0_predictions_used_for_calibration": False,
        "a0_logits_used_for_calibration": False,
        "a0_metrics_used_for_calibration": False,
        "a0_checkpoint_used_for_calibration": False,
        "external_eval_executed": False,
        "normal_training_report_written": False,
        "causal_checkpoint_written": False,
        "final_loss_mean": final_mean,
        "final_applicable_count": row_count,
        "final_loss_sum_reconstructed": final_mean * row_count,
        "final_loss_finite": True,
        "reason_loss_mean": reason_mean,
        "reason_eligible_count": row_count,
        "reason_loss_sum_reconstructed": reason_mean * row_count,
        "reason_loss_finite": True,
        "dataset_path": "data/controlled.jsonl",
        "dataset_sha256": DATA,
        "sidecar_path": "reports/sidecar.jsonl",
        "sidecar_semantic_sha256": SIDE,
        "expected_sidecar_semantic_sha256": SIDE,
        "sidecar_semantic_sha256_verified": True,
        "split_seed": 174,
        "dev_ratio": 0.2,
        "execution_commit": EXEC,
        "declared_execution_commit": EXEC,
        "execution_commit_verified": True,
        "decision": agg.UNIT_DECISION,
    }


def _validate_unit(unit: dict, *, expected_ordered_train_row_count: int = 200):
    return agg.validate_unit_artifact(
        unit,
        expected_execution_commit=EXEC,
        expected_dataset_sha256=DATA,
        expected_sidecar_semantic_sha256=SIDE,
        expected_split_seed=174,
        expected_ordered_train_row_count=expected_ordered_train_row_count,
        expected_ordered_train_row_identity_hash=ROW_HASH,
        expected_dev_ratio=0.2,
    )

def test_unit_schema_validator_accepts_valid_artifact() -> None:
    assert _validate_unit(_unit())["seed"] == 180


def test_missing_dataset_path_rejected() -> None:
    unit = _unit()
    unit.pop("dataset_path")
    with pytest.raises(ValueError, match="missing unit fields"):
        _validate_unit(unit)


def test_missing_sidecar_path_rejected() -> None:
    unit = _unit()
    unit.pop("sidecar_path")
    with pytest.raises(ValueError, match="missing unit fields"):
        _validate_unit(unit)


def test_missing_dev_ratio_rejected() -> None:
    unit = _unit()
    unit.pop("dev_ratio")
    with pytest.raises(ValueError, match="missing unit fields"):
        _validate_unit(unit)


def test_string_seed_rejected() -> None:
    unit = _unit()
    unit["seed"] = "180"
    with pytest.raises(ValueError, match="seed"):
        _validate_unit(unit)


def test_truthy_string_finite_flag_rejected() -> None:
    unit = _unit()
    unit["final_loss_finite"] = "true"
    with pytest.raises(ValueError, match="final_loss_finite"):
        _validate_unit(unit)


def test_unit_artifact_configuration_mismatch_rejected() -> None:
    for key, value in (
        ("backbone", "bert"),
        ("model_name", "state-spaces/mamba-370m-hf"),
        ("max_length", 64),
        ("device", "cpu"),
        ("flag_source", "sidecar"),
        ("reason_router_epsilon", 1e-7),
    ):
        unit = _unit()
        unit[key] = value
        with pytest.raises(ValueError, match=key):
            _validate_unit(unit)


def test_truthy_string_freeze_encoder_rejected() -> None:
    unit = _unit()
    unit["freeze_encoder"] = "true"
    with pytest.raises(ValueError, match="freeze_encoder"):
        _validate_unit(unit)


def test_non_null_train_batch_size_rejected_by_aggregate() -> None:
    unit = _unit()
    unit["train_batch_size"] = 8
    with pytest.raises(ValueError, match="train_batch_size"):
        _validate_unit(unit)


def test_unit_validator_rejects_train_only_isolation_violations() -> None:
    for key in (
        "dev_reason_supervision_built",
        "dev_labels_used_for_calibration",
        "dev_counts_used_for_gate",
        "a0_reference_predictions_accessed",
    ):
        unit = _unit()
        unit[key] = True
        with pytest.raises(ValueError, match=key):
            _validate_unit(unit)


def test_unit_validator_rejects_truthy_string_isolation_flags() -> None:
    for key in (
        "train_reason_supervision_built",
        "dev_reason_supervision_built",
        "a0_reference_predictions_accessed",
    ):
        unit = _unit()
        unit[key] = "true"
        with pytest.raises(ValueError, match=key):
            _validate_unit(unit)


def test_unit_validator_rejects_non_train_only_scope() -> None:
    unit = _unit()
    unit["calibration_data_scope"] = "TRAIN_AND_DEV"
    with pytest.raises(ValueError, match="calibration_data_scope"):
        _validate_unit(unit)


def test_unit_validator_accepts_weight_measurement_with_polarity_not_ready() -> None:
    unit = _unit()
    summary = _validate_unit(unit)
    assert summary["polarity_local_training_ready"] is False
    assert summary["normal_a1_a3_training_ready"] is False


def test_unit_validator_rejects_inconsistent_local_readiness_boolean() -> None:
    unit = _unit()
    unit["local_binary_training_readiness"]["polarity"] = True
    with pytest.raises(ValueError, match="local_binary_training_readiness"):
        _validate_unit(unit)


def test_unit_validator_rejects_primary_reason_count_below_50() -> None:
    unit = _unit()
    unit["primary_reason_class_counts"]["PREDICATE"] = 49
    with pytest.raises(ValueError, match="primary_reason_class_counts"):
        _validate_unit(unit)


def test_observed_sidecar_sha_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="observed sidecar"):
        trainer._p3w1_verify_sidecar_semantic_sha(SIDE, "e" * 64)


def test_declared_observed_execution_commit_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="declared execution commit"):
        trainer._p3w1_verify_execution_commit(EXEC, "e" * 40)


def test_invalid_observed_commit_rejected() -> None:
    with pytest.raises(ValueError, match="40 hexadecimal"):
        trainer._p3w1_verify_execution_commit(EXEC, "not-a-commit")


def test_observed_git_head_helper_rejects_invalid_output(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="not-a-commit\n")

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="observed git HEAD"):
        trainer._p3w1_observed_git_head(Path("."))


def test_sidecar_verified_flag_exact_bool_required() -> None:
    unit = _unit()
    unit["sidecar_semantic_sha256_verified"] = "true"
    with pytest.raises(ValueError, match="sidecar_semantic_sha256_verified"):
        _validate_unit(unit)


def test_execution_commit_verified_flag_exact_bool_required() -> None:
    unit = _unit()
    unit["execution_commit_verified"] = "true"
    with pytest.raises(ValueError, match="execution_commit_verified"):
        _validate_unit(unit)


def test_missing_verified_flags_rejected() -> None:
    for key in ("sidecar_semantic_sha256_verified", "execution_commit_verified"):
        unit = _unit()
        unit.pop(key)
        with pytest.raises(ValueError, match="missing unit fields"):
            _validate_unit(unit)


def test_aggregate_unit_with_mismatched_batch_size_rejected(tmp_path: Path) -> None:
    unit = _unit(180)
    unit["calibration_forward_batch_size"] = 16
    with pytest.raises(ValueError, match="calibration_forward_batch_size"):
        _aggregate(tmp_path, [unit, _unit(181), _unit(182)])


def test_logical_unit_remains_one() -> None:
    unit = _unit()
    summary = _validate_unit(unit)
    assert unit["logical_units_per_seed"] == 1
    assert unit["logical_unit_scope"] == "COMPLETE_AUTHORITATIVE_TRAIN_SPLIT"
    assert summary["row_count"] == 200

def test_zero_reason_count_rejected() -> None:
    unit = _unit()
    unit["reason_eligible_count"] = 0
    unit["reason_loss_sum_reconstructed"] = 0.0
    with pytest.raises(ValueError, match="reason_eligible_count"):
        _validate_unit(unit)


def test_final_applicable_count_must_equal_ordered_train_row_count() -> None:
    unit = _unit()
    unit["ordered_train_row_count"] = 199
    with pytest.raises(ValueError, match="final_applicable_count"):
        _validate_unit(unit, expected_ordered_train_row_count=199)


def test_reason_eligible_count_must_equal_primary_reason_sum() -> None:
    unit = _unit()
    unit["reason_eligible_count"] = 199
    unit["reason_loss_sum_reconstructed"] = unit["reason_loss_mean"] * 199
    with pytest.raises(ValueError, match="reason_eligible_count"):
        _validate_unit(unit)


def test_reason_eligible_count_must_not_exceed_final_applicable_count() -> None:
    unit = _unit()
    unit["ordered_train_row_count"] = 199
    unit["final_applicable_count"] = 199
    unit["final_loss_sum_reconstructed"] = unit["final_loss_mean"] * 199
    with pytest.raises(ValueError, match="reason_eligible_count"):
        _validate_unit(unit, expected_ordered_train_row_count=199)


def test_frame_cohort_counts_must_match_primary_counts() -> None:
    unit = _unit()
    unit["local_binary_cohort_counts"]["frame"][0] = 51
    with pytest.raises(ValueError, match="frame cohort"):
        _validate_unit(unit)


def test_predicate_cohort_counts_must_match_primary_counts() -> None:
    unit = _unit()
    unit["local_binary_cohort_counts"]["predicate"][1] = 101
    with pytest.raises(ValueError, match="predicate cohort"):
        _validate_unit(unit)


def test_sufficiency_cohort_counts_must_match_primary_counts() -> None:
    unit = _unit()
    unit["local_binary_cohort_counts"]["sufficiency"][1] = 51
    with pytest.raises(ValueError, match="sufficiency cohort"):
        _validate_unit(unit)


def test_polarity_applicable_count_must_match_authorized_count() -> None:
    unit = _unit()
    unit["local_binary_cohort_counts"]["polarity"] = {0: 1, 1: 50}
    unit["local_binary_training_readiness"]["polarity"] = True
    unit["all_local_binary_cohorts_training_ready"] = True
    unit["polarity_local_training_ready"] = True
    unit["normal_a1_a3_training_ready"] = True
    with pytest.raises(ValueError, match="polarity applicable count"):
        _validate_unit(unit)


def test_nonfinite_mean_rejected() -> None:
    with pytest.raises(ValueError, match="final_loss_mean"):
        _validate_unit(_unit(final_mean=float("inf")))


def test_reconstructed_sum_mismatch_rejected() -> None:
    unit = _unit()
    unit["reason_loss_sum_reconstructed"] += 0.01
    with pytest.raises(ValueError, match="reason reconstructed sum mismatch"):
        _validate_unit(unit)


def _write_units(tmp_path: Path, units: list[dict]) -> list[Path]:
    paths = []
    for index, unit in enumerate(units):
        path = tmp_path / f"unit{index}.json"
        path.write_text(json.dumps(unit) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _aggregate(tmp_path: Path, units: list[dict], **overrides):
    expected = dict(
        expected_split_seed=174,
        expected_ordered_train_row_count=200,
        expected_ordered_train_row_identity_hash=ROW_HASH,
        expected_dev_ratio=0.2,
    )
    expected.update(overrides)
    return agg.build_aggregate(
        unit_paths=_write_units(tmp_path, units),
        output_json=tmp_path / "aggregate.json",
        expected_execution_commit=EXEC,
        expected_dataset_sha256=DATA,
        expected_sidecar_semantic_sha256=SIDE,
        **expected,
    )


def test_aggregate_requires_exact_seeds_180_181_182(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seeds"):
        _aggregate(tmp_path, [_unit(180), _unit(181), _unit(183)])


def test_duplicate_seed_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seeds"):
        _aggregate(tmp_path, [_unit(180), _unit(180), _unit(182)])


def test_mismatched_train_identity_hash_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity hash"):
        _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182, row_hash="e" * 64)])


def test_mismatched_execution_data_sidecar_split_identity_rejected() -> None:
    for key, value in (("execution_commit", "f" * 40), ("dataset_sha256", "f" * 64), ("sidecar_semantic_sha256", "f" * 64), ("split_seed", 999)):
        unit = _unit()
        unit[key] = value
        with pytest.raises(ValueError):
            _validate_unit(unit)


def test_pooled_estimator_uses_count_weighting(tmp_path: Path) -> None:
    result = _aggregate(
        tmp_path,
        [
            _unit(180, final_mean=10, reason_mean=2),
            _unit(181, final_mean=1, reason_mean=1),
            _unit(182, final_mean=2, reason_mean=2),
        ],
    )
    assert result["mu_final"] == pytest.approx(((10 * 200) + (1 * 200) + (2 * 200)) / 600)
    assert result["mu_reason"] == pytest.approx(((2 * 200) + (1 * 200) + (2 * 200)) / 600)


def test_mean_of_means_counterexample_with_different_counts_is_rejected(tmp_path: Path) -> None:
    units = [
        _unit(180, primary_counts={"FRAME": 50, "PREDICATE": 50, "SUFFICIENCY": 50, "AUTHORIZED": 60}),
        _unit(181, primary_counts={"FRAME": 50, "PREDICATE": 60, "SUFFICIENCY": 50, "AUTHORIZED": 50}),
        _unit(182, primary_counts={"FRAME": 50, "PREDICATE": 50, "SUFFICIENCY": 50, "AUTHORIZED": 60}),
    ]
    with pytest.raises(ValueError, match="primary reason class counts"):
        _aggregate(tmp_path, units, expected_ordered_train_row_count=210)


def test_aggregate_resolved_weight_exact_recomputation(tmp_path: Path) -> None:
    result = _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)])
    assert result["resolved_reason_loss_weight"] == pytest.approx(result["mu_final"] / result["mu_reason"])


def test_wrong_expected_train_row_count_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="row count"):
        _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)], expected_ordered_train_row_count=201)


def test_wrong_expected_train_row_hash_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity hash"):
        _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)], expected_ordered_train_row_identity_hash="e" * 64)


def test_wrong_expected_dev_ratio_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected dev ratio"):
        _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)], expected_dev_ratio=0.25)


def test_expected_split_seed_173_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 174"):
        _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)], expected_split_seed=173)


def test_expected_split_seed_175_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 174"):
        _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)], expected_split_seed=175)


def test_boolean_expected_split_seed_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact integer"):
        _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)], expected_split_seed=True)


def test_aggregate_records_expected_split_seed_and_verified_flag(tmp_path: Path) -> None:
    result = _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)])
    assert result["split_seed"] == 174
    assert result["expected_split_seed"] == 174
    assert result["split_seed_verified"] is True


def test_aggregate_records_fixed_configuration(tmp_path: Path) -> None:
    result = _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)])
    assert result["architecture"] == "v6b_minimal"
    assert result["backbone"] == "mamba"
    assert result["model_name"] == "state-spaces/mamba-130m-hf"
    assert result["max_length"] == 128
    assert result["device"] == "cuda"
    assert result["flag_source"] == "controlled_heuristic"
    assert result["freeze_encoder"] is True
    assert result["reason_router_epsilon"] == 1e-8
    assert result["train_batch_size"] is None
    assert result["balanced_sampler"] is False
    assert result["weighted_label_loss"] is False
    assert result["class_weighting"] == "none"


def test_aggregate_records_all_train_only_a0_isolation_fields(tmp_path: Path) -> None:
    result = _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)])
    assert result["calibration_data_scope"] == "TRAIN_ONLY"
    assert result["all_train_reason_supervision_built"] is True
    assert result["all_dev_reason_supervision_absent"] is True
    assert result["all_dev_inputs_unaccessed"] is True
    assert result["all_dev_labels_unused"] is True
    assert result["all_dev_counts_unused"] is True
    assert result["all_dev_metrics_unused"] is True
    assert result["all_a0_reference_predictions_unrequired"] is True
    assert result["all_a0_reference_predictions_unaccessed"] is True
    assert result["all_a0_predictions_unused"] is True
    assert result["all_a0_logits_unused"] is True
    assert result["all_a0_metrics_unused"] is True
    assert result["all_a0_checkpoints_unused"] is True


def test_aggregate_records_common_supervision_count_authority(tmp_path: Path) -> None:
    result = _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)])
    assert result["primary_reason_class_counts"] == {
        "FRAME": 50,
        "PREDICATE": 50,
        "SUFFICIENCY": 50,
        "AUTHORIZED": 50,
    }
    assert result["local_binary_cohort_counts"] == {
        "frame": {0: 50, 1: 150},
        "predicate": {0: 50, 1: 100},
        "sufficiency": {0: 50, 1: 50},
        "polarity": {0: 0, 1: 50},
    }
    assert result["local_binary_training_readiness"] == {
        "frame": True,
        "predicate": True,
        "sufficiency": True,
        "polarity": False,
    }


def test_aggregate_computes_weight_when_polarity_readiness_false(tmp_path: Path) -> None:
    result = _aggregate(tmp_path, [_unit(180), _unit(181), _unit(182)])
    assert result["all_weight_resolution_measurements_valid"] is True
    assert result["all_primary_reason_min_count_gates_pass"] is True
    assert result["normal_a1_a3_training_ready"] is False
    assert result["all_polarity_local_training_ready"] is False
    assert result["A1_A3_released"] is False
    assert result["resolved_reason_loss_weight"] == pytest.approx(result["mu_final"] / result["mu_reason"])


def test_aggregate_rejects_cross_seed_primary_reason_count_mismatch(tmp_path: Path) -> None:
    units = [
        _unit(180, primary_counts={"FRAME": 50, "PREDICATE": 50, "SUFFICIENCY": 50, "AUTHORIZED": 60}),
        _unit(181, primary_counts={"FRAME": 50, "PREDICATE": 60, "SUFFICIENCY": 50, "AUTHORIZED": 50}),
        _unit(182, primary_counts={"FRAME": 50, "PREDICATE": 50, "SUFFICIENCY": 50, "AUTHORIZED": 60}),
    ]
    with pytest.raises(ValueError, match="primary reason class counts"):
        _aggregate(tmp_path, units, expected_ordered_train_row_count=210)


def test_aggregate_rejects_cross_seed_local_cohort_count_mismatch(tmp_path: Path) -> None:
    unit = _unit(181)
    unit["local_binary_cohort_counts"]["polarity"] = {0: 1, 1: 49}
    unit["local_binary_training_readiness"]["polarity"] = True
    unit["all_local_binary_cohorts_training_ready"] = True
    unit["polarity_local_training_ready"] = True
    unit["normal_a1_a3_training_ready"] = True
    with pytest.raises(ValueError, match="local binary cohort counts"):
        _aggregate(tmp_path, [_unit(180), unit, _unit(182)])


def test_aggregate_rejects_seed_readiness_mismatch_via_unit_validation(tmp_path: Path) -> None:
    unit = _unit(181)
    unit["local_binary_training_readiness"]["polarity"] = True
    with pytest.raises(ValueError, match="local_binary_training_readiness"):
        _aggregate(tmp_path, [_unit(180), unit, _unit(182)])

def test_unit_dev_ratio_mismatch_rejected(tmp_path: Path) -> None:
    unit = _unit(180)
    unit["dev_ratio"] = 0.25
    with pytest.raises(ValueError, match="dev_ratio"):
        _aggregate(tmp_path, [unit, _unit(181), _unit(182)])


def test_cross_seed_consistent_wrong_universe_rejected(tmp_path: Path) -> None:
    wrong = "e" * 64
    with pytest.raises(ValueError, match="expected authority"):
        _aggregate(tmp_path, [_unit(180, row_hash=wrong), _unit(181, row_hash=wrong), _unit(182, row_hash=wrong)])

def test_aggregate_overwrite_rejected_content_unchanged_and_temp_cleaned(tmp_path: Path) -> None:
    output = tmp_path / "aggregate.json"
    output.write_text("original\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        agg.write_json_atomic_no_overwrite(output, {"ok": True})
    assert output.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".aggregate.json.tmp.*")) == []


def test_trainer_unit_writer_overwrite_rejected_content_unchanged_and_temp_cleaned(tmp_path: Path) -> None:
    output = tmp_path / "unit.json"
    output.write_text("original\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        trainer._p3w1_write_json_atomic_no_overwrite(output, {"ok": True})
    assert output.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".unit.json.tmp.*")) == []