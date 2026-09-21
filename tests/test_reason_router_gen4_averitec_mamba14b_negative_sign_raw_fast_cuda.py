from __future__ import annotations

import inspect
import json
import sys
import types

import torch


# The real repository provides this dependency. A minimal import stub keeps the
# candidate tests self-contained while testing runner-local contracts.
bridge_stub = types.ModuleType(
    "scripts.reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda"
)
sys.modules[
    "scripts.reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda"
] = bridge_stub

from scripts import (  # noqa: E402
    reason_router_gen4_averitec_mamba14b_negative_sign_raw_fast_cuda
    as runner,
)


def test_protocol_is_exact() -> None:
    runner.validate_protocol()
    assert runner.SCALE == "mamba14b"
    assert runner.N == 462
    assert runner.CONDITIONS == (
        "native",
        "dominant_neutralized",
        "dominant_control",
    )
    assert runner.SELECTED_PLANE == "P5"
    assert runner.CONTROL_PLANE == "P4"
    assert runner.INTERVENTION_LAYER == 35
    assert runner.TARGET_OFFSET == 2
    assert runner.ITEMS_PER_SHARD == 231
    assert runner.FORWARDS_PER_SHARD == 693
    assert runner.TOTAL_FORWARD_BUDGET == 1386


def test_shards_are_exact_disjoint_and_complete() -> None:
    a = set(range(
        runner.SHARDS[0]["start_index"],
        runner.SHARDS[0]["end_index"],
    ))
    b = set(range(
        runner.SHARDS[1]["start_index"],
        runner.SHARDS[1]["end_index"],
    ))
    assert not (a & b)
    assert sorted(a | b) == list(range(462))
    assert runner.SHARDS[0]["physical_device"] == 0
    assert runner.SHARDS[1]["physical_device"] == 1


def test_frozen_gate_identity_is_exact() -> None:
    assert runner.REQUIRED_GATE_FREEZE_COMMIT == (
        "c260488e91c610eaaea1d7a194a46003c96660c5"
    )
    assert runner.COHORT_GIT_BLOB == (
        "1628526a4eaea3e81497e304f7cb83fd4f911ad6"
    )
    assert runner.GATE_MANIFEST_GIT_BLOB == (
        "8bbd6396341e67f5616914a8f16056017beb369a"
    )
    assert runner.COHORT_SHA256 == (
        "86da8d1435e5722bef5f35db204d097600f152361b147e84d0bb3bbe714c4f00"
    )
    assert runner.GATE_MANIFEST_SHA256 == (
        "88e9f668ac430c7d59796e4b35da790375f6af92349eca3ab021e3935af2e604"
    )


def test_checkpoint_and_model_identity_are_exact() -> None:
    assert runner.CHECKPOINT_SHA256 == (
        "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a"
    )
    assert runner.HF_REPO == "state-spaces/mamba-1.4b-hf"
    assert runner.HF_REVISION == (
        "6e46eae61c27280517feef46f536d16b91076f08"
    )


def test_primary_contract_is_reserved_not_executed() -> None:
    assert runner.PRIMARY_ENDPOINT_RESERVED == (
        "D_EXT_14B=M_native-M_control"
    )
    assert runner.PRIMARY_ALTERNATIVE_RESERVED == "less"
    assert runner.PRIMARY_P_VALUE_COUNT_RESERVED == 1
    source = inspect.getsource(runner).lower()
    assert "scipy" not in source
    assert "ttest_1samp" not in source
    assert "p_value_count_executed\": 0" in source
    assert "scientific_conclusion\": none" in source


def test_rows_do_not_materialize_d_ext() -> None:
    source = inspect.getsource(runner.serialize_output)
    assert '"D_EXT" not in serialized' in source
    assert '"D_EXT_14B" not in serialized' in source


def test_condition_path_uses_frozen_hook_and_historical_forward() -> None:
    source = inspect.getsource(runner.run_condition)
    assert "bridge_scale.install_behavior_hook" in source
    assert "selected_plane=SELECTED_PLANE" in source
    assert "control_plane=CONTROL_PLANE" in source
    assert 'spec["adapter"].historical_forward' in source
    assert "torch.inference_mode()" in source


def test_correct_margin() -> None:
    assert runner.correct_margin([3.0, 1.0, 0.0], 0) == 2.0
    assert runner.correct_margin([3.0, 1.0, 5.0], 1) == -4.0


def test_model_rows_preserve_text() -> None:
    cohort = [
        {
            "example_id": "averitec_dev_001",
            "claim": "claim",
            "evidence": "evidence",
        }
    ] * 462
    # Uniqueness is validated in the real frozen cohort loader; this unit test
    # only verifies the model-row serialization contract.
    rows = runner.model_rows(cohort)
    assert len(rows) == 462
    assert rows[0] == {
        "row_id": "averitec_dev_001",
        "source_pair_id": "averitec_dev_001",
        "contrast_cell_id": "AVERITEC",
        "claim": "claim",
        "evidence": "evidence",
    }


def test_output_contract_is_exactly_four_files() -> None:
    assert runner.ROW_FILE == "external_transfer_rows.jsonl"
    assert runner.SUMMARY_FILE == "raw_external_transfer_summary.json"
    assert runner.MANIFEST_FILE == "artifact_manifest.json"
    assert runner.CHECKSUM_FILE == "SHA256SUMS.txt"


def test_no_historical_averitec_result_dependency() -> None:
    source = inspect.getsource(runner)
    forbidden = (
        "reason_router_gen4_averitec_130m370m_external_transfer",
        "external_transfer_analysis.json",
        "confirmation_inference.json",
        "static_post_result_analysis",
    )
    for token in forbidden:
        assert token not in source
