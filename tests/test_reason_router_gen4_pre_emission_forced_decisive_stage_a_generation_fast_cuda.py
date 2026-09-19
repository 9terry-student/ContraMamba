from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_fast_cuda
    as subject,
)


def fake_gate_artifact():
    return {
        "commitment_grammar": {
            "surfaces": {
                "REFUTE":
                    "Based on the evidence, the verdict is REFUTE.",
                "NOT_ENTITLED":
                    "Based on the evidence, the verdict is NOT_ENTITLED.",
                "SUPPORT":
                    "Based on the evidence, the verdict is SUPPORT.",
            },
            "token_ids": {
                "REFUTE": [
                    15545, 327, 253, 1941, 13, 253, 11844, 310,
                    5689, 39, 23638, 15,
                ],
                "NOT_ENTITLED": [
                    15545, 327, 253, 1941, 13, 253, 11844, 310,
                    5803, 64, 3489, 1433, 23167, 15,
                ],
                "SUPPORT": [
                    15545, 327, 253, 1941, 13, 253, 11844, 310,
                    9242, 27425, 15,
                ],
            },
        },
    }


def fake_cohort():
    rows = []
    counts = {
        "REFUTE": 305,
        "SUPPORT": 122,
        "NOT_ENTITLED": 35,
    }
    index = 0
    for label in subject.GOLD_CLASS_ORDER:
        for _ in range(counts[label]):
            rows.append({
                "example_id": f"fake_{index:03d}",
                "correct_label": label,
            })
            index += 1
    assert len(rows) == subject.N
    return rows


def small_forced_grammar():
    common = [10, 11, 12, 13, 14, 15, 16, 17]
    return {
        "REFUTE": common + [101, 201],
        "SUPPORT": common + [103, 203],
    }


def test_protocol_constants_are_frozen():
    subject.validate_protocol()

    assert subject.REQUIRED_ANCESTOR == (
        "e04179dc8cd1454caf8bd71d36114dc0b1da7b5d"
    )
    assert subject.DESIGN_ARTIFACT_GIT_BLOB == (
        "b4f9c3ebf2e43a556fbfeb43c5146b4a04f4c620"
    )
    assert subject.BASE_GENERATOR_GIT_BLOB == (
        "aa6725bdc1cbfa43547e89293f3b7911a4cdd6ed"
    )

    assert subject.GOLD_CLASS_ORDER == (
        "REFUTE",
        "NOT_ENTITLED",
        "SUPPORT",
    )
    assert subject.FORCED_CLASS_ORDER == (
        "REFUTE",
        "SUPPORT",
    )
    assert subject.base.SELECTED_PLANE == "P3"
    assert subject.base.CONTROL_PLANE == "P5"
    assert subject.base.EARLY_BLOCK == 35
    assert subject.base.LATE_BLOCK == 47
    assert subject.base.LATE_READOUT_REPRO_ATOL == 1.0e-5

    assert subject.OBSERVATION_OFFSETS == (-4, -3, -2, -1)
    assert subject.OBSERVATION_GENERATED_PREFIX_LENGTHS == (
        5, 6, 7, 8,
    )
    assert subject.PRIMARY_STAGE_A_PARTITION == "confirmatory"
    assert subject.PRIMARY_STAGE_A_SIGNAL == "p3_component_l2"
    assert subject.PRIMARY_STAGE_A_TEST == "two_sided_welch_t"
    assert subject.PRIMARY_STAGE_A_MULTIPLICITY == "holm"
    assert subject.PRIMARY_STAGE_A_P_VALUE_COUNT == 4
    assert subject.PRIMARY_STAGE_A_ALPHA == 0.05


def test_forced_grammar_manifest_is_exact_and_excludes_abstention():
    manifest = subject.forced_grammar_manifest(
        fake_gate_artifact()
    )

    assert manifest["class_order"] == ["REFUTE", "SUPPORT"]
    assert manifest["surfaces"] == subject.FORCED_SURFACES
    assert manifest["token_ids"] == subject.FORCED_TOKEN_IDS
    assert manifest["not_entitled_available"] is False
    assert manifest["longest_common_prefix_length"] == 8
    assert manifest[
        "unique_commitment_token_index_zero_based"
    ] == {"REFUTE": 8, "SUPPORT": 8}
    assert manifest["grammar_sha256"] == (
        "62c9c53871f68fcc0f57d38c96b75d2391ee6ddc473a4325a522f91c9249bd00"
    )


def test_partition_logic_remains_three_gold_label_and_response_blind():
    cohort = fake_cohort()
    assignments, manifest = (
        subject.base.build_partition_assignments(cohort)
    )

    assert len(assignments) == 462
    assert manifest["partition_counts"] == {
        "calibration": 231,
        "confirmatory": 231,
    }
    assert manifest["partition_label_counts"] == {
        "calibration": {
            "REFUTE": 153,
            "NOT_ENTITLED": 17,
            "SUPPORT": 61,
        },
        "confirmatory": {
            "REFUTE": 152,
            "NOT_ENTITLED": 18,
            "SUPPORT": 61,
        },
    }

    source = inspect.getsource(
        subject.base.build_partition_assignments
    ).lower()
    for forbidden in (
        "generated",
        "p3_",
        "margin",
        "prediction",
        "correctness",
        "confidence",
    ):
        assert forbidden not in source


def test_grammar_completion_and_compatibility_are_two_class_only():
    grammar = small_forced_grammar()

    assert (
        subject.completed_label(grammar["REFUTE"], grammar)
        == "REFUTE"
    )
    assert (
        subject.completed_label(grammar["REFUTE"][:-1], grammar)
        is None
    )
    assert subject.compatible_labels([], grammar) == (
        "REFUTE",
        "SUPPORT",
    )
    assert subject.compatible_labels(
        grammar["SUPPORT"][:8],
        grammar,
    ) == ("REFUTE", "SUPPORT")
    assert subject.compatible_labels(
        grammar["SUPPORT"][:9],
        grammar,
    ) == ("SUPPORT",)


def test_branch_tokens_are_exactly_refute_and_support():
    assert subject.branch_token_by_label(
        small_forced_grammar()
    ) == {
        "REFUTE": 101,
        "SUPPORT": 103,
    }


def test_greedy_allowed_token_remains_restricted_and_deterministic():
    logits = torch.full((200,), -100.0)
    logits[101] = 9.0
    logits[103] = 9.0
    logits[102] = 1000.0

    chosen, values = subject.base.greedy_allowed_token(
        logits,
        [103, 101],
    )

    assert chosen == 101
    assert set(values) == {"101", "103"}


@pytest.mark.parametrize(
    "gold,emitted,group,category",
    [
        (
            "REFUTE",
            "REFUTE",
            "supported",
            "supported_forced_decisive",
        ),
        (
            "REFUTE",
            "SUPPORT",
            "unsupported",
            "unsupported_forced_decisive",
        ),
        (
            "SUPPORT",
            "SUPPORT",
            "supported",
            "supported_forced_decisive",
        ),
        (
            "SUPPORT",
            "REFUTE",
            "unsupported",
            "unsupported_forced_decisive",
        ),
        (
            "NOT_ENTITLED",
            "REFUTE",
            "unsupported",
            "unsupported_forced_decisive",
        ),
        (
            "NOT_ENTITLED",
            "SUPPORT",
            "unsupported",
            "unsupported_forced_decisive",
        ),
    ],
)
def test_forced_outcome_classification(
    gold,
    emitted,
    group,
    category,
):
    out = subject.classify_forced_outcome(gold, emitted)

    assert out["emitted_is_decisive"] is True
    assert out["primary_stage_a_group"] == group
    assert out["outcome_category"] == category
    assert out["t_star_zero_based"] == 8
    assert (
        out["unsupported_forced_decisive_commitment"]
        != out["supported_forced_decisive_commitment"]
    )


def test_not_entitled_cannot_be_emitted():
    with pytest.raises(
        subject.ForcedDecisiveStageAError,
        match="EMITTED",
    ):
        subject.classify_forced_outcome(
            "REFUTE",
            "NOT_ENTITLED",
        )


def test_m47_margin_is_support_minus_refute():
    logits = {
        "REFUTE": 3.25,
        "SUPPORT": 5.75,
    }
    assert subject.m47_support_minus_refute(logits) == 2.5


def test_observation_schedule_matches_frozen_tstar():
    assert [
        subject.base.observation_offset_for_generated_prefix_length(x)
        for x in range(10)
    ] == [
        None, None, None, None, None,
        -4, -3, -2, -1, None,
    ]


def _synthetic_row(
    *,
    index,
    partition,
    group,
    emitted,
):
    unsupported = group == "unsupported"
    supported = group == "supported"
    return {
        "schema_version": subject.ROW_SCHEMA,
        "example_id": f"row_{index:03d}",
        "partition": partition,
        "emitted_commitment_label": emitted,
        "emitted_is_decisive": True,
        "unsupported_forced_decisive_commitment": unsupported,
        "supported_forced_decisive_commitment": supported,
        "primary_stage_a_group": group,
        "outcome_category": (
            "unsupported_forced_decisive"
            if unsupported
            else "supported_forced_decisive"
        ),
        "t_star_zero_based": 8,
        "observations": [
            {"relative_offset": x}
            for x in subject.OBSERVATION_OFFSETS
        ],
        "generate_api_call_count": 0,
        "manual_constrained_decoding": True,
        "forced_decisive_two_class_grammar": True,
        "not_entitled_available": False,
        "sampling": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "beam_search": False,
        "repetition_penalty": None,
        "logit_bias": None,
        "generated_token_count": 12,
        "scientific_model_forward_count": 12,
    }


def test_raw_row_validation_preserves_groups_without_inference():
    rows = []
    for index in range(462):
        partition = (
            "calibration"
            if index < 231
            else "confirmatory"
        )
        group = (
            "unsupported"
            if index % 3 == 0
            else "supported"
        )
        emitted = (
            "REFUTE"
            if index % 2 == 0
            else "SUPPORT"
        )
        rows.append(
            _synthetic_row(
                index=index,
                partition=partition,
                group=group,
                emitted=emitted,
            )
        )

    out = subject.validate_raw_rows(rows)

    assert sum(
        out["forced_decisive_group_counts"].values()
    ) == 462
    assert sum(
        out[
            "confirmatory_forced_decisive_group_counts"
        ].values()
    ) == 231
    assert (
        out["scientific_model_forward_count_this_run"]
        == 462 * 12
    )
    assert out["generate_api_call_count"] == 0


def test_raw_row_validation_allows_empty_one_group_for_fail_closed_later():
    rows = []
    for index in range(462):
        partition = (
            "calibration"
            if index < 231
            else "confirmatory"
        )
        rows.append(
            _synthetic_row(
                index=index,
                partition=partition,
                group="unsupported",
                emitted="REFUTE",
            )
        )

    out = subject.validate_raw_rows(rows)
    assert out["forced_decisive_group_counts"] == {
        "unsupported": 462,
    }
    assert out[
        "confirmatory_forced_decisive_group_counts"
    ] == {
        "unsupported": 231,
    }


def test_production_source_forbids_training_statistics_and_decode_tuning():
    source = inspect.getsource(subject).lower()

    assert ".generate(" not in source
    assert "optimizer" not in source
    assert ".backward(" not in source
    assert "scipy" not in source
    assert "ttest" not in source
    assert "mannwhitney" not in source

    assert '"sampling": false' in source
    assert '"temperature": none' in source
    assert '"top_k": none' in source
    assert '"top_p": none' in source
    assert '"beam_search": false' in source
    assert '"logit_bias": none' in source
    assert '"not_entitled_available": false' in source

    assert "primary_inference_executed" in source
    assert "p_value_count_added" in source
    assert "estimability_evaluation_deferred" in source


def test_old_stage_a_generator_is_only_a_dependency_not_modified_surface():
    source = inspect.getsource(subject)

    assert (
        "reason_router_gen4_pre_emission_stage_a_generation_fast_cuda"
        in source
    )
    assert subject.BASE_GENERATOR_GIT_BLOB == (
        "aa6725bdc1cbfa43547e89293f3b7911a4cdd6ed"
    )
    assert subject.PRIOR_RAW_FREEZE_COMMIT == (
        "a58b1ebef0957a8437e0255fb4069d2fe9cf8449"
    )
    assert subject.PRIOR_CLOSURE_COMMIT == (
        "130aa76cf3bb0816505a45a18df8334445e6f0ce"
    )


def test_direct_script_entrypoint_help():
    repo_root = Path(subject.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / (
            "reason_router_gen4_pre_emission_"
            "forced_decisive_stage_a_generation_fast_cuda.py"
        )
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--snapshot" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--expected-head" in completed.stdout
