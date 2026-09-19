from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_pre_emission_stage_a_generation_fast_cuda
    as subject,
)


def fake_cohort():
    rows = []
    counts = {
        "REFUTE": 305,
        "SUPPORT": 122,
        "NOT_ENTITLED": 35,
    }
    index = 0
    for label in subject.CLASS_ORDER:
        for _ in range(counts[label]):
            rows.append({
                "example_id": f"fake_{index:03d}",
                "correct_label": label,
            })
            index += 1
    assert len(rows) == subject.N
    return rows


def small_grammar():
    common = [10, 11, 12, 13, 14, 15, 16, 17]
    return {
        "REFUTE": common + [101, 201],
        "NOT_ENTITLED": common + [102, 202, 302],
        "SUPPORT": common + [103, 203],
    }


def test_protocol_constants_are_frozen():
    subject.validate_protocol()
    assert subject.REQUIRED_ANCESTOR == (
        "f3f38b3a96cd2436ffbbe9de0c229a78378abf61"
    )
    assert subject.SELECTED_PLANE == "P3"
    assert subject.CONTROL_PLANE == "P5"
    assert subject.EARLY_BLOCK == 35
    assert subject.LATE_BLOCK == 47
    assert subject.OBSERVATION_OFFSETS == (-4, -3, -2, -1)
    assert subject.OBSERVATION_GENERATED_PREFIX_LENGTHS == (5, 6, 7, 8)
    assert subject.PRIMARY_STAGE_A_PARTITION == "confirmatory"
    assert subject.PRIMARY_STAGE_A_SIGNAL == "p3_component_l2"
    assert subject.PRIMARY_STAGE_A_TEST == "two_sided_welch_t"
    assert subject.PRIMARY_STAGE_A_MULTIPLICITY == "holm"
    assert subject.PRIMARY_STAGE_A_P_VALUE_COUNT == 4
    assert subject.PRIMARY_STAGE_A_ALPHA == 0.05


def test_partition_is_deterministic_response_blind_and_balanced():
    cohort = fake_cohort()
    a, ma = subject.build_partition_assignments(cohort)
    b, mb = subject.build_partition_assignments(list(reversed(cohort)))

    assert a == b
    assert ma["partition_sha256"] == mb["partition_sha256"]
    assert ma["partition_counts"] == {
        "calibration": 231,
        "confirmatory": 231,
    }
    assert ma["partition_label_counts"] == {
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

    source = inspect.getsource(subject.build_partition_assignments).lower()
    for forbidden in (
        "generated",
        "p3_",
        "margin",
        "prediction",
        "correctness",
        "confidence",
    ):
        assert forbidden not in source


def test_partition_digest_binds_gold_label_and_id():
    a = subject.partition_rank_digest("x", "REFUTE")
    b = subject.partition_rank_digest("x", "SUPPORT")
    c = subject.partition_rank_digest("y", "REFUTE")
    assert len(a) == 64
    assert len({a, b, c}) == 3


def test_grammar_completion_and_compatibility():
    grammar = small_grammar()
    assert subject.completed_label(grammar["REFUTE"], grammar) == "REFUTE"
    assert subject.completed_label(grammar["REFUTE"][:-1], grammar) is None
    assert subject.compatible_labels([], grammar) == subject.CLASS_ORDER
    assert subject.compatible_labels(
        grammar["SUPPORT"][:8],
        grammar,
    ) == subject.CLASS_ORDER
    assert subject.compatible_labels(
        grammar["SUPPORT"][:9],
        grammar,
    ) == ("SUPPORT",)


def test_branch_tokens_are_unique_and_at_frozen_index():
    assert subject.branch_token_by_label(small_grammar()) == {
        "REFUTE": 101,
        "NOT_ENTITLED": 102,
        "SUPPORT": 103,
    }


def test_greedy_allowed_token_is_restricted_and_lowest_id_tie_break():
    logits = torch.full((200,), -100.0)
    logits[101] = 2.0
    logits[102] = 9.0
    logits[103] = 9.0
    logits[199] = 1000.0

    chosen, values = subject.greedy_allowed_token(
        logits,
        [103, 102, 101],
    )
    assert chosen == 102
    assert set(values) == {"101", "102", "103"}


@pytest.mark.parametrize(
    "gold,emitted,category,group,tstar",
    [
        ("REFUTE", "REFUTE", "supported_decisive", "supported", 8),
        ("REFUTE", "SUPPORT", "unsupported_decisive", "unsupported", 8),
        (
            "REFUTE",
            "NOT_ENTITLED",
            "nondecisive_abstention_error",
            None,
            None,
        ),
        ("SUPPORT", "SUPPORT", "supported_decisive", "supported", 8),
        ("SUPPORT", "REFUTE", "unsupported_decisive", "unsupported", 8),
        (
            "SUPPORT",
            "NOT_ENTITLED",
            "nondecisive_abstention_error",
            None,
            None,
        ),
        (
            "NOT_ENTITLED",
            "REFUTE",
            "unsupported_decisive",
            "unsupported",
            8,
        ),
        (
            "NOT_ENTITLED",
            "SUPPORT",
            "unsupported_decisive",
            "unsupported",
            8,
        ),
        (
            "NOT_ENTITLED",
            "NOT_ENTITLED",
            "nondecisive_correct_abstention",
            None,
            None,
        ),
    ],
)
def test_outcome_classification(gold, emitted, category, group, tstar):
    out = subject.classify_outcome(gold, emitted)
    assert out["outcome_category"] == category
    assert out["primary_stage_a_group"] == group
    assert out["t_star_zero_based"] == tstar


def test_gold_conditioned_unsupported_branch_margin():
    logits = {
        "REFUTE": 3.0,
        "NOT_ENTITLED": 2.0,
        "SUPPORT": 5.0,
    }
    assert subject.unsupported_branch_margin("REFUTE", logits) == 2.0
    assert subject.unsupported_branch_margin("SUPPORT", logits) == -2.0
    assert subject.unsupported_branch_margin("NOT_ENTITLED", logits) == 3.0
    assert subject.decisive_vs_abstain_margin(logits) == 3.0


def test_observation_schedule_maps_exactly_to_pre_emission_offsets():
    assert [
        subject.observation_offset_for_generated_prefix_length(x)
        for x in range(0, 10)
    ] == [
        None, None, None, None, None,
        -4, -3, -2, -1, None,
    ]


def test_prompt_extraction_requires_left_contiguous_attention():
    encoded = {
        "input_ids": torch.tensor([[4, 5, 6, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
    }
    prompt = subject.extract_prompt_ids(encoded, 0, 3)
    assert prompt.tolist() == [4, 5, 6]

    bad = {
        "input_ids": torch.tensor([[4, 5, 6, 0, 0]]),
        "attention_mask": torch.tensor([[1, 0, 1, 0, 0]]),
    }
    with pytest.raises(
        subject.StageAGenerationError,
        match="PROMPT_LEFT_CONTIGUOUS",
    ):
        subject.extract_prompt_ids(bad, 0, 2)


def test_early_measurement_uses_only_strong_hidden_half_last_token(monkeypatch):
    width = subject.geom.INTERMEDIATE_SIZE
    output = torch.zeros((1, 3, 2 * width), dtype=torch.float32)
    mask = torch.zeros(width, dtype=torch.bool)
    mask[: subject.DIM] = True
    output[0, -1, : subject.DIM] = torch.arange(
        subject.DIM,
        dtype=torch.float32,
    )

    calls = []

    def fake_component(h, plane, planes):
        calls.append((h.clone(), plane))
        return {
            "a": 1.0 if plane == "P3" else 2.0,
            "b": 3.0,
            "component_l2": 4.0,
        }

    monkeypatch.setattr(
        subject.discovery,
        "plane_component",
        fake_component,
    )
    out = subject.early_plane_measurements(
        output,
        strong_mask=mask,
        planes={"P3": {}, "P5": {}},
    )
    assert out["P3"]["component_l2"] == 4.0
    assert out["P5"]["a"] == 2.0
    assert [plane for _, plane in calls] == ["P3", "P5"]
    expected = torch.arange(subject.DIM, dtype=torch.float64)
    assert torch.equal(calls[0][0], expected)


def test_raw_row_validation_adds_no_inference():
    base_obs = [
        {"relative_offset": x}
        for x in subject.OBSERVATION_OFFSETS
    ]
    rows = []
    partitions = ["calibration"] * 231 + ["confirmatory"] * 231
    for i in range(subject.N):
        rows.append({
            "schema_version": subject.ROW_SCHEMA,
            "outcome_category": "supported_decisive",
            "emitted_commitment_label": "REFUTE",
            "partition": partitions[i],
            "observations": base_obs,
            "generate_api_call_count": 0,
            "manual_constrained_decoding": True,
            "sampling": False,
            "beam_search": False,
            "generated_token_count": 12,
            "scientific_model_forward_count": 12,
        })
    out = subject.validate_raw_rows(rows)
    assert out["scientific_model_forward_count_this_run"] == 462 * 12
    assert out["generate_api_call_count"] == 0


def test_production_source_forbids_generate_training_and_statistics():
    source = inspect.getsource(subject).lower()
    assert ".generate(" not in source
    assert "optimizer" not in source
    assert ".backward(" not in source
    assert "scipy" not in source
    assert "ttest" not in source
    assert "mannwhitney" not in source
    assert "primary_inference_executed" in source
    assert "p_value_count_added" in source


def test_direct_script_entrypoint_help():
    repo_root = Path(subject.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "reason_router_gen4_pre_emission_stage_a_generation_fast_cuda.py"
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
