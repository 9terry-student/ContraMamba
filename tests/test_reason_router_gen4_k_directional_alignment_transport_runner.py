import math

import pytest

from scripts import (
    reason_router_gen4_k_directional_alignment_transport_core
    as core,
)
from scripts import (
    reason_router_gen4_k_directional_alignment_transport_runner
    as runner,
)


def test_forward_budget_exact():
    budget = runner.ForwardBudget(3)

    budget.consume()
    budget.consume()
    budget.consume()

    budget.assert_exact()

    with pytest.raises(
        runner.TransportRunnerError
    ):
        budget.consume()


def test_holm_two():
    a, b = runner.holm_two(
        0.01,
        0.04,
    )

    assert math.isclose(
        a,
        0.02,
    )
    assert math.isclose(
        b,
        0.04,
    )


def test_one_sided_greater_t_positive():
    result = (
        runner.one_sided_greater_t(
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ]
        )
    )

    assert result["mean"] > 0.0
    assert result["t_statistic"] > 0.0
    assert (
        0.0
        <= result["raw_p_value"]
        <= 0.5
    )


def test_one_sided_greater_t_negative():
    result = (
        runner.one_sided_greater_t(
            [
                -0.1,
                -0.2,
                -0.3,
                -0.4,
            ]
        )
    )

    assert result["mean"] < 0.0
    assert result["t_statistic"] < 0.0
    assert (
        0.5
        <= result["raw_p_value"]
        <= 1.0
    )


def _synthetic_event(
    pair,
    cell,
    anchor_name,
    anchor,
):
    return {
        "source_pair_id": pair,
        "contrast_cell_id": cell,
        "anchor_name": anchor_name,
        "absolute_anchor_token_index":
            anchor,
        "terminal_index": anchor + 8,
        "post4_eligible": True,
    }


def test_event_plan_allows_branch_local_absolute_coordinates():
    pair = "p0"

    rows = [
        _synthetic_event(
            pair,
            core.TARGET_PLUS_CELL,
            core.ANCHOR_NAME,
            20,
        ),
        _synthetic_event(
            pair,
            core.TARGET_MINUS_CELL,
            core.ANCHOR_NAME,
            21,
        ),
        _synthetic_event(
            pair,
            core.REFERENCE_PLUS_CELL,
            core.ANCHOR_NAME,
            22,
        ),
        _synthetic_event(
            pair,
            core.REFERENCE_MINUS_CELL,
            core.ANCHOR_NAME,
            23,
        ),
        _synthetic_event(
            pair,
            core.TARGET_PLUS_CELL,
            "A_NAME",
            20,
        ),
        _synthetic_event(
            pair,
            core.TARGET_MINUS_CELL,
            "A_NAME",
            21,
        ),
    ]

    lookup = runner.event_lookup(
        rows
    )

    runner.validate_transport_event_plan(
        [pair],
        lookup,
    )

    assert (
        lookup[
            (
                pair,
                core.TARGET_PLUS_CELL,
                core.ANCHOR_NAME,
            )
        ][
            "absolute_anchor_token_index"
        ]
        + core.TARGET_OFFSET
        == 22
    )

    assert (
        lookup[
            (
                pair,
                core.TARGET_MINUS_CELL,
                core.ANCHOR_NAME,
            )
        ][
            "absolute_anchor_token_index"
        ]
        + core.TARGET_OFFSET
        == 23
    )


def _preflight_item(index):
    return {
        "baseline_plus_reproduction_abs_residual":
            1e-14 * (index + 1),
        "baseline_minus_reproduction_abs_residual":
            2e-14 * (index + 1),
        "alignment_cosine_abs_residual":
            1e-13,
        "magnitude_cosine_abs_residual":
            2e-13,
        "alignment_A_preservation_abs_residual":
            1e-13,
        "alignment_B_preservation_abs_residual":
            2e-13,
        "magnitude_A_target_abs_residual":
            3e-13,
        "magnitude_B_target_abs_residual":
            4e-13,
        "alignment_midpoint_max_abs_residual":
            1e-7,
        "magnitude_midpoint_max_abs_residual":
            2e-7,
        "alignment_pair_delta_max_abs_residual":
            3e-7,
        "magnitude_pair_delta_max_abs_residual":
            4e-7,
        "alignment_applied_correction_max_abs_residual":
            3e-7,
        "magnitude_applied_correction_max_abs_residual":
            4e-7,
        # These represent scientific values that
        # must NOT be copied into preflight output.
        "R_ALIGN": 123.0,
        "R_MAG": 456.0,
        "ALIGNMENT_SPECIFICITY": 789.0,
        "alignment_plus_path_efficiency":
            0.11,
    }


def test_preflight_public_does_not_serialize_scientific_values():
    result = (
        runner.build_preflight_public(
            [
                _preflight_item(0),
                _preflight_item(1),
            ],
            forward_count=
                runner.PREFLIGHT_FORWARD_BUDGET,
        )
    )

    assert (
        result[
            "scientific_endpoint_values_serialized"
        ]
        is False
    )
    assert (
        result[
            "inferential_statistics_executed"
        ]
        is False
    )

    forbidden = {
        "R_ALIGN",
        "R_MAG",
        "ALIGNMENT_SPECIFICITY",
        "alignment_plus_path_efficiency",
        "delta_baseline",
        "delta_alignment",
        "delta_magnitude",
    }

    assert not (
        forbidden
        & set(result)
    )


def test_full_and_preflight_forward_budgets_are_frozen():
    assert (
        runner.FULL_FORWARD_BUDGET
        == 2400
    )
    assert (
        runner.PREFLIGHT_FORWARD_BUDGET
        == 16
    )
    assert (
        runner.FORWARDS_PER_PAIR
        == 8
    )
def test_preflight_public_exposes_complete_manipulation_gate():
    result = runner.build_preflight_public(
        [
            _preflight_item(0),
            _preflight_item(1),
        ],
        forward_count=
            runner.PREFLIGHT_FORWARD_BUDGET,
    )

    expected = {
        "max_baseline_reproduction_abs_residual",
        "max_alignment_cosine_abs_residual",
        "max_magnitude_cosine_abs_residual",
        "max_alignment_A_preservation_abs_residual",
        "max_alignment_B_preservation_abs_residual",
        "max_magnitude_A_target_abs_residual",
        "max_magnitude_B_target_abs_residual",
        "max_alignment_midpoint_abs_residual",
        "max_magnitude_midpoint_abs_residual",
        "max_alignment_pair_delta_abs_residual",
        "max_magnitude_pair_delta_abs_residual",
        "max_alignment_applied_correction_abs_residual",
        "max_magnitude_applied_correction_abs_residual",
    }

    assert expected <= set(result)


def test_external_checkpoint_loader_uses_exact_path_and_sha(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "selected_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint")

    calls = {}

    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path:
            runner.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
    )

    monkeypatch.setattr(
        runner.torch,
        "load",
        lambda path, **kwargs: {
            "model_state_dict": {
                "synthetic": 1
            }
        },
    )

    class FakeMamba(runner.torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = runner.torch.nn.Parameter(
                runner.torch.ones(1),
                requires_grad=False,
            )

    class FakeModel(runner.torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mamba = FakeMamba()

    fake_model = FakeModel()

    monkeypatch.setattr(
        runner.r5,
        "build_local_mamba_backbone",
        lambda model_snapshot:
            calls.setdefault(
                "snapshot",
                model_snapshot,
            )
            or object(),
    )

    monkeypatch.setattr(
        runner.adapter,
        "build_historical_model_from_backbone",
        lambda *, backbone, arm:
            fake_model,
    )

    def fake_strict_load(model, payload):
        calls["payload"] = payload
        calls["model"] = model

    monkeypatch.setattr(
        runner.adapter,
        "strict_load_state_dict",
        fake_strict_load,
    )

    model, digest = (
        runner.load_representative_model_external(
            model_snapshot=tmp_path / "model",
            checkpoint_path=checkpoint,
        )
    )

    assert model is fake_model
    assert (
        digest
        == runner.extraction.REPRESENTATIVE_CHECKPOINT_SHA256
    )
    assert calls["model"] is fake_model
    assert (
        calls["payload"]["model_state_dict"]["synthetic"]
        == 1
    )


def test_cli_requires_checkpoint():
    args = runner.parse_args(
        [
            "--mode",
            "preflight",
            "--output-dir",
            "out",
            "--expected-head",
            "a" * 40,
            "--model-snapshot",
            "model",
            "--tokenizer-snapshot",
            "tok",
            "--checkpoint",
            "checkpoint.pt",
        ]
    )

    assert args.checkpoint == "checkpoint.pt"
