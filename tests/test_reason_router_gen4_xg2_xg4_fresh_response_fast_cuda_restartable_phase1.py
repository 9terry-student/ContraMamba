from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1
    as phase1,
)


def _synthetic_items(
    family: str,
) -> list[dict[str, object]]:
    n_large = (
        phase1.FROZEN_REGIME_COUNTS[
            family
        ][phase1.REGIME_LARGE]
    )

    rows: list[
        dict[str, object]
    ] = []

    for index, pair in enumerate(
        phase1._expected_pairs(
            family
        )
    ):
        plus = (
            2.0
            + index * 1.0e-6
        )
        minus = (
            1.0
            + index * 5.0e-7
        )
        delta = plus - minus

        rows.append(
            {
                "schema_version": (
                    phase1.ITEM_SCHEMA
                ),
                "family_key": family,
                "source_pair_id": pair,
                "threshold": (
                    phase1
                    .ALIGNMENT_SHIFT_THRESHOLD
                ),
                "regime": (
                    phase1.REGIME_LARGE
                    if index < n_large
                    else phase1.REGIME_SMALL
                ),
                "alignment_plan_index": (
                    index
                ),
                "baseline_plus_path_efficiency": (
                    plus
                ),
                "baseline_minus_path_efficiency": (
                    minus
                ),
                "delta_baseline": (
                    delta
                ),
                "alignment_delta_h_dtype": (
                    "float32"
                ),
                "alignment_delta_h_shape": [
                    4
                ],
            }
        )

    return rows


def _synthetic_summary(
    family: str,
) -> dict[str, object]:
    counts = (
        phase1.FROZEN_REGIME_COUNTS[
            family
        ]
    )

    return {
        "schema_version": (
            phase1.SUMMARY_SCHEMA
        ),
        "result": (
            phase1.RESULT_PASS
        ),
        "family_key": family,
        "source_pair_count": 300,
        "threshold": (
            phase1
            .ALIGNMENT_SHIFT_THRESHOLD
        ),
        "n_LARGE": (
            counts[
                phase1.REGIME_LARGE
            ]
        ),
        "n_SMALL": (
            counts[
                phase1.REGIME_SMALL
            ]
        ),
        "support_gate_pass": True,
        "baseline_model_forward_count": (
            1200
        ),
        "alignment_model_forward_count": (
            0
        ),
        "total_model_forward_count": (
            1200
        ),
        "phase2_restartable": True,
    }


def test_exact_freeze_identities():
    assert (
        phase1
        .BASELINE_ARTIFACT_FREEZE_COMMIT
        == (
            "40d21032b09560e246b16001"
            "c22837e46d5d3c70"
        )
    )
    assert (
        phase1.CORRECTION_FREEZE_COMMIT
        == (
            "bc8b0e2ca128eb81271761c9"
            "b3f5ed7b0a940455"
        )
    )
    assert (
        phase1.IMPLEMENTATION_SCOPE_COMMIT
        == (
            "917e505cd1156ee870e979300"
            "b1bc912f0aa43c2"
        )
    )


def test_budget_and_threshold_are_exact():
    assert phase1.SOURCE_PAIR_COUNT == 300
    assert (
        phase1.BASELINE_FORWARDS_PER_PAIR
        == 4
    )
    assert (
        phase1
        .FULL_BASELINE_FORWARD_BUDGET
        == 1200
    )
    assert (
        phase1
        .ALIGNMENT_SHIFT_THRESHOLD
        == 0.11228626366380845
    )


def test_frozen_regime_counts_are_exact():
    assert (
        phase1.FROZEN_REGIME_COUNTS[
            "xg2"
        ]
        == {
            phase1.REGIME_LARGE: 92,
            phase1.REGIME_SMALL: 208,
        }
    )
    assert (
        phase1.FROZEN_REGIME_COUNTS[
            "xg4"
        ]
        == {
            phase1.REGIME_LARGE: 57,
            phase1.REGIME_SMALL: 243,
        }
    )


@pytest.mark.parametrize(
    "family",
    ("xg2", "xg4"),
)
def test_item_contract_and_delta_identity(
    family,
):
    rows = _synthetic_items(
        family
    )

    phase1._validate_item_rows(
        family,
        rows,
    )

    for row in rows:
        assert (
            row["delta_baseline"]
            == (
                row[
                    "baseline_plus_path_efficiency"
                ]
                - row[
                    "baseline_minus_path_efficiency"
                ]
            )
        )


@pytest.mark.parametrize(
    "family",
    ("xg2", "xg4"),
)
def test_binary_plan_roundtrip_and_validator(
    tmp_path,
    monkeypatch,
    family,
):
    rows = _synthetic_items(
        family
    )
    plans = [
        torch.tensor(
            [
                float(index),
                float(index) + 0.25,
                float(index) - 0.5,
                1.0,
            ],
            dtype=torch.float32,
        )
        for index in range(300)
    ]

    summary = _synthetic_summary(
        family
    )

    output_dir = (
        tmp_path
        / family
    )

    hashes = phase1._write_outputs(
        output_dir,
        family=family,
        items=rows,
        plans=plans,
        summary=summary,
    )

    assert (
        phase1.PLAN_FILE
        in hashes
    )

    manifest = json.loads(
        (
            output_dir
            / phase1.MANIFEST_FILE
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        phase1.PLAN_FILE
        in manifest["files"]
    )

    def forbidden_forward(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "model forward invoked"
        )

    monkeypatch.setattr(
        phase1.base.prevalence_eq.parent,
        "capture_branch",
        forbidden_forward,
    )

    loaded = (
        phase1
        .validate_restartable_artifact(
            output_dir,
            family,
        )
    )

    tensor = loaded[
        "alignment_delta_h"
    ]

    assert tuple(
        tensor.shape
    ) == (300, 4)
    assert tensor.dtype == torch.float32

    for index in (
        0,
        57,
        299,
    ):
        assert torch.equal(
            tensor[index],
            plans[index],
        )


def test_plan_manifest_tamper_fails(
    tmp_path,
):
    family = "xg2"
    rows = _synthetic_items(
        family
    )
    plans = [
        torch.zeros(
            4,
            dtype=torch.float32,
        )
        for _ in range(300)
    ]

    out = tmp_path / "artifact"

    phase1._write_outputs(
        out,
        family=family,
        items=rows,
        plans=plans,
        summary=_synthetic_summary(
            family
        ),
    )

    plan_path = (
        out
        / phase1.PLAN_FILE
    )

    with plan_path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(
        phase1.RestartablePhase1Error,
        match="ARTIFACT_SHA256",
    ):
        (
            phase1
            .validate_restartable_artifact(
                out,
                family,
            )
        )


def test_response_field_leak_fails():
    rows = _synthetic_items(
        "xg2"
    )
    rows[0]["R_ALIGN"] = -0.5

    with pytest.raises(
        phase1.RestartablePhase1Error,
        match="RESPONSE_FIELD_LEAK",
    ):
        phase1._validate_item_rows(
            "xg2",
            rows,
        )


def test_execution_surface_has_no_alignment_forward():
    source = inspect.getsource(
        phase1
        ._run_restartable_baseline_pair
    )
    tree = ast.parse(source)

    capture_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(
            node.func,
            ast.Attribute,
        )
        and node.func.attr
        == "capture_branch"
    ]

    assert capture_calls

    for call in capture_calls:
        keyword_names = {
            keyword.arg
            for keyword in call.keywords
        }
        assert "delta_h" not in keyword_names
        assert (
            "plus_branch"
            not in keyword_names
        )

    assert (
        "run_prospective_pair"
        not in source
    )


def test_no_training_backward_or_task_head():
    source = inspect.getsource(
        phase1
    )

    assert ".backward(" not in source
    assert ".train(" not in source
    assert "optimizer.step" not in source
    assert "task_head(" not in source


def test_existing_frozen_runner_blob_unchanged():
    observed = phase1.git(
        "rev-parse",
        (
            "HEAD:"
            + phase1.BASELINE_RUNNER_REL
        ),
    )

    assert (
        observed
        == phase1.BASELINE_RUNNER_BLOB
    )


def test_expected_pair_order_is_301_600():
    for family in (
        "xg2",
        "xg4",
    ):
        pairs = (
            phase1._expected_pairs(
                family
            )
        )

        assert len(pairs) == 300
        assert (
            pairs[0]
            == f"{family}_fact_301"
        )
        assert (
            pairs[-1]
            == f"{family}_fact_600"
        )
