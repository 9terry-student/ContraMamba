from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1
    as phase1,
)
from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2
    as phase2,
)


def _synthetic_phase1_items(
    family: str,
) -> list[dict[str, object]]:
    n_large = (
        phase1.FROZEN_REGIME_COUNTS[
            family
        ][phase1.REGIME_LARGE]
    )
    rows: list[dict[str, object]] = []

    for index, pair in enumerate(
        phase1._expected_pairs(family)
    ):
        plus = 2.0 + index * 1.0e-6
        minus = 1.0 + index * 5.0e-7
        rows.append(
            {
                "schema_version": phase1.ITEM_SCHEMA,
                "family_key": family,
                "source_pair_id": pair,
                "threshold": (
                    phase1.ALIGNMENT_SHIFT_THRESHOLD
                ),
                "regime": (
                    phase1.REGIME_LARGE
                    if index < n_large
                    else phase1.REGIME_SMALL
                ),
                "alignment_plan_index": index,
                "baseline_plus_path_efficiency": plus,
                "baseline_minus_path_efficiency": minus,
                "delta_baseline": plus - minus,
                "alignment_delta_h_dtype": "float32",
                "alignment_delta_h_shape": [4],
                "target_plus_anchor": 10,
                "target_minus_anchor": 11,
                "reference_plus_anchor": 12,
                "reference_minus_anchor": 13,
                "alignment_shift_abs": 0.2,
                "alignment_realized_A": 1.0,
                "alignment_realized_B": 2.0,
                "alignment_target_cosine": 0.3,
                "alignment_realized_cosine": 0.3,
                "alignment_cosine_abs_residual": 0.0,
                "alignment_A_preservation_abs_residual": 0.0,
                "alignment_B_preservation_abs_residual": 0.0,
            }
        )
    return rows


def _synthetic_phase1_summary(
    family: str,
) -> dict[str, object]:
    counts = (
        phase1.FROZEN_REGIME_COUNTS[
            family
        ]
    )
    return {
        "schema_version": phase1.SUMMARY_SCHEMA,
        "result": phase1.RESULT_PASS,
        "family_key": family,
        "source_pair_count": 300,
        "threshold": (
            phase1.ALIGNMENT_SHIFT_THRESHOLD
        ),
        "n_LARGE": counts[
            phase1.REGIME_LARGE
        ],
        "n_SMALL": counts[
            phase1.REGIME_SMALL
        ],
        "support_gate_pass": True,
        "baseline_model_forward_count": 1200,
        "alignment_model_forward_count": 0,
        "total_model_forward_count": 1200,
        "phase2_restartable": True,
    }


def _write_synthetic_phase1(
    tmp_path: Path,
    family: str,
) -> Path:
    items = _synthetic_phase1_items(
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
    out = tmp_path / f"phase1-{family}"
    phase1._write_outputs(
        out,
        family=family,
        items=items,
        plans=plans,
        summary=_synthetic_phase1_summary(
            family
        ),
    )
    return out


def _synthetic_phase2_items(
    family: str,
) -> list[dict[str, object]]:
    rows = _synthetic_phase1_items(
        family
    )
    out: list[dict[str, object]] = []

    for row in rows:
        item = dict(row)
        item["phase1_schema_version"] = (
            item["schema_version"]
        )
        item["schema_version"] = (
            phase2.ITEM_SCHEMA
        )
        item[
            "phase1_artifact_freeze_commit"
        ] = (
            phase2.PHASE1_ARTIFACT_FREEZE_COMMIT
        )
        item[
            "alignment_plus_path_efficiency"
        ] = 1.5
        item[
            "alignment_minus_path_efficiency"
        ] = 0.25
        item["delta_alignment"] = 1.25
        item["R_ALIGN"] = (
            1.25 - item["delta_baseline"]
        )
        item[
            "alignment_midpoint_max_abs_residual"
        ] = 0.0
        item[
            "alignment_pair_delta_max_abs_residual"
        ] = 0.0
        item[
            "alignment_applied_correction_max_abs_residual"
        ] = 0.0
        item[
            "alignment_runtime_correction_l2"
        ] = 1.0
        out.append(item)
    return out


def _synthetic_phase2_summary(
    family: str,
) -> dict[str, object]:
    counts = (
        phase2.FROZEN_REGIME_COUNTS[
            family
        ]
    )
    return {
        "schema_version": phase2.SUMMARY_SCHEMA,
        "result": phase2.RESULT_PASS,
        "family_key": family,
        "phase1_artifact_freeze_commit": (
            phase2.PHASE1_ARTIFACT_FREEZE_COMMIT
        ),
        "source_pair_count": 300,
        "n_LARGE": counts[
            phase2.REGIME_LARGE
        ],
        "n_SMALL": counts[
            phase2.REGIME_SMALL
        ],
        "baseline_model_forward_count_this_run": 0,
        "alignment_model_forward_count_this_run": 600,
        "current_run_scientific_forward_count": 600,
        "persisted_phase1_baseline_forward_count": 1200,
        "complete_protocol_forward_count": 1800,
        "response_fields_observed": True,
        "r_align_observed": True,
        "h1_test_executed": False,
        "h2_test_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
    }


def test_exact_freeze_and_budget_identities():
    assert (
        phase2.PHASE1_ARTIFACT_FREEZE_COMMIT
        == "f4f5aef2025a66e8b7e7ed6d077523eec46eb6f0"
    )
    assert (
        phase2.IMPLEMENTATION_SCOPE_COMMIT
        == "7d43d2a9a180c976a23884a0005e83c6a82390c9"
    )
    assert phase2.SOURCE_PAIR_COUNT == 300
    assert phase2.ALIGNMENT_FORWARDS_PER_PAIR == 2
    assert (
        phase2.PHASE2_ALIGNMENT_FORWARD_BUDGET
        == 600
    )
    assert (
        phase2.PERSISTED_PHASE1_BASELINE_FORWARD_COUNT
        == 1200
    )
    assert (
        phase2.COMPLETE_PROTOCOL_FORWARD_COUNT
        == 1800
    )


@pytest.mark.parametrize(
    "family",
    ("xg2", "xg4"),
)
def test_phase1_artifact_load_is_lossless(
    tmp_path,
    family,
):
    artifact = _write_synthetic_phase1(
        tmp_path,
        family,
    )
    loaded = phase2.load_phase1_artifact(
        family,
        artifact,
    )

    assert len(loaded["items"]) == 300
    plans = loaded["alignment_delta_h"]
    assert tuple(plans.shape) == (300, 4)
    assert plans.dtype == torch.float32
    assert loaded["summary"][
        "phase2_restartable"
    ] is True


def test_incomplete_phase1_artifact_fails_closed(
    tmp_path,
):
    artifact = _write_synthetic_phase1(
        tmp_path,
        "xg2",
    )
    (
        artifact
        / phase1.CHECKSUM_FILE
    ).unlink()

    with pytest.raises(Exception):
        phase2.load_phase1_artifact(
            "xg2",
            artifact,
        )


def test_phase1_validation_precedes_runtime_gate(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []

    monkeypatch.setattr(
        phase2,
        "authenticate_repo",
        lambda _head: calls.append(
            "authenticate"
        ),
    )

    def fail_phase1(_family):
        calls.append("phase1")
        raise phase2.RestartablePhase2Error(
            "synthetic phase1 failure"
        )

    monkeypatch.setattr(
        phase2,
        "load_phase1_artifact",
        fail_phase1,
    )
    monkeypatch.setattr(
        phase1.base.prevalence_eq.backend,
        "runtime_gate",
        lambda: calls.append(
            "runtime"
        ),
    )

    with pytest.raises(
        phase2.RestartablePhase2Error
    ):
        phase2.run_restartable_phase2(
            family="xg2",
            expected_head="synthetic",
            model_snapshot=tmp_path,
            tokenizer_snapshot=tmp_path,
            checkpoint_path=tmp_path / "ckpt",
            output_dir=tmp_path / "out",
        )

    assert calls == [
        "authenticate",
        "phase1",
    ]


@pytest.mark.parametrize(
    "field,value",
    (
        ("source_pair_id", "xg2_fact_999"),
        ("alignment_plan_index", 9),
        ("alignment_delta_h_dtype", "float64"),
        ("alignment_delta_h_shape", [5]),
    ),
)
def test_phase1_pair_plan_identity_enforced(
    monkeypatch,
    field,
    value,
):
    family = "xg2"
    items = _synthetic_phase1_items(
        family
    )
    plans = torch.zeros(
        (300, 4),
        dtype=torch.float32,
    )
    items[0][field] = value

    loaded = {
        "summary": _synthetic_phase1_summary(
            family
        ),
        "items": items,
        "alignment_delta_h": plans,
        "manifest": {},
    }

    with pytest.raises(
        phase2.RestartablePhase2Error
    ):
        phase2._validate_phase1_loaded(
            family,
            loaded,
        )


def test_frozen_regime_counts_enforced():
    family = "xg2"
    loaded = {
        "summary": _synthetic_phase1_summary(
            family
        ),
        "items": _synthetic_phase1_items(
            family
        ),
        "alignment_delta_h": torch.zeros(
            (300, 4),
            dtype=torch.float32,
        ),
        "manifest": {},
    }
    loaded["items"][0]["regime"] = (
        phase1.REGIME_SMALL
    )

    with pytest.raises(
        phase2.RestartablePhase2Error
    ):
        phase2._validate_phase1_loaded(
            family,
            loaded,
        )


def test_alignment_pair_uses_exactly_two_intervention_calls(
    monkeypatch,
):
    family = "xg2"
    baseline = _synthetic_phase1_items(
        family
    )[0]
    plan = torch.tensor(
        [0.1, 0.2, 0.3, 0.4],
        dtype=torch.float32,
    )

    monkeypatch.setattr(
        phase1,
        "_cells",
        lambda: {
            "tp": "C2_NAME",
            "tm": "C0_SHAM",
            "rp": "C5_REF",
            "rm": "C3_REF",
        },
    )
    monkeypatch.setattr(
        phase1,
        "_anchors_for_pair",
        lambda _pair, _events: {
            "tp": 10,
            "tm": 11,
            "rp": 12,
            "rm": 13,
        },
    )
    monkeypatch.setattr(
        phase2,
        "_input_row",
        lambda *_args, **_kwargs: (
            torch.zeros(
                (1, 8),
                dtype=torch.long,
            )
        ),
    )

    calls: list[bool] = []

    def fake_capture(
        *_args,
        budget,
        delta_h,
        plus_branch,
        **_kwargs,
    ):
        assert delta_h is not None
        assert torch.equal(
            delta_h,
            plan,
        )
        assert type(plus_branch) is bool
        budget.consume()
        calls.append(plus_branch)
        return {
            "intervention_audit": {
                "synthetic": True,
            },
            "pe": (
                1.5
                if plus_branch
                else 0.25
            ),
        }

    parent = (
        phase1.base.prevalence_eq.parent
    )
    runtime = (
        phase1.base.prevalence_eq
        .transport_runtime
    )

    monkeypatch.setattr(
        parent,
        "capture_branch",
        fake_capture,
    )
    monkeypatch.setattr(
        parent,
        "path_efficiency",
        lambda result: result["pe"],
    )
    monkeypatch.setattr(
        runtime,
        "paired_intervention_audit",
        lambda *_args, **_kwargs: {
            "midpoint_max_abs_residual": 0.0,
            "pair_delta_max_abs_residual": 0.0,
            "applied_correction_max_abs_residual": 0.0,
            "runtime_correction_l2": 1.0,
        },
    )

    budget = parent.ForwardBudget(2)
    item = phase2._run_alignment_pair(
        family,
        baseline,
        plan,
        model=object(),
        runtime_ctx={},
        trace_code=object(),
        trace_line=1,
        encoded={},
        row_index={},
        events={},
        budget=budget,
    )
    budget.assert_exact()

    assert calls == [True, False]
    assert item[
        "alignment_plus_path_efficiency"
    ] == 1.5
    assert item[
        "alignment_minus_path_efficiency"
    ] == 0.25
    assert item["delta_alignment"] == 1.25
    assert item["R_ALIGN"] == (
        item["delta_alignment"]
        - item["delta_baseline"]
    )


def test_response_identities_are_exact():
    rows = _synthetic_phase2_items(
        "xg2"
    )
    phase2._validate_response_rows(
        "xg2",
        rows,
    )

    row = rows[17]
    assert row["delta_alignment"] == (
        row[
            "alignment_plus_path_efficiency"
        ]
        - row[
            "alignment_minus_path_efficiency"
        ]
    )
    assert row["R_ALIGN"] == (
        row["delta_alignment"]
        - row["delta_baseline"]
    )


def test_delta_baseline_is_not_recomputed_by_model_path():
    source = inspect.getsource(
        phase2._run_alignment_pair
    )
    assert "_run_restartable_baseline_pair" not in source
    assert "reconstruct_pair_geometry" not in source
    assert "alignment_delta(" not in source
    assert (
        'baseline_item["delta_baseline"]'
        in source
    )


@pytest.mark.parametrize(
    "family",
    ("xg2", "xg4"),
)
def test_phase2_output_roundtrip_and_checksums(
    tmp_path,
    family,
):
    out = tmp_path / family
    phase2._write_outputs(
        out,
        family=family,
        items=_synthetic_phase2_items(
            family
        ),
        summary=_synthetic_phase2_summary(
            family
        ),
    )

    loaded = (
        phase2.validate_phase2_artifact(
            out,
            family,
        )
    )
    assert len(loaded["items"]) == 300
    assert loaded["summary"][
        "alignment_model_forward_count_this_run"
    ] == 600

    manifest = json.loads(
        (
            out / phase2.MANIFEST_FILE
        ).read_text(
            encoding="utf-8"
        )
    )
    assert set(manifest["files"]) == {
        phase2.ITEM_FILE,
        phase2.SUMMARY_FILE,
    }


def test_phase2_manifest_tamper_fails(
    tmp_path,
):
    out = tmp_path / "xg2"
    phase2._write_outputs(
        out,
        family="xg2",
        items=_synthetic_phase2_items(
            "xg2"
        ),
        summary=_synthetic_phase2_summary(
            "xg2"
        ),
    )

    with (
        out / phase2.ITEM_FILE
    ).open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(
        phase2.RestartablePhase2Error
    ):
        phase2.validate_phase2_artifact(
            out,
            "xg2",
        )


def test_output_collision_fails_closed(
    tmp_path,
):
    out = tmp_path / "exists"
    out.mkdir()

    with pytest.raises(
        phase2.RestartablePhase2Error
    ):
        phase2._write_outputs(
            out,
            family="xg2",
            items=_synthetic_phase2_items(
                "xg2"
            ),
            summary=_synthetic_phase2_summary(
                "xg2"
            ),
        )


def test_no_h1_h2_training_backward_task_head_or_logit_path():
    source = inspect.getsource(
        phase2.run_restartable_phase2
    )

    assert (
        '"h1_test_executed": False'
        in source
    )
    assert (
        '"h2_test_executed": False'
        in source
    )
    assert (
        '"training_executed": False'
        in source
    )
    assert (
        '"backward_executed": False'
        in source
    )
    assert (
        '"task_heads_executed": False'
        in source
    )
    assert '"logits_read": False' in source
    assert (
        '"scientific_conclusion": None'
        in source
    )


def test_no_baseline_forward_helper_in_phase2_execution():
    source = inspect.getsource(
        phase2.run_restartable_phase2
    )
    assert (
        "_run_restartable_baseline_pair"
        not in source
    )
    assert (
        "FULL_BASELINE_FORWARD_BUDGET"
        not in source
    )
