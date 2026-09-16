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
from scripts import reason_router_gen4_xg2_xg4_local_jacobian_fast_cuda as lj


def _baseline_item(family: str, index: int) -> dict[str, object]:
    pair = phase1._expected_pairs(family)[index]
    plus = 2.0 + index * 1.0e-6
    minus = 1.0 + index * 5.0e-7
    return {
        "schema_version": phase1.ITEM_SCHEMA,
        "family_key": family,
        "source_pair_id": pair,
        "alignment_plan_index": index,
        "target_plus_anchor": 10,
        "target_minus_anchor": 11,
        "reference_plus_anchor": 12,
        "reference_minus_anchor": 13,
        "baseline_plus_path_efficiency": plus,
        "baseline_minus_path_efficiency": minus,
        "delta_baseline": plus - minus,
        "alignment_delta_h_dtype": "float64",
        "alignment_delta_h_shape": [4],
    }


def _synthetic_items(family: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(300):
        base = _baseline_item(family, index)
        f0 = float(base["delta_baseline"])
        probes = {}
        for epsilon in lj.EPSILONS:
            f_plus = f0 - 0.5 * epsilon + 0.1 * epsilon * epsilon
            f_minus = f0 + 0.5 * epsilon + 0.1 * epsilon * epsilon
            j = (f_plus - f_minus) / (2.0 * epsilon)
            k = (f_plus + f_minus - 2.0 * f0) / (epsilon * epsilon)

            def signed(orientation: int, f_value: float):
                return {
                    "orientation": orientation,
                    "delta_h_l2": 2.0 * epsilon,
                    "plus_path_efficiency": f_value + 1.0,
                    "minus_path_efficiency": 1.0,
                    "F": f_value,
                    "midpoint_max_abs_residual": 0.0,
                    "pair_delta_max_abs_residual": 0.0,
                    "applied_correction_max_abs_residual": 0.0,
                    "runtime_correction_l2": 2.0 * epsilon,
                    "model_forward_count": 2,
                }

            probes[lj._epsilon_key(epsilon)] = {
                "epsilon": epsilon,
                "F_plus": f_plus,
                "F_minus": f_minus,
                "J": j,
                "K": k,
                "positive_probe": signed(1, f_plus),
                "negative_probe": signed(-1, f_minus),
                "model_forward_count": 4,
            }

        row = dict(base)
        row["phase1_schema_version"] = row["schema_version"]
        row["schema_version"] = lj.ITEM_SCHEMA
        row["implementation_scope_commit"] = lj.IMPLEMENTATION_SCOPE_COMMIT
        row["phase1_artifact_freeze_commit"] = lj.PHASE1_ARTIFACT_FREEZE_COMMIT
        row["original_alignment_plan_l2"] = 2.0
        row["unit_direction_l2"] = 1.0
        row["F0_delta_baseline"] = f0
        row["epsilons"] = list(lj.EPSILONS)
        row["probes"] = probes
        row["baseline_model_forward_count_this_run"] = 0
        row["scientific_model_forward_count_this_run"] = 8
        rows.append(row)
    return rows


def _synthetic_summary(family: str) -> dict[str, object]:
    return {
        "schema_version": lj.SUMMARY_SCHEMA,
        "result": lj.RESULT_PASS,
        "family_key": family,
        "execution_head": "synthetic",
        "implementation_scope_commit": lj.IMPLEMENTATION_SCOPE_COMMIT,
        "phase1_artifact_freeze_commit": lj.PHASE1_ARTIFACT_FREEZE_COMMIT,
        "source_pair_count": 300,
        "epsilons": list(lj.EPSILONS),
        "baseline_model_forward_count_this_run": 0,
        "scientific_model_forward_count_this_run": 2400,
        "primary_inference_executed": False,
        "scale_consistency_inference_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
    }


def test_scope_and_budget_identities():
    assert lj.IMPLEMENTATION_SCOPE_COMMIT == (
        "3a9dd04b3de398069eed3844dfed719e2ba36ca7"
    )
    assert lj.EPSILONS == (0.025, 0.05)
    assert lj.SOURCE_PAIR_COUNT == 300
    assert lj.FORWARDS_PER_SIGNED_PROBE == 2
    assert lj.FORWARDS_PER_PAIR_PER_EPSILON == 4
    assert lj.FORWARDS_PER_PAIR == 8
    assert lj.SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY == 2400
    assert lj.BASELINE_FORWARD_BUDGET_THIS_RUN == 0


def test_unit_direction_exact_normalization():
    plan = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float64)
    unit, norm = lj._unit_direction(plan, "pair")
    assert norm == 5.0
    assert torch.allclose(
        unit,
        torch.tensor([0.6, 0.8, 0.0, 0.0], dtype=torch.float64),
    )
    assert abs(float(torch.linalg.vector_norm(unit)) - 1.0) <= 1e-12


def test_zero_and_nonfinite_plan_fail_closed():
    with pytest.raises(lj.LocalJacobianError, match="ZERO_PLAN_NORM"):
        lj._unit_direction(torch.zeros(4, dtype=torch.float64), "pair")

    bad = torch.ones(4, dtype=torch.float64)
    bad[2] = float("nan")
    with pytest.raises(lj.LocalJacobianError, match="NONFINITE_PLAN"):
        lj._unit_direction(bad, "pair")


def test_only_fixed_epsilons_are_authorized():
    assert lj._epsilon_key(0.025) == "0.025"
    assert lj._epsilon_key(0.05) == "0.05"
    with pytest.raises(lj.LocalJacobianError, match="UNAUTHORIZED_EPSILON"):
        lj._epsilon_key(0.1)


def test_signed_probe_uses_exact_branch_radius(monkeypatch):
    family = "xg2"
    baseline = _baseline_item(family, 0)
    unit = torch.tensor([0.6, 0.8, 0.0, 0.0], dtype=torch.float64)

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
        lj,
        "_input_row",
        lambda *_args, **_kwargs: torch.zeros((1, 8), dtype=torch.long),
    )

    parent = phase1.base.prevalence_eq.parent
    runtime = phase1.base.prevalence_eq.transport_runtime
    seen: list[tuple[bool, torch.Tensor]] = []

    def fake_capture(
        *_args,
        budget,
        delta_h,
        plus_branch,
        **_kwargs,
    ):
        budget.consume()
        seen.append((plus_branch, delta_h.detach().clone()))
        return {
            "intervention_audit": {"synthetic": True},
            "pe": 2.0 if plus_branch else 0.5,
        }

    monkeypatch.setattr(parent, "capture_branch", fake_capture)
    monkeypatch.setattr(parent, "path_efficiency", lambda record: record["pe"])

    def fake_audit(_plus, _minus, delta_h, **_kwargs):
        return {
            "midpoint_max_abs_residual": 0.0,
            "pair_delta_max_abs_residual": 0.0,
            "applied_correction_max_abs_residual": 0.0,
            "runtime_correction_l2": float(torch.linalg.vector_norm(delta_h)),
        }

    monkeypatch.setattr(runtime, "paired_intervention_audit", fake_audit)

    budget = parent.ForwardBudget(2)
    result = lj._run_signed_probe(
        family,
        baseline,
        unit,
        epsilon=0.025,
        orientation=1,
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

    expected = unit * 0.05
    assert [role for role, _ in seen] == [True, False]
    assert torch.equal(seen[0][1], expected)
    assert torch.equal(seen[1][1], expected)
    assert result["delta_h_l2"] == 0.05
    assert result["F"] == 1.5


def test_reverse_probe_negates_delta_h(monkeypatch):
    family = "xg2"
    baseline = _baseline_item(family, 0)
    unit = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    monkeypatch.setattr(
        phase1,
        "_cells",
        lambda: {"tp": "tp", "tm": "tm", "rp": "rp", "rm": "rm"},
    )
    monkeypatch.setattr(
        phase1,
        "_anchors_for_pair",
        lambda _pair, _events: {"tp": 10, "tm": 11, "rp": 12, "rm": 13},
    )
    monkeypatch.setattr(lj, "_input_row", lambda *_args, **_kwargs: torch.zeros((1, 8), dtype=torch.long))

    parent = phase1.base.prevalence_eq.parent
    runtime = phase1.base.prevalence_eq.transport_runtime
    seen: list[torch.Tensor] = []

    def fake_capture(*_args, budget, delta_h, plus_branch, **_kwargs):
        budget.consume()
        seen.append(delta_h.detach().clone())
        return {"intervention_audit": {}, "pe": 1.0 if plus_branch else 0.0}

    monkeypatch.setattr(parent, "capture_branch", fake_capture)
    monkeypatch.setattr(parent, "path_efficiency", lambda record: record["pe"])
    monkeypatch.setattr(
        runtime,
        "paired_intervention_audit",
        lambda _p, _m, delta_h, **_kwargs: {
            "midpoint_max_abs_residual": 0.0,
            "pair_delta_max_abs_residual": 0.0,
            "applied_correction_max_abs_residual": 0.0,
            "runtime_correction_l2": float(torch.linalg.vector_norm(delta_h)),
        },
    )

    budget = parent.ForwardBudget(2)
    lj._run_signed_probe(
        family,
        baseline,
        unit,
        epsilon=0.05,
        orientation=-1,
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
    expected = unit * -0.1
    assert torch.equal(seen[0], expected)
    assert torch.equal(seen[1], expected)


def test_local_pair_j_and_k_identities(monkeypatch):
    family = "xg2"
    baseline = _baseline_item(family, 0)
    baseline["delta_baseline"] = 1.0
    plan = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float64)

    calls: list[tuple[float, int]] = []

    def fake_probe(
        _family,
        _baseline,
        _unit,
        *,
        epsilon,
        orientation,
        **_kwargs,
    ):
        calls.append((epsilon, orientation))
        f = 1.0 + orientation * (-0.4 * epsilon) + 0.2 * epsilon * epsilon
        return {
            "orientation": orientation,
            "delta_h_l2": 2.0 * epsilon,
            "plus_path_efficiency": f + 1.0,
            "minus_path_efficiency": 1.0,
            "F": f,
            "midpoint_max_abs_residual": 0.0,
            "pair_delta_max_abs_residual": 0.0,
            "applied_correction_max_abs_residual": 0.0,
            "runtime_correction_l2": 2.0 * epsilon,
            "model_forward_count": 2,
        }

    monkeypatch.setattr(lj, "_run_signed_probe", fake_probe)
    row = lj._run_local_jacobian_pair(
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
        budget=object(),
    )

    assert calls == [
        (0.025, 1),
        (0.025, -1),
        (0.05, 1),
        (0.05, -1),
    ]
    assert row["original_alignment_plan_l2"] == 5.0
    assert row["baseline_model_forward_count_this_run"] == 0
    assert row["scientific_model_forward_count_this_run"] == 8
    for key in ("0.025", "0.05"):
        assert row["probes"][key]["J"] == pytest.approx(-0.4)
        assert row["probes"][key]["K"] == pytest.approx(0.4)


def test_no_baseline_reexecution_or_response_selection_path():
    source = inspect.getsource(lj.run_local_jacobian)
    pair_source = inspect.getsource(lj._run_local_jacobian_pair)
    probe_source = inspect.getsource(lj._run_signed_probe)

    combined = source + pair_source + probe_source
    assert "_run_restartable_baseline_pair" not in combined
    assert "run_full_baseline" not in combined
    assert "validate_phase2_artifact" not in combined
    assert "R_ALIGN" not in combined
    assert 'baseline_model_forward_count_this_run": 0' in source


def test_phase1_validation_and_plan_norms_precede_runtime(monkeypatch, tmp_path):
    calls: list[str] = []

    monkeypatch.setattr(lj, "authenticate_repo", lambda _head: calls.append("authenticate"))
    monkeypatch.setattr(
        lj.phase2,
        "load_phase1_artifact",
        lambda _family: {
            "summary": {"result": "synthetic"},
            "items": [_baseline_item("xg2", i) for i in range(300)],
            "alignment_delta_h": torch.ones((300, 4), dtype=torch.float64),
            "artifact_manifest_sha256": "a" * 64,
            "artifact_checksum_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(lj, "_validate_all_plan_norms", lambda *_args: calls.append("plan_norms"))

    def stop_runtime():
        calls.append("runtime")
        raise lj.LocalJacobianError("STOP")

    monkeypatch.setattr(
        phase1.base.prevalence_eq.backend,
        "runtime_gate",
        stop_runtime,
    )

    with pytest.raises(lj.LocalJacobianError, match="STOP"):
        lj.run_local_jacobian(
            family="xg2",
            expected_head="synthetic",
            model_snapshot=tmp_path,
            tokenizer_snapshot=tmp_path,
            checkpoint_path=tmp_path / "checkpoint.pt",
            output_dir=tmp_path / "out",
        )

    assert calls == ["authenticate", "plan_norms", "runtime"]


@pytest.mark.parametrize("family", ("xg2", "xg4"))
def test_output_roundtrip_and_checksums(tmp_path, family):
    out = tmp_path / family
    lj._write_outputs(
        out,
        family=family,
        items=_synthetic_items(family),
        summary=_synthetic_summary(family),
    )
    loaded = lj.validate_local_jacobian_artifact(out, family)
    assert len(loaded["items"]) == 300
    assert loaded["summary"]["scientific_model_forward_count_this_run"] == 2400

    manifest = json.loads(
        (out / lj.MANIFEST_FILE).read_text(encoding="utf-8")
    )
    assert set(manifest["files"]) == {lj.ITEM_FILE, lj.SUMMARY_FILE}


def test_artifact_tamper_fails(tmp_path):
    out = tmp_path / "xg2"
    lj._write_outputs(
        out,
        family="xg2",
        items=_synthetic_items("xg2"),
        summary=_synthetic_summary("xg2"),
    )
    with (out / lj.ITEM_FILE).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(lj.LocalJacobianError):
        lj.validate_local_jacobian_artifact(out, "xg2")


def test_output_collision_fails_closed(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(lj.LocalJacobianError, match="OUTPUT_DIR_COLLISION"):
        lj._write_outputs(
            out,
            family="xg2",
            items=_synthetic_items("xg2"),
            summary=_synthetic_summary("xg2"),
        )


def test_no_inference_training_backward_task_head_or_logit_path():
    source = inspect.getsource(lj.run_local_jacobian)
    assert '"primary_inference_executed": False' in source
    assert '"scale_consistency_inference_executed": False' in source
    assert '"training_executed": False' in source
    assert '"backward_executed": False' in source
    assert '"task_heads_executed": False' in source
    assert '"logits_read": False' in source
    assert '"scientific_conclusion": None' in source


def test_cli_has_no_user_epsilon_argument():
    source = inspect.getsource(lj.parse_args)
    assert '"--epsilon"' not in source
    assert "EPSILONS" not in source
