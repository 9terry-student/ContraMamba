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
    reason_router_gen4_family_subspace_sensitivity_fast_cuda
    as sub,
)


def _baseline_item(
    family: str,
    index: int,
) -> dict[str, object]:
    pair = phase1._expected_pairs(
        family
    )[index]
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
        "alignment_delta_h_shape": [6],
    }


def _distinct_axis_plans() -> torch.Tensor:
    counts = [80, 70, 60, 45, 30, 15]
    rows = []
    for axis, count in enumerate(counts):
        for _ in range(count):
            row = torch.zeros(
                6,
                dtype=torch.float64,
            )
            row[axis] = 1.0
            rows.append(row)
    assert len(rows) == 300
    return torch.stack(rows, dim=0)


def _degenerate_axis_plans() -> torch.Tensor:
    rows = []
    for axis in range(6):
        for _ in range(50):
            row = torch.zeros(
                6,
                dtype=torch.float64,
            )
            row[axis] = 1.0
            rows.append(row)
    return torch.stack(rows, dim=0)


def _signed_probe(
    orientation: int,
    f_value: float,
) -> dict[str, object]:
    return {
        "orientation": orientation,
        "delta_h_l2": 2.0 * sub.EPSILON,
        "plus_path_efficiency": f_value + 1.0,
        "minus_path_efficiency": 1.0,
        "F": f_value,
        "midpoint_max_abs_residual": 0.0,
        "pair_delta_max_abs_residual": 0.0,
        (
            "applied_correction_"
            "max_abs_residual"
        ): 0.0,
        "runtime_correction_l2": (
            2.0 * sub.EPSILON
        ),
        "model_forward_count": 2,
    }


def _direction_probe(
    basis_family: str,
    basis_index: int,
    j_value: float,
) -> dict[str, object]:
    f_plus = j_value * sub.EPSILON
    f_minus = -j_value * sub.EPSILON
    realized_j = (
        f_plus - f_minus
    ) / (2.0 * sub.EPSILON)
    return {
        "basis_family": basis_family,
        "basis_index": basis_index,
        "epsilon": sub.EPSILON,
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": realized_j,
        "J_squared": realized_j * realized_j,
        "positive_probe": _signed_probe(
            1,
            f_plus,
        ),
        "negative_probe": _signed_probe(
            -1,
            f_minus,
        ),
        "model_forward_count": 4,
    }


def _synthetic_item(
    family: str,
    index: int,
) -> dict[str, object]:
    cross_family = (
        "xg4"
        if family == "xg2"
        else "xg2"
    )
    own = [
        _direction_probe(
            family,
            j,
            0.1 * (j + 1),
        )
        for j in range(5)
    ]
    cross = [
        _direction_probe(
            cross_family,
            j,
            0.05 * (j + 1),
        )
        for j in range(5)
    ]
    e_own = sum(
        float(row["J_squared"])
        for row in own
    ) / 5.0
    e_cross = sum(
        float(row["J_squared"])
        for row in cross
    ) / 5.0

    row = _baseline_item(
        family,
        index,
    )
    row["phase1_schema_version"] = (
        row["schema_version"]
    )
    row["schema_version"] = sub.ITEM_SCHEMA
    row["implementation_scope_commit"] = (
        sub.IMPLEMENTATION_SCOPE_COMMIT
    )
    row[
        "finite_difference_basis_correction_commit"
    ] = (
        sub.FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT
    )
    row["phase1_artifact_freeze_commit"] = (
        sub.PHASE1_ARTIFACT_FREEZE_COMMIT
    )
    row["epsilon"] = sub.EPSILON
    row["subspace_dim"] = sub.SUBSPACE_DIM
    row["own_basis_family"] = family
    row["cross_basis_family"] = cross_family
    row["own_basis_probes"] = own
    row["cross_basis_probes"] = cross
    row["E_own"] = e_own
    row["E_cross"] = e_cross
    row["D"] = e_own - e_cross
    row[
        "baseline_model_forward_count_this_run"
    ] = 0
    row[
        "scientific_model_forward_count_this_run"
    ] = sub.FORWARDS_PER_PAIR
    return row


def _synthetic_summary(
    family: str,
) -> dict[str, object]:
    def basis_row(
        basis_family: str,
    ) -> dict[str, object]:
        return {
            "phase1_plan_sha256": (
                sub.PHASE1_PLAN_SHA256[
                    basis_family
                ]
            ),
            "subspace_dim": 5,
            "top5_eigenvalues": [
                0.30,
                0.25,
                0.20,
                0.15,
                0.07,
            ],
            "top5_to_6_eigengaps": [
                0.05,
                0.05,
                0.05,
                0.08,
                0.05,
            ],
            "minimum_selected_eigengap": 0.05,
            "second_moment_symmetry_residual": 0.0,
            "orthonormality_max_abs_residual": 0.0,
            "eigenbasis_order": "eigenvalue_descending",
            "sign_canonicalization": (
                "largest_abs_coordinate_positive"
            ),
        }

    return {
        "schema_version": sub.SUMMARY_SCHEMA,
        "result": sub.RESULT_PASS,
        "family_key": family,
        "execution_head": "synthetic",
        "implementation_scope_commit": (
            sub.IMPLEMENTATION_SCOPE_COMMIT
        ),
        (
            "finite_difference_"
            "basis_correction_commit"
        ): (
            sub.FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT
        ),
        "phase1_artifact_freeze_commit": (
            sub.PHASE1_ARTIFACT_FREEZE_COMMIT
        ),
        "source_pair_count": 300,
        "epsilon": sub.EPSILON,
        "subspace_dim": 5,
        "basis_reconstruction": {
            "xg2": basis_row("xg2"),
            "xg4": basis_row("xg4"),
        },
        (
            "baseline_model_forward_"
            "count_this_run"
        ): 0,
        (
            "scientific_model_forward_"
            "count_this_run"
        ): 12000,
        "primary_inference_executed": False,
        "holm_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
    }


def test_scope_correction_and_budget_identities():
    assert sub.IMPLEMENTATION_SCOPE_COMMIT == (
        "4f36483080b6f63b9c9a3dd031cd97a88995e0f3"
    )
    assert (
        sub.FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT
        == "594cb45bdd8740b2dfd63c4780f766f1d0b375bc"
    )
    assert sub.SUBSPACE_DIM == 5
    assert sub.EPSILON == 0.025
    assert sub.EIGENGAP_TOL == 1.0e-10
    assert sub.ORTHONORMALITY_TOL == 1.0e-10
    assert sub.FORWARDS_PER_DIRECTION == 4
    assert sub.DIRECTIONS_PER_PAIR == 10
    assert sub.FORWARDS_PER_PAIR == 40
    assert (
        sub.SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY
        == 12000
    )
    assert sub.BASELINE_FORWARD_BUDGET_THIS_RUN == 0


def test_reconstruct_basis_uses_descending_top5():
    result = sub.reconstruct_family_basis(
        "xg2",
        _distinct_axis_plans(),
    )
    assert result["subspace_dim"] == 5
    assert result["eigenbasis_order"] == (
        "eigenvalue_descending"
    )
    assert result["top5_eigenvalues"] == pytest.approx(
        [
            80 / 300,
            70 / 300,
            60 / 300,
            45 / 300,
            30 / 300,
        ]
    )
    assert result["minimum_selected_eigengap"] > (
        sub.EIGENGAP_TOL
    )

    basis = result["basis"]
    assert tuple(basis.shape) == (6, 5)
    assert torch.allclose(
        basis.T @ basis,
        torch.eye(
            5,
            dtype=torch.float64,
        ),
        atol=1e-12,
        rtol=0.0,
    )


def test_basis_sign_is_canonicalized():
    basis = torch.tensor(
        [
            [-0.2, 0.0],
            [0.9, -0.8],
            [0.1, 0.6],
        ],
        dtype=torch.float64,
    )
    result = (
        sub._canonicalize_eigenvector_signs(
            basis
        )
    )
    assert result[1, 0] > 0
    assert result[1, 1] > 0


def test_degenerate_selected_eigenspace_fails_closed():
    with pytest.raises(
        sub.FamilySubspaceSensitivityError,
        match="EIGENGAP_NOT_STRICT",
    ):
        sub.reconstruct_family_basis(
            "xg2",
            _degenerate_axis_plans(),
        )


def test_zero_and_nonfinite_plan_fail_closed():
    zero = _distinct_axis_plans()
    zero[0].zero_()
    with pytest.raises(
        sub.FamilySubspaceSensitivityError,
        match="ZERO_PLAN_NORM",
    ):
        sub.reconstruct_family_basis(
            "xg2",
            zero,
        )

    bad = _distinct_axis_plans()
    bad[1, 2] = float("nan")
    with pytest.raises(
        sub.FamilySubspaceSensitivityError,
        match="NONFINITE_PLAN",
    ):
        sub.reconstruct_family_basis(
            "xg2",
            bad,
        )


def test_direction_j_uses_exact_fixed_epsilon(
    monkeypatch,
):
    calls = []

    def fake_probe(
        _family,
        _baseline,
        _direction,
        *,
        epsilon,
        orientation,
        **_kwargs,
    ):
        calls.append(
            (epsilon, orientation)
        )
        f = 1.0 + orientation * (
            -0.4 * epsilon
        )
        return _signed_probe(
            orientation,
            f,
        )

    monkeypatch.setattr(
        sub.lj,
        "_run_signed_probe",
        fake_probe,
    )

    result = sub._run_direction_j(
        "xg2",
        _baseline_item("xg2", 0),
        torch.tensor(
            [1.0, 0, 0, 0, 0, 0],
            dtype=torch.float64,
        ),
        basis_family="xg2",
        basis_index=0,
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
    ]
    assert result["J"] == pytest.approx(-0.4)
    assert result["J_squared"] == pytest.approx(0.16)
    assert result["model_forward_count"] == 4


def test_pair_endpoint_is_mean_squared_own_minus_cross(
    monkeypatch,
):
    calls = []

    def fake_direction(
        _family,
        _baseline,
        _direction,
        *,
        basis_family,
        basis_index,
        **_kwargs,
    ):
        calls.append(
            (basis_family, basis_index)
        )
        j = (
            float(basis_index + 1)
            if basis_family == "xg2"
            else 0.5 * float(
                basis_index + 1
            )
        )
        return _direction_probe(
            basis_family,
            basis_index,
            j,
        )

    monkeypatch.setattr(
        sub,
        "_run_direction_j",
        fake_direction,
    )

    eye = torch.eye(
        6,
        dtype=torch.float64,
    )[:, :5]
    result = sub._run_subspace_pair(
        "xg2",
        _baseline_item("xg2", 0),
        own_basis=eye,
        cross_family="xg4",
        cross_basis=eye,
        model=object(),
        runtime_ctx={},
        trace_code=object(),
        trace_line=1,
        encoded={},
        row_index={},
        events={},
        budget=object(),
    )

    expected_own = sum(
        float(j * j)
        for j in range(1, 6)
    ) / 5.0
    expected_cross = sum(
        float((0.5 * j) ** 2)
        for j in range(1, 6)
    ) / 5.0

    assert calls == [
        ("xg2", 0),
        ("xg2", 1),
        ("xg2", 2),
        ("xg2", 3),
        ("xg2", 4),
        ("xg4", 0),
        ("xg4", 1),
        ("xg4", 2),
        ("xg4", 3),
        ("xg4", 4),
    ]
    assert result["E_own"] == pytest.approx(
        expected_own
    )
    assert result["E_cross"] == pytest.approx(
        expected_cross
    )
    assert result["D"] == pytest.approx(
        expected_own - expected_cross
    )
    assert (
        result[
            "scientific_model_forward_count_this_run"
        ]
        == 40
    )


def test_items_validate_exact_endpoint_and_budget():
    items = [
        _synthetic_item(
            "xg2",
            index,
        )
        for index in range(300)
    ]
    sub._validate_items(
        "xg2",
        items,
    )

    items[10]["D"] = float(
        items[10]["D"]
    ) + 1.0
    with pytest.raises(
        sub.FamilySubspaceSensitivityError,
        match="D_IDENTITY",
    ):
        sub._validate_items(
            "xg2",
            items,
        )


def test_no_inference_or_response_guided_basis_path():
    basis_source = inspect.getsource(
        sub.reconstruct_family_basis
    )
    run_source = inspect.getsource(
        sub.run_family_subspace_sensitivity
    )
    combined = basis_source + run_source

    assert "p_value" not in combined
    assert "ttest" not in combined.lower()
    assert "holm" not in basis_source.lower()
    assert "R_ALIGN" not in combined
    assert "delta_alignment" not in combined
    assert "local_jacobian_items" not in combined
    assert (
        "basis_reconstruction_before_"
        in run_source
    )
    assert (
        "primary_inference_executed"
        in run_source
    )


def test_both_bases_are_loaded_before_runtime(
    monkeypatch,
    tmp_path,
):
    calls = []

    monkeypatch.setattr(
        sub,
        "authenticate_repo",
        lambda _head: calls.append(
            "authenticate"
        ),
    )

    def fake_load():
        calls.append("bases")
        raise sub.FamilySubspaceSensitivityError(
            "STOP_BEFORE_RUNTIME"
        )

    monkeypatch.setattr(
        sub,
        "_load_all_phase1_and_bases",
        fake_load,
    )

    def forbidden_runtime():
        calls.append("runtime")
        raise AssertionError(
            "runtime must not execute"
        )

    monkeypatch.setattr(
        phase1.base.prevalence_eq.backend,
        "runtime_gate",
        forbidden_runtime,
    )

    with pytest.raises(
        sub.FamilySubspaceSensitivityError,
        match="STOP_BEFORE_RUNTIME",
    ):
        sub.run_family_subspace_sensitivity(
            family="xg2",
            expected_head="synthetic",
            model_snapshot=tmp_path,
            tokenizer_snapshot=tmp_path,
            checkpoint_path=tmp_path,
            output_dir=(
                tmp_path / "output"
            ),
        )

    assert calls == [
        "authenticate",
        "bases",
    ]


def test_artifact_roundtrip_and_no_inference(
    tmp_path,
):
    items = [
        _synthetic_item(
            "xg4",
            index,
        )
        for index in range(300)
    ]
    summary = _synthetic_summary(
        "xg4"
    )
    out = tmp_path / "artifact"

    sub._write_outputs(
        out,
        family="xg4",
        items=items,
        summary=summary,
    )

    loaded = (
        sub.validate_subspace_sensitivity_artifact(
            out,
            "xg4",
        )
    )
    assert (
        loaded["summary"]["result"]
        == sub.RESULT_PASS
    )
    assert (
        loaded["summary"][
            "primary_inference_executed"
        ]
        is False
    )
    assert (
        loaded["summary"][
            "holm_correction_executed"
        ]
        is False
    )
    assert (
        loaded["summary"][
            "scientific_conclusion"
        ]
        is None
    )


def test_manifest_tamper_fails_closed(
    tmp_path,
):
    items = [
        _synthetic_item(
            "xg2",
            index,
        )
        for index in range(300)
    ]
    summary = _synthetic_summary(
        "xg2"
    )
    out = tmp_path / "artifact"

    sub._write_outputs(
        out,
        family="xg2",
        items=items,
        summary=summary,
    )

    with (
        out / sub.ITEM_FILE
    ).open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("{}\n")

    with pytest.raises(
        sub.FamilySubspaceSensitivityError,
        match="ARTIFACT_SHA256",
    ):
        (
            sub.validate_subspace_sensitivity_artifact(
                out,
                "xg2",
            )
        )


def test_cli_description_has_observation_boundary():
    source = inspect.getsource(
        sub.parse_args
    )
    assert (
        "No baseline forward"
        in source
    )
    assert (
        "inferential test"
        in source
    )
