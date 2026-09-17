from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1
    as phase1,
)
from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2
    as phase2,
)
from scripts import (
    reason_router_gen4_xg2_xg4_local_jacobian_fast_cuda
    as lj,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-family-subspace-sensitivity"
IMPLEMENTATION_SCOPE_COMMIT = (
    "4f36483080b6f63b9c9a3dd031cd97a88995e0f3"
)
FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT = (
    "594cb45bdd8740b2dfd63c4780f766f1d0b375bc"
)
PHASE1_ARTIFACT_FREEZE_COMMIT = phase2.PHASE1_ARTIFACT_FREEZE_COMMIT
PHASE1_ARTIFACT_ROOT = phase2.PHASE1_ARTIFACT_ROOT

PHASE1_PLAN_SHA256 = {
    "xg2": "b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c",
    "xg4": "792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f",
}

SOURCE_PAIR_COUNT = phase2.SOURCE_PAIR_COUNT
FAMILIES = ("xg2", "xg4")
SUBSPACE_DIM = 5
EPSILON = 0.025
EIGENGAP_TOL = 1.0e-10
ORTHONORMALITY_TOL = 1.0e-10
SYMMETRY_TOL = 1.0e-12

FORWARDS_PER_SIGNED_PROBE = 2
SIGNED_PROBES_PER_DIRECTION = 2
FORWARDS_PER_DIRECTION = (
    FORWARDS_PER_SIGNED_PROBE * SIGNED_PROBES_PER_DIRECTION
)
DIRECTIONS_PER_ROLE = SUBSPACE_DIM
DIRECTIONS_PER_PAIR = 2 * DIRECTIONS_PER_ROLE
FORWARDS_PER_PAIR = FORWARDS_PER_DIRECTION * DIRECTIONS_PER_PAIR
SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY = SOURCE_PAIR_COUNT * FORWARDS_PER_PAIR
BASELINE_FORWARD_BUDGET_THIS_RUN = 0

ITEM_SCHEMA = "gen4-k-family-subspace-sensitivity-item-v1"
SUMMARY_SCHEMA = "gen4-k-family-subspace-sensitivity-summary-v1"
MANIFEST_SCHEMA = "gen4-k-family-subspace-sensitivity-manifest-v1"
RESULT_PASS = "PASS_FAMILY_SUBSPACE_SENSITIVITY_OBSERVATION"

ITEM_FILE = "subspace_sensitivity_items.jsonl"
SUMMARY_FILE = "subspace_sensitivity_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

REUSED_PATHS = (
    "reports/reason_router_gen4_family_subspace_sensitivity_implementation_scope.md",
    "reports/reason_router_gen4_family_subspace_sensitivity_finite_difference_basis_correction.md",
    "scripts/reason_router_gen4_xg2_xg4_local_jacobian_fast_cuda.py",
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py",
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2.py",
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_full_baseline.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_core.py",
)


class FamilySubspaceSensitivityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FamilySubspaceSensitivityError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FamilySubspaceSensitivityError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(
        git("status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )

    for ancestor, label in (
        (IMPLEMENTATION_SCOPE_COMMIT, "IMPLEMENTATION_SCOPE"),
        (
            FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT,
            "FINITE_DIFFERENCE_BASIS_CORRECTION",
        ),
        (PHASE1_ARTIFACT_FREEZE_COMMIT, "PHASE1_ARTIFACT_FREEZE"),
    ):
        require(
            _git_is_ancestor(ancestor, expected_head),
            f"{label}_NOT_ANCESTOR",
        )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            PHASE1_ARTIFACT_FREEZE_COMMIT,
            expected_head,
            "--",
            PHASE1_ARTIFACT_ROOT.as_posix(),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_PHASE1_ARTIFACT_TREE_DRIFT")

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT,
            expected_head,
            "--",
            *REUSED_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REUSED_DEPENDENCY_DRIFT")


def _expected_pairs(family: str) -> tuple[str, ...]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    return phase1._expected_pairs(family)


def _phase1_plan_path(family: str) -> Path:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    return (
        ROOT
        / PHASE1_ARTIFACT_ROOT
        / family
        / phase1.PLAN_FILE
    )


def _validate_plan_sha(family: str) -> str:
    path = _phase1_plan_path(family)
    require(
        path.is_file(),
        f"PHASE1_PLAN_FILE_MISSING:{family}",
    )
    observed = phase2.sha256_file(path)
    require(
        observed == PHASE1_PLAN_SHA256[family],
        f"PHASE1_PLAN_SHA256_DRIFT:{family}:{observed}",
    )
    return observed


def _unit_rows(
    plans: torch.Tensor,
    family: str,
) -> torch.Tensor:
    require(
        torch.is_tensor(plans),
        f"PLAN_NOT_TENSOR:{family}",
    )
    require(
        plans.ndim == 2
        and int(plans.shape[0]) == SOURCE_PAIR_COUNT,
        f"PLAN_SHAPE:{family}:{tuple(plans.shape)}",
    )
    require(
        plans.dtype == torch.float64,
        f"PLAN_DTYPE:{family}:{plans.dtype}",
    )

    x = (
        plans.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
        .clone()
    )
    require(
        bool(torch.isfinite(x).all().item()),
        f"NONFINITE_PLAN:{family}",
    )

    norms = torch.linalg.vector_norm(x, ord=2, dim=1)
    require(
        bool(torch.isfinite(norms).all().item()),
        f"NONFINITE_PLAN_NORM:{family}",
    )
    require(
        bool(torch.all(norms > 0.0).item()),
        f"ZERO_PLAN_NORM:{family}",
    )

    unit = (x / norms[:, None]).contiguous()
    residual = float(
        torch.max(
            torch.abs(
                torch.linalg.vector_norm(
                    unit,
                    ord=2,
                    dim=1,
                )
                - 1.0
            )
        ).item()
    )
    require(
        math.isfinite(residual) and residual <= 1.0e-12,
        f"UNIT_ROW_NORM_RESIDUAL:{family}:{residual}",
    )
    return unit


def _canonicalize_eigenvector_signs(
    basis: torch.Tensor,
) -> torch.Tensor:
    require(
        torch.is_tensor(basis)
        and basis.ndim == 2,
        "BASIS_SHAPE",
    )

    out = (
        basis.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
        .clone()
    )

    for column in range(int(out.shape[1])):
        vector = out[:, column]
        pivot = int(
            torch.argmax(torch.abs(vector)).item()
        )
        pivot_value = float(vector[pivot].item())
        require(
            math.isfinite(pivot_value)
            and pivot_value != 0.0,
            f"BASIS_SIGN_PIVOT:{column}:{pivot_value}",
        )
        if pivot_value < 0.0:
            out[:, column].mul_(-1.0)

    return out.contiguous()


def reconstruct_family_basis(
    family: str,
    plans: torch.Tensor,
) -> dict[str, Any]:
    unit = _unit_rows(plans, family)

    second = (
        unit.T @ unit
    ) / float(SOURCE_PAIR_COUNT)
    second = second.to(torch.float64).contiguous()

    require(
        bool(torch.isfinite(second).all().item()),
        f"NONFINITE_SECOND_MOMENT:{family}",
    )

    symmetry_residual = float(
        torch.max(torch.abs(second - second.T)).item()
    )
    require(
        math.isfinite(symmetry_residual)
        and symmetry_residual <= SYMMETRY_TOL,
        (
            f"SECOND_MOMENT_NOT_SYMMETRIC:"
            f"{family}:{symmetry_residual}"
        ),
    )

    eigenvalues_asc, eigenvectors_asc = (
        torch.linalg.eigh(second)
    )
    require(
        bool(torch.isfinite(eigenvalues_asc).all().item())
        and bool(
            torch.isfinite(eigenvectors_asc)
            .all()
            .item()
        ),
        f"NONFINITE_EIGH:{family}",
    )

    order = torch.arange(
        int(eigenvalues_asc.numel()) - 1,
        -1,
        -1,
        dtype=torch.long,
    )
    eigenvalues = eigenvalues_asc[order].contiguous()
    eigenvectors = (
        eigenvectors_asc[:, order]
        .contiguous()
    )

    require(
        int(eigenvalues.numel()) >= SUBSPACE_DIM + 1,
        f"INSUFFICIENT_EIGENVALUES:{family}",
    )

    selected_values = (
        eigenvalues[:SUBSPACE_DIM]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
        .clone()
    )
    basis = _canonicalize_eigenvector_signs(
        eigenvectors[:, :SUBSPACE_DIM]
    )

    gaps = (
        eigenvalues[:SUBSPACE_DIM]
        - eigenvalues[1 : SUBSPACE_DIM + 1]
    )
    require(
        bool(torch.isfinite(gaps).all().item()),
        f"NONFINITE_EIGENGAP:{family}",
    )
    min_gap = float(torch.min(gaps).item())
    require(
        min_gap > EIGENGAP_TOL,
        f"EIGENGAP_NOT_STRICT:{family}:{min_gap}",
    )

    gram = basis.T @ basis
    identity = torch.eye(
        SUBSPACE_DIM,
        dtype=torch.float64,
    )
    orth_residual = float(
        torch.max(torch.abs(gram - identity)).item()
    )
    require(
        math.isfinite(orth_residual)
        and orth_residual <= ORTHONORMALITY_TOL,
        (
            f"BASIS_NOT_ORTHONORMAL:"
            f"{family}:{orth_residual}"
        ),
    )

    basis_norms = torch.linalg.vector_norm(
        basis,
        ord=2,
        dim=0,
    )
    require(
        bool(
            torch.all(
                torch.abs(basis_norms - 1.0)
                <= ORTHONORMALITY_TOL
            ).item()
        ),
        f"BASIS_UNIT_NORM:{family}",
    )

    return {
        "family_key": family,
        "basis": basis,
        "top5_eigenvalues": [
            float(value)
            for value in selected_values.tolist()
        ],
        "top5_to_6_eigengaps": [
            float(value)
            for value in gaps.tolist()
        ],
        "minimum_selected_eigengap": min_gap,
        "second_moment_symmetry_residual": (
            symmetry_residual
        ),
        "orthonormality_max_abs_residual": (
            orth_residual
        ),
        "subspace_dim": SUBSPACE_DIM,
        "eigenbasis_order": "eigenvalue_descending",
        "sign_canonicalization": (
            "largest_abs_coordinate_positive"
        ),
    }


def _load_all_phase1_and_bases() -> dict[str, Any]:
    loaded: dict[str, Any] = {}

    for family in FAMILIES:
        plan_sha = _validate_plan_sha(family)
        artifact = phase2.load_phase1_artifact(
            family
        )
        items = artifact["items"]
        plans = artifact["alignment_delta_h"]

        require(
            tuple(
                str(row["source_pair_id"])
                for row in items
            )
            == _expected_pairs(family),
            f"PHASE1_PAIR_ORDER:{family}",
        )

        basis = reconstruct_family_basis(
            family,
            plans,
        )
        loaded[family] = {
            "artifact": artifact,
            "basis": basis,
            "plan_sha256": plan_sha,
        }

    return loaded


def _run_direction_j(
    family: str,
    baseline_item: Mapping[str, Any],
    unit_direction: torch.Tensor,
    *,
    basis_family: str,
    basis_index: int,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> dict[str, Any]:
    require(
        basis_family in FAMILIES,
        f"BAD_BASIS_FAMILY:{basis_family}",
    )
    require(
        0 <= basis_index < SUBSPACE_DIM,
        f"BAD_BASIS_INDEX:{basis_index}",
    )

    direction = (
        unit_direction.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
        .clone()
    )
    norm = float(
        torch.linalg.vector_norm(
            direction,
            ord=2,
        ).item()
    )
    require(
        math.isfinite(norm)
        and abs(norm - 1.0)
        <= ORTHONORMALITY_TOL,
        (
            f"PROBE_DIRECTION_NORM:"
            f"{basis_family}:{basis_index}:{norm}"
        ),
    )

    positive = lj._run_signed_probe(
        family,
        baseline_item,
        direction,
        epsilon=EPSILON,
        orientation=1,
        model=model,
        runtime_ctx=runtime_ctx,
        trace_code=trace_code,
        trace_line=trace_line,
        encoded=encoded,
        row_index=row_index,
        events=events,
        budget=budget,
    )
    negative = lj._run_signed_probe(
        family,
        baseline_item,
        direction,
        epsilon=EPSILON,
        orientation=-1,
        model=model,
        runtime_ctx=runtime_ctx,
        trace_code=trace_code,
        trace_line=trace_line,
        encoded=encoded,
        row_index=row_index,
        events=events,
        budget=budget,
    )

    f_plus = float(positive["F"])
    f_minus = float(negative["F"])
    j_value = (
        f_plus - f_minus
    ) / (2.0 * EPSILON)

    require(
        all(
            math.isfinite(value)
            for value in (
                f_plus,
                f_minus,
                j_value,
            )
        ),
        (
            f"NONFINITE_DIRECTION_RESPONSE:"
            f"{basis_family}:{basis_index}"
        ),
    )

    return {
        "basis_family": basis_family,
        "basis_index": int(basis_index),
        "epsilon": EPSILON,
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": (
            FORWARDS_PER_DIRECTION
        ),
    }


def _run_subspace_pair(
    family: str,
    baseline_item: Mapping[str, Any],
    *,
    own_basis: torch.Tensor,
    cross_family: str,
    cross_basis: torch.Tensor,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> dict[str, Any]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    require(
        cross_family in FAMILIES
        and cross_family != family,
        (
            f"BAD_CROSS_FAMILY:"
            f"{family}:{cross_family}"
        ),
    )
    require(
        tuple(own_basis.shape)[1:] == ()
        or own_basis.ndim == 2,
        "OWN_BASIS_SHAPE",
    )
    require(
        own_basis.ndim == 2
        and int(own_basis.shape[1])
        == SUBSPACE_DIM,
        f"OWN_BASIS_DIM:{tuple(own_basis.shape)}",
    )
    require(
        cross_basis.ndim == 2
        and int(cross_basis.shape[1])
        == SUBSPACE_DIM,
        (
            f"CROSS_BASIS_DIM:"
            f"{tuple(cross_basis.shape)}"
        ),
    )
    require(
        int(own_basis.shape[0])
        == int(cross_basis.shape[0]),
        "BASIS_WIDTH_MISMATCH",
    )

    pair = str(
        baseline_item["source_pair_id"]
    )
    require(
        baseline_item["family_key"] == family,
        f"BASELINE_ITEM_FAMILY:{pair}",
    )

    own: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []

    for index in range(SUBSPACE_DIM):
        own.append(
            _run_direction_j(
                family,
                baseline_item,
                own_basis[:, index],
                basis_family=family,
                basis_index=index,
                model=model,
                runtime_ctx=runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                budget=budget,
            )
        )

    for index in range(SUBSPACE_DIM):
        cross.append(
            _run_direction_j(
                family,
                baseline_item,
                cross_basis[:, index],
                basis_family=cross_family,
                basis_index=index,
                model=model,
                runtime_ctx=runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                budget=budget,
            )
        )

    e_own = sum(
        float(row["J_squared"])
        for row in own
    ) / float(SUBSPACE_DIM)
    e_cross = sum(
        float(row["J_squared"])
        for row in cross
    ) / float(SUBSPACE_DIM)
    d_value = e_own - e_cross

    require(
        all(
            math.isfinite(value)
            for value in (
                e_own,
                e_cross,
                d_value,
            )
        ),
        f"NONFINITE_PAIR_ENDPOINT:{pair}",
    )

    item = dict(baseline_item)
    item["phase1_schema_version"] = (
        item["schema_version"]
    )
    item["schema_version"] = ITEM_SCHEMA
    item["implementation_scope_commit"] = (
        IMPLEMENTATION_SCOPE_COMMIT
    )
    item[
        "finite_difference_basis_correction_commit"
    ] = FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT
    item["phase1_artifact_freeze_commit"] = (
        PHASE1_ARTIFACT_FREEZE_COMMIT
    )
    item["epsilon"] = EPSILON
    item["subspace_dim"] = SUBSPACE_DIM
    item["own_basis_family"] = family
    item["cross_basis_family"] = cross_family
    item["own_basis_probes"] = own
    item["cross_basis_probes"] = cross
    item["E_own"] = e_own
    item["E_cross"] = e_cross
    item["D"] = d_value
    item[
        "baseline_model_forward_count_this_run"
    ] = 0
    item[
        "scientific_model_forward_count_this_run"
    ] = FORWARDS_PER_PAIR

    return item


def _validate_direction_probe(
    probe: Mapping[str, Any],
    *,
    expected_basis_family: str,
    expected_basis_index: int,
) -> None:
    require(
        probe.get("basis_family")
        == expected_basis_family,
        (
            f"PROBE_BASIS_FAMILY:"
            f"{expected_basis_family}:"
            f"{expected_basis_index}"
        ),
    )
    require(
        probe.get("basis_index")
        == expected_basis_index,
        (
            f"PROBE_BASIS_INDEX:"
            f"{expected_basis_family}:"
            f"{expected_basis_index}"
        ),
    )
    require(
        float(probe["epsilon"]) == EPSILON,
        (
            f"PROBE_EPSILON:"
            f"{expected_basis_family}:"
            f"{expected_basis_index}"
        ),
    )
    require(
        int(probe["model_forward_count"])
        == FORWARDS_PER_DIRECTION,
        (
            f"PROBE_FORWARD_COUNT:"
            f"{expected_basis_family}:"
            f"{expected_basis_index}"
        ),
    )

    f_plus = float(probe["F_plus"])
    f_minus = float(probe["F_minus"])
    j_value = float(probe["J"])
    j_squared = float(probe["J_squared"])
    require(
        all(
            math.isfinite(value)
            for value in (
                f_plus,
                f_minus,
                j_value,
                j_squared,
            )
        ),
        (
            f"PROBE_NONFINITE:"
            f"{expected_basis_family}:"
            f"{expected_basis_index}"
        ),
    )
    require(
        j_value
        == (
            f_plus - f_minus
        ) / (2.0 * EPSILON),
        (
            f"J_IDENTITY:"
            f"{expected_basis_family}:"
            f"{expected_basis_index}"
        ),
    )
    require(
        j_squared == j_value * j_value,
        (
            f"J_SQUARED_IDENTITY:"
            f"{expected_basis_family}:"
            f"{expected_basis_index}"
        ),
    )

    for name, orientation in (
        ("positive_probe", 1),
        ("negative_probe", -1),
    ):
        signed = probe[name]
        require(
            int(signed["orientation"])
            == orientation,
            (
                f"SIGNED_ORIENTATION:"
                f"{expected_basis_family}:"
                f"{expected_basis_index}:{name}"
            ),
        )
        require(
            int(signed["model_forward_count"])
            == FORWARDS_PER_SIGNED_PROBE,
            (
                f"SIGNED_FORWARD_COUNT:"
                f"{expected_basis_family}:"
                f"{expected_basis_index}:{name}"
            ),
        )
        runtime_l2 = float(
            signed["runtime_correction_l2"]
        )
        require(
            math.isfinite(runtime_l2)
            and abs(
                runtime_l2
                - 2.0 * EPSILON
            )
            <= 1.0e-12,
            (
                f"SIGNED_RUNTIME_L2:"
                f"{expected_basis_family}:"
                f"{expected_basis_index}:{name}"
            ),
        )


def _validate_items(
    family: str,
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "ITEM_COUNT",
    )

    cross_family = (
        "xg4"
        if family == "xg2"
        else "xg2"
    )

    for index, (
        expected_pair,
        raw,
    ) in enumerate(
        zip(
            _expected_pairs(family),
            items,
            strict=True,
        )
    ):
        row = dict(raw)
        require(
            row.get("schema_version")
            == ITEM_SCHEMA,
            f"ITEM_SCHEMA:{index}",
        )
        require(
            row.get("phase1_schema_version")
            == phase1.ITEM_SCHEMA,
            f"PHASE1_SCHEMA:{index}",
        )
        require(
            row.get("family_key") == family,
            f"ITEM_FAMILY:{index}",
        )
        require(
            row.get("source_pair_id")
            == expected_pair,
            f"PAIR_ORDER:{index}",
        )
        require(
            row.get("alignment_plan_index")
            == index,
            f"PLAN_INDEX:{index}",
        )
        require(
            row.get(
                "implementation_scope_commit"
            )
            == IMPLEMENTATION_SCOPE_COMMIT,
            f"SCOPE_COMMIT:{index}",
        )
        require(
            row.get(
                "finite_difference_basis_correction_commit"
            )
            == FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT,
            f"CORRECTION_COMMIT:{index}",
        )
        require(
            row.get(
                "phase1_artifact_freeze_commit"
            )
            == PHASE1_ARTIFACT_FREEZE_COMMIT,
            f"PHASE1_FREEZE:{index}",
        )
        require(
            float(row["epsilon"]) == EPSILON,
            f"EPSILON:{index}",
        )
        require(
            int(row["subspace_dim"])
            == SUBSPACE_DIM,
            f"SUBSPACE_DIM:{index}",
        )
        require(
            row["own_basis_family"] == family,
            f"OWN_FAMILY:{index}",
        )
        require(
            row["cross_basis_family"]
            == cross_family,
            f"CROSS_FAMILY:{index}",
        )
        require(
            row[
                "baseline_model_forward_count_this_run"
            ]
            == 0,
            f"BASELINE_FORWARD_COUNT:{index}",
        )
        require(
            row[
                "scientific_model_forward_count_this_run"
            ]
            == FORWARDS_PER_PAIR,
            f"SCIENTIFIC_FORWARD_COUNT:{index}",
        )

        own = row["own_basis_probes"]
        cross = row["cross_basis_probes"]
        require(
            isinstance(own, list)
            and len(own) == SUBSPACE_DIM,
            f"OWN_PROBE_COUNT:{index}",
        )
        require(
            isinstance(cross, list)
            and len(cross) == SUBSPACE_DIM,
            f"CROSS_PROBE_COUNT:{index}",
        )

        for basis_index, probe in enumerate(
            own
        ):
            _validate_direction_probe(
                probe,
                expected_basis_family=family,
                expected_basis_index=basis_index,
            )

        for basis_index, probe in enumerate(
            cross
        ):
            _validate_direction_probe(
                probe,
                expected_basis_family=cross_family,
                expected_basis_index=basis_index,
            )

        e_own = float(row["E_own"])
        e_cross = float(row["E_cross"])
        d_value = float(row["D"])

        expected_e_own = sum(
            float(probe["J_squared"])
            for probe in own
        ) / float(SUBSPACE_DIM)
        expected_e_cross = sum(
            float(probe["J_squared"])
            for probe in cross
        ) / float(SUBSPACE_DIM)

        require(
            e_own == expected_e_own,
            f"E_OWN_IDENTITY:{index}",
        )
        require(
            e_cross == expected_e_cross,
            f"E_CROSS_IDENTITY:{index}",
        )
        require(
            d_value == e_own - e_cross,
            f"D_IDENTITY:{index}",
        )


def _basis_summary(
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for family in FAMILIES:
        basis = loaded[family]["basis"]
        result[family] = {
            "phase1_plan_sha256": (
                loaded[family]["plan_sha256"]
            ),
            "subspace_dim": SUBSPACE_DIM,
            "top5_eigenvalues": list(
                basis["top5_eigenvalues"]
            ),
            "top5_to_6_eigengaps": list(
                basis[
                    "top5_to_6_eigengaps"
                ]
            ),
            "minimum_selected_eigengap": float(
                basis[
                    "minimum_selected_eigengap"
                ]
            ),
            "second_moment_symmetry_residual": float(
                basis[
                    "second_moment_symmetry_residual"
                ]
            ),
            "orthonormality_max_abs_residual": float(
                basis[
                    "orthonormality_max_abs_residual"
                ]
            ),
            "eigenbasis_order": basis[
                "eigenbasis_order"
            ],
            "sign_canonicalization": basis[
                "sign_canonicalization"
            ],
        }

    return result


def _write_outputs(
    output_dir: Path,
    *,
    family: str,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )
    _validate_items(family, items)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    payloads = {
        ITEM_FILE: phase2.jsonl_bytes(items),
        SUMMARY_FILE: phase2.canonical_json_bytes(
            summary
        ),
    }

    hashes: dict[str, str] = {}
    for name, raw in payloads.items():
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = phase2.sha256_bytes(raw)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int(
                    (output_dir / name)
                    .stat()
                    .st_size
                ),
            }
            for name, digest in sorted(
                hashes.items()
            )
        },
    }

    manifest_raw = (
        phase2.canonical_json_bytes(manifest)
    )
    (output_dir / MANIFEST_FILE).write_bytes(
        manifest_raw
    )
    hashes[MANIFEST_FILE] = (
        phase2.sha256_bytes(manifest_raw)
    )

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(
            hashes.items()
        )
    ).encode("utf-8")
    (output_dir / CHECKSUM_FILE).write_bytes(
        checksum_raw
    )

    return hashes


def _load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT_REQUIRED:{line_no}",
        )
        rows.append(value)
    return rows


def validate_subspace_sensitivity_artifact(
    output_dir: Path,
    family: str,
) -> dict[str, Any]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    manifest_path = (
        output_dir / MANIFEST_FILE
    )
    checksum_path = (
        output_dir / CHECKSUM_FILE
    )
    require(
        manifest_path.is_file(),
        "MANIFEST_MISSING",
    )
    require(
        checksum_path.is_file(),
        "CHECKSUM_MISSING",
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8-sig"
        )
    )
    require(
        manifest.get("schema_version")
        == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )
    files = manifest.get("files")
    require(
        isinstance(files, dict),
        "MANIFEST_FILES",
    )
    require(
        set(files)
        == {ITEM_FILE, SUMMARY_FILE},
        "MANIFEST_FILE_SET",
    )

    observed_hashes: dict[str, str] = {}
    for name in (
        ITEM_FILE,
        SUMMARY_FILE,
    ):
        path = output_dir / name
        require(
            path.is_file(),
            f"ARTIFACT_MISSING:{name}",
        )
        observed_sha = phase2.sha256_file(
            path
        )
        observed_bytes = int(
            path.stat().st_size
        )
        require(
            observed_sha
            == files[name]["sha256"],
            f"ARTIFACT_SHA256:{name}",
        )
        require(
            observed_bytes
            == int(files[name]["bytes"]),
            f"ARTIFACT_BYTES:{name}",
        )
        observed_hashes[name] = observed_sha

    observed_hashes[MANIFEST_FILE] = (
        phase2.sha256_file(manifest_path)
    )

    checksum_rows: dict[str, str] = {}
    for line in checksum_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(
            name not in checksum_rows,
            f"CHECKSUM_DUPLICATE:{name}",
        )
        checksum_rows[name] = digest

    require(
        checksum_rows
        == {
            name: digest
            for name, digest in sorted(
                observed_hashes.items()
            )
        },
        "CHECKSUM_CONTENT",
    )

    items = _load_jsonl(
        output_dir / ITEM_FILE
    )
    _validate_items(family, items)

    summary = json.loads(
        (output_dir / SUMMARY_FILE)
        .read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        summary.get("schema_version")
        == SUMMARY_SCHEMA,
        "SUMMARY_SCHEMA",
    )
    require(
        summary.get("result")
        == RESULT_PASS,
        "SUMMARY_RESULT",
    )
    require(
        summary.get("family_key")
        == family,
        "SUMMARY_FAMILY",
    )
    require(
        summary.get("source_pair_count")
        == SOURCE_PAIR_COUNT,
        "SUMMARY_PAIR_COUNT",
    )
    require(
        summary.get(
            "implementation_scope_commit"
        )
        == IMPLEMENTATION_SCOPE_COMMIT,
        "SUMMARY_SCOPE_COMMIT",
    )
    require(
        summary.get(
            "finite_difference_basis_correction_commit"
        )
        == FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT,
        "SUMMARY_CORRECTION_COMMIT",
    )
    require(
        summary.get(
            "phase1_artifact_freeze_commit"
        )
        == PHASE1_ARTIFACT_FREEZE_COMMIT,
        "SUMMARY_PHASE1_FREEZE",
    )
    require(
        float(summary["epsilon"])
        == EPSILON,
        "SUMMARY_EPSILON",
    )
    require(
        int(summary["subspace_dim"])
        == SUBSPACE_DIM,
        "SUMMARY_SUBSPACE_DIM",
    )
    require(
        summary.get(
            "baseline_model_forward_count_this_run"
        )
        == 0,
        "SUMMARY_BASELINE_FORWARD_COUNT",
    )
    require(
        summary.get(
            "scientific_model_forward_count_this_run"
        )
        == SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY,
        "SUMMARY_SCIENTIFIC_FORWARD_COUNT",
    )
    require(
        summary.get(
            "primary_inference_executed"
        )
        is False,
        "SUMMARY_PRIMARY_INFERENCE_BOUNDARY",
    )
    require(
        summary.get(
            "holm_correction_executed"
        )
        is False,
        "SUMMARY_HOLM_BOUNDARY",
    )
    require(
        summary.get(
            "training_executed"
        )
        is False
        and summary.get(
            "backward_executed"
        )
        is False
        and summary.get(
            "task_heads_executed"
        )
        is False
        and summary.get(
            "logits_read"
        )
        is False,
        "SUMMARY_EXECUTION_BOUNDARY",
    )
    require(
        summary.get(
            "scientific_conclusion"
        )
        is None,
        "SUMMARY_CONCLUSION_BOUNDARY",
    )

    basis = summary.get("basis_reconstruction")
    require(
        isinstance(basis, dict)
        and set(basis) == set(FAMILIES),
        "SUMMARY_BASIS_RECONSTRUCTION",
    )
    for basis_family in FAMILIES:
        row = basis[basis_family]
        require(
            row["phase1_plan_sha256"]
            == PHASE1_PLAN_SHA256[
                basis_family
            ],
            (
                "SUMMARY_PLAN_SHA256:"
                f"{basis_family}"
            ),
        )
        require(
            int(row["subspace_dim"])
            == SUBSPACE_DIM,
            (
                "SUMMARY_BASIS_DIM:"
                f"{basis_family}"
            ),
        )
        require(
            len(
                row["top5_eigenvalues"]
            )
            == SUBSPACE_DIM,
            (
                "SUMMARY_EIGENVALUE_COUNT:"
                f"{basis_family}"
            ),
        )
        require(
            len(
                row[
                    "top5_to_6_eigengaps"
                ]
            )
            == SUBSPACE_DIM,
            (
                "SUMMARY_EIGENGAP_COUNT:"
                f"{basis_family}"
            ),
        )
        require(
            float(
                row[
                    "minimum_selected_eigengap"
                ]
            )
            > EIGENGAP_TOL,
            (
                "SUMMARY_EIGENGAP:"
                f"{basis_family}"
            ),
        )

    return {
        "summary": summary,
        "items": items,
        "manifest": manifest,
    }


def run_family_subspace_sensitivity(
    *,
    family: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    authenticate_repo(expected_head)
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    # Both family subspaces are reconstructed and validated before
    # runtime/model setup. No scientific response can influence them.
    loaded = _load_all_phase1_and_bases()

    own_loaded = loaded[family]
    cross_family = (
        "xg4"
        if family == "xg2"
        else "xg2"
    )
    cross_loaded = loaded[cross_family]

    phase1_items = (
        own_loaded["artifact"]["items"]
    )
    own_basis = own_loaded["basis"]["basis"]
    cross_basis = (
        cross_loaded["basis"]["basis"]
    )

    phase1.base.prevalence_eq.backend.runtime_gate()

    with (
        phase1.base.prevalence_eq.backend
        .parent_runtime_rebind()
    ):
        rows, encoded, event_rows = (
            phase1.base.fresh_eq
            .load_family_inputs(
                family,
                tokenizer_snapshot,
            )
        )
        pairs = (
            phase1.base.fresh_eq
            ._pair_order(
                family,
                rows,
            )
        )
        require(
            tuple(pairs)
            == _expected_pairs(family),
            "PAIR_ORDER",
        )
        require(
            tuple(
                str(
                    row[
                        "source_pair_id"
                    ]
                )
                for row in phase1_items
            )
            == tuple(pairs),
            "PHASE1_INPUT_PAIR_ORDER",
        )

        parent = (
            phase1.base.prevalence_eq.parent
        )
        events = parent.event_lookup(
            event_rows
        )
        parent.validate_transport_event_plan(
            pairs,
            events,
        )
        row_index = parent.build_row_index(
            rows
        )
        trace_code, trace_line = (
            phase1.base.prevalence_eq
            .measurement
            ._resolve_and_validate_runtime_binding()
        )

        kernels = (
            phase1.base.prevalence_eq
            .kernel_compat
            .load_exact_fast_kernels()
        )
        with (
            phase1.base.prevalence_eq
            .kernel_compat
            .exact_transformers_kernel_loader(
                kernels
            )
        ) as constructor_kernel_calls:
            model, checkpoint_sha = (
                parent
                .load_representative_model_external(
                    model_snapshot=(
                        model_snapshot
                    ),
                    checkpoint_path=(
                        checkpoint_path
                    ),
                )
            )
            require(
                checkpoint_sha
                == (
                    phase1.base
                    .prevalence_eq
                    .extraction
                    .REPRESENTATIVE_CHECKPOINT_SHA256
                ),
                "CHECKPOINT_IDENTITY",
            )
            runtime_ctx = (
                phase1.base
                .prevalence_eq
                .transport_runtime
                .validate_runtime_components(
                    model
                )
            )

        constructor_counts = Counter(
            constructor_kernel_calls
        )
        require(
            set(constructor_counts)
            == {
                "causal-conv1d",
                "mamba-ssm",
            },
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_NAMES:"
                f"{dict(constructor_counts)}"
            ),
        )
        require(
            constructor_counts[
                "causal-conv1d"
            ]
            > 0
            and constructor_counts[
                "causal-conv1d"
            ]
            == constructor_counts[
                "mamba-ssm"
            ],
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_CALL_COUNT:"
                f"{dict(constructor_counts)}"
            ),
        )
        (
            phase1.base.prevalence_eq
            .kernel_compat
            .validate_transformers_kernel_bindings(
                kernels
            )
        )

        model.to(
            torch.device("cuda:0")
        )
        model.eval()
        require(
            all(
                parameter.device.type
                == "cuda"
                for parameter
                in model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = (
            phase1.base.prevalence_eq
            .backend
            ._make_fast_capture(
                kernels
            )
        )
        original_capture = (
            parent.capture_branch
        )
        budget = parent.ForwardBudget(
            SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY
        )
        items: list[dict[str, Any]] = []

        parent.capture_branch = (
            fast_capture
        )
        try:
            for index, pair in enumerate(
                pairs
            ):
                require(
                    phase1_items[
                        index
                    ][
                        "source_pair_id"
                    ]
                    == pair,
                    (
                        f"PAIR_IDENTITY:"
                        f"{index}"
                    ),
                )

                item = _run_subspace_pair(
                    family,
                    phase1_items[index],
                    own_basis=own_basis,
                    cross_family=(
                        cross_family
                    ),
                    cross_basis=cross_basis,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    trace_code=trace_code,
                    trace_line=trace_line,
                    encoded=encoded,
                    row_index=row_index,
                    events=events,
                    budget=budget,
                )
                items.append(item)

            budget.assert_exact()
            torch.cuda.synchronize()
        finally:
            parent.capture_branch = (
                original_capture
            )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "family_key": family,
        "execution_head": expected_head,
        "implementation_scope_commit": (
            IMPLEMENTATION_SCOPE_COMMIT
        ),
        (
            "finite_difference_"
            "basis_correction_commit"
        ): (
            FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT
        ),
        "phase1_artifact_freeze_commit": (
            PHASE1_ARTIFACT_FREEZE_COMMIT
        ),
        "phase1_artifact_path": (
            PHASE1_ARTIFACT_ROOT
            .joinpath(family)
            .as_posix()
        ),
        "source_pair_count": (
            SOURCE_PAIR_COUNT
        ),
        "pair_id_first": (
            items[0]["source_pair_id"]
        ),
        "pair_id_last": (
            items[-1]["source_pair_id"]
        ),
        "epsilon": EPSILON,
        "subspace_dim": SUBSPACE_DIM,
        "own_basis_family": family,
        "cross_basis_family": (
            cross_family
        ),
        "basis_reconstruction": (
            _basis_summary(loaded)
        ),
        (
            "basis_reconstruction_before_"
            "model_setup"
        ): True,
        (
            "response_used_for_basis_"
            "construction"
        ): False,
        (
            "baseline_model_forward_"
            "count_this_run"
        ): 0,
        (
            "scientific_model_forward_"
            "count_this_run"
        ): (
            SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY
        ),
        "model_forwards_per_pair": (
            FORWARDS_PER_PAIR
        ),
        "model_forwards_per_direction": (
            FORWARDS_PER_DIRECTION
        ),
        "subspace_sensitivity_observed": True,
        "primary_endpoint_D_observed": True,
        "primary_inference_executed": False,
        "holm_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
        "scientific_conclusion_scope": (
            "FAMILY_SUBSPACE_SENSITIVITY_"
            "OBSERVATION_ONLY"
        ),
        (
            "representative_checkpoint_"
            "sha256"
        ): checkpoint_sha,
    }

    _write_outputs(
        output_dir,
        family=family,
        items=items,
        summary=summary,
    )
    validated = (
        validate_subspace_sensitivity_artifact(
            output_dir,
            family,
        )
    )
    require(
        validated["summary"]["result"]
        == RESULT_PASS,
        "POSTWRITE_VALIDATION_RESULT",
    )
    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prospective XG2/XG4 family-specific "
            "top-5 eigenbasis sensitivity observation. "
            "Reconstructs frozen Phase-1 bases before "
            "model setup and probes own versus cross "
            "basis at epsilon=0.025. No baseline forward "
            "or inferential test is executed."
        )
    )
    parser.add_argument(
        "--family",
        choices=FAMILIES,
        required=True,
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    summary = run_family_subspace_sensitivity(
        family=args.family,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=(
            args.tokenizer_snapshot
        ),
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print(
        "RESULT =",
        summary["result"],
    )
    print(
        "FAMILY =",
        summary["family_key"],
    )
    print(
        "SOURCE_PAIR_COUNT =",
        summary["source_pair_count"],
    )
    print(
        "SUBSPACE_DIM =",
        summary["subspace_dim"],
    )
    print(
        "EPSILON =",
        summary["epsilon"],
    )
    print(
        "BASELINE_MODEL_FORWARD_COUNT_THIS_RUN =",
        summary[
            "baseline_model_forward_count_this_run"
        ],
    )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN =",
        summary[
            "scientific_model_forward_count_this_run"
        ],
    )
    print(
        "PRIMARY_INFERENCE = NOT EXECUTED"
    )


if __name__ == "__main__":
    main()
