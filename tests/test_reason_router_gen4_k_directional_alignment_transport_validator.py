import ast
from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_k_directional_alignment_transport_validator
    as validator,
)


def test_validator_does_not_import_runner():
    path = Path(
        r"scripts\reason_router_gen4_k_directional_alignment_transport_validator.py"
    )
    tree = ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    imported = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name
                for alias in node.names
            )
        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            module = node.module or ""
            imported.add(module)
            imported.update(
                f"{module}.{alias.name}"
                for alias in node.names
            )

    assert not any(
        "reason_router_gen4_k_directional_alignment_transport_runner"
        in name
        for name in imported
    )


def test_holm_two():
    first, second = validator.holm_two(
        0.01,
        0.04,
    )

    assert first == pytest.approx(
        0.02
    )
    assert second == pytest.approx(
        0.04
    )


def test_student_t_tail_direction():
    positive = (
        validator.student_t_two_sided_p(
            2.0,
            20,
        )
    )

    assert 0.0 < positive < 1.0

    # Two-sided probability must be
    # symmetric in the t sign.
    negative = (
        validator.student_t_two_sided_p(
            -2.0,
            20,
        )
    )

    assert positive == pytest.approx(
        negative
    )


def test_preflight_public_complete_gate():
    value = {
        "schema_version":
            validator.PREFLIGHT_SCHEMA,
        "pair_count":
            validator.PREFLIGHT_PAIR_COUNT,
        "model_forward_count":
            validator.PREFLIGHT_FORWARD_BUDGET,
        "max_baseline_reproduction_abs_residual":
            1e-14,
        "max_alignment_cosine_abs_residual":
            1e-13,
        "max_magnitude_cosine_abs_residual":
            1e-13,
        "max_alignment_A_preservation_abs_residual":
            1e-13,
        "max_alignment_B_preservation_abs_residual":
            1e-13,
        "max_magnitude_A_target_abs_residual":
            1e-13,
        "max_magnitude_B_target_abs_residual":
            1e-13,
        "max_alignment_midpoint_abs_residual":
            1e-7,
        "max_magnitude_midpoint_abs_residual":
            1e-7,
        "max_alignment_pair_delta_abs_residual":
            1e-7,
        "max_magnitude_pair_delta_abs_residual":
            1e-7,
        "max_alignment_applied_correction_abs_residual":
            1e-7,
        "max_magnitude_applied_correction_abs_residual":
            1e-7,
        "scientific_endpoint_values_serialized":
            False,
        "inferential_statistics_executed":
            False,
        "result":
            "PASS_BOUNDED_PREFLIGHT",
    }

    validator.validate_preflight_public(
        value
    )


def test_preflight_rejects_scientific_inference():
    value = {
        field: 0.0
        for field in validator.PREFLIGHT_FIELDS
    }

    value.update({
        "schema_version":
            validator.PREFLIGHT_SCHEMA,
        "pair_count":
            validator.PREFLIGHT_PAIR_COUNT,
        "model_forward_count":
            validator.PREFLIGHT_FORWARD_BUDGET,
        "scientific_endpoint_values_serialized":
            False,
        "inferential_statistics_executed":
            True,
        "result":
            "PASS_BOUNDED_PREFLIGHT",
    })

    with pytest.raises(
        validator.ValidationError
    ):
        validator.validate_preflight_public(
            value
        )


def test_frozen_code_constants():
    assert (
        validator.RUNNER_FREEZE
        == "2bfcd7b4243832f38e32abf389229d7601b180b2"
    )
    assert (
        validator.RUNNER_BLOB
        == "3677dd83950789e41417c3a1ffaf70b82d7003ad"
    )
    assert (
        validator.CORE_BLOB
        == "4a69339d159338c156c0c0e00fa1b830a63bb997"
    )
    assert (
        validator.RUNTIME_BLOB
        == "989c4a8947560dcf35e9523373d09ba085a9431a"
    )
