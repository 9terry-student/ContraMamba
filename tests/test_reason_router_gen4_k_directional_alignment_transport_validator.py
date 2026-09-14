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
        validator.CORE_FREEZE
        == "3ced19dfcf011ae7b300242723eb1f10b3a3437f"
    )
    assert (
        validator.CORE_BLOB
        == "d98b2dcd3436433c04bb56ecc57dec4240abe820"
    )
    assert (
        validator.RUNTIME_BLOB
        == "989c4a8947560dcf35e9523373d09ba085a9431a"
    )


def _runner_ast():
    path = Path(
        "scripts/"
        "reason_router_gen4_k_directional_alignment_transport_runner.py"
    )
    return ast.parse(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def _function_node(tree, name):
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    ]

    assert len(matches) == 1
    return matches[0]


def _explicit_dict_string_keys(node):
    assert isinstance(node, ast.Dict)

    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant)
        and isinstance(key.value, str)
    }


def _single_return_dict_keys(
    tree,
    function_name,
):
    function = _function_node(
        tree,
        function_name,
    )

    returns = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Dict)
    ]

    assert len(returns) == 1

    return _explicit_dict_string_keys(
        returns[0]
    )


def _assigned_dict_keys(
    tree,
    function_name,
    variable_name,
):
    function = _function_node(
        tree,
        function_name,
    )

    values = []

    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue

        if not isinstance(node.value, ast.Dict):
            continue

        if any(
            isinstance(target, ast.Name)
            and target.id == variable_name
            for target in node.targets
        ):
            values.append(node.value)

    assert len(values) == 1

    return _explicit_dict_string_keys(
        values[0]
    )


def test_runner_validator_schema_contract():
    tree = _runner_ast()

    assert (
        _assigned_dict_keys(
            tree,
            "run_transport",
            "manifest",
        )
        == validator.MANIFEST_FIELDS
    )

    assert (
        _single_return_dict_keys(
            tree,
            "build_preflight_public",
        )
        == validator.PREFLIGHT_FIELDS
    )

    assert (
        _single_return_dict_keys(
            tree,
            "build_full_summary",
        )
        == validator.SUMMARY_FIELDS
    )

    reduction_fields = {
        "delta_baseline",
        "delta_alignment",
        "delta_magnitude",
        "R_ALIGN",
        "R_MAG",
        "ALIGNMENT_SPECIFICITY",
    }

    runner_item_fields = (
        _single_return_dict_keys(
            tree,
            "run_pair",
        )
        | reduction_fields
    )

    assert (
        runner_item_fields
        == validator.ITEM_FIELDS
    )


def test_private_payload_negative_safety_flags():
    validator.reject_private_payload({
        "logits_read": False,
        "raw_vectors_persisted": False,
    })

    for field in (
        "logits_read",
        "raw_vectors_persisted",
    ):
        with pytest.raises(
            validator.ValidationError
        ):
            validator.reject_private_payload({
                field: True,
            })


def test_private_payload_still_rejects_raw_vectors():
    with pytest.raises(
        validator.ValidationError
    ):
        validator.reject_private_payload({
            "raw_vector": [
                1.0,
                2.0,
            ],
        })


def _canonical_json_bytes(value):
    import json

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _synthetic_preflight_value():
    return {
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


def _synthetic_manifest(
    head,
    code_sha,
):
    return {
        "schema_version":
            validator.MANIFEST_SCHEMA,
        "mode":
            "preflight",
        "runtime_branch":
            validator.EXPECTED_BRANCH,
        "runtime_git_head":
            head,
        "implementation_parent":
            validator.IMPLEMENTATION_PARENT,
        "gen4_parent":
            validator.GEN4_PARENT,
        "k_causal_parent":
            validator.K_CAUSAL_PARENT,
        "runner_rel":
            validator.RUNNER_REL,
        "runner_sha256":
            code_sha[
                validator.RUNNER_REL
            ],
        "core_rel":
            validator.CORE_REL,
        "core_sha256":
            code_sha[
                validator.CORE_REL
            ],
        "runtime_rel":
            validator.RUNTIME_REL,
        "runtime_sha256":
            code_sha[
                validator.RUNTIME_REL
            ],
        "checkpoint_sha256":
            validator.CHECKPOINT_SHA256,
        "checkpoint_source_mode":
            "external_exact_sha256",
        "frozen_endpoint_rel":
            validator.FROZEN_ENDPOINT_REL,
        "frozen_endpoint_sha256":
            validator.FROZEN_ENDPOINT_SHA256,
        "event_manifest_rel":
            validator.EVENT_MANIFEST_REL,
        "event_manifest_sha256":
            validator.EVENT_MANIFEST_SHA256,
        "mamba_source_sha256":
            validator.MAMBA_SOURCE_SHA256,
        "source_pair_count":
            validator.PREFLIGHT_PAIR_COUNT,
        "model_forward_count":
            validator.PREFLIGHT_FORWARD_BUDGET,
        "forwards_per_pair":
            validator.FORWARDS_PER_PAIR,
        "source_block":
            validator.SOURCE_BLOCK,
        "target_residual_layer":
            validator.TARGET_RESIDUAL_LAYER,
        "intervention_layer":
            validator.INTERVENTION_LAYER,
        "relative_coordinate":
            validator.RELATIVE_COORDINATE,
        "target_pair": [
            validator.TARGET_PLUS,
            validator.TARGET_MINUS,
        ],
        "reference_pair": [
            validator.REFERENCE_PLUS,
            validator.REFERENCE_MINUS,
        ],
        "anchor_name":
            validator.ANCHOR_NAME,
        "strong_count":
            validator.STRONG_COUNT,
        "weak_count":
            validator.WEAK_COUNT,
        "equal_count":
            validator.EQUAL_COUNT,
        "strong_index_sha256":
            validator.STRONG_INDEX_SHA256,
        "training_executed":
            False,
        "backward_executed":
            False,
        "task_heads_executed":
            False,
        "logits_read":
            False,
        "raw_vectors_persisted":
            False,
        "tokenizer_invoked":
            True,
        "causal_intervention_executed":
            True,
        "statistical_testing":
            False,
    }


def test_synthetic_preflight_bundle_end_to_end(
    tmp_path,
    monkeypatch,
):
    import hashlib

    head = "synthetic-execution-head"

    code_sha = {
        validator.RUNNER_REL:
            "runner-sha",
        validator.CORE_REL:
            "core-sha",
        validator.RUNTIME_REL:
            "runtime-sha",
    }

    manifest = _synthetic_manifest(
        head,
        code_sha,
    )
    preflight = (
        _synthetic_preflight_value()
    )

    files = {
        validator.MANIFEST_FILE:
            _canonical_json_bytes(
                manifest
            ),
        validator.PREFLIGHT_FILE:
            _canonical_json_bytes(
                preflight
            ),
    }

    for name, raw in files.items():
        (tmp_path / name).write_bytes(
            raw
        )

    checksum = "".join(
        (
            hashlib.sha256(
                files[name]
            ).hexdigest()
            + "  "
            + name
            + "\n"
        )
        for name in sorted(files)
    )

    (
        tmp_path
        / validator.CHECKSUM_FILE
    ).write_text(
        checksum,
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(
        validator,
        "validate_frozen_code_identity",
        lambda execution_head: (
            code_sha
            if execution_head == head
            else None
        ),
    )

    result = (
        validator
        .validate_preflight_bundle(
            tmp_path,
            expected_execution_head=head,
        )
    )

    assert (
        result["result"]
        == "PASS_INDEPENDENT_PREFLIGHT_VALIDATION"
    )
    assert (
        result["model_forward_count"]
        == validator.PREFLIGHT_FORWARD_BUDGET
    )
    assert (
        result["scientific_conclusion"]
        == "NONE"
    )
