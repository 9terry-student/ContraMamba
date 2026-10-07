from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "scripts"
    / "run_native_state_kinematics_vitaminc_prediction_census.py"
)

spec = importlib.util.spec_from_file_location(
    "q1census",
    SCRIPT,
)

assert spec is not None
assert spec.loader is not None

q1 = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    q1
)


def source(
    idx: int,
    case_id: str,
    label: str,
):
    return {
        "raw_idx":
            str(idx),

        "unique_id":
            f"u{idx}",

        "case_id":
            case_id,

        "label":
            label,

        "claim":
            "claim",

        "evidence":
            "evidence",
    }


def encoded():
    return {
        "input_token_length":
            20,

        "claim_token_length":
            7,

        "evidence_token_length":
            12,

        "claim_truncated":
            False,

        "evidence_truncated":
            False,
    }


def test_vitaminc_label_canonicalization():
    assert (
        q1.canonicalize_vitaminc_label(
            "SUPPORTS"
        )
        == "SUPPORT"
    )

    assert (
        q1.canonicalize_vitaminc_label(
            "REFUTES"
        )
        == "REFUTE"
    )

    assert (
        q1.canonicalize_vitaminc_label(
            "NOT ENOUGH INFO"
        )
        == "NOT_ENTITLED"
    )


def test_unmapped_label_fails_closed():
    with pytest.raises(
        q1.CensusError
    ):
        q1.canonicalize_vitaminc_label(
            "MAYBE"
        )


def test_external_flags_are_all_zero():
    temporal, predicate = (
        q1.external_flags(
            4
        )
    )

    assert temporal == [
        0,
        0,
        0,
        0,
    ]

    assert predicate == [
        0,
        0,
        0,
        0,
    ]


def test_checkpoint_metadata_exact_a0_replacement_r1():
    metadata = {
        "architecture":
            "v6b_minimal",

        "backbone":
            "mamba",

        "model_name":
            "state-spaces/mamba-130m-hf",

        "training_seed":
            180,

        "resolved_split_seed":
            8192,

        "reason_router_arm":
            "A0",

        "reason_router_mode":
            "explicit_product",

        "gradient_ownership_mode":
            "joint",

        "max_length":
            128,

        "training_args": {
            "freeze_encoder":
                True,
        },
    }

    observed = (
        q1.validate_checkpoint_metadata(
            metadata
        )
    )

    assert (
        observed[
            "split_seed"
        ]
        == 8192
    )


def test_checkpoint_metadata_nested_training_args():
    metadata = {
        "architecture":
            "v6b_minimal",

        "backbone":
            "mamba",

        "model_name":
            "state-spaces/mamba-130m-hf",

        "reason_router_arm":
            "A0",

        "reason_router_composer":
            "explicit_product",

        "gradient_ownership_mode":
            "joint",

        "training_args": {
            "seed":
                180,

            "split_seed":
                8192,

            "max_length":
                128,

            "freeze_encoder":
                True,
        },
    }

    q1.validate_checkpoint_metadata(
        metadata
    )


def test_wrong_split_seed_fails_closed():
    metadata = {
        "architecture":
            "v6b_minimal",

        "backbone":
            "mamba",

        "model_name":
            "state-spaces/mamba-130m-hf",

        "training_seed":
            180,

        "resolved_split_seed":
            174,

        "reason_router_arm":
            "A0",

        "reason_router_mode":
            "explicit_product",

        "gradient_ownership_mode":
            "joint",

        "max_length":
            128,

        "training_args": {
            "freeze_encoder":
                True,
        },
    }

    with pytest.raises(
        q1.CensusError,
        match="split_seed",
    ):
        q1.validate_checkpoint_metadata(
            metadata
        )


def test_confidence_and_decisive_wrong_definition():
    row = q1.make_prediction_row(
        source(
            0,
            "case0",
            "REFUTES",
        ),
        encoded(),
        [
            0.0,
            0.0,
            2.0,
        ],
        [
            0.1,
            0.1,
            0.8,
        ],
    )

    assert (
        row["pred_label"]
        == "SUPPORT"
    )

    assert (
        row["confidence"]
        == 0.8
    )

    assert (
        row[
            "decisive_prediction"
        ]
        is True
    )

    assert (
        row[
            "confident_decisive_wrong"
        ]
        is True
    )


def test_not_entitled_is_not_decisive():
    row = q1.make_prediction_row(
        source(
            0,
            "case0",
            "NOT ENOUGH INFO",
        ),
        encoded(),
        [
            0.1,
            1.8,
            0.1,
        ],
        [
            0.1,
            0.8,
            0.1,
        ],
    )

    assert (
        row["pred_label"]
        == "NOT_ENTITLED"
    )

    assert (
        row[
            "decisive_prediction"
        ]
        is False
    )

    assert (
        row[
            "confident_decisive_correct"
        ]
        is False
    )


def test_serialized_row_has_no_state_outputs():
    row = q1.make_prediction_row(
        source(
            0,
            "case0",
            "SUPPORTS",
        ),
        encoded(),
        [
            0.1,
            0.1,
            2.0,
        ],
        [
            0.1,
            0.1,
            0.8,
        ],
    )

    keys = " ".join(
        row.keys()
    ).lower()

    assert "hidden" not in keys
    assert "state" not in keys
    assert "trajectory" not in keys


def test_static_contract_locks_authority():
    contract = (
        q1.static_contract()
    )

    assert (
        contract[
            "authority_commit"
        ]
        ==
        "7b12527289924c9979fe19231bb8e7e756c0234e"
    )

    assert (
        contract[
            "confidence_threshold"
        ]
        == 0.5
    )

    assert (
        contract[
            "split_seed"
        ]
        == 8192
    )

    assert (
        contract[
            "native_state_access_allowed"
        ]
        is False
    )

    assert (
        contract[
            "training_allowed"
        ]
        is False
    )

    assert (
        contract[
            "auxiliary_gold_labels_forwarded"
        ]
        is False
    )


def test_static_contract_locks_exact_kernel_runtime():
    contract = (
        q1.static_contract()
    )

    assert (
        contract[
            "kernel_runtime_contract"
        ]
        ==
        {
            "kernels_version":
                "0.10.2",

            "build_variant":
                "torch210-cxx11-cu128-x86_64-linux",

            "transport_identity_status":
                "EXACT_FROZEN_BINARY_SHA256_MATCH",

            "mamba_binary_sha256":
                "dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587",

            "causal_conv_binary_sha256":
                "6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6",
        }
    )


def test_kernel_constructor_counts_require_exact_layer_binding():
    observed = (
        q1.validate_kernel_constructor_calls(
            [
                "causal-conv1d",
                "mamba-ssm",
            ]
            * 24,
            24,
        )
    )

    assert observed == {
        "causal-conv1d":
            24,

        "mamba-ssm":
            24,
    }


@pytest.mark.parametrize(
    "calls",
    [
        ["causal-conv1d"] * 24,
        ["mamba-ssm"] * 24,
        (
            ["causal-conv1d"] * 23
            + ["mamba-ssm"] * 24
        ),
        (
            ["causal-conv1d"] * 24
            + ["mamba-ssm"] * 24
            + ["unexpected-kernel"]
        ),
    ],
)
def test_kernel_constructor_counts_fail_closed(
    calls,
):
    with pytest.raises(
        q1.CensusError,
        match="KERNEL_CONSTRUCTOR_COUNTS",
    ):
        q1.validate_kernel_constructor_calls(
            calls,
            24,
        )


def test_source_binds_exact_kernel_runtime_and_provenance():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    assert (
        "load_exact_fast_kernels"
        in text
    )

    assert (
        "exact_transformers_kernel_loader"
        in text
    )

    assert (
        "validate_transformers_kernel_bindings"
        in text
    )

    assert (
        '"kernel_runtime":'
        in text
    )

    assert (
        "NATIVE_Q1_VITAMINC_MAMBA_PREDICTION_CENSUS_PROVENANCE_V2"
        in text
    )


def test_source_has_no_training_or_native_state_request_paths():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    forbidden = (
        "output_hidden_states=True",
        "return_token_diagnostics=True",
        ".backward(",
        "torch.optim",
        "optimizer.step(",
        "model.train(",
    )

    for token in forbidden:
        assert token not in text


def test_capacity_excludes_correct_controls_from_wrong_cases(
    monkeypatch,
):
    monkeypatch.setattr(
        q1,
        "EXPECTED_ROWS",
        4,
    )

    monkeypatch.setattr(
        q1,
        "EXPECTED_CASE_IDS",
        3,
    )

    rows = [
        q1.make_prediction_row(
            source(
                0,
                "A",
                "REFUTES",
            ),
            encoded(),
            [
                0,
                0,
                2,
            ],
            [
                0.1,
                0.1,
                0.8,
            ],
        ),

        q1.make_prediction_row(
            source(
                1,
                "A",
                "SUPPORTS",
            ),
            encoded(),
            [
                0,
                0,
                2,
            ],
            [
                0.1,
                0.1,
                0.8,
            ],
        ),

        q1.make_prediction_row(
            source(
                2,
                "B",
                "SUPPORTS",
            ),
            encoded(),
            [
                0,
                0,
                2,
            ],
            [
                0.1,
                0.1,
                0.8,
            ],
        ),

        q1.make_prediction_row(
            source(
                3,
                "C",
                "REFUTES",
            ),
            encoded(),
            [
                2,
                0,
                0,
            ],
            [
                0.8,
                0.1,
                0.1,
            ],
        ),
    ]

    summary = q1.build_summary(
        rows
    )

    assert (
        summary[
            "wrong_case_id_set"
        ]
        == ["A"]
    )

    assert (
        "A"
        not in summary[
            "correct_control_case_id_set_excluding_all_wrong_cases"
        ]
    )

    assert (
        set(
            summary[
                "correct_control_case_id_set_excluding_all_wrong_cases"
            ]
        )
        == {
            "B",
            "C",
        }
    )

    assert (
        summary[
            "total_case_level_pair_capacity"
        ]
        == 1
    )

    assert (
        summary[
            "total_case_level_pair_capacity_upper_bound_without_predicted_class_constraint"
        ]
        == 1
    )


def test_total_capacity_requires_predicted_class_compatibility(
    monkeypatch,
):
    monkeypatch.setattr(
        q1,
        "EXPECTED_ROWS",
        2,
    )

    monkeypatch.setattr(
        q1,
        "EXPECTED_CASE_IDS",
        2,
    )

    rows = [
        q1.make_prediction_row(
            source(
                0,
                "wrong_case",
                "REFUTES",
            ),
            encoded(),
            [0.0, 0.0, 2.0],
            [0.1, 0.1, 0.8],
        ),
        q1.make_prediction_row(
            source(
                1,
                "control_case",
                "REFUTES",
            ),
            encoded(),
            [2.0, 0.0, 0.0],
            [0.8, 0.1, 0.1],
        ),
    ]

    summary = q1.build_summary(rows)

    assert (
        summary[
            "total_case_level_pair_capacity_upper_bound_without_predicted_class_constraint"
        ]
        == 1
    )

    # Wrong predicts SUPPORT while the only correct control predicts REFUTE.
    # Final predicted class is a frozen matching control, so capacity is zero.
    assert (
        summary[
            "total_case_level_pair_capacity"
        ]
        == 0
    )

    assert (
        summary[
            "pair_capacity_respects_predicted_class"
        ]
        is True
    )


def test_maximum_capacity_enforces_case_uniqueness_with_reassignment():
    wrong_rows = [
        {
            "case_id": "W1",
            "pred_label": "REFUTE",
        },
        {
            "case_id": "W1",
            "pred_label": "SUPPORT",
        },
        {
            "case_id": "W2",
            "pred_label": "SUPPORT",
        },
    ]

    control_rows = [
        {
            "case_id": "C1",
            "pred_label": "REFUTE",
        },
        {
            "case_id": "C2",
            "pred_label": "SUPPORT",
        },
    ]

    assert (
        q1.maximum_case_level_pair_capacity(
            wrong_rows,
            control_rows,
        )
        == 2
    )


def test_parser_static_mode_needs_no_checkpoint():
    args = (
        q1.build_parser()
        .parse_args(
            [
                "--static-contract-verify",
            ]
        )
    )

    assert (
        args.static_contract_verify
        is True
    )

    assert (
        args.checkpoint
        is None
    )