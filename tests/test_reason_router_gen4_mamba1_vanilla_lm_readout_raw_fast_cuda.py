from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import torch

from scripts import (
    reason_router_gen4_mamba1_vanilla_lm_readout_raw_fast_cuda
    as subject,
)


def test_protocol_constants() -> None:
    assert subject.SCALES == (
        "mamba370m",
        "mamba790m",
        "mamba14b",
        "mamba28b",
    )
    assert subject.TARGET_CELLS == (
        "C0_SHAM",
        "C2_NAME",
    )
    assert subject.COMMON_SELECTED_PLANE == "P3"
    assert subject.COMMON_CONTROL_PLANE == "P5"
    assert subject.ITEMS_PER_SCALE == 600
    assert subject.FORWARDS_PER_SCALE == 600
    assert subject.BACKWARDS_PER_SCALE == 600


def test_scale_registry() -> None:
    expected = {
        "mamba370m":
            ("P3", "P5", 650, 35),
        "mamba790m":
            ("P2", "P5", 975, 35),
        "mamba14b":
            ("P5", "P4", 829, 35),
        "mamba28b":
            ("P3", "P5", 1003, 47),
    }

    for scale, values in expected.items():
        spec = subject.scale_spec(scale)
        assert (
            spec["selected_plane"],
            spec["control_plane"],
            spec["dim"],
            spec["geom"].INTERVENTION_LAYER,
        ) == values
        assert len(spec["pair_ids"]) == 300


def test_cosine() -> None:
    assert subject.cosine(
        2.0,
        2.0,
        1.0,
    ) == 1.0

    assert subject.cosine(
        -2.0,
        2.0,
        1.0,
    ) == -1.0

    assert subject.cosine(
        0.0,
        0.0,
        1.0,
    ) is None


def test_matched_components() -> None:
    planes = {}

    for index in range(5):
        plus = torch.zeros(
            10,
            dtype=torch.float64,
        )
        minus = torch.zeros(
            10,
            dtype=torch.float64,
        )

        plus[2 * index] = 1.0
        minus[2 * index + 1] = 1.0

        planes[
            f"P{index + 1}"
        ] = {
            "plus": plus,
            "minus": minus,
        }

    native = torch.zeros(
        10,
        dtype=torch.float64,
    )
    native[0] = 3.0
    native[1] = -4.0

    result = subject.matched_components(
        native,
        selected_plane="P1",
        control_plane="P2",
        planes=planes,
        dim=10,
        tol=1e-12,
    )

    assert result["a"] == 3.0
    assert result["b"] == -4.0
    assert result["component_norm"] == 5.0
    assert (
        result[
            "norm_mismatch"
        ]
        == 0.0
    )


def test_contrast_identity() -> None:
    planes = {}

    for index in range(5):
        plus = torch.zeros(
            10,
            dtype=torch.float64,
        )
        minus = torch.zeros(
            10,
            dtype=torch.float64,
        )
        plus[2 * index] = 1.0
        minus[2 * index + 1] = 1.0

        planes[
            f"P{index + 1}"
        ] = {
            "plus": plus,
            "minus": minus,
        }

    native = torch.zeros(
        10,
        dtype=torch.float64,
    )
    native[0] = 2.0
    native[1] = 1.0

    grad = torch.zeros(
        10,
        dtype=torch.float64,
    )
    grad[0] = 1.0
    grad[2] = -0.5

    result = subject.contrast_readout(
        grad,
        native,
        selected_plane="P1",
        control_plane="P2",
        planes=planes,
        dim=10,
        tol=1e-12,
    )

    assert (
        result["Delta_L_LM"]
        == result["L_selected_LM"]
        - result["L_control_LM"]
    )


def test_plan_freeze_identity() -> None:
    assert (
        subject.PLAN_FREEZE_COMMIT
        == "1fe9a198a15c9cea0e5451d918cd949bc21bf7e0"
    )


def test_raw_runner_is_descriptive_only() -> None:
    source = inspect.getsource(
        subject
    ).lower()

    assert "scipy" not in source
    assert "ttest" not in source
    assert "p_value_count=0" in source
    assert (
        "pair_aggregation_performed=false"
        in source
    )
    assert (
        "mean_delta_l_computed=false"
        in source
    )
    assert (
        "sign_vector_computed=false"
        in source
    )


def test_no_contramamba_functional_path() -> None:
    source = inspect.getsource(
        subject
    )

    forbidden = (
        "ContraMambaV6BMinimal",
        "historical_forward(",
        "_active_margin(",
        "LABEL_ID_BY_CELL",
        "D_EDGE_OWNERSHIP_LAMBDA",
        "selected_downstream_checkpoint",
        "COMPACT_CHECKPOINT_REL",
    )

    for token in forbidden:
        assert token not in source


def test_vanilla_lm_path_is_explicit() -> None:
    source = inspect.getsource(
        subject
    )

    assert "AutoModelForCausalLM" in source
    assert '"MambaForCausalLM"' in source
    assert "torch.log_softmax" in source
    assert "next_token_id" in source
    assert "use_cache=False" in source
    assert (
        '"contra_downstream_loaded":\n'
        '            False'
    ) in source


def test_no_parameter_backward_call() -> None:
    source = inspect.getsource(
        subject
    )

    assert ".backward(" not in source
    assert "torch.autograd.grad(" in source

def test_direct_script_entrypoint_can_import_repo_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "scripts"
        / "reason_router_gen4_mamba1_vanilla_lm_readout_raw_fast_cuda.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--scale",
            "mamba370m",
            "--expected-head",
            "0" * 40,
            "--model-snapshot",
            str(root),
            "--output-dir",
            str(root / "_never_created_vanilla_lm_test_output"),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert completed.returncode != 0
    assert "No module named 'scripts'" not in completed.stdout
    assert "ModuleNotFoundError" not in completed.stdout
    assert "HEAD:" in completed.stdout

def test_exact_frozen_kernel_binding_is_mandatory() -> None:
    source = inspect.getsource(subject)

    assert "load_exact_fast_kernels()" in source
    assert "exact_transformers_kernel_loader(" in source
    assert "validate_transformers_kernel_bindings(" in source
    assert '"KERNEL_CONSTRUCTOR:"' in source
    assert '"kernel_transport_identity_status"' in source


def test_lm_head_must_be_storage_tied() -> None:
    source = inspect.getsource(subject)

    assert "lm_head.weight" in source
    assert "embeddings.weight" in source
    assert "output_weight.data_ptr()" in source
    assert "input_weight.data_ptr()" in source
    assert '"LM_HEAD_NOT_STORAGE_TIED"' in source
    assert '"lm_head_storage_tied_to_embeddings"' in source

def test_lm_head_tie_reconstruction_matches_historical_mamba() -> None:
    source = inspect.getsource(subject)

    assert (
        subject.ORIGINAL_MAMBA_LM_REFERENCE_COMMIT
        == "009bec5ee37f586844a3fc89c040a9c1a9d8badf"
    )
    assert (
        subject.ORIGINAL_MAMBA_LM_REFERENCE_PATH
        == "mamba_ssm/models/mixer_seq_simple.py"
    )
    assert (
        subject.ORIGINAL_MAMBA_LM_HEAD_CONTRACT
        == "lm_head.weight=backbone.embedding.weight"
    )

    required = (
        "HF_TIE_METADATA_UNEXPECTED",
        "HF_RUNTIME_TIE_STATE_UNEXPECTED",
        "CHECKPOINT_EMBEDDING_WEIGHT_MISSING",
        "CHECKPOINT_LM_HEAD_WEIGHT_UNEXPECTED",
        "model.lm_head.weight = (",
        "model.backbone.embeddings.weight",
        "LM_HEAD_PARAMETER_NOT_IDENTICAL",
        "LM_HEAD_ALIAS_DRIFT_AFTER_FREEZE",
        '"historical_state_spaces_mamba_direct_parameter_alias"',
        '"checkpoint_lm_head_weight_present"',
        '"lm_head_tie_reference_commit"',
    )

    for token in required:
        assert token in source

    assert "model.tie_weights()" not in source
    assert "LM_TIE_WORD_EMBEDDINGS_DISABLED" not in source

def test_checkpoint_head_gate_supports_safetensor_shards() -> None:
    source = inspect.getsource(subject)

    required = (
        'if name.endswith(".safetensors")',
        "CHECKPOINT_WEIGHT_FILES_MISSING",
        "CHECKPOINT_WEIGHT_FILE_MISSING:",
        "CHECKPOINT_DUPLICATE_KEYS:",
        "checkpoint_key_set.update(",
        '"checkpoint_weight_files"',
        '"checkpoint_weight_file_count"',
        "CHECKPOINT_EMBEDDING_WEIGHT_MISSING",
        "CHECKPOINT_LM_HEAD_WEIGHT_UNEXPECTED",
    )

    for token in required:
        assert token in source

    assert 'snapshot / "model.safetensors"' not in source
