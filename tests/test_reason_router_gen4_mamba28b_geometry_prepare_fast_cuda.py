from __future__ import annotations

import inspect
import math

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba28b_geometry_prepare_fast_cuda
    as subject,
)


def test_prospective_layer_mapping_and_forward_budget():
    subject.validate_protocol_constants()

    assert subject.INTERVENTION_LAYER == 47
    assert (
        subject.SOURCE_BLOCK,
        subject.TARGET_RESIDUAL_LAYER,
        subject.INTERVENTION_LAYER,
    ) == (45, 46, 47)

    assert subject.FAMILIES == ("xg2", "xg4")
    assert subject.SOURCE_PAIR_COUNT == 300
    assert subject.FORWARDS_PER_PAIR == 4
    assert subject.FORWARDS_PER_FAMILY == 1200
    assert subject.TOTAL_FORWARD_BUDGET == 2400


def test_checkout_identity_accepts_named_branch_or_detached_exact_head():
    head = "a" * 40

    subject.validate_checkout_identity(
        branch=subject.EXPECTED_BRANCH,
        head=head,
        status="",
        expected_head=head,
    )
    subject.validate_checkout_identity(
        branch="",
        head=head,
        status="",
        expected_head=head,
    )

    with pytest.raises(
        subject.GeometryPreparationError,
        match="BRANCH_MISMATCH",
    ):
        subject.validate_checkout_identity(
            branch="unexpected-branch",
            head=head,
            status="",
            expected_head=head,
        )

    with pytest.raises(
        subject.GeometryPreparationError,
        match="HEAD_MISMATCH",
    ):
        subject.validate_checkout_identity(
            branch="",
            head="b" * 40,
            status="",
            expected_head=head,
        )

    with pytest.raises(
        subject.GeometryPreparationError,
        match="WORKTREE_NOT_CLEAN",
    ):
        subject.validate_checkout_identity(
            branch="",
            head=head,
            status="?? unexpected.txt",
            expected_head=head,
        )


def test_strong_partition_is_fail_closed_and_deterministic():
    weight = torch.zeros(
        subject.INTERMEDIATE_SIZE,
        1,
        subject.CONV_KERNEL,
        dtype=torch.float64,
    )
    weight[:, 0, subject.LAG0_KERNEL_INDEX] = torch.arange(
        1,
        subject.INTERMEDIATE_SIZE + 1,
        dtype=torch.float64,
    )

    first = subject.strong_partition(weight)
    second = subject.strong_partition(weight.clone())

    assert 0 < first["strong_count"] < subject.INTERMEDIATE_SIZE
    assert first["strong_indices"] == second["strong_indices"]
    assert (
        first["strong_index_sha256"]
        == second["strong_index_sha256"]
    )

    reconstructed = torch.zeros(
        subject.INTERMEDIATE_SIZE,
        dtype=torch.bool,
    )
    reconstructed[first["strong_indices"]] = True
    assert (
        subject.strong_index_sha256(
            torch.nonzero(
                reconstructed,
                as_tuple=False,
            ).flatten().tolist()
        )
        == first["strong_index_sha256"]
    )

    with pytest.raises(
        subject.GeometryPreparationError,
        match="ZERO_SELECTED_CHANNELS",
    ):
        subject.strong_partition(
            torch.ones(
                subject.INTERMEDIATE_SIZE,
                1,
                subject.CONV_KERNEL,
            )
        )

    bad = weight.clone()
    bad[3, 0, subject.LAG0_KERNEL_INDEX] = float("nan")
    with pytest.raises(
        subject.GeometryPreparationError,
        match="NONFINITE_KERNEL_STATISTIC",
    ):
        subject.strong_partition(bad)

    with pytest.raises(
        subject.GeometryPreparationError,
        match="CONV_WEIGHT_SHAPE",
    ):
        subject.strong_partition(
            torch.zeros(
                subject.INTERMEDIATE_SIZE - 1,
                1,
                subject.CONV_KERNEL,
            )
        )


def _synthetic_plans(width: int, offset: int) -> torch.Tensor:
    rows = []
    for index in range(subject.SOURCE_PAIR_COUNT):
        vector = torch.zeros(width, dtype=torch.float64)
        vector[(index + offset) % 6] = 4.0
        vector[(2 * index + offset + 1) % 6] += 2.0
        vector[(3 * index + offset + 2) % 6] += 1.0
        vector[6] += 0.20 + 0.001 * index
        vector[7] += 0.11 + 0.0007 * index
        vector[8] += 0.05 + 0.0003 * index
        rows.append(vector)
    return torch.stack(rows, dim=0)


def test_family_basis_and_projector_contrast_geometry():
    width = 12

    # Basis reconstruction is tested independently from principal-plane
    # geometry so the projector test has prospectively nondegenerate angles.
    xg2_plans = _synthetic_plans(width, 0)
    xg4_plans = _synthetic_plans(width, 1)
    b2_info = subject.reconstruct_family_basis(
        "xg2",
        xg2_plans,
    )
    b4_info = subject.reconstruct_family_basis(
        "xg4",
        xg4_plans,
    )
    assert b2_info["basis"].shape == (width, subject.K)
    assert b4_info["basis"].shape == (width, subject.K)
    assert (
        b2_info["minimum_selected_eigengap"]
        > subject.BASIS_EIGENGAP_TOL
    )
    assert (
        b4_info["minimum_selected_eigengap"]
        > subject.BASIS_EIGENGAP_TOL
    )

    # Explicitly construct two 5D subspaces with five distinct nonzero
    # principal angles. This avoids accidental shared directions in a unit
    # test while preserving exact projector-contrast geometry.
    b2 = torch.zeros(width, subject.K, dtype=torch.float64)
    b4 = torch.zeros(width, subject.K, dtype=torch.float64)
    angles = (0.10, 0.20, 0.30, 0.40, 0.50)

    for index, theta in enumerate(angles):
        b2[index, index] = 1.0
        b4[index, index] = math.cos(theta)
        b4[subject.K + index, index] = math.sin(theta)

    geometry = subject.build_projector_geometry(b2, b4)

    assert geometry["width"] == width
    assert len(geometry["planes"]) == subject.K
    assert len(geometry["lambda_plus_by_plane"]) == subject.K
    assert (
        geometry[
            "full_principal_10_vector_gram_max_abs_residual"
        ]
        <= 5e-10
    )

    contrast = geometry["projector_contrast"]

    for plane_number in range(1, subject.K + 1):
        plane = geometry["planes"][f"P{plane_number}"]
        assert plane["lambda_plus"] > 0.0
        assert plane["lambda_minus"] < 0.0

        plus = plane["plus"]
        minus = plane["minus"]
        lam = float(plane["lambda_plus"])

        assert torch.linalg.vector_norm(
            contrast @ plus - lam * plus
        ).item() <= subject.PLANE_TOL
        assert torch.linalg.vector_norm(
            contrast @ minus + lam * minus
        ).item() <= subject.PLANE_TOL


def test_alignment_delta_preserves_norm_and_hits_target_cosine():
    x = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    y = torch.tensor(
        [0.4, math.sqrt(1.0 - 0.4**2), 0.0, 0.0],
        dtype=torch.float64,
    )

    delta, audit = subject.alignment_delta(
        x,
        y,
        0.75,
    )
    y2 = y + delta

    assert abs(
        torch.linalg.vector_norm(y2).item()
        - torch.linalg.vector_norm(y).item()
    ) <= 5e-11
    assert abs(subject.cosine(x, y2) - 0.75) <= 5e-11
    assert abs(audit["realized_C"] - 0.75) <= 5e-11


def test_pinned_cross_backbone_identity_constants():
    assert subject.EXPECTED_BRANCH == "gen4-mamba1-five-scale-ladder-extension"
    assert subject.HF_REPO == "state-spaces/mamba-2.8b-hf"
    assert (
        subject.HF_REVISION
        == "96c48e0292b63f5346b6d30061af2551f7101e26"
    )
    assert (
        subject.COMPACT_CHECKPOINT_SHA256
        == "c253235b8555f9d259a33061a1055e6f5da60d5616e24c1760f2487c386cce33"
    )
    assert (
        subject.SOURCE_SELECTED_CHECKPOINT_SHA256
        == "5f470a799522f2bc17a3f52293028f20688f8ee56d90943b9c7d20a76f4faf52"
    )
    assert (
        subject.PRETRAINED_BACKBONE_STATE_STREAM_SHA256
        == "d79e785f2c03bcf3a3bd94d3947156519926687f4bb431b996af7d0083c1204f"
    )
    assert (
        subject.DOWNSTREAM_STATE_CANONICAL_SHA256
        == "d4fa0cb88af9a01fe13ca05829b5b87e73ac52b7bf761dc3971f813d98650782"
    )
    assert (
        subject.TRAINING_EXECUTION_COMMIT
        == "aaafcf16a8efa204e90517e8e0b7e864196a2bd4"
    )
    assert (
        subject.TRAINING_ARTIFACT_FREEZE_COMMIT
        == "ee1e066ef330297d2d0e0c996c57c47d9ee99bd7"
    )

    assert subject.TRAINING_SEED == 181
    assert subject.SPLIT_SEED == 8192
    assert subject.SELECTED_EPOCH == 20
    assert subject.ARM == "G3-GROUP-D-HALF"

    assert subject.HIDDEN_SIZE == 2560
    assert subject.INTERMEDIATE_SIZE == 5120
    assert subject.STATE_SIZE == 16
    assert subject.LAYER_COUNT == 64

    assert subject.TOKENIZER_FILE_SHA256 == {
        "tokenizer.json":
            "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
        "tokenizer_config.json":
            "027cbbe0813aefd46e0037bf828acf22b9b591ce845153ddbf409e99770a3a79",
    }

    assert set(subject.HF_FILES) == {
        "config.json",
        "generation_config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }



def test_geometry_stage_has_no_xg1_response_gate():
    assert subject.RESULT_PASS == "PASS_MAMBA28B_GEOMETRY_PREPARATION"
    assert subject.TARGET_PLUS_CELL == "C2_NAME"
    assert subject.TARGET_MINUS_CELL == "C0_SHAM"
    assert subject.REFERENCE_PLUS_CELL == "C5_TITLE_NAME"
    assert subject.REFERENCE_MINUS_CELL == "C1_TITLE"


def test_no_stale_backbone_identity_in_28b_geometry_source():
    source = inspect.getsource(subject)

    assert "1.4B" not in source
    assert "1.4b" not in source
    assert "mamba14b" not in source.lower()
    assert "790M" not in source
    assert "mamba790m" not in source.lower()

    assert (
        "Independent Mamba-2.8B geometry reconstruction only; "
        in source
    )
    assert "contramamba_mamba28b_compact_checkpoint_v2" in source
    assert "source_selected_checkpoint" in source
    assert "SOURCE_FULL_CHECKPOINT_SHA256" not in source
    assert "FULL_STATE_CANONICAL_SHA256" not in source

