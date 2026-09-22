from __future__ import annotations

import inspect

import pytest
import torch

from scripts import (
    reason_router_gen4_mamba28b_readout_alignment_raw_fast_cuda
    as subject,
)


def test_protocol_is_exact() -> None:
    subject.validate_protocol()

    assert subject.PAIR_IDS[0] == "xg1_fact_6601"
    assert subject.PAIR_IDS[-1] == "xg1_fact_6900"
    assert subject.PAIR_COUNT == 300

    assert subject.TARGET_CELLS == (
        "C0_SHAM",
        "C2_NAME",
    )

    assert subject.SELECTED_PLANE == "P3"
    assert subject.CONTROL_PLANE == "P5"

    assert subject.DIM == 1003
    assert subject.ARM == "G3-GROUP-D-HALF"

    assert subject.D_EDGE_OWNERSHIP_LAMBDA == 0.5
    assert subject.FORWARD_EQUIVALENT_SCALE == 2.0

    assert subject.NATIVE_MODEL_FORWARD_COUNT == 600
    assert subject.LOCAL_BACKWARD_COUNT == 600

    assert subject.SHARDS[0]["pair_first"] == "xg1_fact_6601"
    assert subject.SHARDS[0]["pair_last"] == "xg1_fact_6750"
    assert subject.SHARDS[1]["pair_first"] == "xg1_fact_6751"
    assert subject.SHARDS[1]["pair_last"] == "xg1_fact_6900"


def test_frozen_readout_population_is_exact() -> None:
    facts, rows, manifest = (
        subject.load_readout_population()
    )

    assert len(facts) == 300
    assert len(rows) == 1800

    assert facts[0]["pair_id"] == "xg1_fact_6601"
    assert facts[-1]["pair_id"] == "xg1_fact_6900"

    assert manifest["role"] == "readout"
    assert manifest["scale"] == "mamba28b"

    assert manifest["planned_quantity"] == (
        "Delta_L_owned_with_forward_equivalent_recorded_if_applicable"
    )

    assert manifest["selection_allowed"] is False
    assert (
        manifest["response_guided_selection_allowed"]
        is False
    )
    assert (
        manifest["cohort_replacement_allowed"]
        is False
    )
    assert (
        manifest["row_filtering_allowed"]
        is False
    )
    assert manifest["rescue_policy"] == "none"

    assert (
        manifest["discovery_raw_response_access_allowed"]
        is False
    )
    assert (
        manifest["confirmation_raw_response_access_allowed"]
        is False
    )


def test_readout_full_hashes_are_exact() -> None:
    assert subject.READOUT_SOURCE_SHA256 == (
        "6bb5e24bd9ebd3fa1f308f023610ec1f21560e2493c6bc08267667edc8fcf681"
    )
    assert subject.READOUT_ROWS_SHA256 == (
        "2d9b5ecc2f0f3ead3dc308ffe918d7692caed054477d13569e321895e76f053b"
    )
    assert subject.READOUT_MANIFEST_SHA256 == (
        "747064cd4af4112aa5a2de8cc5500d2d7734253f6a3920f27a4b129d749882b3"
    )


def test_core_confirmation_is_frozen_before_readout() -> None:
    manifest = (
        subject.load_frozen_confirmation_manifest()
    )

    assert manifest["core_supported"] is True
    assert (
        manifest["selected_dominant_candidate"]
        == "P3"
    )
    assert (
        manifest["response_blind_control_plane"]
        == "P5"
    )
    assert manifest["selection_reopened"] is False
    assert manifest["readout_response_accessed"] is False
    assert manifest["rescue_performed"] is False

    assert subject.CONFIRMATION_FREEZE_COMMIT == (
        "d44de45051b7c4ec73932724f41187846fdb0917"
    )
    assert subject.CONFIRMATION_MANIFEST_SHA256 == (
        "1cfc69be4ab3edce540acee64bafb43b96089ab3d1b91d61143cb8bcd773eaac"
    )


def test_model_checkpoint_and_layer_identity() -> None:
    assert subject.geom.HF_REPO == (
        "state-spaces/mamba-2.8b-hf"
    )
    assert subject.geom.HF_REVISION == (
        "96c48e0292b63f5346b6d30061af2551f7101e26"
    )
    assert subject.geom.COMPACT_CHECKPOINT_SHA256 == (
        "c253235b8555f9d259a33061a1055e6f5da60d5616e24c1760f2487c386cce33"
    )

    assert subject.geom.SOURCE_BLOCK == 45
    assert subject.geom.TARGET_RESIDUAL_LAYER == 46
    assert subject.geom.INTERVENTION_LAYER == 47
    assert subject.geom.TARGET_OFFSET == 2
    assert subject.geom.INTERMEDIATE_SIZE == 5120


def test_shards_exactly_partition_readout_pairs() -> None:
    observed = []

    for shard in subject.SHARDS:
        observed.extend(
            subject.PAIR_IDS[
                shard["start_index"]:
                shard["end_index"]
            ]
        )

    assert tuple(observed) == subject.PAIR_IDS
    assert subject.SHARDS[0]["pair_count"] == 150
    assert subject.SHARDS[1]["pair_count"] == 150

    assert subject.SHARDS[0]["native_forward_count"] == 300
    assert subject.SHARDS[1]["native_forward_count"] == 300
    assert subject.SHARDS[0]["local_backward_count"] == 300
    assert subject.SHARDS[1]["local_backward_count"] == 300


def test_local_leaf_hook_cuts_upstream_and_preserves_values(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "DIM",
        2,
    )

    class Mixer:
        def __init__(self):
            self.in_proj = torch.nn.Identity()

    mixer = Mixer()
    capture = {}

    mask = torch.tensor([
        True,
        False,
        True,
        False,
    ])

    handle = subject._install_local_leaf_hook(
        mixer,
        token_index=1,
        intermediate_size=4,
        strong_mask=mask,
        capture=capture,
    )

    try:
        x = (
            torch.arange(24.0)
            .view(1, 3, 8)
            .requires_grad_(True)
        )

        y = mixer.in_proj(x)

        assert torch.equal(
            y.detach(),
            x.detach(),
        )

        loss = y[0, 1, :4].sum()

        grad = torch.autograd.grad(
            loss,
            capture["leaf"],
        )[0]

        assert torch.equal(
            grad,
            torch.ones(4),
        )

        assert x.grad is None

    finally:
        handle.remove()


def test_active_margin_exact() -> None:
    logits = torch.tensor(
        [[1.0, 2.0, 5.0]],
        requires_grad=True,
    )

    margin, wrong = (
        subject._active_margin(
            logits,
            2,
        )
    )

    assert wrong == 1
    assert margin.item() == 3.0


def test_plane_components_use_p3_coefficients_for_p5_control(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "DIM",
        4,
    )
    monkeypatch.setattr(
        subject.confirmation,
        "TOL",
        1e-12,
    )

    planes = {
        "P3": {
            "plus":
                torch.tensor(
                    [1.0, 0.0, 0.0, 0.0],
                    dtype=torch.float64,
                ),
            "minus":
                torch.tensor(
                    [0.0, 1.0, 0.0, 0.0],
                    dtype=torch.float64,
                ),
        },
        "P5": {
            "plus":
                torch.tensor(
                    [0.0, 0.0, 1.0, 0.0],
                    dtype=torch.float64,
                ),
            "minus":
                torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    dtype=torch.float64,
                ),
        },
    }

    h = torch.tensor(
        [3.0, 4.0, 9.0, 8.0],
        dtype=torch.float64,
    )

    out = subject._plane_components(
        h,
        planes=planes,
    )

    assert out["a"] == 3.0
    assert out["b"] == 4.0

    assert torch.equal(
        out["selected_component"],
        torch.tensor(
            [3.0, 4.0, 0.0, 0.0],
            dtype=torch.float64,
        ),
    )

    assert torch.equal(
        out["control_component"],
        torch.tensor(
            [0.0, 0.0, 3.0, 4.0],
            dtype=torch.float64,
        ),
    )


def test_owned_forward_equivalent_relation_is_exact() -> None:
    values = [
        -1.25,
        -0.0,
        0.0,
        2.75,
        1e-9,
    ]

    for owned in values:
        forward = (
            subject._owned_to_forward_equivalent(
                owned
            )
        )
        assert forward == 2.0 * owned


def test_all_model_parameters_are_frozen_for_local_leaf() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.Linear(3, 2),
    )

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(
            parameter
        )

    subject.freeze_all_parameters_for_local_leaf(
        model
    )

    assert all(
        not parameter.requires_grad
        for parameter in model.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_raw_source_contains_no_statistical_inference() -> None:
    src = " ".join(
        inspect.getsource(subject).lower().split()
    )

    assert "scipy" not in src
    assert "ttest" not in src

    assert '"p_value_count_executed": 0' in src
    assert '"inferential_test_performed": false' in src
    assert '"scientific_conclusion": none' in src
    assert (
        '"pair_level_sign_aggregation_performed": false'
        in src
    )


def test_raw_contract_has_exact_execution_counts() -> None:
    src = inspect.getsource(
        subject.write_outputs
    )

    assert '"native_model_forward_count":' in src
    assert '"local_leaf_backward_count":' in src
    assert '"parameter_gradient_count":' in src
    assert '"intervention_condition_forward_count":' in src
    assert '"p_value_count_executed":' in src

    assert subject.NATIVE_MODEL_FORWARD_COUNT == 600
    assert subject.LOCAL_BACKWARD_COUNT == 600


def test_no_discovery_or_confirmation_raw_reopening() -> None:
    src = " ".join(
        inspect.getsource(subject).lower().split()
    )

    assert "discovery_items.jsonl" not in src
    assert "confirmation_items.jsonl" not in src

    assert (
        '"discovery_raw_response_accessed": false'
        in src
    )
    assert (
        '"confirmation_raw_response_accessed": false'
        in src
    )


def test_no_stale_historical_scale_constants() -> None:
    src = inspect.getsource(
        subject
    )

    assert "xg1_fact_4801" not in src
    assert "xg1_fact_5100" not in src
    assert "xg1_fact_2701" not in src
    assert "xg1_fact_3000" not in src

    assert subject.DIM != 395
    assert subject.geom.INTERVENTION_LAYER != 17


def test_result_schema_keeps_owned_and_forward_equivalent_names() -> None:
    src = " ".join(
        inspect.getsource(
            subject.run_native_gradient_row
        ).split()
    )

    assert '"Delta_L_owned"' in src
    assert '"Delta_L_forward_equivalent"' in src
    assert (
        "delta_l_forward == 2.0 * delta_l_owned"
        in src
    )
