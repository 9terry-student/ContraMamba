from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from contramamba.gen5_phase2_state_update_ownership import (
    CorrectionShape,
    parent_parameter_fingerprint,
    reference_correction_mixer_contribution,
)
from contramamba.gen5_stage_e_causal_plane_bottleneck import (
    ARMS,
    EXPECTED_TRAINABLE_NUMEL,
    FixedPlaneStateWriteCorrection,
    StageELayer22MixerWrapper,
    accelerated_fixed_plane_mixer_contribution,
    fixed_plane_geometry,
    install_stage_e_layer22_wrapper,
    stage_e_active_mask,
    stage_e_parameter_audit,
    validate_arm,
)
from scripts import (
    train_reason_router_gen5_stage_e_causal_plane_bottleneck
    as train,
)


TEST_SHAPE = CorrectionShape(
    hidden_size=4,
    intermediate_size=3,
    state_size=2,
    rank=2,
)


def orthogonal_bases(state_width: int):
    eye = torch.eye(state_width, dtype=torch.float64)
    return eye[:, :2].clone(), eye[:, 2:4].clone()


def make_correction(arm: str, seed: int = 6201):
    r22, c22 = orthogonal_bases(TEST_SHAPE.state_width)
    return FixedPlaneStateWriteCorrection(
        arm=arm,
        r22=r22,
        c22=c22,
        seed=seed,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )


class TinyMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = TEST_SHAPE.hidden_size
        self.intermediate_size = TEST_SHAPE.intermediate_size
        self.ssm_state_size = TEST_SHAPE.state_size
        self.time_step_rank = 2
        self.layer_idx = 22

        self.in_proj = nn.Linear(4, 6, bias=False)
        self.conv1d = nn.Conv1d(
            3,
            3,
            kernel_size=2,
            padding=1,
            groups=3,
            bias=True,
        )
        self.x_proj = nn.Linear(3, 2 + 2 * 2, bias=False)
        self.dt_proj = nn.Linear(2, 3, bias=True)
        self.A_log = nn.Parameter(torch.zeros(3, 2))
        self.D = nn.Parameter(torch.ones(3))
        self.out_proj = nn.Linear(3, 4, bias=True)
        self.act = F.silu

    def forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        del cache_params, cache_position, attention_mask
        x = self.in_proj(hidden_states)[..., :3]
        return self.out_proj(torch.tanh(x))


class TinyBlock(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer


class TinyMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyBlock(nn.Identity()) for _ in range(23)]
        )
        self.layers[22] = TinyBlock(TinyMixer())


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = TinyMamba()


def base_args(**updates):
    values = {
        "static_verify_only": True,
        "cuda_preflight_only": False,
        "run_cell": False,
        "run_matrix": False,
        "expected_head": train.IMPLEMENTATION_AUTHORITY_COMMIT,
        "allow_opening_worktree": True,
        "implementation_freeze_commit": None,
        "execution_authority_commit": None,
        "seed": None,
        "arm": None,
        "tokenizer_snapshot": None,
        "model_snapshot": None,
        "checkpoint": None,
        "preflight_output": None,
        "output_root": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_arm_validation():
    assert ARMS == ("E-R22", "E-C22")
    for arm in ARMS:
        assert validate_arm(arm) == arm
    with pytest.raises(Exception):
        validate_arm("E-FREE")


def test_matched_seed_initialization_and_zero_m():
    r = make_correction("E-R22", seed=6201)
    c = make_correction("E-C22", seed=6201)

    assert torch.equal(r.A_theta.weight, c.A_theta.weight)
    assert torch.equal(r.M_theta.weight, c.M_theta.weight)
    assert torch.count_nonzero(r.M_theta.weight) == 0
    assert torch.count_nonzero(c.M_theta.weight) == 0

    x = torch.randn(2, 5, 4)
    assert torch.count_nonzero(r(x)) == 0
    assert torch.count_nonzero(c(x)) == 0


def test_different_seed_changes_a():
    a = make_correction("E-R22", seed=6201)
    b = make_correction("E-R22", seed=6202)
    assert not torch.equal(a.A_theta.weight, b.A_theta.weight)


def test_fixed_plane_output_and_cross_plane_leakage():
    torch.manual_seed(7)
    x = torch.randn(3, 4, TEST_SHAPE.hidden_size)

    for arm in ARMS:
        module = make_correction(arm)
        with torch.no_grad():
            module.M_theta.weight.copy_(
                torch.tensor(
                    [[1.2, -0.3], [0.4, 0.7]],
                    dtype=module.M_theta.weight.dtype,
                )
            )

        out = module(x)
        q = module.selected_basis().to(out.dtype)
        other = module.other_basis().to(out.dtype)

        flat = out.reshape(-1, TEST_SHAPE.state_width)
        residual = flat - (flat @ q) @ q.T

        assert torch.max(torch.abs(residual)).item() < 1e-6
        assert torch.max(torch.abs(flat @ other)).item() < 1e-6

        geometry = fixed_plane_geometry(module)
        assert geometry["fixed_plane_residual_max_abs"] < 1e-12
        assert geometry["cross_plane_max_abs"] < 1e-12
        assert geometry["effective_output_rank"] == 2


def test_active_mask_zeroes_new_writes():
    module = make_correction("E-R22")
    with torch.no_grad():
        module.M_theta.weight.copy_(torch.eye(2))

    x = torch.randn(2, 4, 4)
    mask = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0]],
        dtype=torch.bool,
    )
    out = module(x, attention_mask=mask)

    assert torch.count_nonzero(out[~mask]) == 0
    assert torch.linalg.vector_norm(out[mask]).item() > 0


def test_bases_are_buffers_not_parameters():
    module = make_correction("E-R22")
    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    assert set(parameters) == {
        "A_theta.weight",
        "M_theta.weight",
    }
    assert "R22" in buffers
    assert "C22" in buffers


def test_trainable_numel_contract():
    assert EXPECTED_TRAINABLE_NUMEL == 1540
    assert 2 * 768 + 2 * 2 == EXPECTED_TRAINABLE_NUMEL


def test_reference_and_accelerated_forward_and_gradients_match():
    torch.manual_seed(11)
    mixer = TinyMixer()
    for parameter in mixer.parameters():
        parameter.requires_grad_(False)

    ref = make_correction("E-R22")
    acc = make_correction("E-R22")

    with torch.no_grad():
        value = torch.tensor(
            [[0.08, -0.03], [0.02, 0.05]],
            dtype=ref.M_theta.weight.dtype,
        )
        ref.M_theta.weight.copy_(value)
        acc.M_theta.weight.copy_(value)
        acc.A_theta.weight.copy_(ref.A_theta.weight)

    x = torch.randn(2, 5, 4)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.bool,
    )

    ref_out = reference_correction_mixer_contribution(
        mixer,
        x,
        ref,
        attention_mask=mask,
    )
    acc_out = accelerated_fixed_plane_mixer_contribution(
        mixer,
        x,
        acc,
        attention_mask=mask,
        checkpoint_recompute=True,
    )

    assert torch.allclose(
        ref_out,
        acc_out,
        atol=1e-6,
        rtol=1e-6,
    )

    ref_out.square().sum().backward()
    acc_out.square().sum().backward()

    assert torch.allclose(
        ref.A_theta.weight.grad,
        acc.A_theta.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        ref.M_theta.weight.grad,
        acc.M_theta.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_wrapper_install_parent_fingerprint_and_backward_isolation():
    model = TinyModel()
    original = model.mamba.layers[22].mixer
    before = parent_parameter_fingerprint(model)

    r22, c22 = orthogonal_bases(TEST_SHAPE.state_width)
    wrapper = install_stage_e_layer22_wrapper(
        model,
        arm="E-C22",
        r22=r22,
        c22=c22,
        seed=6201,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )

    assert isinstance(wrapper, StageELayer22MixerWrapper)
    assert wrapper.native_mixer is original
    assert parent_parameter_fingerprint(model) == before

    audit = stage_e_parameter_audit(model)
    assert audit["trainable_tensor_count"] == 2
    assert audit["trainable_numel"] == 12
    assert audit["all_trainable_are_correction"]

    x = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        native = original(x)
        wrapped_zero = wrapper(x, attention_mask=mask)
    assert torch.equal(native, wrapped_zero)

    with torch.no_grad():
        wrapper.correction.M_theta.weight.copy_(torch.eye(2))

    out = wrapper(x, attention_mask=mask)
    out.square().mean().backward()

    assert wrapper.correction.A_theta.weight.grad is not None
    assert wrapper.correction.M_theta.weight.grad is not None
    assert all(
        parameter.grad is None
        for parameter in original.parameters()
    )


def test_out_of_band_active_mask_bridge():
    model = TinyModel()
    r22, c22 = orthogonal_bases(TEST_SHAPE.state_width)
    wrapper = install_stage_e_layer22_wrapper(
        model,
        arm="E-R22",
        r22=r22,
        c22=c22,
        seed=6201,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )

    x = torch.randn(1, 3, 4)
    with pytest.raises(Exception):
        wrapper(x)

    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    with stage_e_active_mask(model, mask):
        out = wrapper(x)
    assert tuple(out.shape) == tuple(x.shape)


def test_static_mode_forbids_runtime_inputs():
    train.validate_mode_args(base_args())

    with pytest.raises(Exception):
        train.validate_mode_args(
            base_args(checkpoint=Path("parent.pt"))
        )
    with pytest.raises(Exception):
        train.validate_mode_args(
            base_args(arm="E-R22")
        )
    with pytest.raises(Exception):
        train.validate_mode_args(
            base_args(
                implementation_freeze_commit="a" * 40
            )
        )


def test_runtime_modes_require_future_authority_binding():
    args = base_args(
        static_verify_only=False,
        cuda_preflight_only=True,
        allow_opening_worktree=False,
        seed=6201,
        arm="E-R22",
        checkpoint=Path("parent.pt"),
        preflight_output=Path("preflight.json"),
    )
    with pytest.raises(Exception):
        train.validate_mode_args(args)

    args.implementation_freeze_commit = "a" * 40
    args.execution_authority_commit = "b" * 40
    train.validate_mode_args(args)


def test_matrix_scope_is_exact_six_cells_and_p0():
    flattened = [
        cell
        for worker in train.MATRIX_WORKER_CELLS
        for cell in worker
    ]
    assert len(flattened) == 6
    assert len(set(flattened)) == 6
    assert set(flattened) == {
        (seed, arm)
        for seed in (6201, 6202, 6203)
        for arm in ("E-R22", "E-C22")
    }
    assert train.PRESSURE == "P0"


def test_frozen_training_contract_and_reference_values():
    assert train.TRAIN_ROWS == 3360
    assert train.DEV_ROWS == 840
    assert train.SPLIT_SEED == 16384
    assert train.TOTAL_OPTIMIZER_STEPS == 20
    assert train.LEARNING_RATE == 0.001
    assert train.WEIGHT_DECAY == 0.0001
    assert train.GRADIENT_CLIP_NORM == 5.0

    assert train.ZERO_DEV_CE_P0 == 1.3409655094146729
    assert train.FREE_P0_GAIN_BY_SEED == {
        6201: 0.4984860420227051,
        6202: 0.5020102858543396,
        6203: 0.4994615912437439,
    }


def test_no_unrestricted_training_arm():
    assert "FREE" not in train.TRAINING_ARMS
    source = inspect.getsource(train)
    assert "E-FREE" not in source


def test_static_source_does_not_execute_scientific_model():
    source = inspect.getsource(train.run_static_verify)
    assert "_prepare_runtime_model(" not in source
    assert "_streamed_forward(" not in source
    assert "torch.optim" not in source


def test_execution_authority_is_fail_closed():
    source = inspect.getsource(train._validate_execution_authority)
    assert (
        "SCIENTIFIC_EXECUTION_ALLOWED="
        "YES_STAGE_E_FIXED_PLANE_SIX_CELL_MATRIX"
    ) in source
    assert "CONFIRMATORY_9601_9900_ALLOWED=NO" in source
    assert (
        "GPU_TOPOLOGY="
        "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP"
    ) in source


def test_matrix_workers_use_repo_external_scratch_before_merge():
    source = inspect.getsource(train.run_matrix)
    assert "tempfile.mkdtemp" in source
    assert "shutil.copytree" in source
    assert "MATRIX_WORKER_CELLS" in source


def test_parser_modes_are_mutually_exclusive():
    parser = train.build_parser()
    parsed = parser.parse_args([
        "--static-verify-only",
        "--expected-head",
        "abc",
    ])
    assert parsed.static_verify_only

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--static-verify-only",
            "--run-matrix",
            "--expected-head",
            "abc",
        ])
