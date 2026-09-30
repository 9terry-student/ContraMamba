from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from contramamba.gen5_phase2_state_update_ownership import (
    ARMS,
    accelerated_correction_mixer_contribution,
    CorrectionShape,
    Phase2Layer22MixerWrapper,
    StateWriteCorrection,
    correction_parameter_audit,
    explicit_total_recurrence,
    parent_parameter_fingerprint,
    phase2_active_mask,
    phase2_final_three_way_ce,
    reference_correction_mixer_contribution,
    reference_state_stack_bytes,
    streaming_state_bytes,
    reference_correction_recurrence,
    remove_phase2_layer22_wrapper,
    validate_arm,
)
from scripts import train_reason_router_gen5_phase2_state_update_ownership as train


TEST_SHAPE = CorrectionShape(
    hidden_size=4,
    intermediate_size=3,
    state_size=2,
    rank=2,
)


def orthogonal_bases(state_width: int):
    eye = torch.eye(state_width, dtype=torch.float64)
    return eye[:, :2].clone(), eye[:, 2:4].clone()


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
            3, 3, kernel_size=2, padding=1, groups=3, bias=True
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
        x = self.in_proj(hidden_states)[..., :3]
        return self.out_proj(torch.tanh(x))


class TinyBlock(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [TinyBlock(nn.Identity()) for _ in range(23)]
        layers[22] = TinyBlock(TinyMixer())
        self.mamba = type("TinyMambaHolder", (), {"layers": layers})()
        self.target = layers[22].mixer


def make_correction(arm: str, seed: int = 5201):
    r, c = orthogonal_bases(TEST_SHAPE.state_width)
    return StateWriteCorrection(
        arm=arm,
        r22=r,
        c22=c,
        seed=seed,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )


def test_arm_validation():
    for arm in ARMS:
        assert validate_arm(arm) == arm
    with pytest.raises(Exception):
        validate_arm("G5-X")


def test_initialization_same_seed_and_zero_output():
    x = torch.randn(2, 5, 4)
    a = make_correction("G5-C0", 5201)
    b = make_correction("G5-M1", 5201)
    assert torch.equal(a.A_theta.weight, b.A_theta.weight)
    assert torch.equal(a.B_theta.weight, b.B_theta.weight)
    assert torch.count_nonzero(a.B_theta.weight) == 0
    assert torch.count_nonzero(a(x)) == 0


def test_different_seed_changes_a():
    a = make_correction("G5-C0", 5201)
    b = make_correction("G5-C0", 5202)
    assert not torch.equal(a.A_theta.weight, b.A_theta.weight)


def test_projector_and_autograd():
    r, c = orthogonal_bases(TEST_SHAPE.state_width)
    raw = torch.randn(7, TEST_SHAPE.state_width, requires_grad=True)

    m1 = make_correction("G5-M1")
    c1 = make_correction("G5-C1")
    c0 = make_correction("G5-C0")

    m1_out = m1.project(raw)
    c1_out = c1.project(raw)
    c0_out = c0.project(raw)

    assert torch.max(torch.abs(m1_out.double() @ r)).item() < 1e-5
    assert torch.max(torch.abs(c1_out.double() @ c)).item() < 1e-5
    assert torch.equal(c0_out, raw)

    (m1_out.square().sum() + c1_out.square().sum()).backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert m1.R22.grad is None
    assert m1.C22.grad is None


def test_recurrence_superposition_and_persistence():
    torch.manual_seed(1)
    b, i, l, n = 2, 3, 4, 2
    discrete_a = torch.sigmoid(torch.randn(b, i, l, n))
    native_write = torch.randn(b, l, i, n)
    correction_write = torch.randn(b, l, i, n)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)

    native, correction, direct = explicit_total_recurrence(
        discrete_a,
        native_write,
        correction_write,
        active_mask=mask,
    )
    assert torch.allclose(native + correction, direct, atol=1e-6, rtol=1e-6)

    zero = reference_correction_recurrence(
        discrete_a,
        torch.zeros_like(correction_write),
        active_mask=mask,
    )
    assert torch.count_nonzero(zero) == 0
    assert torch.linalg.vector_norm(correction[0, 2]).item() > 0


def test_wrapper_zero_equivalence_and_backward_isolation():
    mixer = TinyMixer()
    r, c = orthogonal_bases(TEST_SHAPE.state_width)
    corr = StateWriteCorrection(
        arm="G5-M1",
        r22=r,
        c22=c,
        seed=5201,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )
    for p in mixer.parameters():
        p.requires_grad_(False)

    wrapper = Phase2Layer22MixerWrapper(
        native_mixer=mixer,
        correction=corr,
        strict_frozen_dimensions=False,
    )
    x = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        native = mixer(x, attention_mask=mask)
        wrapped = wrapper(x, attention_mask=mask)
    assert torch.equal(native, wrapped)

    with torch.no_grad():
        corr.B_theta.weight.normal_(mean=0.0, std=0.01)

    out = wrapper(x, attention_mask=mask)
    loss = out.square().mean()
    loss.backward()
    assert corr.A_theta.weight.grad is not None
    assert corr.B_theta.weight.grad is not None
    assert torch.isfinite(corr.A_theta.weight.grad).all()
    assert torch.isfinite(corr.B_theta.weight.grad).all()
    assert all(p.grad is None for p in mixer.parameters())


def test_final_ce_is_plain_cross_entropy():
    logits = torch.tensor(
        [[1.0, 0.5, -0.5], [0.1, -0.2, 0.8]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 2])
    observed = phase2_final_three_way_ce(logits, labels)
    expected = F.cross_entropy(logits, labels)
    assert torch.equal(observed, expected)


class FrozenShapeMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.intermediate_size = 1536
        self.ssm_state_size = 16
        self.time_step_rank = 48
        self.layer_idx = 22
        self.in_proj = nn.Linear(768, 3072, bias=False)
        self.conv1d = nn.Conv1d(
            1536, 1536, kernel_size=4, padding=3, groups=1536, bias=True
        )
        self.x_proj = nn.Linear(1536, 48 + 32, bias=False)
        self.dt_proj = nn.Linear(48, 1536, bias=True)
        self.A_log = nn.Parameter(torch.zeros(1536, 16))
        self.D = nn.Parameter(torch.ones(1536))
        self.out_proj = nn.Linear(1536, 768, bias=False)
        self.act = F.silu

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        return hidden_states


class RegisteredBlock(nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer


class RegisteredMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [RegisteredBlock(nn.Identity()) for _ in range(23)]
        )
        self.layers[22] = RegisteredBlock(FrozenShapeMixer())


class RegisteredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = RegisteredMamba()


def test_production_install_scope_mask_bridge_and_remove():
    from contramamba.gen5_phase2_state_update_ownership import (
        EXPECTED_TRAINABLE_NUMEL,
        Phase2Layer22MixerWrapper,
        install_phase2_layer22_wrapper,
    )

    model = RegisteredModel()
    original = model.mamba.layers[22].mixer
    untouched = [model.mamba.layers[i].mixer for i in range(22)]
    before = parent_parameter_fingerprint(model)

    r22 = torch.zeros(24576, 2, dtype=torch.float64)
    c22 = torch.zeros(24576, 2, dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0

    wrapper = install_phase2_layer22_wrapper(
        model,
        arm="G5-M1",
        r22=r22,
        c22=c22,
        seed=5201,
    )
    assert isinstance(model.mamba.layers[22].mixer, Phase2Layer22MixerWrapper)
    assert wrapper.native_mixer is original
    assert all(model.mamba.layers[i].mixer is untouched[i] for i in range(22))
    assert parent_parameter_fingerprint(model) == before

    audit = correction_parameter_audit(model)
    assert audit["trainable_tensor_count"] == 2
    assert audit["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL
    assert audit["all_trainable_are_correction"]

    x = torch.randn(1, 3, 768)
    with pytest.raises(Exception):
        wrapper(x)

    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    with phase2_active_mask(model, mask):
        out = wrapper(x)
    assert torch.equal(out, x)

    restored = remove_phase2_layer22_wrapper(model)
    assert restored is original
    assert model.mamba.layers[22].mixer is original
    assert parent_parameter_fingerprint(model) == before


def test_bases_are_buffers_not_parameters():
    module = make_correction("G5-M1")
    parameter_names = dict(module.named_parameters())
    buffer_names = dict(module.named_buffers())
    assert "R22" not in parameter_names
    assert "C22" not in parameter_names
    assert "R22" in buffer_names
    assert "C22" in buffer_names


def _copy_nonzero_correction(source, target):
    with torch.no_grad():
        target.A_theta.weight.copy_(source.A_theta.weight)
        target.B_theta.weight.copy_(source.B_theta.weight)


def test_accelerated_matches_reference_forward_and_gradients():
    torch.manual_seed(17)
    mixer = TinyMixer()
    for parameter in mixer.parameters():
        parameter.requires_grad_(False)

    r, c = orthogonal_bases(TEST_SHAPE.state_width)
    ref_corr = StateWriteCorrection(
        arm="G5-M1",
        r22=r,
        c22=c,
        seed=5201,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )
    acc_corr = StateWriteCorrection(
        arm="G5-M1",
        r22=r,
        c22=c,
        seed=5201,
        shape=TEST_SHAPE,
        strict_frozen_dimensions=False,
    )
    with torch.no_grad():
        ref_corr.B_theta.weight.normal_(0.0, 0.02)
    _copy_nonzero_correction(ref_corr, acc_corr)

    x = torch.randn(2, 5, 4)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
        dtype=torch.bool,
    )

    ref = reference_correction_mixer_contribution(
        mixer,
        x,
        ref_corr,
        attention_mask=mask,
    )
    acc = accelerated_correction_mixer_contribution(
        mixer,
        x,
        acc_corr,
        attention_mask=mask,
        checkpoint_recompute=True,
    )
    assert torch.allclose(ref, acc, atol=1e-6, rtol=1e-6)

    ref.square().sum().backward()
    acc.square().sum().backward()

    assert torch.allclose(
        ref_corr.A_theta.weight.grad,
        acc_corr.A_theta.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        ref_corr.B_theta.weight.grad,
        acc_corr.B_theta.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_streaming_backend_removes_sequence_state_stack():
    batch = 2880
    seq = 128
    reference = reference_state_stack_bytes(batch, seq, dtype_bytes=4)
    streaming = streaming_state_bytes(batch, dtype_bytes=4)
    assert reference == 36_238_786_560
    assert streaming == 283_115_520
    assert reference == streaming * seq


# ---------------------------------------------------------------------------
# Training-runner opening tests
# ---------------------------------------------------------------------------


def test_cuda_runtime_defers_transformers_module_binding_validation(monkeypatch):
    from scripts import (
        reason_router_gen4_generator_family_prevalence_kernel_compat
        as kernel_compat,
    )
    from scripts import (
        reason_router_gen4_k_fast_cuda_one_pair_equivalence
        as backend,
    )

    monkeypatch.setattr(backend, "runtime_gate", lambda: None)
    monkeypatch.setattr(
        train,
        "runtime_versions",
        lambda: {
            "python": "3.12.13",
            "torch": "2.10.0+cu128",
            "transformers": "5.0.0",
            "cuda_runtime": "12.8",
            "gpu0_name": "Tesla T4",
            "gpu0_capability": (7, 5),
            "default_dtype": "torch.float32",
            "autocast_enabled": False,
        },
    )

    def fail_if_called(_kernels):
        pytest.fail(
            "Transformers kernel module bindings were validated "
            "before Mamba construction"
        )

    monkeypatch.setattr(
        kernel_compat,
        "validate_transformers_kernel_bindings",
        fail_if_called,
    )

    runtime, observed_compat, observed_backend = train.validate_cuda_runtime()

    assert runtime["transformers"] == "5.0.0"
    assert observed_compat is kernel_compat
    assert observed_backend is backend


def test_kaggle_preflight_binding_failure_is_preconstruction_state():
    from types import SimpleNamespace
    from scripts import (
        reason_router_gen4_generator_family_prevalence_kernel_compat
        as kernel_compat,
    )

    mamba = SimpleNamespace(
        selective_scan_fn=lambda: None,
        selective_state_update=lambda: None,
        mamba_inner_fn=lambda: None,
    )
    conv = SimpleNamespace(
        causal_conv1d_fn=lambda: None,
        causal_conv1d_update=lambda: None,
    )
    modeling = SimpleNamespace()

    functions = kernel_compat.patch_transformers_mamba(
        mamba,
        conv,
        modeling_module=modeling,
    )
    kernels = {
        "mamba": mamba,
        "conv": conv,
        **functions,
        "transport_identity_status": "EXACT_FROZEN_BINARY_SHA256_MATCH",
    }

    with pytest.raises(kernel_compat.KernelCompatibilityError) as exc_info:
        kernel_compat.validate_transformers_kernel_bindings(
            kernels,
            modeling_module=modeling,
        )

    assert str(exc_info.value) == (
        "TRANSFORMERS_KERNEL_BINDING:mamba_ssm,causal_conv1d"
    )

    # This is what Transformers 5.0.0 MambaMixer construction performs
    # through the exact lazy_load_kernel interception.
    modeling.causal_conv1d = conv
    modeling.mamba_ssm = mamba

    kernel_compat.validate_transformers_kernel_bindings(
        kernels,
        modeling_module=modeling,
    )


def test_training_matrix_defers_binding_validation_to_parent_construction():
    import inspect

    source = inspect.getsource(train.run_matrix)
    assert "validate_transformers_kernel_bindings" not in source


def test_training_authority_is_bound():
    assert train.TRAINING_EXECUTION_AUTHORITY_COMMIT == (
        "5ab3174cc24731f867e512c9381b9abcc3263915"
    )
    assert train.ACCELERATED_IMPLEMENTATION_COMMIT == (
        "a69a0dd47853e3ec68f7c5aa21bb8d692c89a86a"
    )
    assert train.RECOVERY_AMENDMENT_COMMIT == (
        "360670cc657dbb7d2ee3ee894d9109846631a838"
    )
    assert train.RECOVERY_AMENDMENT_BLOB == (
        "f52603e014e46a4a62a42571121605a1f6574879"
    )
    assert train.RECOVERY_AMENDMENT_PATH in train.AUTHORIZED_POST_AUTHORITY_PATHS


def test_training_matrix_is_exact():
    assert train.TRAINING_SEEDS == (5201, 5202, 5203)
    assert train.TRAINING_ARMS == ("G5-C0", "G5-C1", "G5-M1")
    assert train.TRAINING_MATRIX == (
        (5201, "G5-C0"),
        (5201, "G5-C1"),
        (5201, "G5-M1"),
        (5202, "G5-C0"),
        (5202, "G5-C1"),
        (5202, "G5-M1"),
        (5203, "G5-C0"),
        (5203, "G5-C1"),
        (5203, "G5-M1"),
    )


def test_dual_gpu_worker_partition_is_exact_and_seed_grouped():
    assert train.MATRIX_GPU_COUNT == 2
    assert len(train.MATRIX_WORKER_CELLS) == 2

    flattened = [
        cell
        for worker_cells in train.MATRIX_WORKER_CELLS
        for cell in worker_cells
    ]
    assert len(flattened) == len(train.TRAINING_MATRIX)
    assert len(set(flattened)) == len(train.TRAINING_MATRIX)
    assert set(flattened) == set(train.TRAINING_MATRIX)

    for seed in train.TRAINING_SEEDS:
        owners = [
            worker_index
            for worker_index, worker_cells
            in enumerate(train.MATRIX_WORKER_CELLS)
            if any(cell_seed == seed for cell_seed, _ in worker_cells)
        ]
        assert len(owners) == 1
        arms = {
            arm
            for cell_seed, arm in train.MATRIX_WORKER_CELLS[owners[0]]
            if cell_seed == seed
        }
        assert arms == set(train.TRAINING_ARMS)


def test_frozen_training_contract_exact():
    c = train.FROZEN_TRAINING_CONTRACT
    assert c["train_rows"] == 2880
    assert c["dev_rows"] == 720
    assert c["split_seed"] == 8192
    assert c["optimizer"] == "torch.optim.AdamW"
    assert c["learning_rate"] == 0.001
    assert c["weight_decay"] == 0.0001
    assert c["scheduler"] is None
    assert c["epochs"] == 20
    assert c["optimizer_steps"] == 20
    assert c["gradient_clip_norm"] == 5.0
    assert c["objective"] == "FINAL_3WAY_CROSS_ENTROPY_ONLY"
    assert c["microbatching"] is False
    assert c["gradient_accumulation"] is False
    assert c["feasibility_recovery_amendment_commit"] == train.RECOVERY_AMENDMENT_COMMIT
    assert c["backbone_streaming"] is True
    assert c["backbone_stream_rows"] == 240
    assert c["downstream_full_batch_once"] is True
    assert c["loss_backward_calls_per_epoch"] == 1
    assert c["matrix_gpu_count"] == 2
    assert c["matrix_parallelism"] == "TWO_T4_INDEPENDENT_SEED_GROUP_WORKERS"
    assert c["matrix_worker_seeds"] == [[5201, 5203], [5202]]


def test_git_lf_sha256_normalizes_crlf(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"alpha\nbeta\n")
    b.write_bytes(b"alpha\r\nbeta\r\n")
    assert train.git_lf_sha256(a) == train.git_lf_sha256(b)


def test_semantic_jsonl_hash_is_canonical_array(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"b":2,"a":1}\n{"a":3,"b":4}\n',
        encoding="utf-8",
        newline="\n",
    )
    expected = hashlib.sha256(
        json.dumps(
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert train.semantic_jsonl_sha256(path) == expected


def test_ordered_row_identity_is_order_sensitive():
    rows = [
        {
            "id": "p1__none",
            "pair_id": "p1",
            "intervention_type": "none",
            "final_label": "SUPPORT",
        },
        {
            "id": "p2__none",
            "pair_id": "p2",
            "intervention_type": "none",
            "final_label": "REFUTE",
        },
    ]
    assert train.ordered_row_identity_sha256(rows) != train.ordered_row_identity_sha256(
        list(reversed(rows))
    )


def test_encoded_bundle_hash_is_tensor_and_order_sensitive():
    bundle = {
        "pair_ids": ["a", "b"],
        "intervention_types": ["none", "none"],
        "model_inputs": {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            "claim_mask": torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
            "evidence_mask": torch.tensor([[0, 1], [0, 0]], dtype=torch.bool),
            "final_labels": torch.tensor([2, 0], dtype=torch.long),
        },
    }
    first = train.encoded_bundle_sha256(bundle)
    changed = {
        **bundle,
        "pair_ids": ["b", "a"],
    }
    assert first != train.encoded_bundle_sha256(changed)


def test_expected_dev_pair_contract():
    assert len(train.EXPECTED_DEV_PAIR_IDS) == 60
    assert "clinic_expansion" in train.EXPECTED_DEV_PAIR_IDS
    assert "satellite_launch" in train.EXPECTED_DEV_PAIR_IDS
    assert "orion_approval" not in train.EXPECTED_DEV_PAIR_IDS


def test_fresh_assay_paths_are_rejected():
    with pytest.raises(Exception):
        train.validate_no_fresh_assay_path(
            "data/reason_router_gen5_phase2_xg1_ownership_assay_v1/foo.jsonl"
        )
    with pytest.raises(Exception):
        train.validate_no_fresh_assay_path("xg1_fact_8701")
    train.validate_no_fresh_assay_path("reports/phase2_training")


def test_status_paths_preserves_porcelain_fixed_width_prefix(monkeypatch):
    raw = (
        " M scripts/train_reason_router_gen5_phase2_state_update_ownership.py\n"
        "M  tests/test_reason_router_gen5_phase2_state_update_ownership.py\n"
        "?? scratch.txt\n"
    )

    def fake_check_output(command, **kwargs):
        assert command == ["git", "status", "--porcelain=v1"]
        assert kwargs["cwd"] == train.ROOT
        assert kwargs["text"] is True
        return raw

    monkeypatch.setattr(train.subprocess, "check_output", fake_check_output)

    assert train._status_paths() == {
        "scripts/train_reason_router_gen5_phase2_state_update_ownership.py",
        "tests/test_reason_router_gen5_phase2_state_update_ownership.py",
        "scratch.txt",
    }


def _base_args(**updates):
    values = dict(
        static_preflight_only=True,
        cuda_preflight_only=False,
        run_matrix=False,
        run_matrix_worker=False,
        expected_head="abc",
        execution_authority_commit=train.TRAINING_EXECUTION_AUTHORITY_COMMIT,
        model_snapshot=None,
        checkpoint=None,
        preflight_output=None,
        streamed_preflight_report=None,
        output_root=None,
        matrix_worker_index=None,
        expected_train_order_sha256_v2=None,
        expected_dev_order_sha256_v2=None,
        expected_train_encoding_sha256=None,
        expected_dev_encoding_sha256=None,
        allow_opening_worktree=True,
        print_frozen_contract=False,
    )
    values.update(updates)
    return argparse.Namespace(**values)


def test_static_mode_forbids_checkpoint_and_outputs():
    train.validate_mode_args(_base_args())
    with pytest.raises(Exception):
        train.validate_mode_args(_base_args(checkpoint=Path("x.pt")))
    with pytest.raises(Exception):
        train.validate_mode_args(_base_args(output_root=Path("out")))


def test_cuda_execution_requires_pinned_identities():
    args = _base_args(
        static_preflight_only=False,
        cuda_preflight_only=True,
        allow_opening_worktree=False,
        checkpoint=Path("parent.pt"),
        preflight_output=Path("preflight.json"),
    )
    with pytest.raises(Exception):
        train.validate_mode_args(args)

    args.expected_train_order_sha256_v2 = "a" * 64
    args.expected_dev_order_sha256_v2 = "b" * 64
    args.expected_train_encoding_sha256 = "c" * 64
    args.expected_dev_encoding_sha256 = "d" * 64
    train.validate_mode_args(args)


def test_matrix_mode_requires_output_root_and_no_preflight_output():
    args = _base_args(
        static_preflight_only=False,
        run_matrix=True,
        allow_opening_worktree=False,
        checkpoint=Path("parent.pt"),
        expected_train_order_sha256_v2="a" * 64,
        expected_dev_order_sha256_v2="b" * 64,
        expected_train_encoding_sha256="c" * 64,
        expected_dev_encoding_sha256="d" * 64,
    )
    with pytest.raises(Exception):
        train.validate_mode_args(args)

    args.output_root = Path("out")
    args.streamed_preflight_report = Path("streamed_preflight.json")
    train.validate_mode_args(args)
    args.preflight_output = Path("preflight.json")
    with pytest.raises(Exception):
        train.validate_mode_args(args)


def test_matrix_worker_mode_requires_index_and_pinned_inputs():
    args = _base_args(
        static_preflight_only=False,
        run_matrix_worker=True,
        allow_opening_worktree=False,
        checkpoint=Path("parent.pt"),
        streamed_preflight_report=Path("streamed_preflight.json"),
        output_root=Path("out"),
        expected_train_order_sha256_v2="a" * 64,
        expected_dev_order_sha256_v2="b" * 64,
        expected_train_encoding_sha256="c" * 64,
        expected_dev_encoding_sha256="d" * 64,
    )
    with pytest.raises(Exception):
        train.validate_mode_args(args)

    args.matrix_worker_index = 1
    train.validate_mode_args(args)

    args.matrix_worker_index = 2
    with pytest.raises(Exception):
        train.validate_mode_args(args)


def test_matrix_worker_command_pins_same_inputs():
    args = _base_args(
        static_preflight_only=False,
        run_matrix=True,
        allow_opening_worktree=False,
        checkpoint=Path("/tmp/parent.pt"),
        streamed_preflight_report=Path("/tmp/preflight.json"),
        output_root=Path("outputs/matrix"),
        expected_train_order_sha256_v2="a" * 64,
        expected_dev_order_sha256_v2="b" * 64,
        expected_train_encoding_sha256="c" * 64,
        expected_dev_encoding_sha256="d" * 64,
    )
    command = train._matrix_worker_command(args, 1)
    assert "--run-matrix-worker" in command
    assert command[command.index("--matrix-worker-index") + 1] == "1"
    assert command[command.index("--expected-head") + 1] == "abc"
    assert command[command.index("--checkpoint") + 1] == str(Path("/tmp/parent.pt"))
    assert (
        command[command.index("--streamed-preflight-report") + 1]
        == str(Path("/tmp/preflight.json"))
    )


def test_cross_arm_initialization_manifest_exact():
    r22 = torch.zeros(24576, 2, dtype=torch.float64)
    c22 = torch.zeros(24576, 2, dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    manifest = train.correction_initialization_manifest(r22, c22)
    assert set(manifest) == {"5201", "5202", "5203"}
    for row in manifest.values():
        assert row["cross_arm_identical"] is True
        assert set(row["arms"]) == set(train.TRAINING_ARMS)


def test_checkpoint_payload_contains_only_correction_state():
    correction = make_correction("G5-M1", 5201)
    wrapper = type("Wrapper", (), {"correction": correction})()
    args = argparse.Namespace(expected_head="deadbeef")
    payload, hashes = train._correction_checkpoint_payload(
        wrapper=wrapper,
        args=args,
        seed=5201,
        arm="G5-M1",
    )
    assert set(payload["state_dict"]) == {"A_theta.weight", "B_theta.weight"}
    assert set(hashes) == {"A_theta.weight", "B_theta.weight"}
    assert "model_state_dict" not in payload


def test_parser_modes_are_mutually_exclusive():
    parser = train.build_parser()
    parsed = parser.parse_args(
        ["--static-preflight-only", "--expected-head", "abc"]
    )
    assert parsed.static_preflight_only
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--static-preflight-only",
                "--cuda-preflight-only",
                "--expected-head",
                "abc",
            ]
        )


def test_phase2_sidecar_semantic_hash_excludes_created_at(tmp_path):
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as training

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"

    row_a = {
        "row_id": "r1",
        "pair_id": "p1",
        "integrity_status": "ELIGIBLE",
        "created_at": "2026-01-01T00:00:00Z",
    }
    row_b = {
        **row_a,
        "created_at": "2026-09-30T00:00:00Z",
    }

    first.write_text(json.dumps(row_a, sort_keys=True) + "\n", encoding="utf-8")
    second.write_text(json.dumps(row_b, sort_keys=True) + "\n", encoding="utf-8")

    assert training.semantic_sidecar_sha256(first) == training.semantic_sidecar_sha256(second)
    assert training.semantic_jsonl_sha256(first) != training.semantic_jsonl_sha256(second)


def test_phase2_sidecar_semantic_hash_changes_on_non_timestamp_field(tmp_path):
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as training

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"

    row_a = {
        "row_id": "r1",
        "pair_id": "p1",
        "integrity_status": "ELIGIBLE",
        "created_at": "2026-01-01T00:00:00Z",
    }
    row_b = {
        **row_a,
        "integrity_status": "INELIGIBLE",
    }

    first.write_text(json.dumps(row_a, sort_keys=True) + "\n", encoding="utf-8")
    second.write_text(json.dumps(row_b, sort_keys=True) + "\n", encoding="utf-8")

    assert training.semantic_sidecar_sha256(first) != training.semantic_sidecar_sha256(second)


class _StreamingToyMamba(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [RegisteredBlock(nn.Identity()) for _ in range(23)]
        native = TinyMixer()
        for parameter in native.parameters():
            parameter.requires_grad_(False)
        correction = make_correction("G5-M1", 5201)
        with torch.no_grad():
            correction.B_theta.weight.normal_(0.0, 0.02)
        layers[22] = RegisteredBlock(
            Phase2Layer22MixerWrapper(
                native_mixer=native,
                correction=correction,
                strict_frozen_dimensions=False,
            )
        )
        self.layers = nn.ModuleList(layers)
        self.config = type("Config", (), {"use_cache": False})()

    def forward(self, input_ids):
        hidden = F.one_hot(
            torch.remainder(input_ids, 4),
            num_classes=4,
        ).to(torch.float32)
        hidden = self.layers[22].mixer(hidden)
        return type("Output", (), {"last_hidden_state": hidden})()


class _StreamingToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = _StreamingToyMamba()
        self.dropout = nn.Dropout(p=0.25)
        self.head = nn.Linear(4, 3, bias=False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        self.downstream_calls = 0

    def forward(
        self,
        input_ids,
        attention_mask,
        claim_mask,
        evidence_mask,
        encoder_hidden_states=None,
        **kwargs,
    ):
        del attention_mask, claim_mask, evidence_mask, kwargs
        if encoder_hidden_states is None:
            hidden = self.mamba(input_ids=input_ids).last_hidden_state
        else:
            hidden = encoder_hidden_states
        self.downstream_calls += 1
        pooled = hidden.mean(dim=1)
        return {"logits": self.head(self.dropout(pooled))}


def test_streamed_backbone_preserves_objective_and_gradients_cpu():
    torch.manual_seed(73)
    model = _StreamingToyModel()
    model.train()
    wrapper = model.mamba.layers[22].mixer
    features = {
        "input_ids": torch.tensor(
            [
                [0, 1, 2, 3, 0],
                [1, 2, 3, 0, 1],
                [2, 3, 0, 1, 2],
                [3, 0, 1, 2, 3],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(4, 5, dtype=torch.bool),
        "claim_mask": torch.ones(4, 5, dtype=torch.bool),
        "evidence_mask": torch.ones(4, 5, dtype=torch.bool),
    }
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    torch.manual_seed(991)
    initial_rng = train._capture_rng_state()
    model.zero_grad(set_to_none=True)
    with phase2_active_mask(model, features["attention_mask"]):
        monolithic = train._historical_forward(model, features)
    mono_logits = monolithic["logits"]
    mono_loss = phase2_final_three_way_ce(mono_logits, labels)
    mono_loss.backward()
    mono_a = wrapper.correction.A_theta.weight.grad.detach().clone()
    mono_b = wrapper.correction.B_theta.weight.grad.detach().clone()
    mono_logits = mono_logits.detach().clone()
    mono_loss_value = mono_loss.detach().clone()

    model.zero_grad(set_to_none=True)
    train._restore_rng_state(initial_rng)
    streamed, chunk_count = train._streamed_historical_forward(
        model,
        features,
        stream_rows=2,
    )
    stream_logits = streamed["logits"]
    stream_loss = phase2_final_three_way_ce(stream_logits, labels)
    stream_loss.backward()

    assert chunk_count == 2
    assert model.downstream_calls == 2
    assert torch.allclose(
        mono_logits,
        stream_logits.detach(),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        mono_loss_value,
        stream_loss.detach(),
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.allclose(
        mono_a,
        wrapper.correction.A_theta.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        mono_b,
        wrapper.correction.B_theta.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if ".correction." not in name
    )



def test_streamed_backbone_alone_preserves_rng_cpu():
    torch.manual_seed(101)
    model = _StreamingToyModel()
    model.train()
    features = {
        "input_ids": torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
                [2, 3, 0, 1],
                [3, 0, 1, 2],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones(4, 4, dtype=torch.bool),
        "claim_mask": torch.ones(4, 4, dtype=torch.bool),
        "evidence_mask": torch.ones(4, 4, dtype=torch.bool),
    }

    before = train._capture_rng_state()
    hidden, chunk_count = train._streamed_backbone_hidden(
        model,
        features,
        stream_rows=2,
    )
    after = train._capture_rng_state()

    assert chunk_count == 2
    assert hidden.shape == (4, 4, 4)
    assert torch.equal(before["cpu"], after["cpu"])
    assert before["cuda"] == after["cuda"]
    assert model.downstream_calls == 0

def test_streamed_backbone_helper_has_no_per_chunk_downstream_call():
    import inspect

    source = inspect.getsource(train._streamed_backbone_hidden)
    assert "_historical_forward" not in source
    source = inspect.getsource(train._streamed_historical_forward)
    assert source.count("_historical_forward_from_hidden") == 1


def test_training_cell_is_one_streamed_loss_backward_per_epoch():
    import inspect

    source = inspect.getsource(train.run_training_cell)
    assert "_streamed_historical_forward" in source
    assert "with phase2_active_mask(model, features" not in source
    assert source.count("loss.backward()") == 1
    assert source.count("optimizer.step()") == 1


def test_streamed_preflight_report_gate(tmp_path: Path):
    static = {
        "train_order_sha256_v2": "a" * 64,
        "dev_order_sha256_v2": "b" * 64,
    }
    encoded = {
        "train_encoding_sha256": "c" * 64,
        "dev_encoding_sha256": "d" * 64,
    }
    args = argparse.Namespace(expected_head="head123")
    report = {
        "schema_version": "GEN5_PHASE2_STREAMED_CUDA_PREFLIGHT_V1",
        "result": "PASS_GEN5_PHASE2_STREAMED_FULL_BATCH_CUDA_PREFLIGHT",
        "execution_head": "head123",
        "authority_commit": train.TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "feasibility_recovery_amendment_commit": train.RECOVERY_AMENDMENT_COMMIT,
        "checkpoint_sha256": train.PARENT_CHECKPOINT_SHA256,
        "r22_sha256": train.R22_SHA256,
        "c22_sha256": train.C22_SHA256,
        "full_batch_rows": train.TRAIN_ROWS,
        "backbone_stream_rows": train.BACKBONE_STREAM_ROWS,
        "backward_completed": True,
        "backward_executed": True,
        "optimizer_step_executed": False,
        "training_executed": False,
        "fresh_xg1_loaded": False,
        "scientific_p_value_count": 0,
        "train_order_sha256_v2": static["train_order_sha256_v2"],
        "dev_order_sha256_v2": static["dev_order_sha256_v2"],
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "dev_encoding_sha256": encoded["dev_encoding_sha256"],
        "bounded_equivalence": {"pass": True},
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    observed = train.validate_streamed_preflight_report(
        path,
        args=args,
        static=static,
        encoded=encoded,
    )
    assert observed["result"] == report["result"]

    report["optimizer_step_executed"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(Exception):
        train.validate_streamed_preflight_report(
            path,
            args=args,
            static=static,
            encoded=encoded,
        )


def test_cuda_preflight_has_backward_but_no_optimizer_step():
    import inspect

    source = inspect.getsource(train.run_cuda_preflight)
    assert "_bounded_streaming_equivalence" in source
    assert "_streamed_historical_forward" in source
    assert source.count("loss.backward()") == 1
    assert "optimizer.step()" not in source
