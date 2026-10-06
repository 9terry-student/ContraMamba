from __future__ import annotations

import argparse
import inspect
import json
import math

import pytest
import torch

from scripts import audit_reason_router_gen5_ainit_temporal_birth as mod
from scripts import train_reason_router_gen5_ainit_rng_causal_intervention as base


def _args(**overrides):
    values = {
        "static_verify_only": False,
        "cuda_preflight_only": False,
        "replay_auth_recovery_only": False,
        "replay_auth_recovery_worker": False,
        "run_replay_matrix": False,
        "run_replay_worker": False,
        "phase_b_cuda_preflight_only": False,
        "run_phase_b": False,
        "phase_b_worker": False,
        "temporal_mechanism_static_verify_only": False,
        "temporal_mechanism_cuda_preflight_only": False,
        "run_temporal_mechanism": False,
        "temporal_mechanism_worker": False,
        "expected_head": mod.AUTHORITY_COMMIT,
        "allow_opening_worktree": False,
        "implementation_freeze_commit": None,
        "model_snapshot": None,
        "tokenizer_snapshot": None,
        "checkpoint": None,
        "output_root": None,
        "scratch_root": None,
        "worker_id": None,
        "phase_b_step": None,
        "phase_b_compute_geometry": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_full_factorial_grid_and_exact_two_gpu_parity_sharding():
    assert mod.FACTOR_SEEDS == (6201, 6202, 6203)
    assert mod.FULL_FACTORIAL_CELLS == (
        (6201, 6201),
        (6201, 6202),
        (6201, 6203),
        (6202, 6201),
        (6202, 6202),
        (6202, 6203),
        (6203, 6201),
        (6203, 6202),
        (6203, 6203),
    )
    assert mod.GPU_QUEUES == {
        0: (
            (6201, 6201),
            (6201, 6203),
            (6202, 6202),
            (6203, 6201),
            (6203, 6203),
        ),
        1: (
            (6201, 6202),
            (6202, 6201),
            (6202, 6203),
            (6203, 6202),
        ),
    }
    mod.validate_gpu_queues()


@pytest.mark.parametrize("cell", mod.FULL_FACTORIAL_CELLS)
def test_low_level_factor_validation_accepts_full_3x3(cell):
    base.validate_factor_cell(*cell)
    mod.validate_full_factorial_cell(*cell)


@pytest.mark.parametrize("seed", mod.FACTOR_SEEDS)
def test_historical_offdiagonal_validation_still_rejects_diagonal(seed):
    with pytest.raises(
        base.AInitRNGCausalInterventionError,
        match="NOT_AUTHORIZED_OFFDIAGONAL",
    ):
        base.validate_offdiagonal_cell(seed, seed)


def test_runtime_model_low_level_uses_full_factor_validation():
    source = inspect.getsource(base._prepare_runtime_model)
    executable_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "validate_factor_cell(a_init_seed, training_rng_seed)" in executable_lines
    assert not any(
        line.startswith("validate_offdiagonal_cell(")
        for line in executable_lines
    )


def test_exact_frozen_checkpoint_source_matrix():
    assert set(mod.FROZEN_FINAL_SOURCES) == set(mod.FULL_FACTORIAL_CELLS)
    assert len(mod.FROZEN_FINAL_SOURCES) == 9
    expected_hashes = {
        "157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf",
        "ef03fbbedf3fab8efb255f2ccb6ec33cdb65ee6881a92e6b4d43fe1e719e40d4",
        "15582eda034befb1c8d202f04c494f7fd5882bf9fd60ddc963232761057b3df7",
        "3c217b43eb164583980bee91c39d53a7a3dc20d181e0341db35ffe8ece162ed3",
        "1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214",
        "d1c478b448c5f53e2a97552196a454f63f9d9081de614a6d9a5cd4e389e30ee2",
        "d9e1062baf554867b212da57c9d30fe8efeccafc77255ba869affac691606359",
        "32943df3558ef72eb7f7b7bfb70c6a1a03185a1ea6ca4dd83b9677cd59d13698",
        "c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770",
    }
    assert {row[1] for row in mod.FROZEN_FINAL_SOURCES.values()} == expected_hashes


def test_operator_inner_matches_explicit_matrix_product():
    generator = torch.Generator().manual_seed(123)
    a1 = torch.randn((2, 5), generator=generator)
    b1 = torch.randn((7, 2), generator=generator)
    a2 = torch.randn((2, 5), generator=generator)
    b2 = torch.randn((7, 2), generator=generator)

    explicit1 = b1.to(torch.float64) @ a1.to(torch.float64)
    explicit2 = b2.to(torch.float64) @ a2.to(torch.float64)

    expected_inner = float(torch.sum(explicit1 * explicit2).item())
    expected_distance_sq = float(
        torch.sum((explicit1 - explicit2) ** 2).item()
    )

    assert math.isclose(
        mod.operator_inner(a1, b1, a2, b2),
        expected_inner,
        rel_tol=0.0,
        abs_tol=1e-10,
    )
    assert math.isclose(
        mod.operator_distance_sq(a1, b1, a2, b2),
        expected_distance_sq,
        rel_tol=0.0,
        abs_tol=1e-10,
    )


def test_operator_snapshot_singular_values_match_explicit_product():
    generator = torch.Generator().manual_seed(321)
    a = torch.randn((2, 5), generator=generator)
    b = torch.randn((7, 2), generator=generator)
    observed = mod.operator_snapshot_stats(a, b)
    expected = torch.linalg.svdvals(
        b.to(torch.float64) @ a.to(torch.float64)
    )
    assert observed["nonzero_singular_values"] == pytest.approx(
        expected[:2].tolist(),
        rel=1e-12,
        abs=1e-12,
    )
    assert expected[2:].tolist() == pytest.approx(
        [0.0] * int(expected.numel() - 2),
        rel=0.0,
        abs=1e-12,
    )
    assert observed["frobenius_norm"] == pytest.approx(
        float(torch.linalg.vector_norm(
            b.to(torch.float64) @ a.to(torch.float64)
        ).item()),
        rel=1e-12,
        abs=1e-12,
    )


def test_factorial_energy_detects_pure_a_effect():
    cells = list(mod.FULL_FACTORIAL_CELLS)
    # Complete grid with identical values within each A level.
    complete = {
        (a, r): torch.tensor([[float(a - 6200)]], dtype=torch.float64)
        for a, r in cells
    }
    ordered, gram = mod._tensor_gram(complete)
    result = mod.factorial_energy_from_gram(ordered, gram)
    fractions = result["normalized_fractions"]
    assert fractions is not None
    assert fractions["A"] == pytest.approx(1.0, abs=1e-12)
    assert fractions["R"] == pytest.approx(0.0, abs=1e-12)
    assert fractions["AR"] == pytest.approx(0.0, abs=1e-12)
    assert result["closure_abs"] <= 1e-10


def test_zero_grid_has_no_normalized_pair_residual_or_factor_fraction():
    values = {
        cell: torch.zeros((3, 2), dtype=torch.float32)
        for cell in mod.FULL_FACTORIAL_CELLS
    }
    ordered, gram = mod._tensor_gram(values)
    grouped = mod.grouped_pair_stats_from_gram(ordered, gram)
    factorial = mod.factorial_energy_from_gram(ordered, gram)

    assert (
        grouped["same_training_rng_different_a"]["mean_squared_distance"]
        == 0.0
    )
    assert (
        grouped["same_training_rng_different_a"]["mean_normalized_residual"]
        is None
    )
    assert (
        grouped["same_a_different_training_rng"]["mean_normalized_residual"]
        is None
    )
    assert grouped["A_over_R_pair_distance_ratio"] is None
    assert factorial["normalized_fractions"] is None



def test_replay_auth_recovery_cells_cover_two_historical_lineages():
    assert mod.REPLAY_AUTH_RECOVERY_CELLS == {0: (6201, 6201), 1: (6201, 6202)}


@pytest.mark.parametrize("cell", mod.FULL_FACTORIAL_CELLS)
def test_historical_training_trace_is_frozen_for_full_3x3(cell):
    trace = mod.load_historical_training_trace(*cell)
    assert len(trace["training_losses"]) == mod.TOTAL_OPTIMIZER_STEPS
    assert len(trace["gradient_norms_before_clip"]) == mod.TOTAL_OPTIMIZER_STEPS
    assert trace["file_sha256"] == mod.HISTORICAL_TRAINING_REPORT_SHA256[cell]



def test_replay_authentication_correction_constants_are_preregistered():
    assert (
        mod.REPLAY_AUTHENTICATION_CORRECTION_COMMIT
        == "341e2e59668ea9b575007ddf591424b831170577"
    )
    assert mod.EPS32 == float(torch.finfo(torch.float32).eps)
    assert mod.SCALAR_TRACE_EPS_MULTIPLIER == 32.0
    assert mod.PARAMETER_MAX_ABS_BOUND == 1.0e-5
    assert mod.PARAMETER_RELATIVE_L2_BOUND == 1.0e-4
    assert mod.OPERATOR_RELATIVE_FROBENIUS_BOUND == 1.0e-4


def test_scalar_trace_authentication_accepts_observed_recovery_drift():
    diag = mod.scalar_trace_diagnostics(
        observed=1.25958251953125,
        historical=1.2595824003219604,
    )
    assert diag["absolute_error"] == pytest.approx(
        1.1920928955078125e-07,
        rel=0.0,
        abs=1e-15,
    )
    assert diag["pass"] is True


def test_scalar_trace_authentication_rejects_out_of_bound_difference():
    historical = 1.0
    observed = historical + 2.0 * mod.scalar_trace_tolerance(historical)
    diag = mod.scalar_trace_diagnostics(
        observed=observed,
        historical=historical,
    )
    assert diag["pass"] is False


def test_tensor_replay_diagnostics_keep_sha_as_diagnostic():
    reference = torch.ones((2, 3), dtype=torch.float32)
    replay = reference.clone()
    replay[0, 0] += 1.0e-6
    diag = mod.tensor_replay_diagnostics(replay, reference)
    assert diag["torch_equal"] is False
    assert diag["replay_sha256"] != diag["historical_sha256"]
    assert diag["pass"] is True


def test_tensor_replay_diagnostics_reject_out_of_bound_max_abs():
    reference = torch.ones((2, 3), dtype=torch.float32)
    replay = reference.clone()
    replay[0, 0] += 2.0e-5
    diag = mod.tensor_replay_diagnostics(replay, reference)
    assert diag["pass"] is False
    assert diag["max_abs_pass"] is False


def test_operator_replay_diagnostics_match_explicit_residual():
    a_ref = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b_ref = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    a_rep = a_ref.clone()
    b_rep = b_ref.clone()
    b_rep[0, 0] += 1.0e-6
    diag = mod.operator_replay_diagnostics(
        replay_a=a_rep,
        replay_b=b_rep,
        historical_a=a_ref,
        historical_b=b_ref,
    )
    w_ref = b_ref.to(torch.float64) @ a_ref.to(torch.float64)
    w_rep = b_rep.to(torch.float64) @ a_rep.to(torch.float64)
    expected = float(
        torch.linalg.vector_norm(w_rep - w_ref).item()
        / torch.linalg.vector_norm(w_ref).item()
    )
    assert diag["relative_frobenius_residual"] == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


def test_full_worker_authenticates_historical_trace_for_every_cell():
    source = inspect.getsource(mod.run_replay_worker)
    assert (
        "historical_trace = load_historical_training_trace(a_seed, r_seed)"
        in source
    )
    assert "historical_trace=historical_trace" in source


def test_exact_endpoint_identity_is_diagnostic_not_gate():
    source = inspect.getsource(mod._run_replay_cell)
    assert "FINAL_A_REPLAY_MISMATCH" not in source
    assert "FINAL_B_REPLAY_MISMATCH" not in source
    assert "FINAL_A_SHA_MISMATCH" not in source
    assert "FINAL_B_SHA_MISMATCH" not in source
    assert "TEMPORAL_REPLAY_SCALAR_AUTHENTICATION_FAILED" in source
    assert "TEMPORAL_REPLAY_PARAMETER_AUTHENTICATION_FAILED" in source
    assert "TEMPORAL_REPLAY_OPERATOR_AUTHENTICATION_FAILED" in source

def test_replay_instrumentation_occurs_after_historical_sync():
    source = inspect.getsource(mod._run_replay_cell)
    assert source.index("optimizer.step()") < source.index("torch.cuda.synchronize()") < source.index("a_grad_cpu =") < source.index("public, private = snapshot_state(")
    assert "torch.linalg.vector_norm(a_grad).detach().cpu()" not in source
    assert "torch.linalg.vector_norm(b_grad).detach().cpu()" not in source


def test_replay_auth_recovery_mode_forbids_outputs():
    args = _args(replay_auth_recovery_only=True, implementation_freeze_commit="freeze", checkpoint="parent.pt", output_root="forbidden")
    with pytest.raises(mod.TemporalBirthError, match="RECOVERY_OUTPUT_FORBIDDEN"):
        mod.validate_args(args)


def test_replay_auth_recovery_worker_requires_fixed_id():
    args = _args(replay_auth_recovery_worker=True, implementation_freeze_commit="freeze", checkpoint="parent.pt", worker_id=3)
    with pytest.raises(mod.TemporalBirthError, match="RECOVERY_WORKER_ID_REQUIRED"):
        mod.validate_args(args)

def test_static_mode_forbids_runtime_fields():
    args = _args(
        static_verify_only=True,
        implementation_freeze_commit="freeze",
    )
    with pytest.raises(mod.TemporalBirthError, match="STATIC_RUNTIME_ARG"):
        mod.validate_args(args)


def test_runtime_requires_exact_implementation_freeze_argument():
    args = _args(
        run_replay_matrix=True,
        checkpoint="parent.pt",
        output_root="reports/reason_router_gen5_ainit_temporal_birth_replay_runs/x",
    )
    with pytest.raises(mod.TemporalBirthError, match="IMPLEMENTATION_FREEZE_REQUIRED"):
        mod.validate_args(args)


def test_authority_identity_and_collection_policy_are_frozen():
    assert mod.AUTHORITY_COMMIT == "20ae761dbff10ad70853b10910cbe12e51e0666a"
    assert (
        mod.REPLAY_AUTHENTICATION_CORRECTION_COMMIT
        == "341e2e59668ea9b575007ddf591424b831170577"
    )
    assert mod.GPU_TOPOLOGY == "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP"
    authority_text = (
        mod.ROOT / mod.AUTHORITY_PATH
    ).read_text(encoding="utf-8")
    assert "PREFLIGHT_COLLECTION=FORBIDDEN" in authority_text
    assert "FAILED_RUN_COLLECTION=FORBIDDEN" in authority_text
    assert "SUCCESSFUL_SCIENTIFIC_RUN_COLLECTION=REQUIRED" in authority_text

def test_phase_b_raw_control_identity_is_exactly_frozen():
    observed = [
        (
            row["index"],
            row["seed"],
            row["sha256_label"],
        )
        for row in (
            mod.phase_b_control_identity(index)
            for index in range(mod.PHASE_B_CONTROL_COUNT)
        )
    ]
    assert tuple(observed) == mod.PHASE_B_RAW_CONTROL_IDENTITY


def test_phase_b_signed_permutation_preserves_token_norms_exactly():
    residual = torch.arange(
        2 * 3 * 16,
        dtype=torch.float32,
    ).reshape(2, 3, 16)
    controlled = mod.phase_b_apply_signed_permutation(
        residual,
        control_index=0,
    )
    assert torch.equal(
        torch.sum(controlled * controlled, dim=-1),
        torch.sum(residual * residual, dim=-1),
    )


def test_phase_b_worker_sharding_matches_frozen_source_index_parity():
    assert mod._phase_b_worker_source_cells(0) == (
        (6201, 6201),
        (6201, 6203),
        (6202, 6202),
        (6203, 6201),
        (6203, 6203),
    )
    assert mod._phase_b_worker_source_cells(1) == (
        (6201, 6202),
        (6202, 6201),
        (6202, 6203),
        (6203, 6202),
    )
    assert len(mod._phase_b_worker_orientations(0)) == 10
    assert len(mod._phase_b_worker_orientations(1)) == 8


def test_phase_b_two_row_projection_exact_span_case():
    g0 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    g1 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    residual = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    visible, energy, gain = mod._phase_b_two_row_projection(
        grad_refute=g0,
        grad_support=g1,
        residual=residual,
    )
    assert torch.allclose(visible, residual)
    assert energy == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert gain == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_phase_b_gate_uses_only_frozen_thresholds():
    value = {
        "full_replay_max_abs": 0.0,
        "task_row_energy_enrichment": 5.0,
        "finite_intervention": {
            "centered_logits": {
                "R_visible": 0.60,
                "R_complement": 0.05,
                "R_interaction": 0.05,
            },
            "two_margins": {
                "R_visible": 1.40,
                "R_complement": 0.05,
                "R_interaction": 0.05,
            },
        },
    }
    assert mod._phase_b_gate(value)["pass"] is True
    value["task_row_energy_enrichment"] = 4.999
    assert mod._phase_b_gate(value)["pass"] is False


def test_phase_b_geometric_birth_is_first_strictly_positive_postupdate():
    def row(t, d):
        return {
            "t": t,
            "grouped": {
                "same_training_rng_different_a": {
                    "mean_squared_distance": d
                }
            },
        }
    geometry = [row(0, 0.0), row(1, 0.0), row(2, 1e-12)]
    assert mod.phase_b_geometric_birth_step(geometry) == 2


def test_phase_b_endpoint_tolerances_are_preregistered_and_not_gate_thresholds():
    assert mod.PHASE_B_ENDPOINT_TOLERANCE == {
        "normalized_residual_atol": 5.0e-4,
        "task_row_energy_atol": 5.0e-5,
        "control_task_row_energy_atol": 5.0e-6,
        "enrichment_atol": 5.0e-1,
        "effect_ratio_atol": 5.0e-3,
    }
    assert mod.PHASE_B_PROJECTOR_PINV_RTOL == 1.0e-12
    assert mod.PHASE_B_FULL_REPLAY_ATOL == 5.0e-5


def test_phase_b_interpretation_cases_are_bounded():
    assert mod._phase_b_interpretation(
        geometric_birth=1,
        functional_birth=1,
    ) == "IMMEDIATE_FIRST_UPDATE_BIRTH"
    assert mod._phase_b_interpretation(
        geometric_birth=1,
        functional_birth=3,
    ) == "GEOMETRY_FIRST_FUNCTION_LATER"
    assert mod._phase_b_interpretation(
        geometric_birth=2,
        functional_birth=3,
    ) == "DELAYED_GEOMETRIC_BIRTH"


def test_phase_b_parser_modes_are_present():
    parser = mod.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--phase-b-cuda-preflight-only" in option_strings
    assert "--run-phase-b" in option_strings
    assert "--phase-b-worker" in option_strings
    assert "--phase-b-step" in option_strings
    assert "--phase-b-compute-geometry" in option_strings


def test_phase_b_does_not_use_backward_or_optimizer_in_phase_b_functions():
    for fn in (
        mod.run_phase_b_cuda_preflight,
        mod.run_phase_b_worker,
        mod.run_phase_b_analysis,
    ):
        source = inspect.getsource(fn)
        assert ".backward(" not in source
        assert "torch.optim" not in source

def test_phase_b_git_show_text_forces_utf8_decode():
    source = inspect.getsource(mod._phase_b_git_show_text)
    assert "encoding=\"utf-8\"" in source
    assert "errors=\"strict\"" in source
    assert "subprocess.check_output" in source



def test_phase_b_static_contract_uses_exact_sha_wrapped_control_token():
    source = inspect.getsource(mod.validate_phase_b_static_contract)
    assert '`SHA256(\\"GEN5_INTERNAL_PRECURSOR_V1|raw_write|<control_index>\\")`' in source


def test_phase_b_internal_correction_is_authenticated_by_exact_blob():
    assert mod.INTERNAL_PRECURSOR_CORRECTION_BLOB == "f4a45513712c9a6380b309f0a8171fbeb499a925"
    source = inspect.getsource(mod.validate_phase_b_static_contract)
    assert "PHASE_B_PRECURSOR_CORRECTION_BLOB" in source
    assert "PHASE_B_PRECURSOR_CORRECTION_TOKEN" not in source

def test_phase_b_joint_analysis_runtime_neutralizes_and_restores_historical_edge_state():
    class Dummy:
        pass

    model = Dummy()
    original_edges = {"F_TO_P": 0.25, "F_TO_S": 0.5}
    model.gradient_ownership_mode = "edge_specific"
    model.gradient_ownership_lambda = None
    model.edge_gradient_lambdas = original_edges
    model.return_q_diagnostics = True

    with mod._phase_b_joint_analysis_runtime(model):
        assert model.gradient_ownership_mode == "joint"
        assert model.gradient_ownership_lambda is None
        assert model.edge_gradient_lambdas is None
        assert model.return_q_diagnostics is True

    assert model.gradient_ownership_mode == "edge_specific"
    assert model.gradient_ownership_lambda is None
    assert model.edge_gradient_lambdas is original_edges
    assert model.return_q_diagnostics is True


def test_phase_b_joint_analysis_runtime_restores_absent_attributes():
    class Dummy:
        pass

    model = Dummy()
    with mod._phase_b_joint_analysis_runtime(model):
        assert model.gradient_ownership_mode == "joint"
        assert model.gradient_ownership_lambda is None
        assert model.edge_gradient_lambdas is None
        assert model.return_q_diagnostics is True

    assert not hasattr(model, "gradient_ownership_mode")
    assert not hasattr(model, "gradient_ownership_lambda")
    assert not hasattr(model, "edge_gradient_lambdas")
    assert not hasattr(model, "return_q_diagnostics")


def test_phase_b_joint_forward_explicitly_neutralizes_edge_specific_arguments():
    source = inspect.getsource(mod._phase_b_joint_forward_from_hidden)
    assert 'gradient_ownership_mode="joint"' in source
    assert "gradient_ownership_lambda=None" in source
    assert "edge_gradient_lambdas=None" in source
    assert "with _phase_b_joint_analysis_runtime(model):" in source

def test_phase_b_projector_uses_frozen_float64_cpu_backend():
    assert mod.PHASE_B_PROJECTOR_BACKEND == "float64_cpu"
    helper = inspect.getsource(mod._phase_b_float64_cpu_flatten)
    projector = inspect.getsource(mod._phase_b_two_row_projection)
    control = inspect.getsource(mod._phase_b_control_energy_and_gain)
    assert '.to(device="cpu", dtype=torch.float64)' in helper
    assert "_phase_b_float64_cpu_flatten(grad_refute)" in projector
    assert "_phase_b_float64_cpu_flatten(grad_support)" in projector
    assert "_phase_b_float64_cpu_flatten(residual)" in projector
    assert "_phase_b_float64_cpu_flatten(grad_refute)" in control
    assert "_phase_b_float64_cpu_flatten(grad_support)" in control
    assert "_phase_b_float64_cpu_flatten(residual)" in control


def test_phase_b_float64_cpu_projection_preserves_output_surface():
    grad_refute = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    grad_support = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
    residual = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)

    visible, energy, gain = mod._phase_b_two_row_projection(
        grad_refute=grad_refute,
        grad_support=grad_support,
        residual=residual,
    )
    control_energy, control_gain = mod._phase_b_control_energy_and_gain(
        grad_refute=grad_refute,
        grad_support=grad_support,
        residual=residual,
    )

    assert visible.device == residual.device
    assert visible.dtype == residual.dtype
    assert torch.allclose(
        visible,
        torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float32),
    )
    assert energy == pytest.approx(25.0 / 86.0)
    assert gain == pytest.approx(5.0 / math.sqrt(86.0))
    assert control_energy == pytest.approx(energy)
    assert control_gain == pytest.approx(gain)


def test_phase_b_summary_and_provenance_bind_projector_backend():
    source = inspect.getsource(mod.run_phase_b_analysis)
    assert '"projector_backend": PHASE_B_PROJECTOR_BACKEND' in source
    assert '"projector_pinv_rtol": PHASE_B_PROJECTOR_PINV_RTOL' in source



# ---------------------------------------------------------------------------
# Temporal mechanism scan implementation pass 1
# ---------------------------------------------------------------------------

def test_temporal_mechanism_authority_identity_and_fixed_times():
    assert (
        mod.TEMPORAL_MECHANISM_AUTHORITY_COMMIT
        == "adad44c6fb304500c700b9ea59d727cefd71c2d9"
    )
    assert mod.TEMPORAL_MECHANISM_CRITICAL_TIMES == (
        1, 2, 4, 10, 11, 16, 17, 20
    )
    assert mod.TEMPORAL_MECHANISM_TASK_VISIBLE_TIMES == (1, 17, 20)
    assert mod.TEMPORAL_MECHANISM_T1_STATES == (
        "T0",
        "A_DECAY_ONLY",
        "B_UPDATE_ONLY",
        "FULL_T1",
    )
    assert mod.TEMPORAL_MECHANISM_INTERNAL_STAGES == (
        "raw_write",
        "recurrent_state",
        "c_readout_pre_gate",
        "gated_scan",
        "layer22_out_proj",
    )


def test_temporal_mechanism_authority_is_no_training_contract():
    text = (
        mod.ROOT / mod.TEMPORAL_MECHANISM_AUTHORITY_PATH
    ).read_text(encoding="utf-8")
    for token in (
        "TRAINING_ALLOWED=NO",
        "OPTIMIZER_CONSTRUCTION_ALLOWED=NO",
        "OPTIMIZER_STEP_ALLOWED=NO",
        "PARAMETER_UPDATE_ALLOWED=NO",
        "BACKWARD_ALLOWED=NO",
        "PARAMETER_GRADIENTS_ALLOWED=NO",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
    ):
        assert token in text


def test_temporal_mechanism_static_parser_mode_is_present():
    parser = mod.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--temporal-mechanism-static-verify-only" in options


def test_temporal_mechanism_static_mode_forbids_runtime_fields():
    args = _args(
        temporal_mechanism_static_verify_only=True,
        implementation_freeze_commit="forbidden",
    )
    with pytest.raises(
        mod.TemporalBirthError,
        match="TEMPORAL_MECHANISM_STATIC_RUNTIME_ARG",
    ):
        mod.validate_args(args)


def test_temporal_mechanism_centered_logits_are_row_centered():
    logits = torch.tensor(
        [[1.0, 2.0, 6.0], [3.0, -1.0, 4.0]],
        dtype=torch.float32,
    )
    centered = mod.temporal_mechanism_centered_logits(logits)
    assert torch.allclose(
        centered.mean(dim=-1),
        torch.zeros(2),
        atol=1e-7,
        rtol=0.0,
    )


def test_temporal_mechanism_margin_order_matches_authority():
    logits = torch.tensor([[7.0, 2.0, -1.0]], dtype=torch.float32)
    margins = mod.temporal_mechanism_margin_vector(logits)
    assert margins.tolist() == [[5.0, -3.0]]


def test_temporal_mechanism_static_verify_has_no_runtime_science_calls():
    source = inspect.getsource(mod.run_temporal_mechanism_static_verify)
    assert "torch.autograd.grad" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "_prepare_runtime(" not in source


def test_temporal_mechanism_implementation_scope_is_exactly_two_files():
    assert mod.TEMPORAL_MECHANISM_IMPLEMENTATION_PATHS == frozenset({
        "scripts/audit_reason_router_gen5_ainit_temporal_birth.py",
        "tests/test_reason_router_gen5_ainit_temporal_birth.py",
    })



# ---------------------------------------------------------------------------
# Temporal mechanism scan implementation pass 2
# ---------------------------------------------------------------------------

def test_temporal_mechanism_runtime_parser_modes_are_present():
    parser = mod.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--temporal-mechanism-cuda-preflight-only" in options
    assert "--run-temporal-mechanism" in options
    assert "--temporal-mechanism-worker" in options


def test_temporal_mechanism_geometry_pairs_are_exact_factor_controls():
    rows = mod.temporal_mechanism_geometry_pairs()
    assert len(rows) == 18
    assert sum(group == "A" for group, _, _ in rows) == 9
    assert sum(group == "R" for group, _, _ in rows) == 9
    assert len(mod.temporal_mechanism_worker_geometry_pairs(0)) == 9
    assert len(mod.temporal_mechanism_worker_geometry_pairs(1)) == 9
    assert (
        set(mod.temporal_mechanism_worker_geometry_pairs(0))
        | set(mod.temporal_mechanism_worker_geometry_pairs(1))
    ) == set(rows)
    assert not (
        set(mod.temporal_mechanism_worker_geometry_pairs(0))
        & set(mod.temporal_mechanism_worker_geometry_pairs(1))
    )


def test_temporal_mechanism_raw_control_matches_frozen_phase_b_identity():
    for index in range(mod.PHASE_B_CONTROL_COUNT):
        assert (
            mod.temporal_mechanism_stage_control_identity(
                "raw_write",
                index,
            )
            == mod.phase_b_control_identity(index)
        )


def test_temporal_mechanism_stage_control_is_deterministic_and_stage_specific():
    raw = mod.temporal_mechanism_stage_control_identity(
        "raw_write", 0
    )
    state = mod.temporal_mechanism_stage_control_identity(
        "recurrent_state", 0
    )
    assert raw == mod.temporal_mechanism_stage_control_identity(
        "raw_write", 0
    )
    assert raw["sha256_label"] != state["sha256_label"]
    assert raw["seed"] != state["seed"]


def test_temporal_mechanism_stage_control_preserves_token_norms():
    residual = torch.arange(
        2 * 3 * 16,
        dtype=torch.float32,
    ).reshape(2, 3, 16)
    controlled = mod.temporal_mechanism_apply_stage_control(
        residual,
        stage="gated_scan",
        control_index=3,
    )
    assert torch.equal(
        torch.sum(controlled * controlled, dim=-1),
        torch.sum(residual * residual, dim=-1),
    )


def test_temporal_mechanism_prediction_factor_disagreement_pure_a():
    predictions = torch.empty(
        (9, mod.DEV_ROWS),
        dtype=torch.long,
    )
    for index, (a_seed, _r_seed) in enumerate(
        mod.FULL_FACTORIAL_CELLS
    ):
        predictions[index].fill_(
            mod.FACTOR_SEEDS.index(a_seed)
        )
    result = mod.temporal_mechanism_prediction_factor_disagreement(
        predictions
    )
    assert (
        result[
            "same_training_rng_different_a_disagreement_sum"
        ]
        == 9 * mod.DEV_ROWS
    )
    assert (
        result[
            "same_a_different_training_rng_disagreement_sum"
        ]
        == 0
    )


def test_temporal_mechanism_runtime_modes_enforce_output_boundaries():
    with pytest.raises(
        mod.TemporalBirthError,
        match="TEMPORAL_MECHANISM_PREFLIGHT_OUTPUT_FORBIDDEN",
    ):
        mod.validate_args(
            _args(
                temporal_mechanism_cuda_preflight_only=True,
                implementation_freeze_commit="freeze",
                checkpoint="parent.pt",
                output_root="forbidden",
            )
        )

    with pytest.raises(
        mod.TemporalBirthError,
        match="TEMPORAL_MECHANISM_OUTPUT_REQUIRED",
    ):
        mod.validate_args(
            _args(
                run_temporal_mechanism=True,
                implementation_freeze_commit="freeze",
                checkpoint="parent.pt",
            )
        )

    with pytest.raises(
        mod.TemporalBirthError,
        match="TEMPORAL_MECHANISM_WORKER_SCRATCH_REQUIRED",
    ):
        mod.validate_args(
            _args(
                temporal_mechanism_worker=True,
                implementation_freeze_commit="freeze",
                checkpoint="parent.pt",
                worker_id=0,
            )
        )


def test_temporal_mechanism_scientific_runtime_has_no_training_backward_optimizer():
    for fn in (
        mod.run_temporal_mechanism_cuda_preflight,
        mod.run_temporal_mechanism_worker,
        mod.run_temporal_mechanism_analysis,
    ):
        source = inspect.getsource(fn)
        assert ".backward(" not in source
        assert "torch.optim" not in source
        assert "optimizer.step(" not in source


def test_temporal_mechanism_stage_order_is_used_without_posthoc_extension():
    worker_source = inspect.getsource(
        mod.run_temporal_mechanism_worker
    )
    assert (
        "for t in TEMPORAL_MECHANISM_CRITICAL_TIMES"
        in worker_source
    )
    assert (
        "for t in TEMPORAL_MECHANISM_TASK_VISIBLE_TIMES"
        in worker_source
    )
    assert (
        "for stage in TEMPORAL_MECHANISM_INTERNAL_STAGES"
        in worker_source
    )


def test_temporal_mechanism_output_contract_is_exact():
    source = inspect.getsource(
        mod.run_temporal_mechanism_analysis
    )
    assert "temporal_mechanism_summary.json" in source
    assert "temporal_behavioral_coordinates.pt" in source
    assert "temporal_internal_stage_metrics.pt" in source
    assert "run_provenance.json" in source
    assert (
        "GEN5_AINIT_TEMPORAL_MECHANISM_SCAN_PASS"
        in source
    )


def test_temporal_mechanism_endpoint_authentication_uses_frozen_precursor():
    source = inspect.getsource(
        mod._temporal_mechanism_merge_task_visible
    )
    assert (
        'precursor_summary["grouped"]['
        in source
    )
    assert (
        '"same_training_rng_different_a_init"'
        in source
    )
    assert (
        "TEMPORAL_MECHANISM_T20_PRECURSOR_AUTHENTICATION_FAILED"
        in source
    )

def test_temporal_mechanism_frozen_t1_factor_aggregate_is_144_0():
    payload = json.loads(
        (
            mod.ROOT / mod.TEMPORAL_MECHANISM_BEHAVIORAL_CELL_METRICS_PATH
        ).read_text(encoding="utf-8")
    )
    predictions = torch.empty(
        (9, mod.DEV_ROWS),
        dtype=torch.long,
    )
    for cell_index, cell in enumerate(mod.FULL_FACTORIAL_CELLS):
        name = mod.cell_name(*cell)
        predictions[cell_index] = torch.tensor(
            payload["cells"][name]["steps"][1]["predictions"],
            dtype=torch.long,
        )
    result = mod.temporal_mechanism_prediction_factor_disagreement(
        predictions
    )
    assert mod.TEMPORAL_MECHANISM_T1_A_DISAGREEMENT_EXPECTED == 144
    assert mod.TEMPORAL_MECHANISM_T1_R_DISAGREEMENT_EXPECTED == 0
    assert (
        result["same_training_rng_different_a_disagreement_sum"]
        == mod.TEMPORAL_MECHANISM_T1_A_DISAGREEMENT_EXPECTED
    )
    assert (
        result["same_a_different_training_rng_disagreement_sum"]
        == mod.TEMPORAL_MECHANISM_T1_R_DISAGREEMENT_EXPECTED
    )


def test_temporal_mechanism_valid_stage_tensor_masks_padding():
    value = torch.arange(
        2 * 3 * 4,
        dtype=torch.float32,
    ).reshape(2, 3, 4)
    attention_mask = torch.tensor(
        [[1, 1, 0], [1, 0, 0]],
        dtype=torch.long,
    )
    masked = mod._temporal_mechanism_valid_stage_tensor(
        value,
        attention_mask,
    )
    valid = attention_mask.bool()
    assert torch.equal(masked[valid], value[valid])
    assert torch.count_nonzero(masked[~valid]).item() == 0


def test_temporal_mechanism_task_visible_accumulator_uses_valid_token_mask():
    source = inspect.getsource(
        mod._temporal_mechanism_accumulate_stage_orientation
    )
    assert "attention_mask" in source
    assert "_temporal_mechanism_valid_stage_tensor" in source
    worker_source = inspect.getsource(
        mod.run_temporal_mechanism_worker
    )
    assert (
        'attention_mask=batch_features["attention_mask"]'
        in worker_source
    )
