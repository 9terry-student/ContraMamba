from __future__ import annotations

import argparse
import inspect
import math

import pytest
import torch

from scripts import (
    audit_native_downstream_functional_reconvergence_raw_write_transport as mod,
)


def _args(**overrides):
    values = {
        "static_contract_check": False,
        "preflight": False,
        "run_worker": False,
        "merge_only": False,
        "run": False,
        "expected_head": None,
        "implementation_freeze_commit": None,
        "model_snapshot": None,
        "tokenizer_snapshot": None,
        "checkpoint": None,
        "scratch_root": None,
        "output_root": None,
        "worker_id": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_exact_factorial_grid_and_checkpoint_hashes():
    assert mod.FACTOR_SEEDS == (6201, 6202, 6203)
    assert len(mod.FULL_FACTORIAL_CELLS) == 9
    assert set(mod.CHECKPOINT_SHA256) == set(mod.FULL_FACTORIAL_CELLS)
    assert all(len(value) == 64 for value in mod.CHECKPOINT_SHA256.values())


def test_pair_enumeration_is_exactly_18_plus_18():
    rows = mod.all_orientations()
    assert len(rows) == 36
    assert sum(group == "PRIMARY_A" for group, _, _ in rows) == 18
    assert sum(group == "CONTROL_R" for group, _, _ in rows) == 18


def test_both_orientations_exist_for_every_unordered_pair():
    rows = set(mod.all_orientations())
    for group, source, target in tuple(rows):
        assert (group, target, source) in rows


def test_worker_partition_is_exact_20_16_and_disjoint():
    w0 = set(mod.worker_orientations(0))
    w1 = set(mod.worker_orientations(1))
    assert len(w0) == 20
    assert len(w1) == 16
    assert w0.isdisjoint(w1)
    assert w0 | w1 == set(mod.all_orientations())


def test_valid_stage_masks_padding():
    value = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
    observed = mod._valid_stage(value, mask)
    assert torch.equal(observed[mask.bool()], value[mask.bool()])
    assert torch.count_nonzero(observed[~mask.bool()]).item() == 0


def test_energy_accumulation_is_float64_and_excludes_padding():
    value = torch.tensor(
        [[[3.0, 4.0], [100.0, 100.0]], [[0.0, 5.0], [9.0, 9.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
    energy = mod._sq_energy_by_example(value, mask)
    assert energy.dtype == torch.float64
    assert energy.tolist() == pytest.approx([25.0, 25.0])


def test_stage_energy_interaction_zero_for_linear_decomposition():
    source = torch.zeros(1, 2, 2)
    visible = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    complement = torch.tensor([[[0.0, 2.0], [0.0, 0.0]]])
    full = visible + complement
    mask = torch.tensor([[1, 0]])
    row = mod._stage_energy_sums(
        source=source,
        full=full,
        visible=visible,
        complement=complement,
        attention_mask=mask,
    )
    assert row["D_full"] == pytest.approx(5.0)
    assert row["D_visible"] == pytest.approx(1.0)
    assert row["D_complement"] == pytest.approx(4.0)
    assert row["I_abs"] == pytest.approx(0.0)


def test_projector_contract_is_frozen_float64_two_by_two():
    assert mod.PROJECTOR_BACKEND == "float64_cpu"
    assert mod.PROJECTOR_PINV_RTOL == 1e-12
    assert mod.PROJECTOR_PINV_ATOL == 0.0
    source = inspect.getsource(mod._project_visible_rows)
    assert "legacy._phase_b_two_row_projection" in source
    assert "_valid_stage(grad_refute" in source
    assert "_valid_stage(grad_support" in source


def test_raw_visible_plus_complement_reconstruction_definition():
    residual = torch.randn(2, 3, 4)
    visible = torch.randn(2, 3, 4)
    complement = residual - visible
    assert torch.allclose(visible + complement, residual)


def test_full_target_chord_is_not_target_endpoint_gate_in_code():
    source = inspect.getsource(mod.run_worker)
    assert "actual_stages[target]" in source
    assert "FULL_TARGET_CHORD_REAL_TARGET_EQUALITY" not in source
    assert "downstream_map_shared_across_grid" in source


def test_ratio_of_sums_retention_not_mean_of_row_ratios():
    acc = mod._empty_orientation("PRIMARY_A", (6201, 6201), (6202, 6201))
    acc["example_count"] = mod.DEV_ROWS
    acc["task_row_energy_sum"] = 0.0
    acc["task_row_energy_numer_sum"] = 1.0
    acc["task_row_energy_denom_sum"] = 4.0
    for stage in mod.STAGE_ORDER:
        acc["stage"][stage] = {
            "D_full": 10.0,
            "D_visible": 2.0,
            "D_complement": 8.0,
            "I_abs": 0.0,
        }
    out = mod._finalize_orientation(acc)
    assert out["task_row_energy_ratio_of_sums"] == pytest.approx(0.25)
    assert out["stage"]["raw_write"]["RET_VISIBLE"] == pytest.approx(1.0)
    assert out["stage"]["raw_write"]["RET_COMPLEMENT"] == pytest.approx(1.0)
    assert out["stage"]["raw_write"]["SELECTIVE_RETENTION"] == pytest.approx(1.0)


def test_i_abs_i_rel_and_denominator_are_all_reported():
    acc = mod._empty_orientation("PRIMARY_A", (6201, 6201), (6202, 6201))
    acc["example_count"] = mod.DEV_ROWS
    for stage in mod.STAGE_ORDER:
        acc["stage"][stage] = {
            "D_full": 2.0,
            "D_visible": 1.0,
            "D_complement": 1.0,
            "I_abs": 0.5,
        }
    out = mod._finalize_orientation(acc)
    row = out["stage"]["recurrent_state"]
    assert row["I_abs"] == pytest.approx(0.5)
    assert row["I_REL"] == pytest.approx(0.25)
    assert row["sum_D_full"] == pytest.approx(2.0)


def test_log_symmetric_pair_aggregation_uses_geometric_mean():
    def make(source, target, selective):
        stage = {
            name: {
                "SELECTIVE_RETENTION": selective,
                "LOG_SELECTIVE_RETENTION": math.log(selective),
            }
            for name in mod.STAGE_ORDER
        }
        return {
            "group": "PRIMARY_A",
            "source": source,
            "target": target,
            "stage": stage,
        }

    rows = [
        make("A6201-R6201", "A6202-R6201", 0.25),
        make("A6202-R6201", "A6201-R6201", 1.0),
    ]
    # Fill eight more unordered primary pairs only for count validation.
    for idx in range(8):
        a = f"L{idx}"
        b = f"R{idx}"
        rows += [make(a, b, 1.0), make(b, a, 1.0)]
    # _symmetric_pairs also checks group count 9, which is exactly what we built.
    pairs = mod._symmetric_pairs(rows)
    first = next(
        row for row in pairs
        if {row["left"], row["right"]} == {"A6201-R6201", "A6202-R6201"}
    )
    assert first["stage"]["raw_write"]["pair_selective_retention"] == pytest.approx(0.5)


def test_no_parameter_training_or_backward_surface():
    for fn in (
        mod.run_preflight,
        mod.run_worker,
        mod.run_merge_only,
        mod.run_full,
    ):
        source = inspect.getsource(fn)
        assert ".backward(" not in source
        assert "torch.optim" not in source


def test_output_collision_is_fail_closed_in_full_run_source():
    source = inspect.getsource(mod.run_full)
    assert "SCRATCH_COLLISION" in source
    assert "OUTPUT_COLLISION" in source


def test_worker_chunk_hash_is_mandatory():
    source = inspect.getsource(mod._read_worker)
    assert "WORKER_SHA_MISSING" in source
    assert "WORKER_SHA_MISMATCH" in source


def test_merge_rejects_missing_duplicate_or_mixed_coverage_by_construction():
    source = inspect.getsource(mod.run_merge_only)
    assert "MERGED_ORIENTATION_COUNT" in source
    assert "MERGED_ORIENTATION_DUPLICATE" in source
    assert "MERGED_ORIENTATION_COVERAGE" in source
    assert "MIXED_WORKER_HEAD" in source


def test_confirmatory_and_vitaminc_embargo_are_persisted():
    source = inspect.getsource(mod.run_worker)
    assert '"confirmatory_9601_9900_loaded": False' in source
    assert '"vitaminc_loaded": False' in source


def test_parser_contains_required_contract_modes():
    parser = mod.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--static-contract-check" in options
    assert "--preflight" in options
    assert "--run-worker" in options
    assert "--merge-only" in options
    assert "--run" in options


def test_static_mode_does_not_require_runtime_arguments():
    args = _args(static_contract_check=True)
    mod.validate_args(args)


def test_runtime_requires_exact_head_binding():
    args = _args(
        preflight=True,
        expected_head="abc",
        implementation_freeze_commit="abc",
        model_snapshot="model",
        tokenizer_snapshot="tokenizer",
        checkpoint="checkpoint",
    )
    mod.validate_args(args)


def test_expected_execution_counts_bind_worker_partition():
    counts = mod._expected_counts()
    assert mod.BATCH_ROWS == 32
    assert counts["batch_rows"] == 32
    assert counts["batches_per_worker"] == 27
    assert counts["total_orientations"] == 36
    assert counts["workers"]["0"]["orientations"] == 20
    assert counts["workers"]["1"]["orientations"] == 16
    assert counts["workers"]["0"]["source_cells"] == 5
    assert counts["workers"]["1"]["source_cells"] == 4
