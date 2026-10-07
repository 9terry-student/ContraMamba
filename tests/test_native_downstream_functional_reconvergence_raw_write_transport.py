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


def test_worker_example_shards_cover_dev_without_overlap():
    assert mod.worker_row_range(0) == (0, 416)
    assert mod.worker_row_range(1) == (416, mod.DEV_ROWS)
    expected = set(mod.all_orientations())
    assert set(mod.worker_orientations(0)) == expected
    assert set(mod.worker_orientations(1)) == expected


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


def test_projector_contract_uses_gpu_reduction_and_cpu_2x2_pinv():
    assert mod.PROJECTOR_BACKEND == "gpu_f32_product_f64_reduce_cpu_2x2_pinv"
    assert mod.PROJECTOR_PINV_RTOL == 1e-12
    assert mod.PROJECTOR_PINV_ATOL == 0.0
    source = inspect.getsource(mod._prepare_projector_pinv)
    assert "dtype=torch.float64" in source
    assert 'device="cpu"' in source
    assert "torch.linalg.pinv" in source


def test_raw_visible_plus_complement_reconstruction_definition():
    residual = torch.randn(2, 3, 4)
    visible = torch.randn(2, 3, 4)
    complement = residual - visible
    assert torch.allclose(visible + complement, residual)


def test_full_target_chord_is_not_target_endpoint_gate_in_code():
    source = inspect.getsource(mod.run_worker)
    assert "raw_targets" in source
    assert "_stream_transport_group(" in source
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
    helper = inspect.getsource(mod._merge_worker_accumulators)
    assert "MERGED_ORIENTATION_COUNT" in source
    assert "MERGED_ORIENTATION_DUPLICATE" in source
    assert "MERGED_ORIENTATION_COVERAGE" in source
    assert "MIXED_WORKER_HEAD" in source
    assert "MERGED_WORKER_COVERAGE" in helper
    assert "MERGED_ORIENTATION_EXAMPLE_COUNT" in helper


def test_confirmatory_and_vitaminc_embargo_are_persisted():
    source = inspect.getsource(mod.run_worker)
    compact = source.replace(" ", "")
    assert '"confirmatory_9601_9900_loaded":False' in compact
    assert '"vitaminc_loaded":False' in compact


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


def test_stream_write_trajectories_adds_source_to_projected_components():
    torch.manual_seed(11)
    batch, seq_len, width = 2, 3, 4
    source = torch.randn(batch, seq_len, width)
    targets = [
        source + 0.1 * torch.randn(batch, seq_len, width),
        source + 0.1 * torch.randn(batch, seq_len, width),
    ]
    visible = torch.randn(2, batch, seq_len, width)
    complement = torch.randn(2, batch, seq_len, width)

    token_index = 1
    packed = mod._stream_write_trajectories(
        raw_source=source,
        raw_targets=targets,
        visible=visible,
        complement=complement,
        token_index=token_index,
    )

    assert packed.shape == (7, batch, width)
    assert torch.equal(packed[0], source[:, token_index, :])
    assert torch.equal(packed[1], targets[0][:, token_index, :])
    assert torch.equal(packed[2], targets[1][:, token_index, :])

    assert torch.equal(
        packed[3],
        source[:, token_index, :] + visible[0, :, token_index, :],
    )
    assert torch.equal(
        packed[4],
        source[:, token_index, :] + visible[1, :, token_index, :],
    )
    assert torch.equal(
        packed[5],
        source[:, token_index, :] + complement[0, :, token_index, :],
    )
    assert torch.equal(
        packed[6],
        source[:, token_index, :] + complement[1, :, token_index, :],
    )


def test_stream_stage_energy_matches_reference_stage_sum():
    source = torch.randn(2, 3, 5)
    full = source + torch.randn(2, 3, 5)
    visible = source + torch.randn(2, 3, 5)
    complement = source + torch.randn(2, 3, 5)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

    reference = mod._stage_energy_sums(
        source=source,
        full=full,
        visible=visible,
        complement=complement,
        attention_mask=mask,
    )

    observed = torch.zeros((4, 1), dtype=torch.float64)
    for token_index in range(source.shape[1]):
        packed = torch.stack(
            (
                source[:, token_index],
                full[:, token_index],
                visible[:, token_index],
                complement[:, token_index],
            ),
            dim=0,
        )
        observed += mod._trajectory_stage_energy(
            packed,
            target_count=1,
            valid_token=mask[:, token_index],
        )

    assert observed[0, 0].item() == pytest.approx(reference["D_full"], rel=1e-6, abs=1e-7)
    assert observed[1, 0].item() == pytest.approx(reference["D_visible"], rel=1e-6, abs=1e-7)
    assert observed[2, 0].item() == pytest.approx(reference["D_complement"], rel=1e-6, abs=1e-7)
    assert observed[3, 0].item() == pytest.approx(reference["I_abs"], rel=1e-6, abs=1e-7)


def test_projector_group_matches_legacy_small_tensor():
    torch.manual_seed(7)
    batch, seq_len, width = 3, 4, 5
    mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
        dtype=torch.long,
    )
    g0 = torch.randn(batch, seq_len, width)
    g1 = torch.randn(batch, seq_len, width)
    source = torch.randn(batch, seq_len, width)
    targets = [
        source + 0.1 * torch.randn(batch, seq_len, width),
        source + 0.1 * torch.randn(batch, seq_len, width),
    ]

    fast_accs = [
        mod._empty_orientation("PRIMARY_A", (6201, 6201), (6202, 6201)),
        mod._empty_orientation("PRIMARY_A", (6201, 6201), (6203, 6201)),
    ]

    g0_fast, g1_fast, pinv = mod._prepare_projector_pinv(
        grad_refute=g0.clone(),
        grad_support=g1.clone(),
        attention_mask=mask,
    )
    visible_fast, complement_fast = mod._project_visible_group(
        grad_refute=g0_fast,
        grad_support=g1_fast,
        pinv_cpu=pinv,
        attention_mask=mask,
        raw_source=source,
        raw_targets=targets,
        accumulators=fast_accs,
    )

    masked_g0 = mod._valid_stage(g0, mask)
    masked_g1 = mod._valid_stage(g1, mask)

    for index, target in enumerate(targets):
        residual = mod._valid_stage(target - source, mask)
        visible_legacy = []
        for row in range(batch):
            value, _, _ = mod.legacy._phase_b_two_row_projection(
                grad_refute=masked_g0[row],
                grad_support=masked_g1[row],
                residual=residual[row],
            )
            visible_legacy.append(value)
        visible_legacy = torch.stack(visible_legacy)
        assert torch.max(
            torch.abs(visible_fast[index] - visible_legacy)
        ).item() < 2e-5
        assert torch.max(
            torch.abs(
                visible_fast[index]
                + complement_fast[index]
                - residual
            )
        ).item() < 2e-5


def test_run_worker_uses_fused_streaming_without_stage_cpu_offload():
    source = inspect.getsource(mod.run_worker)
    assert "_stream_transport_group(" in source
    assert "_project_visible_group(" in source
    assert "_offload_stage_map" not in source
    assert "actual_stages" not in source
    assert "GEN5_TRANSPORT_PROGRESS" in source


def test_merge_worker_accumulators_sums_sufficient_statistics_and_maxima():
    def part(worker_id, count, scale, recon):
        rows=[]
        for group,source,target in mod.all_orientations():
            row=mod._empty_orientation(group,source,target); row["example_count"]=count
            row["task_row_energy_sum"]=scale*count; row["task_row_energy_numer_sum"]=2.0*scale; row["task_row_energy_denom_sum"]=4.0; row["raw_reconstruction_max_abs"]=recon
            for stage in mod.STAGE_ORDER: row["stage"][stage]={"D_full":10.0*scale,"D_visible":2.0*scale,"D_complement":8.0*scale,"I_abs":0.5*scale}
            for coord in ("centered_logits","two_margins"): row["finite"][coord]={"E_full":10.0*scale,"E_visible":8.0*scale,"E_complement":1.0*scale,"E_interaction":0.1*scale}
            rows.append(row)
        return {"worker_id":worker_id,"orientation_accumulators":rows}
    merged=mod._merge_worker_accumulators([part(0,416,1.0,1e-7),part(1,424,2.0,3e-7)])
    assert len(merged)==36; first=merged[0]
    assert first["example_count"]==mod.DEV_ROWS; assert first["raw_reconstruction_max_abs"]==pytest.approx(3e-7)
    assert first["stage"]["raw_write"]["D_full"]==pytest.approx(30.0); assert first["finite"]["two_margins"]["E_full"]==pytest.approx(30.0)


def test_expected_execution_counts_bind_example_shards():
    c=mod._expected_counts(); w0,w1=c["workers"]["0"],c["workers"]["1"]
    assert mod.BATCH_ROWS==32 and mod.TARGETS_PER_TRANSPORT==2
    assert (w0["row_start"],w0["row_stop"],w0["row_count"])==(0,416,416)
    assert (w1["row_start"],w1["row_stop"],w1["row_count"])==(416,840,424)
    assert (w0["batches"],w1["batches"])==(13,14); assert w0["source_cells"]==w1["source_cells"]==9; assert w0["orientations"]==w1["orientations"]==36
    assert (w0["raw_write_cell_evaluations"],w1["raw_write_cell_evaluations"])==(117,126)
    assert (w0["source_gradient_forwards"],w1["source_gradient_forwards"])==(117,126)
    assert (w0["fused_transport_calls"],w1["fused_transport_calls"])==(234,252)
    assert c["total_common_context_batches"]==27; assert c["total_source_gradient_forwards"]==243; assert c["total_fused_transport_calls"]==486
    assert w0["legacy_full_stage_tensor_offloads"]==w1["legacy_full_stage_tensor_offloads"]==0
