#!/usr/bin/env python3
"""Stage C CPU-mock-only, single-group frozen/v10 callable seam.

No model loader, filesystem writes, subprocesses, or CUDA entrypoint. Existing
frozen worker, Stage A/B, V1/V2 and hybrid modules remain immutable. The seam
can be exercised with injected mocks in this implementation phase only.
"""
from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

BASE = "6683d4c0dc9e742f5ac503fe4851e1c5218b99d8"
SCHEMA = "GEN5_V10_BOUNDED_RUNTIME_BRIDGE_CANDIDATE_V1"
ENGINE_FROZEN = "FROZEN_FFT_V1"
ENGINE_V10 = "V10_HYBRID_CANDIDATE"
GUARD_LABELS = (
    "frozen_original_full_fine_stats", "frozen_coarse_replay",
    "frozen_fine_parent_replay", "v10_coarse_replay",
    "v10_fine_parent_replay", "raw_reconstruction_auth",
    "visible_raw_energy_internal_replay", "complement_raw_energy_internal_replay",
)


class StageCBlocked(ValueError):
    pass


def demand(cond: bool, why: str) -> None:
    if not cond:
        raise StageCBlocked("GEN5_V10_STAGE_C_BRIDGE_BLOCKED:" + why)


def _cpu_only(tensors: Mapping[str, Any]) -> None:
    import torch
    for key, tensor in tensors.items():
        demand(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu",
               "CPU_MOCK_ONLY:" + key)


def flat_inputs(component: Any, log_a: Any, survival: Any,
                attention_mask: Any, role_masks: Mapping[str, Any],
                fine_masks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as stage_a
    flat = stage_a._flat_tensors(
        component=component, log_a=log_a, survival=survival,
        attention_mask=attention_mask, role_masks=role_masks, fine_masks=fine_masks,
    )
    demand(set(flat) == set(stage_a.TENSOR_KEYS), "SEVENTEEN_KEYS")
    _cpu_only(flat)
    return flat


def capture_phase(tensors: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """Stage A-compatible C-order value digests without exposing a GPU mode."""
    from scripts import gen5_v10_bounded_worker_export_candidate as b
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as a
    demand(phase in ("PRE_FROZEN", "POST_FROZEN", "POST_V10"), "CAPTURE_PHASE")
    demand(set(tensors) == set(a.TENSOR_KEYS), "CAPTURE_KEYS")
    _cpu_only(tensors)
    result = {}
    for key in a.TENSOR_KEYS:
        row = b.capture_value_digest(tensors[key], phase=phase)
        demand(row["source_device"] == "cpu", "CAPTURE_DEVICE")
        result[key] = row
    return result


def compare_captures(pre: Mapping[str, Any], post: Mapping[str, Any]) -> None:
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as stage_a
    demand(set(pre) == set(post) == set(stage_a.TENSOR_KEYS), "CAPTURE_SET")
    for key in stage_a.TENSOR_KEYS:
        lhs, rhs = pre[key], post[key]
        for field in ("content_sha256", "byte_count", "shape", "dtype", "source_device"):
            demand(lhs[field] == rhs[field], f"INPUT_MUTATION:{key}:{field}")


def bounded_selector(frozen: Any, row_ids: list[str]) -> dict[str, Any]:
    """Resolve original frozen enumeration before *any* model-side work."""
    from scripts import gen5_v10_bounded_worker_export_candidate as b
    source_cells = tuple(frozen.FULL_FACTORIAL_CELLS)
    orientations = tuple(frozen.all_orientations())
    source = source_cells[0]
    group = tuple(row[2] for row in orientations if row[1] == source)[:2]
    demand(frozen.TARGETS_PER_TRANSPORT == 2, "TARGET_GROUP_SIZE")
    demand(tuple(frozen.worker_row_range(1)) == (416, 840), "WORKER_RANGE")
    return b.choose_bounded_locus(
        worker_id=1, batch_index=13, row_start=800, row_stop=832,
        row_ids=row_ids, source_cells=source_cells, orientations=orientations,
        selected_source=source, selected_targets=group, target_group_index=0,
    )


def _post_original_reducer(frozen: Any, coarse_rows: list[dict[str, Any]],
                           gap: dict[str, Any], half_span: float, component: Any) -> list[dict[str, Any]]:
    """Copy the frozen post-reducer reconstruction, using its real _replay_vector.

    Frozen `_component_fine_pair_stats` recomputes coarse once and uses a
    float64 reconstructed parent and frozen relative replay for each component.
    """
    import torch
    demand(type(coarse_rows) is list and len(coarse_rows) == 4, "COARSE_K4")
    demand(isinstance(half_span, (int, float)) and math.isfinite(half_span), "HALF_SPAN")
    demand(type(gap) is dict and set(gap) == set(frozen.PARENT_PAIR_NAMES), "PARENT_KEYS")
    rows = []
    for k in range(4):
        rec = dict(coarse_rows[k]); fine_out = {}; max_abs = max_rel = 0.0
        for parent in frozen.PARENT_PAIR_NAMES:
            demand(type(gap[parent]) is dict and
                   set(gap[parent]) == set(frozen.FINE_PAIR_NAMES), "FINE_KEYS")
            reconstructed = torch.zeros(
                frozen.p3a.MAX_LENGTH, device=component.device, dtype=torch.float64
            )
            parent_out = {}
            for name in frozen.FINE_PAIR_NAMES:
                v = gap[parent][name]
                demand(isinstance(v, torch.Tensor) and
                       tuple(v.shape) == (4, frozen.p3a.MAX_LENGTH) and
                       v.dtype == torch.float64 and v.device == component.device,
                       "FINE_VECTOR_SHAPE_DTYPE")
                demand(bool(torch.all(torch.isfinite(v)).item()) and
                       bool(torch.all(v[:, 0] == 0).item()), "FINE_VECTOR_FINITE_ZERO")
                reconstructed += v[k]
                parent_out[name] = [float(x) for x in v[k, 1:].detach().cpu().tolist()]
            ref = [0.0] + [float(x) for x in rec["region_pair_C_by_gap"][parent]]
            expected = torch.tensor(ref, device=component.device, dtype=torch.float64)
            s = float(rec["S"])
            demand(math.isfinite(s) and s > 0.0, "SELF_ENERGY")
            observed_h = [float(x) / s for x in reconstructed[1:].detach().cpu().tolist()]
            expected_h = [float(x) / s for x in expected[1:].detach().cpu().tolist()]
            demand(all(math.isfinite(x) for x in observed_h + expected_h), "H_NONFINITE")
            replay = frozen.coarse._replay_vector(
                name=f"FINE_PARENT_H_RECON_{parent}_{k}",
                observed=observed_h, expected=expected_h,
            )
            demand(replay["pass"] is True, "FROZEN_REPLAY_FAILED")
            max_abs = max(max_abs, float(replay["max_abs_error"]))
            max_rel = max(max_rel, float(replay["max_rel_error"]))
            fine_out[parent] = parent_out
        rec["fine_pair_C_by_gap"] = fine_out
        rec["fine_parent_reconstruction_abs_max"] = max_abs
        rec["fine_parent_reconstruction_rel_max"] = max_rel
        rec["fine_pair_fft_half_log_span_max"] = float(half_span)
        rows.append(rec)
    return rows


def paired_cpu_seam(*, frozen: Any, hybrid: Any, component: Any, log_a: Any,
                    survival: Any, attention_mask: Any, role_masks: Any,
                    fine_masks: Any, shape: Any, selector: Mapping[str, Any],
                    source_row_ids: list[str]) -> dict[str, Any]:
    """CPU synthetic seam only, given inputs already produced by injected mocks.

    The actual original full frozen outer stats are called first. V10 reuses
    frozen coarse + original parent replay. No input or guard is mutated.
    """
    import torch
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as stage_a
    demand(dict(selector) == bounded_selector(frozen, source_row_ids), "SELECTOR_DRIFT")
    demand(isinstance(component, torch.Tensor) and component.ndim == 4 and
           component.shape[0] == 4 and component.shape[1] == 32 and
           component.shape[2] == 128 and component.shape[3] != 24576,
           "CPU_SYNTHETIC_GEOMETRY_ONLY")
    inp = dict(component=component, log_a=log_a, survival=survival,
               attention_mask=attention_mask, role_masks=role_masks,
               fine_masks=fine_masks)
    flat = flat_inputs(**inp)
    manifest = stage_a.manifest_for_cpu_inputs(**inp, shape=shape,
                                               synthetic_cpu_geometry=True)
    pre = capture_phase(flat, "PRE_FROZEN")
    with torch.no_grad():
        original = frozen._component_fine_pair_stats(**inp, shape=shape)
        demand(type(original) is list and len(original) == 4, "FROZEN_K4")
        after_frozen = capture_phase(flat, "POST_FROZEN")
        compare_captures(pre, after_frozen)
        coarse_rows = frozen.coarse._component_region_pair_stats(
            component=component, log_a=log_a, survival=survival,
            attention_mask=attention_mask, role_masks=role_masks, shape=shape,
        )
        view = SimpleNamespace(
            p3a=frozen.p3a, PARENT_PAIRS=frozen.PARENT_PAIRS,
            PARENT_PAIR_NAMES=frozen.PARENT_PAIR_NAMES,
            FINE_CLASSES=frozen.FINE_CLASSES, FINE_PAIR_NAMES=frozen.FINE_PAIR_NAMES,
            FAST_FLOAT32_HALF_SPAN_LIMIT=frozen.FAST_FLOAT32_HALF_SPAN_LIMIT,
            FFT_CHUNK_WIDTH=4,
            _fine_pair_gap_interference_fft=frozen._fine_pair_gap_interference_fft,
        )
        fine_gap, span, dispatch = hybrid.hybrid_fine_pair_gap(
            view, component=component, log_a=log_a, survival=survival,
            attention_mask=attention_mask, fine_masks=fine_masks, shape=shape,
            synthetic_cpu_geometry=True,
        )
        after_v10 = capture_phase(flat, "POST_V10")
        compare_captures(pre, after_v10)
        candidate = _post_original_reducer(frozen, coarse_rows, fine_gap, span, component)
    demand(len(candidate) == len(original) == 4, "ENGINE_K4")
    # Independent V3 row replay and exact schema inspect both engine outputs.
    # This does NOT authenticate source, upstream guards, or any GPU execution.
    from scripts import gen5_v10_e0e3_v3_runtime_verifier_candidate as v3
    for engine, records in (("frozen", original), ("v10", candidate)):
        for index, record in enumerate(records):
            v3.validate_record(record, f"{engine}[{index}]")
    return {
        "selector": dict(selector), "frozen_records": original,
        "v10_records": candidate, "v10_dispatch": dispatch,
        "input_manifest": manifest,
        "input_capture": {"pre_frozen": pre, "post_frozen": after_frozen,
                          "post_v10": after_v10},
        "guard_coverage": "FINE_PARENT_AND_COARSE_ONLY_UPSTREAM_NOT_ATTESTED",
        "V10_E0_PROVENANCE": "NOT_EVALUATED",
        "V10_E1_FROZEN_GUARDS": "NOT_EVALUATED",
        "V10_E2_PAYLOAD": "NOT_EVALUATED",
        "V10_E3_DIAGNOSTICS": "NOT_EVALUATED",
        "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED", "PRODUCTION": "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-contract-check", action="store_true", required=True)
    args = parser.parse_args()
    demand(args.static_contract_check, "STATIC_ONLY")
    demand(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"), "GPU_NOT_DISABLED")
    print("GEN5_V10_STAGE_C_RUNTIME_BRIDGE_CPU_STATIC_CONTRACT_PASS")
    print("SCHEMA=" + SCHEMA + ";CANDIDATE_BASE=" + BASE)
    print("GPU=OFF;MODEL_LOAD=NO;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED;PRODUCTION=BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# This callable is deliberately not referenced by the CLI. The user-controlled
# execution boundary remains CLOSED; dependency-injected CPU mocks only.
def construct_one_group_from_mocked_frozen_worker(*, frozen: Any,
                                                  row_ids: list[str],
                                                  static: Any, encoded: Any,
                                                  tokenizer_snapshot: Any,
                                                  model: Any, wrapper: Any,
                                                  strong_mask: Any, planes: Any,
                                                  features: Any, active: Any,
                                                  targets: Any) -> dict[str, Any]:
    """One selected frozen-worker source/group control-flow transcription.

    `features`, `model`, etc. must be ALREADY injected by CPU-only test harness.
    No file/model loading is performed by this function. All upstream helper
    calls delegate to frozen objects; exceptions propagate without suppression.
    """
    import torch
    selector = bounded_selector(frozen, row_ids)  # prior to any model work
    source = tuple(frozen.FULL_FACTORIAL_CELLS)[0]
    source_rows = tuple(row for row in frozen.all_orientations() if row[1] == source)
    demand(len(source_rows) == 4 and tuple(row[2] for row in source_rows[:2])
           == ((6201, 6202), (6201, 6203)), "ORIENTATION_IDENTITY")
    batch_features = frozen.transport.legacy._phase_b_batch_features(features, 800, 832)
    batch_mask = batch_features["attention_mask"]
    demand(isinstance(batch_mask, torch.Tensor) and
           batch_mask.device.type == "cpu" and
           tuple(batch_mask.shape) == (32, 128), "CPU_MOCK_BATCH_ONLY")
    role_masks = frozen.coarse._serialization_role_masks(batch_features)
    partition, _anchor_evidence = frozen._build_anchor_partition(
        static=static, encoded=encoded, tokenizer_snapshot=tokenizer_snapshot,
        row_start=416, row_stop=840,
    )
    fine_masks = frozen._batch_anchor_masks(
        partition, local_start=384, local_stop=416, device=batch_mask.device,
    )
    coarse_counts = frozen.coarse._pair_opportunity_counts(role_masks)
    frozen._fine_pair_opportunity_counts(fine_masks, coarse_counts)
    context = frozen.transport.legacy._phase_b_prepare_common_context(
        model=model, wrapper=wrapper, features=batch_features,
        stressor_active=active[800:832], target_indices=targets[800:832],
        strong_mask=strong_mask, planes=planes,
    )
    shape = wrapper.correction.shape
    log_a = frozen.kernel._discrete_log_a(context, shape=shape).to(device=batch_mask.device)
    survival = frozen.pairgap._backward_survival_factor(
        log_a_flat=log_a.reshape(log_a.shape[0], log_a.shape[1], -1),
        attention_mask=batch_mask,
    )
    weights = frozen.transport._cell_weights()
    raw = {}
    with torch.no_grad():
        # Frozen worker materializes all nine raw writes before the selected
        # source/group gradient; retain this exact precursor sequence.
        for cell in frozen.FULL_FACTORIAL_CELLS:
            demand(cell in weights and cell not in raw, "RAW_CELL_IDENTITY")
            a_weight, b_weight = weights[cell]
            raw[cell] = frozen.transport.legacy._phase_b_raw_write(
                context["mixer_input"], batch_mask, a_weight, b_weight,
            ).detach()
    _, grad_refute, grad_support = frozen.transport._source_gradient_bundle(
        model=model, wrapper=wrapper, features=batch_features,
        context=context, raw_source=raw[source],
    )
    grad_refute, grad_support, pinv_cpu = frozen.transport._prepare_projector_pinv(
        grad_refute=grad_refute, grad_support=grad_support,
        attention_mask=batch_mask,
    )
    acc = [frozen.transport._empty_orientation(*row) for row in source_rows[:2]]
    visible, complement = frozen.transport._project_visible_group(
        grad_refute=grad_refute, grad_support=grad_support, pinv_cpu=pinv_cpu,
        attention_mask=batch_mask, raw_source=raw[source],
        raw_targets=[raw[source_rows[0][2]], raw[source_rows[1][2]]],
        accumulators=acc,
    )
    demand(isinstance(visible, torch.Tensor) and
           isinstance(complement, torch.Tensor) and
           visible.device.type == complement.device.type == "cpu" and
           tuple(visible.shape[:3]) == (2, 32, 128) and
           tuple(complement.shape) == tuple(visible.shape), "PROJECTOR_OUTPUT_CPU")
    combined = torch.cat((visible, complement), dim=0)
    flat_inputs(component=combined, log_a=log_a, survival=survival,
                attention_mask=batch_mask, role_masks=role_masks,
                fine_masks=fine_masks)
    demand(not any(p.grad is not None for p in model.parameters()),
           "PARAMETER_GRADIENT_ACCUMULATED")
    return {
        "selector": selector, "component": combined, "log_a": log_a,
        "survival": survival, "attention_mask": batch_mask,
        "role_masks": role_masks, "fine_masks": fine_masks, "shape": shape,
        "raw_projector_accumulators": acc,
        "upstream_guard_coverage": "NOT_AUTHENTICATED_CPU_MOCK",
        "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
    }


def verify_projector_original_guards(*, frozen: Any, projector_accumulators: list[Any],
                                     frozen_records: list[dict[str, Any]]) -> None:
    """Call original relative-error and bound semantics after original full stats."""
    demand(len(projector_accumulators) == 2 and len(frozen_records) == 4,
           "RAW_GUARD_K4")
    for i, acc in enumerate(projector_accumulators):
        max_err = float(acc["raw_reconstruction_max_abs"])
        demand(math.isfinite(max_err) and max_err <= frozen.transport.RAW_RECON_ATOL,
               "RAW_RECONSTRUCTION_AUTH")
        raw = acc["stage"]["raw_write"]
        visible = float(raw["D_visible"])
        complement = float(raw["D_complement"])
        demand(frozen._relative_error(float(frozen_records[i]["R"]), visible)
               <= frozen.RAW_COMPONENT_MATCH_RTOL, "VISIBLE_RAW_ENERGY_INTERNAL_REPLAY")
        demand(frozen._relative_error(float(frozen_records[2+i]["R"]), complement)
               <= frozen.RAW_COMPONENT_MATCH_RTOL,
               "COMPLEMENT_RAW_ENERGY_INTERNAL_REPLAY")


def execute_one_group_cpu_mock(*, frozen: Any, hybrid: Any, row_ids: list[str],
                               static: Any, encoded: Any, tokenizer_snapshot: Any,
                               model: Any, wrapper: Any, strong_mask: Any, planes: Any,
                               features: Any, active: Any, targets: Any) -> dict[str, Any]:
    """CPU injected seam integration, no loading and no claimed upstream E1 PASS."""
    made = construct_one_group_from_mocked_frozen_worker(
        frozen=frozen,row_ids=row_ids,static=static,encoded=encoded,
        tokenizer_snapshot=tokenizer_snapshot,model=model,wrapper=wrapper,
        strong_mask=strong_mask,planes=planes,features=features,active=active,
        targets=targets,
    )
    kwargs = {key:made[key] for key in ("component", "log_a", "survival",
              "attention_mask", "role_masks", "fine_masks", "shape")}
    result = paired_cpu_seam(frozen=frozen, hybrid=hybrid, **kwargs,
                             selector=made["selector"],source_row_ids=row_ids)
    verify_projector_original_guards(
        frozen=frozen,projector_accumulators=made["raw_projector_accumulators"],
        frozen_records=result["frozen_records"],
    )
    result["raw_projector_guard_status"] = "PASS_CPU_MOCK_ONLY_NOT_AUTHENTICATED"
    return result
