#!/usr/bin/env python3
"""CPU-only paired frozen/v10 reducer adapter for already-provided tensors.

No CLI, data/model loading, CUDA, file output, worker or scientific evaluation.
This module intentionally does NOT modify any function in the frozen audit.
A paired result is a local numerical candidate, not authenticated E0-E3 evidence.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from scripts import gen5_v10_hybrid_fine_pair_candidate as hybrid

PARENTS = ("CLAIM->CLAIM", "EVIDENCE->EVIDENCE", "CLAIM->EVIDENCE")
FINE_CLASSES = ("A_TITLE", "A_NAME", "A_ROLE", "A_PREDICATE", "RESIDUAL")


class BoundedAdapterBlocked(RuntimeError):
    """Fail-closed static adapter guard."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise BoundedAdapterBlocked("GEN5_V10_PAIRED_ADAPTER_BLOCKED:" + reason)


def _checked_inputs(audit: Any, *, component: torch.Tensor, log_a: torch.Tensor,
                    survival: torch.Tensor, attention_mask: torch.Tensor,
                    role_masks: Mapping[str, torch.Tensor],
                    fine_masks: Mapping[str, Mapping[str, torch.Tensor]],
                    shape: Any, synthetic_cpu_geometry: bool) -> tuple[int, int, int, int]:
    require(type(synthetic_cpu_geometry) is bool, "SYNTHETIC_FLAG_TYPE")
    require(isinstance(component, torch.Tensor) and component.ndim == 4,
            "COMPONENT_SHAPE")
    k, b, seq, width = component.shape
    require(k == 4 and b in (8, 32) and seq == 128, "K_BATCH_SEQUENCE")
    require(width > 0 and width == shape.intermediate_size * shape.state_size,
            "STATE_WIDTH")
    require(synthetic_cpu_geometry or width == 24576, "PRODUCTION_WIDTH")
    require(component.dtype == torch.float32 and component.device.type == "cpu",
            "COMPONENT_CPU_FLOAT32")
    tensors = (log_a, survival, attention_mask)
    require(all(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in tensors),
            "CPU_INPUTS_ONLY")
    require(log_a.dtype == torch.float32 and survival.dtype == torch.float32,
            "LOG_SURVIVAL_FLOAT32")
    require(tuple(log_a.shape) == (b, seq, shape.intermediate_size, shape.state_size),
            "LOG_A_SHAPE")
    require(tuple(survival.shape) == (b, seq, width), "SURVIVAL_SHAPE")
    require(tuple(attention_mask.shape) == (b, seq) and
            attention_mask.dtype == torch.bool, "ATTENTION_MASK_GEOMETRY")
    require(all(bool(torch.all(torch.isfinite(t)).item())
                for t in (component, log_a, survival)), "INPUT_NONFINITE")
    require(bool(torch.all(log_a <= 0).item()), "POSITIVE_LOG_A")
    require(bool(torch.all(survival >= 0).item()), "NEGATIVE_SURVIVAL")
    require(tuple(audit.PARENT_PAIR_NAMES) == PARENTS and
            tuple(audit.FINE_CLASSES) == FINE_CLASSES and
            len(audit.FINE_PAIR_NAMES) == 25 and
            audit.p3a.MAX_LENGTH == seq, "FROZEN_GEOMETRY")
    require(audit.FAST_FLOAT32_HALF_SPAN_LIMIT == 60.0, "FROZEN_DISPATCH_THRESHOLD")
    require(audit.FFT_CHUNK_WIDTH == 256, "FROZEN_CHUNK_WIDTH")
    require(set(role_masks) == {"CLAIM", "EVIDENCE", "EOS"}, "ROLE_MASK_KEYS")
    for name, mask in role_masks.items():
        require(isinstance(mask, torch.Tensor) and
                mask.device.type == "cpu" and mask.dtype == torch.bool and
                tuple(mask.shape) == (b, seq), "ROLE_MASK:" + name)
    require(set(fine_masks) == {"CLAIM", "EVIDENCE"}, "FINE_SIDE_KEYS")
    union = torch.zeros_like(attention_mask)
    for side in ("CLAIM", "EVIDENCE"):
        require(set(fine_masks[side]) == set(FINE_CLASSES), "FINE_KEYS:" + side)
        for cell in FINE_CLASSES:
            mask = fine_masks[side][cell]
            require(isinstance(mask, torch.Tensor) and mask.device.type == "cpu"
                    and mask.dtype == torch.bool and tuple(mask.shape) == (b, seq),
                    "FINE_MASK:" + side + ":" + cell)
            require(not bool(torch.any(mask & union).item()), "FINE_MASK_OVERLAP")
            require(not bool(torch.any(mask & ~attention_mask).item()),
                    "FINE_MASK_OUTSIDE_ATTENTION")
            union |= mask
    return k, b, seq, width


def _validate_gap(audit: Any, gap: Any, *, k: int, seq: int, engine: str) -> None:
    require(isinstance(gap, dict) and set(gap) == set(audit.PARENT_PAIR_NAMES),
            engine + "_PARENT_KEYS")
    for parent in audit.PARENT_PAIR_NAMES:
        require(isinstance(gap[parent], dict) and
                set(gap[parent]) == set(audit.FINE_PAIR_NAMES),
                engine + "_FINE_KEYS:" + parent)
        for cell in audit.FINE_PAIR_NAMES:
            vector = gap[parent][cell]
            require(isinstance(vector, torch.Tensor) and
                    tuple(vector.shape) == (k, seq) and
                    vector.device.type == "cpu" and vector.dtype == torch.float64,
                    engine + "_GAP_SHAPE_DTYPE")
            require(bool(torch.all(torch.isfinite(vector)).item()),
                    engine + "_NONFINITE")
            require(bool(torch.all(vector[:, 0] == 0).item()),
                    engine + "_ZERO_GAP")


def _post_reducer_records(audit: Any, *, coarse_rows: list[dict[str, Any]],
                          fine_gap: dict[str, dict[str, torch.Tensor]],
                          fine_span: float, k: int, seq: int) -> list[dict[str, Any]]:
    """Byte-semantic transcription of frozen _component_fine_pair_stats post-reducer.

    Reuses actual frozen coarse._replay_vector for every parent and row.
    Only CPU already-emitted tensors are accepted; no replacement guard exists.
    """
    require(len(coarse_rows) == k, "COARSE_ROW_COUNT")
    require(math.isfinite(float(fine_span)), "FINE_SPAN_NONFINITE")
    output = []
    for index in range(k):
        fine_out: dict[str, dict[str, list[float]]] = {}
        max_abs = 0.0
        max_rel = 0.0
        for parent in audit.PARENT_PAIR_NAMES:
            reconstructed = torch.zeros(seq, device="cpu", dtype=torch.float64)
            parent_out: dict[str, list[float]] = {}
            for fine in audit.FINE_PAIR_NAMES:
                vector = fine_gap[parent][fine][index]
                reconstructed += vector
                parent_out[fine] = [
                    float(value) for value in vector[1:].detach().cpu().tolist()
                ]
            expected = torch.tensor(
                [0.0] + [float(x) for x in coarse_rows[index]
                         ["region_pair_C_by_gap"][parent]],
                device="cpu", dtype=torch.float64,
            )
            require(tuple(expected.shape) == (seq,), "COARSE_GAP_GEOMETRY")
            self_energy = float(coarse_rows[index]["S"])
            require(math.isfinite(self_energy) and self_energy > 0.0,
                    "FINE_PARENT_SELF_ENERGY")
            reconstructed_h = [float(value) / self_energy
                               for value in reconstructed[1:].detach().cpu().tolist()]
            expected_h = [float(value) / self_energy
                          for value in expected[1:].detach().cpu().tolist()]
            replay = audit.coarse._replay_vector(
                name=f"FINE_PARENT_H_RECON_{parent}_{index}",
                observed=reconstructed_h, expected=expected_h,
            )
            require(replay["pass"] is True, "FROZEN_REPLAY_NOT_PASS")
            max_abs = max(max_abs, float(replay["max_abs_error"]))
            max_rel = max(max_rel, float(replay["max_rel_error"]))
            fine_out[parent] = parent_out
        row = dict(coarse_rows[index])
        row["fine_pair_C_by_gap"] = fine_out
        row["fine_parent_reconstruction_abs_max"] = max_abs
        row["fine_parent_reconstruction_rel_max"] = max_rel
        row["fine_pair_fft_half_log_span_max"] = fine_span
        output.append(row)
    return output


def paired_engine_records(audit: Any, *, component: torch.Tensor,
                          log_a: torch.Tensor, survival: torch.Tensor,
                          attention_mask: torch.Tensor,
                          role_masks: Mapping[str, torch.Tensor],
                          fine_masks: Mapping[str, Mapping[str, torch.Tensor]],
                          shape: Any, synthetic_cpu_geometry: bool = False) -> dict[str, Any]:
    """Produce engine-separated K4 rows for one provided CPU batch, no I/O.

    `audit` should be the original frozen audit module. A CPU-only reduced-width
    synthetic geometry is explicitly marked and can NEVER promote science.
    """
    k, b, seq, width = _checked_inputs(
        audit, component=component, log_a=log_a, survival=survival,
        attention_mask=attention_mask, role_masks=role_masks,
        fine_masks=fine_masks, shape=shape,
        synthetic_cpu_geometry=synthetic_cpu_geometry,
    )
    kwargs = dict(component=component, log_a=log_a, survival=survival,
                  attention_mask=attention_mask, fine_masks=fine_masks, shape=shape)
    with torch.no_grad():
        # Shared, genuine coarse computation, never a fabricated engine proxy.
        coarse_rows = audit.coarse._component_region_pair_stats(
            component=component, log_a=log_a, survival=survival,
            attention_mask=attention_mask, role_masks=role_masks, shape=shape,
        )
        require(isinstance(coarse_rows, list) and len(coarse_rows) == k,
                "COARSE_ROWS")
        original_gap, original_span = audit._fine_pair_gap_interference_fft(**kwargs)
        _validate_gap(audit, original_gap, k=k, seq=seq, engine="FROZEN")
        view = SimpleNamespace(
            p3a=audit.p3a, FINE_CLASSES=audit.FINE_CLASSES,
            PARENT_PAIRS=audit.PARENT_PAIRS,
            PARENT_PAIR_NAMES=audit.PARENT_PAIR_NAMES,
            FINE_PAIR_NAMES=audit.FINE_PAIR_NAMES,
            FFT_CHUNK_WIDTH=(4 if synthetic_cpu_geometry else audit.FFT_CHUNK_WIDTH),
            FAST_FLOAT32_HALF_SPAN_LIMIT=audit.FAST_FLOAT32_HALF_SPAN_LIMIT,
            _fine_pair_gap_interference_fft=audit._fine_pair_gap_interference_fft,
        )
        v10_gap, v10_span, route = hybrid.hybrid_fine_pair_gap(
            view, **kwargs, synthetic_cpu_geometry=synthetic_cpu_geometry,
        )
        _validate_gap(audit, v10_gap, k=k, seq=seq, engine="V10")
        frozen_rows = _post_reducer_records(
            audit, coarse_rows=coarse_rows, fine_gap=original_gap,
            fine_span=original_span, k=k, seq=seq,
        )
        v10_rows = _post_reducer_records(
            audit, coarse_rows=coarse_rows, fine_gap=v10_gap,
            fine_span=v10_span, k=k, seq=seq,
        )
    require(len(frozen_rows) == k and len(v10_rows) == k, "RECORD_COUNT")
    return {
        "frozen_records": frozen_rows,
        "v10_records": v10_rows,
        "v10_route": route,
        "batch_rows": b,
        "k": k,
        "sequence_length": seq,
        "state_width": width,
        "synthetic_cpu_geometry": synthetic_cpu_geometry,
        "status": "CPU_BOUNDED_PAIRED_RECORDS_ONLY_NOT_AUTHENTICATED",
        "V10_E0_PROVENANCE": "NOT_EVALUATED",
        "V10_E1_FROZEN_GUARDS": "NOT_EVALUATED",
        "V10_E2_PAYLOAD": "NOT_EVALUATED",
        "V10_E3_DIAGNOSTICS": "NOT_EVALUATED",
        "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
        "PRODUCTION": "BLOCKED",
    }
