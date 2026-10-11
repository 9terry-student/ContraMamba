#!/usr/bin/env python3
"""Gen5 v10 Stage B static-only bounded worker/export design.

No model loading, CUDA worker, scientific execution, or --run-bounded mode.
This module can perform CPU synthetic seam checks and capture canonical value
hashes; those hashes by themselves do not authenticate data provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any, Callable

SCHEMA = "GEN5_V10_BOUNDED_PAIRED_EXPORT_V2"
VERIFICATION_SCHEMA = "GEN5_V10_BOUNDED_PAIRED_VERIFICATION_V2"
FROZEN_BASE = "f8e60f173d3a72aa17779449ec9b931c7fa4d043"
SOURCE_WORKER = 1
BATCH_ONE_BASED = 13
ROW_START = 800
ROW_STOP = 832
BATCH_ROWS = 32
TARGETS_PER_TRANSPORT = 2
FINE_CELLS = 75
FACTOR_SEEDS = (6201, 6202, 6203)
SOURCE_CELLS = tuple((a, r) for a in FACTOR_SEEDS for r in FACTOR_SEEDS)

def frozen_pair_group(source, target):
    if source == target:
        return None
    if source[1] == target[1] and source[0] != target[0]:
        return "PRIMARY_A"
    if source[0] == target[0] and source[1] != target[1]:
        return "CONTROL_R"
    return None

FROZEN_ORIENTATIONS = tuple(
    (frozen_pair_group(source, target), source, target)
    for source in SOURCE_CELLS for target in SOURCE_CELLS
    if frozen_pair_group(source, target) is not None
)
MAX_CAPTURE_BYTES = 8 * 1024 * 1024

class StageBBlocked(ValueError):
    pass


def demand(condition: bool, code: str) -> None:
    if not condition:
        raise StageBBlocked("GEN5_V10_STAGE_B_BLOCKED:" + code)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise StageBBlocked("CANONICAL_JSON_INVALID") from exc


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def choose_bounded_locus(*, worker_id: int, batch_index: int,
                         row_start: int, row_stop: int, row_ids: list[str],
                         source_cells: tuple[Any, ...],
                         orientations: tuple[tuple[Any, Any, Any], ...],
                         selected_source: Any, selected_targets: tuple[Any, Any],
                         target_group_index: int = 0) -> dict[str, Any]:
    """Select exactly first source's first 2-target group in worker1 batch13.

    Identity must come from original enumeration at the frozen runtime seam;
    numeric positions without cell/row identities are insufficient.
    """
    demand(type(worker_id) is int and worker_id == SOURCE_WORKER, "WORKER_ID")
    demand(type(batch_index) is int and batch_index == BATCH_ONE_BASED, "BATCH_ID")
    demand(type(row_start) is int and type(row_stop) is int and
           (row_start, row_stop) == (ROW_START, ROW_STOP), "ROW_INTERVAL")
    demand(type(row_ids) is list and len(row_ids) == BATCH_ROWS and
           all(type(v) is str and bool(v) for v in row_ids) and
           len(set(row_ids)) == BATCH_ROWS, "ROW_IDS")
    demand(type(source_cells) is tuple and source_cells == SOURCE_CELLS,
           "SOURCE_ORDER")
    demand(type(orientations) is tuple and orientations == FROZEN_ORIENTATIONS,
           "ORIENTATION_ORDER")
    demand(type(target_group_index) is int and target_group_index == 0,
           "FIRST_GROUP_ONLY")
    demand(type(selected_targets) is tuple and len(selected_targets) == 2,
           "TARGET_GROUP_GEOMETRY")
    source = source_cells[0]
    demand(source == selected_source, "SOURCE_NOT_FIRST")
    targets = tuple(row[2] for row in orientations if row[1] == source)
    demand(len(targets) == 4 and len(set(targets)) == 4,
           "SOURCE_ORIENTATION_GEOMETRY")
    demand(tuple(selected_targets) == targets[:TARGETS_PER_TRANSPORT],
           "TARGET_GROUP_IDENTITY")
    demand(all(len(row) == 3 for row in orientations), "ORIENTATION_SHAPE")
    group_names = tuple(row[0] for row in orientations if row[1] == source)[:2]
    demand(group_names == ("CONTROL_R", "CONTROL_R"), "FIRST_GROUP_NOT_CONTROL_R")
    def cell_id(cell: Any) -> str:
        demand(type(cell) is tuple and len(cell) == 2 and
               all(type(x) is int for x in cell), "CELL_FORMAT")
        return f"{cell[0]}_{cell[1]}"
    return {
        "worker_id": worker_id, "batch_index_one_based": batch_index,
        "row_start": row_start, "row_stop": row_stop,
        "row_ids": list(row_ids), "source_cell": cell_id(source),
        "target_cells": [cell_id(x) for x in targets[:2]],
        "target_group_index": 0, "group": group_names[0],
        "expected_component_count": 4, "expected_batch_rows": 32,
    }


def _bounded_flat_bytes(tensor: Any, *, max_chunk_bytes: int):
    """Yield C-order chunks, recursively split *logical* tensor slices.

    A non-contiguous view never gets flattened into a full-size GPU clone.
    At most max_chunk_bytes of contiguous GPU scratch is requested per leaf.
    """
    import torch
    import numpy as np
    demand(type(max_chunk_bytes) is int and max_chunk_bytes >= 4,
           "CAPTURE_CHUNK_CAP")
    demand(isinstance(tensor, torch.Tensor) and tensor.dtype in
           (torch.float32, torch.bool), "CAPTURE_DTYPE")
    per_item = 4 if tensor.dtype == torch.float32 else 1
    demand(tensor.ndim >= 2 and tensor.numel() > 0, "CAPTURE_GEOMETRY")
    def visit(piece):
        if piece.numel() * per_item <= max_chunk_bytes:
            # CUDA execution of this leaf is not exposed by this module's CLI.
            host = piece.detach().contiguous().to(device="cpu", copy=True)
            arr = host.numpy()
            if tensor.dtype == torch.float32:
                arr = arr.astype(np.dtype("<f4"), copy=False)
            else:
                arr = arr.astype(np.uint8, copy=False)
            demand(arr.flags.c_contiguous, "CAPTURE_NONCONTIGUOUS_LEAF")
            yield memoryview(arr).cast("B").tobytes()
            return
        dim = next((i for i, n in enumerate(piece.shape) if n > 1), None)
        demand(dim is not None, "IMPOSSIBLE_CAPTURE_SPLIT")
        # This splitting is C-order because all preceding dimensions are 1.
        half = piece.shape[dim] // 2
        for a, b in ((0, half), (half, piece.shape[dim])):
            yield from visit(piece.narrow(dim, a, b - a))
    yield from visit(tensor)


def capture_value_digest(tensor: Any, *, phase: str,
                         max_chunk_bytes: int = MAX_CAPTURE_BYTES) -> dict[str, Any]:
    """Stream Stage-A-compatible tensor bytes; no source attestation.

    This API is future CUDA-capture *code*, not an authorized GPU entrypoint.
    Actual GPU use requires separate execution gate and memory validation.
    """
    import torch
    demand(type(phase) is str and phase in
           ("PRE_FROZEN", "POST_FROZEN", "POST_V10"), "CAPTURE_PHASE")
    demand(isinstance(tensor, torch.Tensor), "CAPTURE_TENSOR")
    demand(tensor.device.type in ("cpu", "cuda"), "CAPTURE_DEVICE")
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)
    observed = hashlib.sha256()
    seen = 0
    for part in _bounded_flat_bytes(tensor, max_chunk_bytes=max_chunk_bytes):
        observed.update(part)
        seen += len(part)
    demand(seen == tensor.numel() * (4 if tensor.dtype == torch.float32 else 1),
           "CAPTURE_BYTE_COUNT")
    return {"content_sha256": observed.hexdigest(), "byte_count": seen,
            "shape": list(tensor.shape),
            "dtype": "float32" if tensor.dtype == torch.float32 else "bool",
            "source_device": str(tensor.device), "source_strides": list(tensor.stride()),
            "phase": phase, "synchronized": True,
            "chunk_cap_bytes": max_chunk_bytes,
            "V10_E0_PROVENANCE": "NOT_EVALUATED"}


def cpu_stub_paired_capture(*, inputs: dict[str, Any], shape: Any,
                            frozen_fn: Callable[..., Any],
                            v10_fn: Callable[..., Any]) -> dict[str, Any]:
    """CPU-only fake seam: test input immutability and independent call order.

    Functions are injected and may only be CPU stubs. No model/gradient load.
    """
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as stage_a
    import torch
    demand(type(inputs) is dict and set(inputs) ==
           {"component", "log_a", "survival", "attention_mask", "role_masks", "fine_masks"},
           "CPU_STUB_INPUT_KEYS")
    demand(inputs["component"].device.type == "cpu", "CPU_STUB_ONLY")
    kw = dict(inputs, shape=shape, synthetic_cpu_geometry=True)
    before = stage_a.manifest_for_cpu_inputs(**kw)
    with torch.no_grad():
        old = frozen_fn(**inputs, shape=shape)
        after_old = stage_a.manifest_for_cpu_inputs(**kw)
        demand(before == after_old, "FROZEN_MUTATED_INPUTS")
        new = v10_fn(**inputs, shape=shape)
        after_new = stage_a.manifest_for_cpu_inputs(**kw)
        demand(before == after_new, "V10_MUTATED_INPUTS")
    demand(type(old) is list and len(old) == 4 and
           type(new) is list and len(new) == 4, "PAIR_RECORD_K4")
    return {"frozen_records": old, "v10_records": new,
            "manifest_sha256": before["manifest_sha256"],
            "capture_equal": True, "authentic_model_inputs": False,
            "V10_E0_PROVENANCE": "NOT_EVALUATED", "V10_E1_FROZEN_GUARDS": "NOT_EVALUATED",
            "V10_E2_PAYLOAD": "NOT_EVALUATED", "V10_E3_DIAGNOSTICS": "NOT_EVALUATED",
            "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED", "PRODUCTION": "BLOCKED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-contract-check", action="store_true", required=True)
    args = parser.parse_args()
    demand(args.static_contract_check, "STATIC_ONLY")
    demand(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
           "CUDA_DISABLED_REQUIRED")
    print("GEN5_V10_STAGE_B_WORKER_CPU_STATIC_CONTRACT_PASS")
    print("SCHEMA=" + SCHEMA + ";EXPECTED_BASE=" + FROZEN_BASE)
    print("BOUNDED=WORKER1_BATCH13_ROWS800_832_FIRST_SOURCE_FIRST_GROUP")
    print("GPU=OFF;MODEL_LOAD=NO;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED;PRODUCTION=BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
