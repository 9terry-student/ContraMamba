#!/usr/bin/env python3
"""Gen5 v10 D4 CPU-only in-process callback wiring, NOT a CUDA worker.

Callbacks are injected CPU test doubles, not authenticated frozen/model calls.
This module emits events AFTER successful callback return, hashes supplied CPU
values itself, and never treats a callback return as a source attestation.
No model loading, GPU query, disk artifact, or run command exists.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from collections.abc import Mapping
from typing import Any, Callable

from scripts import gen5_v10_stage_d3_untrusted_instrumentation_candidate as d3

SCHEMA = "GEN5_V10_D4_CPU_INPROCESS_WIRING_V1"
BASE = "da6c5480e4506afa840ccefac72dbd1bd0fbc320"
SOURCE_ORDER = tuple((a, r) for a in (6201, 6202, 6203) for r in (6201, 6202, 6203))
CAPTURE_LABELS = {
    "PRE_FROZEN_CAPTURED": "pre_frozen",
    "POST_FROZEN_CAPTURED": "post_frozen",
    "POST_V10_CAPTURED": "post_v10",
}
CAPTURE_PHASES = {"PRE_FROZEN_CAPTURED": "PRE_FROZEN",
                  "POST_FROZEN_CAPTURED": "POST_FROZEN",
                  "POST_V10_CAPTURED": "POST_V10"}
NO_ARTIFACT = "CPU_SIMULATED_PUBLICATION_NO_FILE_WRITTEN"
SOURCE_MARKER = "INJECTED_CPU_FIXTURE_NOT_AUTHENTICATED"


class D4Blocked(ValueError):
    """Never upgrade a failed synthetic seam to evidence."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise D4Blocked("GEN5_V10_D4_BLOCKED:" + reason)


def _cpu(tensor: Any, name: str) -> Any:
    import torch
    require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu", "CPU_ONLY:" + name)
    return tensor


def _valid_callbacks(callbacks: Any) -> None:
    require(type(callbacks) is dict and set(callbacks) == set(d3.EVENTS), "CALLBACK_EXACT_SET")
    require(all(callable(x) for x in callbacks.values()), "CALLBACK_NOT_CALLABLE")
    require(callbacks["FROZEN_FULL_REDUCER_RETURNED"] is not
            callbacks["V10_HYBRID_REDUCER_RETURNED"], "ENGINE_CALLBACK_NOT_INDEPENDENT")


def _capture_values(value: Any, *, component: Any, batch: Any, dynamics: Any) -> dict[str, Any]:
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as stage_a
    import torch
    require(type(value) is dict and set(value) == {"component", "log_a", "survival",
             "attention_mask", "role_masks", "fine_masks", "shape"}, "CAPTURE_INPUT_FIELDS")
    require(value["component"] is component, "CAPTURE_COMPONENT_NOT_SHARED")
    require(value["log_a"] is dynamics["log_a"] and
            value["survival"] is dynamics["survival"], "CAPTURE_DYNAMICS_NOT_SHARED")
    require(value["attention_mask"] is batch, "CAPTURE_MASK_NOT_SHARED")
    k4 = _cpu(component, "component")
    require(tuple(k4.shape[:3]) == (4, 32, 128) and k4.shape[3] != 24576,
            "SYNTHETIC_WIDTH_KBS")
    assert isinstance(k4, torch.Tensor)
    try:
        manifest = stage_a.manifest_for_cpu_inputs(**value, synthetic_cpu_geometry=True)
    except (stage_a.ProvenanceBlocked, TypeError, AttributeError) as exc:
        raise D4Blocked("GEN5_V10_D4_BLOCKED:INVALID_CPU_CAPTURE") from exc
    return manifest


def _validate_records(value: Any, label: str) -> None:
    from scripts import gen5_v10_e0e3_v3_runtime_verifier_candidate as v3
    require(type(value) is list and len(value) == 4, "K4_RECORDS:" + label)
    for i, row in enumerate(value):
        try:
            v3.validate_record(row, label + f"[{i}]")
        except (v3.V3Blocked, TypeError, ValueError, KeyError) as exc:
            raise D4Blocked("GEN5_V10_D4_BLOCKED:V3_RECORD:" + label) from exc


def run_injected_cpu_fixture(*, callbacks: dict[str, Callable[[], Any]],
                             clock_ns: Callable[[], int] = time.monotonic_ns) -> dict[str, Any]:
    """Execute 17 CPU-stub callbacks in original stage order (exactly once).

    No arbitrary event list or pre-sealed manifest accepted. CPU tensors are
    rehashed by Stage A on each capture. This is deliberately NOT an interface
    for injecting the real original model/worker or for executing CUDA.
    """
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"), "CUDA_NOT_DISABLED")
    _valid_callbacks(callbacks)
    require(callable(clock_ns), "CLOCK_CALLBACK")
    t0 = clock_ns()
    require(type(t0) is int and t0 >= 0, "CLOCK_START")
    batch = dynamics = component = None
    frozen_records = v10_records = None
    pre_manifest = None
    last_inputs = None
    rows: list[dict[str, Any]] = []
    captures: dict[str, Any] = {}
    guards = {name: "NOT_AUTHENTICATED" for name in d3.GUARD_KEYS}
    for ordinal, label in enumerate(d3.EVENTS):
        # Callback is invoked first. A failing/raising step never yields an
        # apparently completed trace or advances to later callbacks.
        value = callbacks[label]()
        digest = None
        if label == "CODE_SOURCE_CHECKED":
            require(value == SOURCE_MARKER and type(value) is str,
                    "CPU_SOURCE_MARKER")
        elif label == "SELECTOR_LOCKED":
            require(type(value) is dict and value == d3.BOUNDS and
                    all(type(value[k]) is type(d3.BOUNDS[k]) for k in d3.BOUNDS),
                    "SELECTOR_MISMATCH")
        elif label == "BATCH_FEATURES_MASKS_READY":
            import torch
            batch = _cpu(value, "batch_attention")
            require(batch.dtype == torch.bool and tuple(batch.shape) == (32, 128),
                    "BATCH_MASK_GEOMETRY")
        elif label == "COMMON_CONTEXT_LOG_A_SURVIVAL_READY":
            require(type(value) is dict and set(value) == {"log_a", "survival"},
                    "DYNAMICS_KEYS")
            for k in value:
                _cpu(value[k], "dynamics." + k)
            dynamics = value
        elif label == "RAW_WRITE_ALL_NINE_READY":
            require(type(value) is dict and tuple(value) == SOURCE_ORDER,
                    "NINE_RAW_CELLS_ORDER")
            for cell, raw in value.items():
                _cpu(raw, "raw." + str(cell))
        elif label == "FIRST_SOURCE_GRADIENT_READY":
            require(type(value) is tuple and len(value) == 2, "GRADIENT_PAIR")
            for i, grad in enumerate(value):
                _cpu(grad, "gradient." + str(i))
        elif label == "FIRST_PROJECTOR_READY":
            require(value == "CPU_MOCK_PROJECTOR_READY" and type(value) is str,
                    "PROJECTOR_MOCK_MARKER")
            guards["source_gradient_projector"] = "OBSERVED_CLAIM_ONLY"
        elif label == "FIRST_TARGET_GROUP_PROJECTED":
            import torch
            require(type(value) is tuple and len(value) == 2, "PROJECTED_PAIR")
            visible, complement = (_cpu(x, "projected") for x in value)
            require(visible.dtype == complement.dtype == torch.float32 and
                    tuple(visible.shape[:3]) == (2, 32, 128) and
                    tuple(visible.shape) == tuple(complement.shape),
                    "PROJECTED_GEOMETRY")
            projected = (visible, complement)
        elif label == "K4_COMBINED_READY":
            import torch
            component = _cpu(value, "K4")
            require(component.dtype == torch.float32 and
                    tuple(component.shape) == (4, 32, 128, projected[0].shape[3]) and
                    torch.equal(component, torch.cat(projected, dim=0)),
                    "K4_MISMATCH")
        elif label in CAPTURE_LABELS:
            manifest = _capture_values(value, component=component,
                                       batch=batch, dynamics=dynamics)
            digest = manifest["manifest_sha256"]
            captures[CAPTURE_LABELS[label]] = manifest
            last_inputs = value
            if pre_manifest is None:
                pre_manifest = manifest
            else:
                require(manifest == pre_manifest, "INPUT_MUTATION:" + label)
        elif label in ("FROZEN_FULL_REDUCER_RETURNED", "V10_HYBRID_REDUCER_RETURNED"):
            _validate_records(value, label)
            if label.startswith("FROZEN"):
                frozen_records = value
                for k in ("frozen_original_full_fine_stats", "frozen_coarse_replay",
                          "frozen_fine_parent_replay"):
                    guards[k] = "OBSERVED_CLAIM_ONLY"
            else:
                v10_records = value
                for k in ("v10_coarse_replay", "v10_fine_parent_replay"):
                    guards[k] = "OBSERVED_CLAIM_ONLY"
        elif label == "ORIGINAL_RAW_PROJECTOR_GUARDS_EVALUATED":
            require(value == "CPU_MOCK_ORIGINAL_GUARDS_RETURNED" and type(value) is str,
                    "GUARDS_MOCK_MARKER")
            for k in ("raw_reconstruction_auth", "visible_raw_energy_internal_replay",
                      "complement_raw_energy_internal_replay"):
                guards[k] = "OBSERVED_CLAIM_ONLY"
        elif label == "V3_OFFLINE_RECORDS_VERIFIED":
            from scripts import gen5_v10_e0e3_v3_runtime_verifier_candidate as v3
            require(value == "CPU_MOCK_OFFLINE_REPLAY" and type(value) is str,
                    "OFFLINE_MOCK_MARKER")
            require(frozen_records is not None and v10_records is not None,
                    "MISSING_ENGINE_OUTPUT")
            try:
                stats = v3._normalized(frozen_records, v10_records)
            except (v3.V3Blocked, TypeError, ValueError, KeyError) as exc:
                raise D4Blocked("GEN5_V10_D4_BLOCKED:OFFLINE_NORMALIZATION") from exc
        elif label == "D1_ARTIFACT_PUBLISHED":
            # Synthetic-only: no file exists and no assertion of publication.
            require(value == NO_ARTIFACT and type(value) is str, "FALSE_PUBLICATION")
        else:
            raise D4Blocked("GEN5_V10_D4_BLOCKED:UNEXPECTED_EVENT")
        now = clock_ns()
        require(type(now) is int and now > t0, "NONMONOTONIC_CLOCK")
        rows.append({"event": label, "ordinal": ordinal, "elapsed_ns": now - t0,
                     "memory": None, "capture_manifest_sha256": digest})
    require(set(captures) == set(CAPTURE_LABELS.values()), "CAPTURE_COVERAGE")
    # The later guard/offline/publication-stub callbacks may also mutate a
    # borrowed tensor. Hash once more before returning a successful trace.
    final_manifest = _capture_values(last_inputs, component=component,
                                     batch=batch, dynamics=dynamics)
    require(final_manifest == pre_manifest, "INPUT_MUTATION:FINAL_CALLBACKS")
    # D3 is an untrusted record schema. The trace is not signed evidence of
    # lineage/FFT memory/guard execution, regardless of callback success.
    obs = d3.observation(source_kind="CPU_SYNTHETIC_FIXTURE",
                         claimed_code_head=BASE, selector=copy.deepcopy(d3.BOUNDS),
                         entries=rows, source_claims=None, guard_trace=guards)
    verdict = d3.validate(obs)
    require(verdict["E0"] == verdict["E1"] == verdict["E2"] == verdict["E3"] == "NOT_EVALUATED",
            "D3_FALSE_PROMOTION")
    return {"schema_version": SCHEMA, "candidate_base": BASE,
            "observation": obs, "trace_verdict": verdict,
            "capture_manifest_sha256": pre_manifest["manifest_sha256"],
            "synthetic_offline_H_counts": stats,
            "callbacks_invoked": len(d3.EVENTS),
            "artifact_written": False, "model_loaded": False,
            "input_source_authenticated": False,
            "upstream_guard_authenticated": False,
            "GPU_EXECUTION": "NOT_AUTHORIZED", "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
            "V10_E0_PROVENANCE": "NOT_EVALUATED", "V10_E1_FROZEN_GUARDS": "NOT_EVALUATED",
            "V10_E2_PAYLOAD": "NOT_EVALUATED", "V10_E3_DIAGNOSTICS": "NOT_EVALUATED",
            "PRODUCTION": "BLOCKED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-contract-check", action="store_true", required=True)
    args = parser.parse_args()
    require(args.static_contract_check, "STATIC_ONLY")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"), "CUDA_NOT_DISABLED")
    print("GEN5_V10_STAGE_D4_CPU_WIRING_STATIC_CONTRACT_PASS")
    print("SCHEMA=" + SCHEMA + ";BASE=" + BASE)
    print("EVENTS=" + str(len(d3.EVENTS)) + ";SYNTHETIC_CALLBACKS_ONLY=YES")
    print("GPU=OFF;MODEL_LOAD=NO;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED;E0_E3=NOT_EVALUATED")
    print("RUNTIME_ORIGIN=NOT_AUTHENTICATED;ACTUAL_D1_ARTIFACT=NOT_WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
