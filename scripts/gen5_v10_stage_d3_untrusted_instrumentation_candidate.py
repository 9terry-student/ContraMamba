#!/usr/bin/env python3
"""Gen5 v10 D3: CPU-only observer event/measurement contract (not a GPU runner).

An event log or SHA256 produced by the same source that asserts its provenance is
untrusted. This module never probes hardware or loads inputs. Real-GPU execution,
scientific E0-E3 verdicts, and an execution entrypoint are NOT AUTHORIZED.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from typing import Any

SCHEMA = "GEN5_V10_D3_UNTRUSTED_RUNTIME_OBSERVATION_V1"
BASE = "8b2a9ad0be96c45e18883cab951a6414bfb179c3"
BRANCH = "gen5-causal-role-state-ownership"
HASH_DOMAIN = b"GEN5_V10_D3_RUNTIME_OBSERVATION_V1\x00"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")

# Audited against the frozen worker: all 9 raw writes precede source gradients;
# one selected source/projector and one 2-target group only. No model loaded here.
EVENTS = (
    "CODE_SOURCE_CHECKED",
    "SELECTOR_LOCKED",
    "BATCH_FEATURES_MASKS_READY",
    "COMMON_CONTEXT_LOG_A_SURVIVAL_READY",
    "RAW_WRITE_ALL_NINE_READY",
    "FIRST_SOURCE_GRADIENT_READY",
    "FIRST_PROJECTOR_READY",
    "FIRST_TARGET_GROUP_PROJECTED",
    "K4_COMBINED_READY",
    "PRE_FROZEN_CAPTURED",
    "FROZEN_FULL_REDUCER_RETURNED",
    "POST_FROZEN_CAPTURED",
    "V10_HYBRID_REDUCER_RETURNED",
    "POST_V10_CAPTURED",
    "ORIGINAL_RAW_PROJECTOR_GUARDS_EVALUATED",
    "V3_OFFLINE_RECORDS_VERIFIED",
    "D1_ARTIFACT_PUBLISHED",
)
CAPTURE_EVENTS = frozenset(("PRE_FROZEN_CAPTURED", "POST_FROZEN_CAPTURED", "POST_V10_CAPTURED"))
WORK_COUNTS = {
    "raw_write_cells": 9, "source_gradient_calls": 1,
    "projector_preparations": 1, "projected_target_groups": 1,
    "selected_targets": 2, "k4_components": 4,
    "input_tensor_keys": 17, "input_capture_phases": 3,
}
BOUNDS = {
    "worker_id": 1, "batch_index_one_based": 13, "row_start": 800,
    "row_stop": 832, "source_cell": "6201_6201",
    "target_cells": ["6201_6202", "6201_6203"],
    "K": 4, "B": 32, "S": 128, "W": 24576,
}
GUARD_KEYS = (
    "frozen_original_full_fine_stats", "frozen_coarse_replay",
    "frozen_fine_parent_replay", "v10_coarse_replay",
    "v10_fine_parent_replay", "raw_reconstruction_auth",
    "visible_raw_energy_internal_replay", "complement_raw_energy_internal_replay",
    "source_gradient_projector",
)
ENTRY_FIELDS = frozenset(("event", "ordinal", "elapsed_ns", "memory", "capture_manifest_sha256"))
MEMORY_FIELDS = frozenset(("allocated_bytes", "reserved_bytes", "peak_allocated_bytes",
                           "driver_free_bytes", "driver_total_bytes"))
ROOT_FIELDS = frozenset(("schema_version", "candidate_base", "branch", "claimed_code_head",
                         "source_kind", "selector", "source_claims", "work_counts", "events", "guard_trace",
                         "attestation", "GPU_EXECUTION", "E0", "E1", "E2", "E3",
                         "readiness", "observation_sha256"))

class D3Blocked(ValueError):
    """Fail closed on malformed or overpromoted untrusted observation."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise D3Blocked("GEN5_V10_D3_BLOCKED:" + reason)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise D3Blocked("NONCANONICAL_JSON") from exc


def _hash(value: Any) -> str:
    return hashlib.sha256(HASH_DOMAIN + _canonical(value)).hexdigest()


def _uint(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, name + "_NONNEG_INT")
    return value


def _memory(sample: Any, kind: str, event: str) -> None:
    if kind == "CPU_SYNTHETIC_FIXTURE":
        require(sample is None, "CPU_FAKE_MEMORY_MUST_BE_NULL:" + event)
        return
    require(type(sample) is dict and set(sample) == MEMORY_FIELDS,
            "RUNTIME_MEMORY_FIELDS:" + event)
    for name in MEMORY_FIELDS:
        _uint(sample[name], "MEMORY_" + name)
    require(sample["allocated_bytes"] <= sample["reserved_bytes"] and
            sample["reserved_bytes"] <= sample["driver_total_bytes"] and
            sample["driver_free_bytes"] <= sample["driver_total_bytes"] and
            sample["peak_allocated_bytes"] >= sample["allocated_bytes"],
            "MEMORY_INCONSISTENT:" + event)
    # These counters are not a proof of the *whole* unmeasured FFT/GEMM peak.
    # No safety threshold or GPU-ready verdict is derived here.


def _check_selector(obj: Any) -> None:
    require(type(obj) is dict and set(obj) == set(BOUNDS), "SELECTOR_FIELDS")
    for name, value in BOUNDS.items():
        require(type(obj[name]) is type(value) and obj[name] == value,
                "SELECTOR_MISMATCH:" + name)


def observation(*, source_kind: str, claimed_code_head: str,
                selector: dict[str, Any], entries: list[dict[str, Any]],
                source_claims: dict[str, Any] | None = None,
                guard_trace: dict[str, str] | None = None) -> dict[str, Any]:
    """Revalidate caller-INJECTED values; does not acquire runtime readings."""
    value = {
        "schema_version": SCHEMA, "candidate_base": BASE, "branch": BRANCH,
        "claimed_code_head": claimed_code_head, "source_kind": source_kind,
        "selector": copy.deepcopy(selector), "source_claims": copy.deepcopy(source_claims),
        "work_counts": dict(WORK_COUNTS),
        "guard_trace": (copy.deepcopy(guard_trace) if guard_trace is not None
                        else {name: "NOT_AUTHENTICATED" for name in GUARD_KEYS}),
        "events": copy.deepcopy(entries), "attestation": "SELF_ASSERTED_UNAUTHENTICATED",
        "GPU_EXECUTION": "NOT_AUTHORIZED", "E0": "NOT_EVALUATED",
        "E1": "NOT_EVALUATED", "E2": "NOT_EVALUATED", "E3": "NOT_EVALUATED",
        "readiness": "BLOCKED_INDEPENDENT_RUNTIME_AND_MEMORY_EVIDENCE",
    }
    value["observation_sha256"] = _hash(value)
    validate(value)
    return value


def validate(value: Any) -> dict[str, Any]:
    require(type(value) is dict and set(value) == ROOT_FIELDS, "OBSERVATION_FIELDS")
    require(value["schema_version"] == SCHEMA and value["candidate_base"] == BASE and
            value["branch"] == BRANCH, "SCHEMA_OR_BASE")
    kind = value["source_kind"]
    require(type(kind) is str and kind in ("CPU_SYNTHETIC_FIXTURE", "UNVERIFIED_RUNTIME_CLAIM"),
            "SOURCE_KIND")
    head = value["claimed_code_head"]
    require(type(head) is str and HEX40.fullmatch(head) is not None and head != BASE,
            "EXECUTION_HEAD_UNPINNED")
    _check_selector(value["selector"])
    from scripts import gen5_v10_stage_d2_static_runtime_preflight_candidate as d2
    claims = value["source_claims"]
    if kind == "CPU_SYNTHETIC_FIXTURE":
        require(claims is None, "CPU_FAKE_SOURCE_CLAIMS")
    else:
        try:
            checked = d2.validate_source_claims(claims)
        except (d2.D2Blocked, TypeError, KeyError) as exc:
            raise D3Blocked("GEN5_V10_D3_BLOCKED:D2_SOURCE_CLAIMS") from exc
        require(checked["claimed_code_head"] == head and not checked["source_authenticated"],
                "CLAIMED_HEAD_OR_SOURCE_AUTHENTICATION")
    guards = value["guard_trace"]
    require(type(guards) is dict and set(guards) == set(GUARD_KEYS), "GUARD_FIELDS")
    require(all(type(x) is str and x in ("NOT_AUTHENTICATED", "OBSERVED_CLAIM_ONLY")
                for x in guards.values()), "GUARD_FALSE_ATTESTATION")
    require(type(value["work_counts"]) is dict and value["work_counts"] == WORK_COUNTS and
            all(type(v) is int for v in value["work_counts"].values()), "WORK_COUNTS")
    rows = value["events"]
    require(type(rows) is list and len(rows) == len(EVENTS), "EVENT_COUNT")
    previous = -1
    cap_hashes = []
    prior_peak = -1
    for ordinal, label in enumerate(EVENTS):
        row = rows[ordinal]
        require(type(row) is dict and set(row) == ENTRY_FIELDS, "EVENT_FIELDS:" + label)
        require(type(row["event"]) is str and row["event"] == label and
                type(row["ordinal"]) is int and row["ordinal"] == ordinal,
                "EVENT_ORDER:" + label)
        elapsed = _uint(row["elapsed_ns"], "ELAPSED_NS")
        require(elapsed > previous, "NONMONOTONIC_EVENT_TIME:" + label)
        previous = elapsed
        _memory(row["memory"], kind, label)
        if kind == "UNVERIFIED_RUNTIME_CLAIM":
            peak = row["memory"]["peak_allocated_bytes"]
            require(peak >= prior_peak, "RESET_OR_NONMONOTONIC_PEAK:" + label)
            prior_peak = peak
        h = row["capture_manifest_sha256"]
        if label in CAPTURE_EVENTS:
            require(type(h) is str and HEX64.fullmatch(h) is not None,
                    "CAPTURE_DIGEST_MISSING:" + label)
            cap_hashes.append(h)
        else:
            require(h is None, "NONCAPTURE_DIGEST_INJECTION:" + label)
    require(len(cap_hashes) == 3 and len(set(cap_hashes)) == 1,
            "INPUT_CAPTURE_VALUE_DRIFT")
    for name, literal in (("attestation", "SELF_ASSERTED_UNAUTHENTICATED"),
                          ("GPU_EXECUTION", "NOT_AUTHORIZED"),
                          ("E0", "NOT_EVALUATED"), ("E1", "NOT_EVALUATED"),
                          ("E2", "NOT_EVALUATED"), ("E3", "NOT_EVALUATED"),
                          ("readiness", "BLOCKED_INDEPENDENT_RUNTIME_AND_MEMORY_EVIDENCE")):
        require(type(value[name]) is str and value[name] == literal,
                "FORBIDDEN_PROMOTION:" + name)
    sha = value["observation_sha256"]
    require(type(sha) is str and HEX64.fullmatch(sha) is not None and
            sha == _hash({k:v for k,v in value.items() if k != "observation_sha256"}),
            "OBSERVATION_HASH")
    return {"structure": "VALID_UNTRUSTED_TRACE_ONLY", "event_count": len(rows),
            "capture_phases": 3, "claimed_code_head": head,
            "observed_gpu_peak_authenticated": False,
            "source_tensor_bytes_authenticated": False,
            "guard_execution_authenticated": False,
            "GPU_EXECUTION": "NOT_AUTHORIZED", "E0": "NOT_EVALUATED",
            "E1": "NOT_EVALUATED", "E2": "NOT_EVALUATED", "E3": "NOT_EVALUATED",
            "readiness": "BLOCKED_INDEPENDENT_RUNTIME_AND_MEMORY_EVIDENCE"}


def synthetic_events(*, memory: bool = False) -> list[dict[str, Any]]:
    """Static demo fixture ONLY. Real instrumentation never uses this helper."""
    rows = []
    for i, event in enumerate(EVENTS):
        sample = ({"allocated_bytes": 10, "reserved_bytes": 20,
                   "peak_allocated_bytes": 30, "driver_free_bytes": 50,
                   "driver_total_bytes": 100} if memory else None)
        rows.append({"event": event, "ordinal": i, "elapsed_ns": i + 1,
                     "memory": sample,
                     "capture_manifest_sha256": "1" * 64 if event in CAPTURE_EVENTS else None})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--static-contract-check", action="store_true", required=True)
    a = p.parse_args()
    require(a.static_contract_check, "STATIC_ONLY")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
            "CUDA_DISABLED_REQUIRED")
    report = observation(source_kind="CPU_SYNTHETIC_FIXTURE", claimed_code_head="0"*40,
                         selector=copy.deepcopy(BOUNDS), entries=synthetic_events())
    validate(report)
    print("GEN5_V10_STAGE_D3_CPU_INSTRUMENTATION_STATIC_PASS")
    print("SCHEMA=" + SCHEMA + ";BASE=" + BASE)
    print("EVENTS=" + str(len(EVENTS)) + ";CAPTURES=3;SELECTED_GROUPS=1")
    print("GPU=OFF;MODEL_LOAD=NO;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED;E0_E3=NOT_EVALUATED")
    print("READINESS=BLOCKED_INDEPENDENT_RUNTIME_AND_MEMORY_EVIDENCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
