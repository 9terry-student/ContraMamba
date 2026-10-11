#!/usr/bin/env python3
"""Gen5 v10 Stage D2: CPU-only static memory and runtime-identity preflight.

No model import/load, torch CUDA probe, runtime execution or output artifact.
Checking a Git tree here never authenticates model inputs or scientific E0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

BASE = "6927a3ae96985a50ed1017edd522d6bebd679b6a"
BRANCH = "gen5-causal-role-state-ownership"
SCHEMA = "GEN5_V10_D2_CPU_STATIC_RUNTIME_PREFLIGHT_V1"
CHECKPOINT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b3357f9f8cdbe2b30399e7d2f"
DEV_ORDER_SHA256 = "b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25"
ACTIVE_ENCODING_SHA256 = "e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51"
MODEL_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
EXPECTED_BLOBS = {
    "scripts/audit_native_recurrent_generator_anchor_pair_localization.py": "f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635",
    "scripts/gen5_v10_hybrid_fine_pair_candidate.py": "56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0",
    "scripts/gen5_v10_bounded_tensor_provenance_candidate.py": "703e0f6faf15ee499ec6a97f1aa8edc8d2a93333",
    "scripts/gen5_v10_bounded_worker_export_candidate.py": "cbfc6004c6ab177eea8e7def1cc001049ec4bc4e",
    "scripts/gen5_v10_bounded_runtime_bridge_candidate.py": "7c134614d0e9b4f35e0af70e058a7d184c6332ca",
    "scripts/gen5_v10_e0e3_v3_runtime_verifier_candidate.py": "c506c45eaea394c6ef7aa047aed40e6c0d5fd4f0",
    "scripts/gen5_v10_stage_d_immutable_artifact_candidate.py": "3779066ce63d8ed20ef016d01d52261fe66d6f07",
}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
MAX_CAPTURE_CHUNK_BYTES = 8 * 1024 * 1024
# 3 float32: component K*B*S*W, log_a B*S*W, survival B*S*W.
FLOAT_KEYS = ("component", "log_a", "survival")
MASK_COUNT = 14  # attention + CLAIM/EVIDENCE/EOS + two sides * 5 fine masks
INPUT_KEYS = 17
CPU_CAPTURE_PHASES = ("PRE_FROZEN", "POST_FROZEN", "POST_V10")
STOP_CODES = (
    "CUDA_ACTUAL_PEAK_MEMORY_UNMEASURED",
    "FULL_FROZEN_FFT_TEMPORARY_ALLOCATION_UNMEASURED",
    "V10_GEMM_PEAK_ALLOCATION_UNMEASURED",
    "CUDA_CAPTURE_TEMPORARY_ALLOCATION_UNMEASURED",
    "SOURCE_TENSOR_LINEAGE_NOT_INDEPENDENTLY_AUTHENTICATED",
    "KAGGLE_CUDA_EXECUTION_GATE_NOT_AUTHORIZED",
)


class D2Blocked(ValueError):
    """Fail closed on invalid source or budget input."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise D2Blocked("GEN5_V10_D2_STATIC_BLOCKED:" + reason)


def _posint(value: Any, label: str) -> int:
    require(type(value) is int and value > 0, label + "_POSITIVE_INT")
    return value


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise D2Blocked("INVALID_CANONICAL_JSON") from exc


def memory_floor(*, intermediate_size: int, state_size: int,
                 capture_chunk_bytes: int = MAX_CAPTURE_CHUNK_BYTES) -> dict[str, Any]:
    """Exact logical input bytes; deliberately not a GPU peak allocation bound."""
    i = _posint(intermediate_size, "INTERMEDIATE")
    s = _posint(state_size, "STATE")
    require(i * s == 24576, "EXACT_STATE_WIDTH")
    require(type(capture_chunk_bytes) is int and
            4 <= capture_chunk_bytes <= MAX_CAPTURE_CHUNK_BYTES and
            capture_chunk_bytes % 4 == 0 and
            (capture_chunk_bytes & (capture_chunk_bytes - 1)) == 0,
            "CAPTURE_CHUNK_POLICY")
    k, b, seq, w = 4, 32, 128, 24576
    single_float = b * seq * w * 4
    sizes = {"component": k * single_float,
             "log_a": single_float, "survival": single_float}
    mask_each = b * seq
    input_bytes = sum(sizes.values()) + MASK_COUNT * mask_each
    # The number is theoretical traffic for 3 complete scans; actual PCIe
    # bandwidth/time and memory peaks require independent runtime evidence.
    report = {
        "schema_version": SCHEMA,
        "fixed_locus": {"worker_id": 1, "batch_index_one_based": 13,
                        "row_start": 800, "row_stop": 832, "K": k, "B": b,
                        "S": seq, "W": w, "source_cell": "6201_6201",
                        "target_cells": ["6201_6202", "6201_6203"]},
        "shape": {"intermediate_size": i, "state_size": s},
        "float32_input_bytes": sizes,
        "mask_tensor_count": MASK_COUNT,
        "mask_input_bytes": MASK_COUNT * mask_each,
        "logical_input_bytes_per_capture": input_bytes,
        "logical_input_bytes_three_captures": 3 * input_bytes,
        "logical_tensor_keys": INPUT_KEYS,
        "capture_phase_count": len(CPU_CAPTURE_PHASES),
        "max_leaf_payload_bytes": capture_chunk_bytes,
        "additional_full_cuda_clone_allowed": False,
        "maximum_resident_gpu_memory_certified": False,
        "observed_gpu_free_bytes": None,
        "observed_cuda_peak_bytes": None,
        "source_attestation": "NOT_ESTABLISHED_BY_STATIC_INPUT",
        "readiness": "BLOCKED_PENDING_INDEPENDENT_RUNTIME_EVIDENCE",
        "GPU_EXECUTION": "NOT_AUTHORIZED",
        "V10_E0_PROVENANCE": "NOT_EVALUATED",
        "V10_E1_FROZEN_GUARDS": "NOT_EVALUATED",
        "V10_E2_PAYLOAD": "NOT_EVALUATED",
        "V10_E3_DIAGNOSTICS": "NOT_EVALUATED",
        "PRODUCTION": "BLOCKED",
        "missing_evidence": list(STOP_CODES),
    }
    report["static_contract_sha256"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report


def validate_memory_floor(report: Any) -> None:
    """Recompute every field; self-edited SHA256 cannot forge a readiness PASS."""
    require(type(report) is dict and
            type(report.get("shape")) is dict and
            set(report["shape"]) == {"intermediate_size", "state_size"},
            "MEMORY_REPORT_SHAPE")
    expected = memory_floor(intermediate_size=report["shape"]["intermediate_size"],
                            state_size=report["shape"]["state_size"],
                            capture_chunk_bytes=report.get("max_leaf_payload_bytes"))
    require(report == expected, "MEMORY_REPORT_MISMATCH")


def validate_source_claims(claims: Any) -> dict[str, Any]:
    """Check declared values against frozen references, never attest their origin."""
    fields = {"checkpoint_sha256", "dev_order_sha256", "active_encoding_sha256",
              "model_revision", "worker_id", "batch_index_one_based",
              "row_start", "row_stop", "source_cell", "target_cells",
              "code_head", "claimed_branch", "claimed_clean_worktree",
              "source_status", "physical_device", "torch_version",
              "cuda_runtime", "cublas_version", "allow_tf32"}
    require(type(claims) is dict and set(claims) == fields, "SOURCE_CLAIMS_KEYS")
    for field, expected in (("checkpoint_sha256", CHECKPOINT_SHA256),
                            ("dev_order_sha256", DEV_ORDER_SHA256),
                            ("active_encoding_sha256", ACTIVE_ENCODING_SHA256),
                            ("model_revision", MODEL_REVISION),
                            ("worker_id", 1), ("batch_index_one_based", 13),
                            ("row_start", 800), ("row_stop", 832),
                            ("source_cell", "6201_6201"),
                            ("claimed_branch", BRANCH),
                            ("source_status", "SELF_ASSERTED_NOT_AUTHENTICATED")):
        require(type(claims[field]) is type(expected) and claims[field] == expected,
                "PINNED_CLAIM:" + field)
    require(type(claims["target_cells"]) is list and
            claims["target_cells"] == ["6201_6202", "6201_6203"],
            "TARGET_ORDER")
    require(type(claims["code_head"]) is str and
            HEX40.fullmatch(claims["code_head"]) is not None and
            claims["code_head"] != BASE, "EXECUTION_HEAD_NOT_NEW")
    require(type(claims["claimed_clean_worktree"]) is bool and
            claims["claimed_clean_worktree"] is True, "CLEAN_WORKTREE_CLAIM")
    require(type(claims["allow_tf32"]) is bool and
            claims["allow_tf32"] is False, "TF32_DISABLED_CLAIM")
    for field in ("physical_device", "torch_version", "cuda_runtime", "cublas_version"):
        value = claims[field]
        require(type(value) is str and 0 < len(value.strip()) <= 128 and
                value != "NOT_AUTHENTICATED", "RUNTIME_CLAIM:" + field)
    return {"static_pins_consistent": True,
            "source_authenticated": False,
            "metadata_independently_observed": False,
            "claimed_code_head": claims["code_head"],
            "E0": "NOT_EVALUATED",
            "GPU_EXECUTION": "NOT_AUTHORIZED"}


def inspect_clean_local_git(repo: Path, *, expected_head: str) -> dict[str, Any]:
    """Read-only local Git source check; does NOT attest checkpoint/CUDA data.

    Caller must provide a separately pinned exact 40-char HEAD. No remote
    fetch, mutation, login, model execution, or environmental GPU inspection.
    """
    require(isinstance(repo, Path) and repo.is_dir() and not repo.is_symlink(),
            "GIT_REPOSITORY_PATH")
    require(type(expected_head) is str and HEX40.fullmatch(expected_head) is not None,
            "EXPECTED_HEAD_FORMAT")
    def git(*args: str) -> str:
        try:
            cp = subprocess.run(["git", "-C", str(repo), *args],
                                capture_output=True, text=True, timeout=15, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise D2Blocked("GEN5_V10_D2_STATIC_BLOCKED:GIT_IO_FAILURE") from exc
        require(cp.returncode == 0, "GIT_COMMAND_FAILED:" + args[0])
        return cp.stdout.strip()
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("branch", "--show-current") == BRANCH, "BRANCH_MISMATCH")
    require(git("status", "--porcelain=v1", "--untracked-files=all") == "", "DIRTY_WORKTREE")
    require(git("merge-base", "--is-ancestor", BASE, expected_head) == "",
            "D1_ANCESTOR_MISSING")
    for path, sha in EXPECTED_BLOBS.items():
        require(git("rev-parse", "HEAD:" + path) == sha,
                "FROZEN_GIT_BLOB_DRIFT:" + path)
    return {"local_git_source_checked": True,
            "local_head": expected_head, "branch": BRANCH,
            "pinned_blobs_checked": len(EXPECTED_BLOBS),
            "model_checkpoint_authenticated": False,
            "runtime_tensor_origin_authenticated": False,
            "GPU_EXECUTION": "NOT_AUTHORIZED", "E0": "NOT_EVALUATED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-contract-check", action="store_true", required=True)
    args = parser.parse_args()
    require(args.static_contract_check, "STATIC_ONLY")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
            "CUDA_MUST_BE_DISABLED")
    result = memory_floor(intermediate_size=1, state_size=24576)
    validate_memory_floor(result)
    print("GEN5_V10_STAGE_D2_CPU_STATIC_PREFLIGHT_PASS")
    print("SCHEMA=" + SCHEMA + ";BASE=" + BASE)
    print("B32_INPUT_LOGICAL_BYTES=" + str(result["logical_input_bytes_per_capture"]))
    print("THREE_CAPTURE_LOGICAL_BYTES=" + str(result["logical_input_bytes_three_captures"]))
    print("GPU=OFF;MODEL_LOAD=NO;GPU_EXECUTION=NOT_AUTHORIZED;E0=NOT_EVALUATED")
    print("READINESS=BLOCKED_PENDING_INDEPENDENT_RUNTIME_EVIDENCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
