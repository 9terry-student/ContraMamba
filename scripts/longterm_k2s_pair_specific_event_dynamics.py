"""K2S pair-specific exact-prefix native Mamba event dynamics.

This is a fail-closed, preregistered scientific runner.  It uses the frozen
300-item K2W construction-valid population, deterministic reciprocal blocks,
and direct post-consumption selective-SSM recurrent states.  A0 task-head
predictions are never used for selection, primary endpoints, or promotion.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Frozen scientific / provenance constants
# ---------------------------------------------------------------------------
K2S_PREREG_AUTHORITY_COMMIT = "c9cb68c2a48a19c3874d059ca00f5c297d929ada"
K2W_CLOSURE_COMMIT = "b94f81b411bcbc74e32ad0ad6564b8978016eec0"
A0_COMMIT = "55debe94f0d19d16a334395e8561901fed6b52fa"
SOURCE_BLOB_COMMIT = "8eb7386e0344117d026c0e6ab172018bb98a698e"

POPULATION_REL = (
    "reports/longterm_k2w_fixed_window_phase_a_c7c7a0c218bb/"
    "candidate_pool.jsonl"
)
POPULATION_SHA256 = "abf693d3267cc4e3dd27a8127d2948b36fdf8ba24e135a643215f0f31a26d808"
SOURCE_REL = (
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458/"
    "controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
)
SOURCE_PHYSICAL_SHA256 = "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
SOURCE_SEMANTIC_SHA256 = "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"

SEED = "seed180"
EXPECTED_ZIP_SHA256 = "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
EXPECTED_CHECKPOINT_SHA256 = "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
HANDOFF_SCHEMA = "contramamba-handoff-v3"
HANDOFF_JSON_MAX_BYTES = 2 * 1024 * 1024

COMMON_ENCODER_CANONICAL_SHA256 = "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
COMMON_ENCODER_RAW_CONCAT_SHA256 = "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"
COMMON_ENCODER_TENSOR_COUNT = 242
COMMON_ENCODER_NUMEL = 129_135_360
COMMON_ENCODER_RAW_BYTES = 516_541_440

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"
HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
K2S_UNTRACKED = {
    "scripts/longterm_k2s_pair_specific_event_dynamics.py",
    "tests/test_longterm_k2s_pair_specific_event_dynamics.py",
}

W = 8
EPSILON = 1e-12
PRIMARY_LAYER = 23
N_LAYERS = 24
N_ITEMS = 300
N_BLOCKS = 150
PRIMARY_ORDER = ("R", "D", "P")

MANIFEST_SCHEMA = "k2s-pair-specific-manifest-v1"
ITEM_SCHEMA = "k2s-item-metrics-v1"
LAYER_SCHEMA = "k2s-layer-metrics-v1"
BLOCK_SCHEMA = "k2s-block-metrics-v1"
PRIMARY_SCHEMA = "k2s-primary-stats-v1"

POSITIVE_VERDICT = "PAIR_SPECIFIC_EVENT_ALIGNED_NATIVE_STATE_RESPONSE_OBSERVED"
NULL_VERDICT = "PRIMARY_PAIR_SPECIFICITY_NULL_AFTER_VALID_EXECUTION"
REVERSED_VERDICT = "PAIR_SPECIFICITY_REVERSED_AFTER_VALID_EXECUTION"
MIXED_VERDICT = "MIXED_PRIMARY_PAIR_SPECIFICITY_RESULT"
INSTRUMENTATION_FAILURE = "INSTRUMENTATION_OR_PROVENANCE_FAILURE"

A0_MODEL_REL = "src/contramamba/modeling_v6b_minimal.py"
A0_HEADS_REL = "src/contramamba/heads"


class ContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_line(value: Any) -> bytes:
    return canonical_json(value) + b"\n"


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_line(dict(row)) for row in rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"git failure: {' '.join(args)}") from exc


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(["git", *args], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"git failure: {' '.join(args)}") from exc


def git_provenance(root: Path, instrumentation_preflight: bool = False) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")
    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", K2S_PREREG_AUTHORITY_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "K2S_PREREG_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()
    allowed = set(HISTORICAL_K1_UNTRACKED)
    if instrumentation_preflight:
        allowed |= K2S_UNTRACKED
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(line[:2] == "??" and path in allowed, "GIT_DIRTY_CONTRACT_MISMATCH")

    # The full A0 model is used only for the synthetic noninterference gate.
    # Fail closed if its task-model source or heads drifted from the A0 lineage.
    current_model_blob = _git(root, "rev-parse", f"{head}:{A0_MODEL_REL}")
    a0_model_blob = _git(root, "rev-parse", f"{A0_COMMIT}:{A0_MODEL_REL}")
    current_heads_tree = _git(root, "rev-parse", f"{head}:{A0_HEADS_REL}")
    a0_heads_tree = _git(root, "rev-parse", f"{A0_COMMIT}:{A0_HEADS_REL}")
    require(current_model_blob == a0_model_blob, "A0_MODEL_SOURCE_DRIFT")
    require(current_heads_tree == a0_heads_tree, "A0_HEADS_SOURCE_DRIFT")

    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "runtime_dirty_contract": status,
        "a0_model_blob_sha": current_model_blob,
        "a0_heads_tree_sha": current_heads_tree,
    }


# ---------------------------------------------------------------------------
# Frozen population and reciprocal design
# ---------------------------------------------------------------------------
def parse_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    require(not raw.startswith(b"\xef\xbb\xbf"), "JSONL_BOM_FORBIDDEN")
    require(b"\r" not in raw, "JSONL_CR_FORBIDDEN")
    require(raw.endswith(b"\n"), "JSONL_FINAL_LF_REQUIRED")
    lines = raw[:-1].split(b"\n")
    require(bool(lines) and all(lines), "JSONL_BLANK_LINE_FORBIDDEN")
    rows: list[dict[str, Any]] = []
    try:
        for line in lines:
            value = json.loads(line.decode("utf-8", "strict"))
            require(isinstance(value, dict), "JSONL_ROW_NOT_OBJECT")
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("JSONL_DECODE_FAILURE") from exc
    return rows


def load_frozen_population(root: Path) -> list[dict[str, Any]]:
    raw = _git_bytes(root, "show", f"{K2W_CLOSURE_COMMIT}:{POPULATION_REL}")
    require(sha256_bytes(raw) == POPULATION_SHA256, "POPULATION_SHA256_MISMATCH")
    rows = parse_jsonl_bytes(raw)
    require(len(rows) == N_ITEMS, "POPULATION_COUNT_MISMATCH")
    required = {
        "stable_item_id",
        "base_claim_sha256",
        "pair_id",
        "prefix_text",
        "correction_text",
        "control_text",
        "construction_status",
        "source_dataset_physical_sha256",
        "source_dataset_semantic_sha256",
    }
    for row in rows:
        require(required <= set(row), "POPULATION_FIELD_MISSING")
        require(row["construction_status"] == "valid", "POPULATION_NOT_ALL_VALID")
        require(
            row["source_dataset_physical_sha256"] == SOURCE_PHYSICAL_SHA256
            and row["source_dataset_semantic_sha256"] == SOURCE_SEMANTIC_SHA256,
            "POPULATION_SOURCE_IDENTITY_MISMATCH",
        )
        for key in ("stable_item_id", "base_claim_sha256", "pair_id", "prefix_text", "correction_text", "control_text"):
            require(isinstance(row[key], str) and row[key] != "", f"POPULATION_{key}_INVALID")
    require(len({r["stable_item_id"] for r in rows}) == N_ITEMS, "STABLE_ID_NOT_UNIQUE")
    require(len({r["base_claim_sha256"] for r in rows}) == N_ITEMS, "BASE_CLAIM_NOT_UNIQUE")
    return sorted(rows, key=lambda r: r["stable_item_id"])


def donor_index(index: int) -> int:
    require(type(index) is int and 0 <= index < N_ITEMS, "DONOR_INDEX_INVALID")
    return index ^ 1


def reciprocal_blocks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(rows) == N_ITEMS, "RECIPROCAL_POPULATION_COUNT")
    blocks: list[dict[str, Any]] = []
    for block_index in range(N_BLOCKS):
        a = 2 * block_index
        b = a + 1
        ra, rb = rows[a], rows[b]
        require(donor_index(a) == b and donor_index(b) == a, "RECIPROCAL_MAPPING_BROKEN")
        require(ra["stable_item_id"] != rb["stable_item_id"], "RECIPROCAL_STABLE_ID_COLLISION")
        require(ra["base_claim_sha256"] != rb["base_claim_sha256"], "RECIPROCAL_BASE_CLAIM_COLLISION")
        blocks.append(
            {
                "block_index": block_index,
                "block_id": f"k2s-block-{block_index:03d}",
                "item_a_index": a,
                "item_b_index": b,
                "item_a_stable_id": ra["stable_item_id"],
                "item_b_stable_id": rb["stable_item_id"],
            }
        )
    return blocks


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, Mapping) else out.input_ids
    return list(ids)


def first_divergence_after_prefix(corr: Sequence[int], ctrl: Sequence[int], prefix_len: int) -> int | None:
    stop = min(len(corr), len(ctrl), prefix_len + W)
    return next((i for i in range(prefix_len, stop) if corr[i] != ctrl[i]), None)


def build_input_contracts(rows: Sequence[Mapping[str, Any]], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(getattr(tokenizer, "is_fast", False), "TOKENIZER_MUST_BE_FAST")
    require(len(rows) == N_ITEMS, "INPUT_CONTRACT_COUNT")

    contracts: list[dict[str, Any]] = []
    matched_d: dict[str, int] = {}
    swapped_d: dict[str, int] = {}
    availability: dict[str, list[int]] = {
        "matched_corr": [], "matched_ctrl": [], "swapped_corr": [], "swapped_ctrl": []
    }

    for i, row in enumerate(rows):
        donor = rows[donor_index(i)]
        prefix = row["prefix_text"]
        p_ids = _token_ids(tokenizer, prefix)
        require(len(p_ids) >= 2, "PREFIX_TOO_SHORT_FOR_PRESTATE")
        p = len(p_ids) - 1

        texts = {
            "matched_corr": prefix + row["correction_text"],
            "matched_ctrl": prefix + row["control_text"],
            "swapped_corr": prefix + donor["correction_text"],
            "swapped_ctrl": prefix + donor["control_text"],
        }
        token_arrays: dict[str, list[int]] = {}
        for name, text in texts.items():
            ids = _token_ids(tokenizer, text)
            require(ids[: len(p_ids)] == p_ids, "INVALID_EXACT_PREFIX")
            require(len(ids) - len(p_ids) >= W, "INVALID_WINDOW8")
            token_arrays[name] = ids
            availability[name].append(len(ids) - len(p_ids))

        md = first_divergence_after_prefix(token_arrays["matched_corr"], token_arrays["matched_ctrl"], len(p_ids))
        sd = first_divergence_after_prefix(token_arrays["swapped_corr"], token_arrays["swapped_ctrl"], len(p_ids))
        require(md is not None, "MATCHED_NO_DIVERGENCE_WITHIN_W8")
        require(sd is not None, "SWAPPED_NO_DIVERGENCE_WITHIN_W8")
        matched_d[str(md - p)] = matched_d.get(str(md - p), 0) + 1
        swapped_d[str(sd - p)] = swapped_d.get(str(sd - p), 0) + 1

        contracts.append(
            {
                "item_index": i,
                "block_index": i // 2,
                "block_id": f"k2s-block-{i // 2:03d}",
                "stable_item_id": row["stable_item_id"],
                "pair_id": row["pair_id"],
                "base_claim_sha256": row["base_claim_sha256"],
                "donor_index": donor_index(i),
                "donor_stable_item_id": donor["stable_item_id"],
                "donor_base_claim_sha256": donor["base_claim_sha256"],
                "prefix_text": prefix,
                "correction_text": row["correction_text"],
                "control_text": row["control_text"],
                "donor_correction_text": donor["correction_text"],
                "donor_control_text": donor["control_text"],
                "p": p,
                "prefix_token_ids": p_ids,
                "branch_texts": texts,
                "branch_token_ids": token_arrays,
                "matched_d_minus_p": md - p,
                "swapped_d_minus_p": sd - p,
            }
        )

    # Exact marginal preservation is text-level and prospective.
    require(
        sorted(r["correction_text"] for r in rows)
        == sorted(rows[donor_index(i)]["correction_text"] for i in range(N_ITEMS)),
        "CORRECTION_MARGINAL_NOT_PRESERVED",
    )
    require(
        sorted(r["control_text"] for r in rows)
        == sorted(rows[donor_index(i)]["control_text"] for i in range(N_ITEMS)),
        "CONTROL_MARGINAL_NOT_PRESERVED",
    )

    summary = {
        "N_total": N_ITEMS,
        "N_blocks": N_BLOCKS,
        "N_matched_valid": N_ITEMS,
        "N_swapped_valid": N_ITEMS,
        "matched_d_minus_p": dict(sorted(matched_d.items(), key=lambda kv: int(kv[0]))),
        "swapped_d_minus_p": dict(sorted(swapped_d.items(), key=lambda kv: int(kv[0]))),
        "matched_corr_available_minmax": [min(availability["matched_corr"]), max(availability["matched_corr"])],
        "matched_ctrl_available_minmax": [min(availability["matched_ctrl"]), max(availability["matched_ctrl"])],
        "swapped_corr_available_minmax": [min(availability["swapped_corr"]), max(availability["swapped_corr"])],
        "swapped_ctrl_available_minmax": [min(availability["swapped_ctrl"]), max(availability["swapped_ctrl"])],
        "prefix_marginal_preserved": True,
        "correction_marginal_preserved": True,
        "control_marginal_preserved": True,
    }
    expected = {
        "N_total": 300,
        "N_blocks": 150,
        "N_matched_valid": 300,
        "N_swapped_valid": 300,
        "matched_d_minus_p": {"2": 149, "3": 151},
        "swapped_d_minus_p": {"2": 149, "3": 151},
        "matched_corr_available_minmax": [20, 27],
        "matched_ctrl_available_minmax": [17, 26],
        "swapped_corr_available_minmax": [20, 27],
        "swapped_ctrl_available_minmax": [17, 26],
        "prefix_marginal_preserved": True,
        "correction_marginal_preserved": True,
        "control_marginal_preserved": True,
    }
    require(summary == expected, "K2S_FROZEN_TOKEN_FEASIBILITY_MISMATCH")
    return contracts, summary


# ---------------------------------------------------------------------------
# Handoff / checkpoint / HF identity (ported from validated K2W machinery)
# ---------------------------------------------------------------------------
def _safe_member(name: str) -> str:
    if not name or "\\" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ":" in name:
        raise ContractError("HANDOFF_MEMBER_PATH_INVALID")
    parts = PurePosixPath(name).parts
    if not parts or any(part in (".", "..") for part in parts) or "/".join(parts) != name:
        raise ContractError("HANDOFF_MEMBER_PATH_INVALID")
    return name


def _safe_zip_names(archive: zipfile.ZipFile) -> list[str]:
    names = [_safe_member(info.filename) for info in archive.infolist()]
    require(len(names) == len(set(names)), "HANDOFF_MEMBER_DUPLICATE")
    return names


def discover_handoff_manifest(archive: zipfile.ZipFile) -> tuple[str, dict[str, Any]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for info in archive.infolist():
        name = _safe_member(info.filename)
        if name.endswith(".json") and info.file_size <= HANDOFF_JSON_MAX_BYTES:
            try:
                value = json.loads(archive.read(info).decode("utf-8", "strict"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("schema") == HANDOFF_SCHEMA:
                found.append((name, value))
    require(len(found) == 1, "HANDOFF_MANIFEST_MISSING_OR_AMBIGUOUS")
    return found[0]


def audit_handoff(path: Path) -> dict[str, Any]:
    require(path.is_file(), "HANDOFF_FILE_MISSING")
    require(file_sha256(path) == EXPECTED_ZIP_SHA256, "HANDOFF_ZIP_SHA256_MISMATCH")
    with zipfile.ZipFile(path) as archive:
        names = _safe_zip_names(archive)
        manifest_member, manifest = discover_handoff_manifest(archive)
        require(
            manifest.get("expected_commit") == A0_COMMIT and manifest.get("actual_commit") == A0_COMMIT,
            "HANDOFF_COMMIT_MISMATCH",
        )
        files = manifest.get("files")
        matches = [
            row
            for row in files
            if isinstance(row, Mapping) and row.get("sha256") == EXPECTED_CHECKPOINT_SHA256
        ] if isinstance(files, list) else []
        require(len(matches) == 1, "HANDOFF_CHECKPOINT_MANIFEST_MISSING_OR_AMBIGUOUS")
        row = matches[0]
        require(isinstance(row.get("path"), str) and isinstance(row.get("size_bytes"), int), "HANDOFF_CHECKPOINT_MANIFEST_INVALID")
        checkpoint_member = _safe_member("files/" + _safe_member(row["path"]))
        require(checkpoint_member in names, "HANDOFF_CHECKPOINT_MEMBER_MISSING")
        data = archive.read(checkpoint_member)
        require(len(data) == row["size_bytes"], "HANDOFF_CHECKPOINT_SIZE_MISMATCH")
        require(sha256_bytes(data) == EXPECTED_CHECKPOINT_SHA256, "HANDOFF_CHECKPOINT_SHA256_MISMATCH")
    return {
        "seed": SEED,
        "zip_path": str(path.resolve()),
        "zip_sha256": EXPECTED_ZIP_SHA256,
        "manifest_member": manifest_member,
        "checkpoint_member": checkpoint_member,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": len(data),
    }


def load_authenticated_checkpoint(handoff: Mapping[str, Any]) -> Mapping[str, Any]:
    import torch

    with zipfile.ZipFile(handoff["zip_path"]) as archive:
        data = archive.read(handoff["checkpoint_member"])
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, Mapping), "CHECKPOINT_NOT_MAPPING")
    require(checkpoint.get("schema_version") == "stage176a0_selected_checkpoint_v1", "CHECKPOINT_SCHEMA_INVALID")
    require(isinstance(checkpoint.get("model_state_dict"), Mapping), "CHECKPOINT_STATE_DICT_MISSING")
    require(
        isinstance(checkpoint.get("metadata"), Mapping)
        and isinstance(checkpoint["metadata"].get("training_args"), Mapping),
        "CHECKPOINT_TRAINING_ARGS_MISSING",
    )
    return checkpoint


def encoder_fingerprint(state: Mapping[str, Any]) -> dict[str, Any]:
    tensor_digests: dict[str, str] = {}
    raw_chunks: list[bytes] = []
    total_numel = 0
    dtypes: set[str] = set()
    for key, value in sorted(state.items()):
        if not key.startswith("mamba."):
            continue
        tensor = value.detach().cpu().contiguous()
        raw = tensor.numpy().tobytes()
        tensor_digests[key] = sha256_bytes(raw)
        raw_chunks.append(raw)
        total_numel += tensor.numel()
        dtypes.add(str(tensor.dtype))
    result = {
        "canonical_digest": sha256_bytes(canonical_json(tensor_digests)),
        "raw_concat_digest": sha256_bytes(b"".join(raw_chunks)),
        "tensor_count": len(tensor_digests),
        "total_numel": total_numel,
        "total_raw_bytes": sum(len(x) for x in raw_chunks),
        "dtypes": sorted(dtypes),
    }
    require(result["canonical_digest"] == COMMON_ENCODER_CANONICAL_SHA256, "COMMON_ENCODER_CANONICAL_DIGEST_MISMATCH")
    require(result["raw_concat_digest"] == COMMON_ENCODER_RAW_CONCAT_SHA256, "COMMON_ENCODER_RAW_CONCAT_DIGEST_MISMATCH")
    require(result["tensor_count"] == COMMON_ENCODER_TENSOR_COUNT, "COMMON_ENCODER_TENSOR_COUNT_MISMATCH")
    require(result["total_numel"] == COMMON_ENCODER_NUMEL, "COMMON_ENCODER_NUMEL_MISMATCH")
    require(result["total_raw_bytes"] == COMMON_ENCODER_RAW_BYTES, "COMMON_ENCODER_RAW_BYTES_MISMATCH")
    require(result["dtypes"] == ["torch.float32"], "COMMON_ENCODER_DTYPE_MISMATCH")
    return result


def resolve_hf_snapshot(revision: str) -> tuple[Path, dict[str, Any]]:
    require(revision == HF_REVISION, "HF_REVISION_MISMATCH")
    from huggingface_hub import snapshot_download, __version__ as hub_version
    from transformers import AutoConfig, AutoTokenizer, __version__ as transformers_version

    snapshot = Path(
        snapshot_download(
            repo_id=HF_MODEL,
            revision=revision,
            allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json", "vocab.*", "merges.txt"],
        )
    )
    require(snapshot.name == revision, "HF_RESOLVED_REVISION_MISMATCH")
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    require(tokenizer.is_fast, "TOKENIZER_MUST_BE_FAST")
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and (
            path.name == "config.json"
            or "tokenizer" in path.name
            or path.name in {"vocab.json", "merges.txt", "special_tokens_map.json"}
        ):
            files.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return snapshot, {
        "hf_model_id": HF_MODEL,
        "requested_hf_revision": revision,
        "resolved_hf_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_backend_class": type(tokenizer.backend_tokenizer).__name__,
        "tokenizer_is_fast": True,
        "tokenizer_files": files,
        "transformers_version": transformers_version,
        "huggingface_hub_version": hub_version,
        "config": config,
        "tokenizer": tokenizer,
    }


def normalized_a0_constructor_args(checkpoint: Mapping[str, Any], backbone: Any) -> dict[str, Any]:
    state = checkpoint["model_state_dict"]
    args = checkpoint["metadata"]["training_args"]
    names = set(state)
    flags = {
        "use_boundary_head": "boundary_head.",
        "use_frame_violation_head": "frame_violation_head.",
        "use_predicate_isolation_head": "predicate_isolation_head.",
        "use_preservation_entitlement_head": "preservation_entitlement_head.",
        "use_temporal_diagnostic_head": "temporal_diagnostic_head.",
        "use_temporal_residual_adapter": "temporal_residual_adapter.",
        "use_temporal_channel": "temporal_channel_v1.",
    }
    return {
        "backbone": backbone,
        "frame_size": 128,
        "predicate_size": 128,
        "sufficiency_size": 128,
        "energy_size": 64,
        "dropout": 0.1,
        "freeze_a_log": False,
        "decision_mode": "explicit_product",
        "reason_router_epsilon": float(args.get("reason_router_epsilon", 1e-8)),
        "use_temporal_comparator": "alpha_temporal_raw" in names,
        "use_predicate_comparator": "alpha_predicate_raw" in names,
        "alpha_temporal_init": 1.25,
        "alpha_predicate_init": 1.25,
        **{name: any(key.startswith(prefix) for key in names) for name, prefix in flags.items()},
    }


def build_a0_model(root: Path, snapshot: Path, checkpoint: Mapping[str, Any]) -> Any:
    from transformers import MambaConfig, MambaModel

    sys.path[:0] = [str(root), str(root / "src")]
    from contramamba.modeling_v6b_minimal import ContraMambaV6BMinimal

    backbone = MambaModel(
        MambaConfig.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    )
    model = ContraMambaV6BMinimal(**normalized_a0_constructor_args(checkpoint, backbone))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# O0c-compatible native recurrent-state instrumentation
# ---------------------------------------------------------------------------
def _target_name(node: ast.AST) -> str:
    target: ast.AST | None = None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


def analyze_mamba_source(source: bytes) -> dict[str, int]:
    """Prove sequential recurrence update -> readout and return readout line.

    The trace line is the readout statement. CPython emits its line event before
    executing that statement, hence after the immediately preceding recurrent
    update and before the state is consumed by the readout. This is the O0c
    post-consumption s_t timing convention.
    """
    try:
        tree = ast.parse(source.decode("utf-8", "strict"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ContractError("MAMBA_SOURCE_PARSE_FAILURE") from exc
    mixers = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MambaMixer"]
    require(len(mixers) == 1, "MAMBA_MIXER_CLASS_AMBIGUOUS")
    slow = [n for n in mixers[0].body if isinstance(n, ast.FunctionDef) and n.name == "slow_forward"]
    forward = [n for n in mixers[0].body if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    require(len(slow) == 1 and len(forward) == 1, "MAMBA_FORWARD_SOURCE_AMBIGUOUS")

    candidates: list[tuple[ast.For, ast.AST, ast.AST]] = []
    for node in ast.walk(slow[0]):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name) or node.target.id != "i":
            continue
        for pos in range(len(node.body) - 1):
            update = node.body[pos]
            readout = node.body[pos + 1]
            update_value = update.value if isinstance(update, (ast.Assign, ast.AnnAssign)) else None
            readout_value = readout.value if isinstance(readout, (ast.Assign, ast.AnnAssign)) else None
            if _target_name(update) != "ssm_state" or _target_name(readout) != "scan_output":
                continue
            update_names = _names(update_value)
            readout_names = _names(readout_value)
            update_shape_ok = (
                isinstance(update_value, ast.BinOp)
                and isinstance(update_value.op, ast.Add)
                and isinstance(update_value.left, ast.BinOp)
                and isinstance(update_value.left.op, ast.Mult)
                and {"discrete_A", "ssm_state", "i"} <= _names(update_value.left)
                and {"deltaB_u", "i"} <= _names(update_value.right)
            )
            readout_shape_ok = (
                isinstance(readout_value, ast.Call)
                and isinstance(readout_value.func, ast.Attribute)
                and readout_value.func.attr == "matmul"
                and len(readout_value.args) >= 2
                and "ssm_state" in _names(readout_value.args[0])
                and {"C", "i"} <= _names(readout_value.args[1])
            )
            if (
                update_shape_ok
                and readout_shape_ok
                and {"discrete_A", "ssm_state", "deltaB_u", "i"} <= update_names
                and {"ssm_state", "C", "i"} <= readout_names
            ):
                candidates.append((node, update, readout))
    require(len(candidates) == 1, "MAMBA_SEQUENTIAL_RECURRENCE_AMBIGUOUS")
    _, update, readout = candidates[0]

    # Prove ordinary forward has an explicit slow-forward fallback.
    slow_calls = [
        n
        for n in ast.walk(forward[0])
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "slow_forward"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    ]
    require(len(slow_calls) == 1, "MAMBA_FORWARD_SLOW_FALLBACK_MISSING")
    return {
        "recurrence_update_line": int(update.lineno),
        "capture_line": int(readout.lineno),
        "slow_forward_line": int(slow[0].lineno),
        "forward_line": int(forward[0].lineno),
    }


@dataclass(frozen=True)
class CaptureBinding:
    code: Any
    capture_line: int
    source_path: Path
    source_sha256: str
    source_bytes: int
    recurrence_update_line: int
    slow_forward_line: int
    forward_line: int
    qualname: str


def resolve_capture_binding() -> CaptureBinding:
    import transformers.models.mamba.modeling_mamba as module

    function = module.MambaMixer.slow_forward
    require(inspect.isfunction(function), "MAMBA_SLOW_FORWARD_NOT_FUNCTION")
    source_path = Path(inspect.getsourcefile(function) or "").resolve()
    require(source_path.is_file(), "MAMBA_SOURCE_FILE_MISSING")
    raw = source_path.read_bytes()
    analysis = analyze_mamba_source(raw)
    require(function.__code__.co_filename == str(source_path), "MAMBA_CODE_SOURCE_MISMATCH")
    code_lines = {line for _, _, line in function.__code__.co_lines() if line is not None}
    require(analysis["capture_line"] in code_lines, "MAMBA_CAPTURE_LINE_NOT_EXECUTABLE")
    return CaptureBinding(
        code=function.__code__,
        capture_line=analysis["capture_line"],
        source_path=source_path,
        source_sha256=sha256_bytes(raw),
        source_bytes=len(raw),
        recurrence_update_line=analysis["recurrence_update_line"],
        slow_forward_line=analysis["slow_forward_line"],
        forward_line=analysis["forward_line"],
        qualname="MambaMixer.slow_forward",
    )


class TraceCollector:
    """O0c-style CPython line collector for post-consumption s_t snapshots."""

    def __init__(
        self,
        binding: CaptureBinding,
        registered_layers: Mapping[int, int],
        target_indices: Iterable[int],
        enabled: bool = True,
    ) -> None:
        self.binding = binding
        self.layers = dict(registered_layers)
        require(len(self.layers) == N_LAYERS, "REGISTERED_LAYER_COUNT_MISMATCH")
        require(len(set(self.layers.values())) == N_LAYERS, "REGISTERED_LAYER_IDENTITY_DUPLICATE")
        self.target_indices = frozenset(int(i) for i in target_indices)
        require(self.target_indices and min(self.target_indices) >= 0, "TRACE_TARGET_INDICES_INVALID")
        self.enabled = bool(enabled)
        self.snapshots: dict[tuple[int, int], Any] | None = None
        self._prior = None

    def _trace(self, frame: Any, event: str, arg: Any):
        if frame.f_code is self.binding.code and event == "line" and frame.f_lineno == self.binding.capture_line:
            layer_index = self.layers.get(id(frame.f_locals.get("self")))
            if layer_index is None:
                return self._trace
            token_index = frame.f_locals.get("i")
            require(type(token_index) is int and token_index >= 0, "AMBIGUOUS_TOKEN_INDEX")
            if token_index not in self.target_indices:
                return self._trace
            state = frame.f_locals.get("ssm_state")
            require(state is not None and hasattr(state, "detach") and hasattr(state, "clone"), "NOT_NATIVE_RECURRENT_STATE")
            snapshot = state.detach().cpu().contiguous().clone()
            require(snapshot is not state, "SNAPSHOT_ALIAS")
            key = (int(layer_index), int(token_index))
            require(self.snapshots is not None and key not in self.snapshots, "DUPLICATE_STATE_COORDINATE")
            self.snapshots[key] = snapshot
        return self._trace

    @contextmanager
    def capture(self):
        if not self.enabled:
            yield self
            return
        require(self.snapshots is None, "TRACE_COLLECTOR_REUSE")
        self.snapshots = {}
        self._prior = sys.gettrace()
        sys.settrace(self._trace)
        try:
            yield self
        finally:
            sys.settrace(self._prior)


def registered_mamba_layers(model: Any) -> dict[int, int]:
    layers = getattr(getattr(model, "mamba", None), "layers", None)
    require(layers is not None and len(layers) == N_LAYERS, "MAMBA_LAYER_COUNT_MISMATCH")
    mapping: dict[int, int] = {}
    for index, block in enumerate(layers):
        mixer = getattr(block, "mixer", None)
        require(mixer is not None, "MAMBA_MIXER_MISSING")
        mapping[id(mixer)] = index
    require(len(mapping) == N_LAYERS, "MAMBA_MIXER_ID_COLLISION")
    return mapping


def tensor_sha256(tensor: Any) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def validate_snapshot_tensor(snapshot: Any, mixer: Any) -> None:
    import torch

    expected = (1, int(mixer.intermediate_size), int(mixer.ssm_state_size))
    require(tuple(snapshot.shape) == expected, "RECURRENT_STATE_SHAPE_MISMATCH")
    require(snapshot.dtype == torch.float32, "RECURRENT_STATE_DTYPE_MISMATCH")
    require(snapshot.device.type == "cpu", "RECURRENT_STATE_DEVICE_MISMATCH")
    require(bool(torch.isfinite(snapshot).all().item()), "RECURRENT_STATE_NONFINITE")


def task_mask_bundle(tokenizer: Any, text: str) -> dict[str, list[int]]:
    """Synthetic preflight-only masks for the frozen A0 task model."""
    require(getattr(tokenizer, "is_fast", False), "TOKENIZER_MUST_BE_FAST")
    out = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(out["input_ids"])
    offsets = [tuple(v) for v in out["offset_mapping"]]
    claim_start = len("Claim: ")
    claim_end = text.index("\nEvidence: ")
    evidence_start = claim_end + len("\nEvidence: ")
    evidence_end = text.index("\nAdditional evidence:\n")
    claim = [int(a < claim_end and b > claim_start) for a, b in offsets]
    evidence = [int(a < evidence_end and b > evidence_start) for a, b in offsets]
    require(any(claim) and any(evidence), "SYNTHETIC_MASK_EMPTY")
    require(not any(a and b for a, b in zip(claim, evidence)), "SYNTHETIC_MASK_OVERLAP")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "claim_mask": claim,
        "evidence_mask": evidence,
    }


def _full_model_forward(model: Any, bundle: Mapping[str, Sequence[int]]) -> Any:
    import torch

    tensors = {
        key: torch.tensor(value, dtype=torch.long).unsqueeze(0)
        for key, value in bundle.items()
        if key in {"input_ids", "attention_mask", "claim_mask", "evidence_mask"}
    }
    with torch.inference_mode():
        return model(**tensors)


def _logits(output: Any) -> Any:
    if isinstance(output, Mapping):
        return output["logits"]
    return output.logits


def run_instrumentation_preflight(model: Any, tokenizer: Any, binding: CaptureBinding) -> dict[str, Any]:
    import torch

    model.eval()
    layer_map = registered_mamba_layers(model)
    mixers = [model.mamba.layers[i].mixer for i in range(N_LAYERS)]
    prefix = "Claim: synthetic blorp\nEvidence: synthetic snarp\nAdditional evidence:\n"
    continuations = (
        " synthetic corrective alpha beta gamma delta epsilon zeta eta theta iota.",
        " synthetic control kappa lambda mu nu xi omicron pi rho sigma.",
        " synthetic swapped tau upsilon phi chi psi omega amber cobalt jade.",
        " synthetic alternate cedar quartz river summit valley willow ember frost.",
    )
    texts = [prefix + continuation for continuation in continuations]
    bundles = [task_mask_bundle(tokenizer, text) for text in texts]
    prefix_ids = _token_ids(tokenizer, prefix)
    require(all(bundle["input_ids"][: len(prefix_ids)] == prefix_ids for bundle in bundles), "SYNTHETIC_PREFIX_TOKEN_MISMATCH")
    p = len(prefix_ids) - 1
    require(p >= 1 and all(len(bundle["input_ids"]) - len(prefix_ids) >= W for bundle in bundles), "SYNTHETIC_WINDOW_INVALID")

    # Exact ordinary-logit noninterference on the first synthetic branch.
    baseline = _full_model_forward(model, bundles[0])
    baseline_logits = _logits(baseline).detach().cpu().clone()
    target_all = range(len(bundles[0]["input_ids"]))
    collector = TraceCollector(binding, layer_map, target_all, enabled=True)
    with collector.capture():
        traced = _full_model_forward(model, bundles[0])
    traced_logits = _logits(traced).detach().cpu()
    require(torch.equal(baseline_logits, traced_logits), "TRACE_LOGIT_NONINTERFERENCE_FAILURE")
    require(collector.snapshots is not None, "TRACE_SNAPSHOTS_MISSING")
    expected_count = N_LAYERS * len(bundles[0]["input_ids"])
    require(len(collector.snapshots) == expected_count, "TRACE_CAPTURE_INCOMPLETE")
    for (layer_index, _), snapshot in collector.snapshots.items():
        validate_snapshot_tensor(snapshot, mixers[layer_index])

    # Fresh-state isolation: same input, fresh collector, byte-identical snapshots.
    collector2 = TraceCollector(binding, layer_map, target_all, enabled=True)
    with collector2.capture():
        _full_model_forward(model, bundles[0])
    require(collector2.snapshots is not None, "TRACE_SECOND_SNAPSHOTS_MISSING")
    require(set(collector.snapshots) == set(collector2.snapshots), "TRACE_FRESH_COORDINATE_MISMATCH")
    for key in collector.snapshots:
        require(torch.equal(collector.snapshots[key], collector2.snapshots[key]), "TRACE_FRESH_STATE_MISMATCH")

    # Exact causal-prefix identity across four independent full forwards.
    prefix_targets = range(p + 1)
    reference_hashes: dict[tuple[int, int], str] | None = None
    for branch_index, bundle in enumerate(bundles):
        c = TraceCollector(binding, layer_map, prefix_targets, enabled=True)
        with c.capture():
            _full_model_forward(model, bundle)
        require(c.snapshots is not None and len(c.snapshots) == N_LAYERS * (p + 1), "SYNTHETIC_PREFIX_CAPTURE_INCOMPLETE")
        hashes = {key: tensor_sha256(value) for key, value in c.snapshots.items()}
        if branch_index == 0:
            reference_hashes = hashes
        else:
            require(hashes == reference_hashes, "SYNTHETIC_PREFIX_STATE_IDENTITY_FAILURE")

    return {
        "status": "PASS_INSTRUMENTATION_PREFLIGHT",
        "capture_state_source": "native_selective_ssm_recurrent_state",
        "capture_state_timing": "post_consumption_s_t",
        "capture_source_qualname": binding.qualname,
        "capture_source_path": str(binding.source_path),
        "capture_source_sha256": binding.source_sha256,
        "capture_source_bytes": binding.source_bytes,
        "recurrence_update_line": binding.recurrence_update_line,
        "capture_line": binding.capture_line,
        "layer_count": N_LAYERS,
        "synthetic_token_count": len(bundles[0]["input_ids"]),
        "synthetic_prefix_token_count": len(prefix_ids),
        "trace_logit_noninterference": "PASS_EXACT",
        "fresh_state_isolation": "PASS_EXACT",
        "causal_prefix_state_identity": "PASS_EXACT_ALL_PREFIX_TOKENS_ALL_LAYERS",
        "state_dtype": "torch.float32",
        "device": "cpu",
    }


# ---------------------------------------------------------------------------
# Kinematics and primary statistics
# ---------------------------------------------------------------------------
def compute_branch_metrics(states: Mapping[int, Any], p: int) -> dict[str, Any]:
    import torch

    required = list(range(p - 1, p + W + 1))
    require(set(required) <= set(states), "BRANCH_STATE_WINDOW_INCOMPLETE")
    vectors = {idx: states[idx].to(torch.float64) for idx in required}
    v_pre = vectors[p] - vectors[p - 1]
    velocities = [vectors[p + k] - vectors[p + k - 1] for k in range(1, W + 1)]
    speeds = [float(torch.linalg.vector_norm(v).item()) for v in velocities]
    require(all(math.isfinite(v) for v in speeds), "SPEED_NONFINITE")

    turns: list[float | None] = []
    previous = v_pre
    for velocity in velocities:
        prev_norm = float(torch.linalg.vector_norm(previous).item())
        cur_norm = float(torch.linalg.vector_norm(velocity).item())
        if prev_norm <= EPSILON or cur_norm <= EPSILON:
            turns.append(None)
        else:
            cosine = float(torch.sum(previous * velocity).item() / (prev_norm * cur_norm))
            cosine = max(-1.0, min(1.0, cosine))
            turns.append(1.0 - cosine)
        previous = velocity
    valid_turns = [value for value in turns if value is not None]
    mean_turn = None if not valid_turns else float(sum(valid_turns) / len(valid_turns))

    path_length = float(sum(speeds))
    displacement = float(torch.linalg.vector_norm(vectors[p + W] - vectors[p]).item())
    eta = displacement / (path_length + EPSILON)
    mean_speed = path_length / W
    for value in (path_length, displacement, eta, mean_speed):
        require(math.isfinite(value), "KINEMATIC_METRIC_NONFINITE")

    window_bytes = b"".join(
        states[idx].detach().cpu().contiguous().numpy().tobytes() for idx in required
    )
    return {
        "R_mean_speed": mean_speed,
        "D_mean_turn": mean_turn,
        "P_efficiency": eta,
        "path_length": path_length,
        "displacement": displacement,
        "speed_steps": speeds,
        "turn_steps": turns,
        "n_valid_turns": len(valid_turns),
        "state_window_sha256": sha256_bytes(window_bytes),
        "state_p_minus_1_sha256": tensor_sha256(states[p - 1]),
        "state_p_sha256": tensor_sha256(states[p]),
    }


def run_native_branch(
    model: Any,
    token_ids: Sequence[int],
    p: int,
    binding: CaptureBinding,
    layer_map: Mapping[int, int],
) -> dict[int, dict[str, Any]]:
    import torch

    require(p >= 1 and len(token_ids) >= p + W + 1, "SCIENTIFIC_BRANCH_WINDOW_INVALID")
    # Capture the entire common prefix as well as the fixed post-event window.
    # The preregistration requires exact recurrent-state identity through p_i,
    # not merely at the two kinematic pre-event coordinates.
    target_indices = range(0, p + W + 1)
    collector = TraceCollector(binding, layer_map, target_indices, enabled=True)
    input_tensor = torch.tensor(list(token_ids), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode(), collector.capture():
        # No padding and no attention mask. This matches the A0 encoder call path.
        model.mamba(input_ids=input_tensor)
    require(collector.snapshots is not None, "SCIENTIFIC_TRACE_MISSING")
    require(len(collector.snapshots) == N_LAYERS * (p + W + 1), "SCIENTIFIC_TRACE_INCOMPLETE")

    out: dict[int, dict[str, Any]] = {}
    for layer in range(N_LAYERS):
        mixer = model.mamba.layers[layer].mixer
        state_map: dict[int, Any] = {}
        for index in target_indices:
            snapshot = collector.snapshots.get((layer, index))
            require(snapshot is not None, "SCIENTIFIC_STATE_COORDINATE_MISSING")
            validate_snapshot_tensor(snapshot, mixer)
            state_map[index] = snapshot
        metrics = compute_branch_metrics(state_map, p)
        prefix_bytes = b"".join(
            state_map[index].detach().cpu().contiguous().numpy().tobytes()
            for index in range(0, p + 1)
        )
        metrics["prefix_state_sequence_sha256"] = sha256_bytes(prefix_bytes)
        metrics["prefix_state_token_count"] = p + 1
        out[layer] = metrics
    del collector
    return out


def recipient_layer_result(branches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = {"matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl"}
    require(set(branches) == required, "RECIPIENT_BRANCH_SET_INVALID")

    prefix_p = {branches[name]["state_p_sha256"] for name in required}
    prefix_pm1 = {branches[name]["state_p_minus_1_sha256"] for name in required}
    prefix_sequences = {branches[name]["prefix_state_sequence_sha256"] for name in required}
    prefix_counts = {branches[name]["prefix_state_token_count"] for name in required}
    require(
        len(prefix_p) == 1
        and len(prefix_pm1) == 1
        and len(prefix_sequences) == 1
        and len(prefix_counts) == 1,
        "SCIENTIFIC_PREFIX_STATE_IDENTITY_FAILURE",
    )

    def delta(metric: str, a: str, b: str) -> float | None:
        x = branches[a][metric]
        y = branches[b][metric]
        if x is None or y is None:
            return None
        value = float(x - y)
        require(math.isfinite(value), "RECIPIENT_DELTA_NONFINITE")
        return value

    deltas = {
        "R_matched": delta("R_mean_speed", "matched_corr", "matched_ctrl"),
        "R_swapped": delta("R_mean_speed", "swapped_corr", "swapped_ctrl"),
        "D_matched": delta("D_mean_turn", "matched_corr", "matched_ctrl"),
        "D_swapped": delta("D_mean_turn", "swapped_corr", "swapped_ctrl"),
        "P_matched": delta("P_efficiency", "matched_corr", "matched_ctrl"),
        "P_swapped": delta("P_efficiency", "swapped_corr", "swapped_ctrl"),
    }

    x_values: dict[str, float | None] = {}
    for metric in PRIMARY_ORDER:
        matched = deltas[f"{metric}_matched"]
        swapped = deltas[f"{metric}_swapped"]
        if matched is None or swapped is None:
            x_values[metric] = None
        else:
            x = abs(matched) - abs(swapped)
            require(math.isfinite(x), "RECIPIENT_X_NONFINITE")
            x_values[metric] = float(x)

    return {
        "branches": {name: dict(branches[name]) for name in sorted(branches)},
        "signed_contrasts": deltas,
        "X_pair_specificity": x_values,
        "common_state_p_sha256": next(iter(prefix_p)),
        "common_state_p_minus_1_sha256": next(iter(prefix_pm1)),
        "common_prefix_state_sequence_sha256": next(iter(prefix_sequences)),
        "common_prefix_state_token_count": next(iter(prefix_counts)),
    }


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    require(positive >= 0 and negative >= 0, "SIGN_COUNTS_INVALID")
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    numerator = sum(math.comb(n, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**n))


def holm_adjust(raw: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    require(set(raw) == set(PRIMARY_ORDER), "HOLM_PRIMARY_SET_MISMATCH")
    tie = {name: i for i, name in enumerate(PRIMARY_ORDER)}
    ordered = sorted(PRIMARY_ORDER, key=lambda name: (float(raw[name]), tie[name]))
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(raw[name]))
        running = max(running, candidate)
        adjusted[name] = min(1.0, running)
    return {
        name: {
            "raw_p": float(raw[name]),
            "holm_adjusted_p": float(adjusted[name]),
            "holm_reject": bool(adjusted[name] <= 0.05),
        }
        for name in PRIMARY_ORDER
    }


def summarize_endpoint(values: Sequence[float | None]) -> dict[str, Any]:
    undefined = sum(value is None for value in values)
    valid = [float(value) for value in values if value is not None]
    require(all(math.isfinite(value) for value in valid), "BLOCK_VALUE_NONFINITE")
    positive = sum(value > 0.0 for value in valid)
    negative = sum(value < 0.0 for value in valid)
    zero = sum(value == 0.0 for value in valid)
    n_valid = len(valid)
    n_eff = positive + negative
    floor_pass = n_valid >= 120 and n_eff >= 30
    raw_p = exact_two_sided_sign_p(positive, negative) if floor_pass else 1.0
    effect = None if n_eff == 0 else (positive - negative) / n_eff
    return {
        "n_valid": n_valid,
        "n_eff": n_eff,
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "undefined_count": undefined,
        "promotion_floor_pass": floor_pass,
        "raw_p": raw_p,
        "rank_biserial_sign_effect": effect,
    }


def primary_statistics(block_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(block_rows) == N_BLOCKS, "BLOCK_COUNT_MISMATCH")
    endpoint = {
        metric: summarize_endpoint([row[f"B_{metric}"] for row in block_rows])
        for metric in PRIMARY_ORDER
    }
    holm = holm_adjust({metric: endpoint[metric]["raw_p"] for metric in PRIMARY_ORDER})
    positive_metrics: list[str] = []
    negative_metrics: list[str] = []
    for metric in PRIMARY_ORDER:
        endpoint[metric].update(holm[metric])
        effect = endpoint[metric]["rank_biserial_sign_effect"]
        if endpoint[metric]["promotion_floor_pass"] and endpoint[metric]["holm_reject"]:
            if effect is not None and effect > 0:
                positive_metrics.append(metric)
            elif effect is not None and effect < 0:
                negative_metrics.append(metric)

    if positive_metrics and negative_metrics:
        verdict = MIXED_VERDICT
    elif positive_metrics:
        verdict = POSITIVE_VERDICT
    elif negative_metrics:
        verdict = REVERSED_VERDICT
    else:
        verdict = NULL_VERDICT
    return {
        "schema_version": PRIMARY_SCHEMA,
        "primary_order": list(PRIMARY_ORDER),
        "holm_m": 3,
        "holm_alpha": 0.05,
        "holm_tie_order": list(PRIMARY_ORDER),
        "endpoints": endpoint,
        "positive_endpoints": positive_metrics,
        "negative_endpoints": negative_metrics,
        "scientific_verdict": verdict,
    }


def block_rows_from_items(item_primary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(item_primary) == N_ITEMS, "ITEM_PRIMARY_COUNT_MISMATCH")
    rows: list[dict[str, Any]] = []
    for block_index in range(N_BLOCKS):
        a = item_primary[2 * block_index]
        b = item_primary[2 * block_index + 1]
        require(a["block_index"] == block_index and b["block_index"] == block_index, "BLOCK_ITEM_ALIGNMENT_FAILURE")
        row: dict[str, Any] = {
            "schema_version": BLOCK_SCHEMA,
            "block_index": block_index,
            "block_id": f"k2s-block-{block_index:03d}",
            "item_a_stable_id": a["stable_item_id"],
            "item_b_stable_id": b["stable_item_id"],
        }
        for metric in PRIMARY_ORDER:
            xa = a[f"X_{metric}"]
            xb = b[f"X_{metric}"]
            if xa is None or xb is None:
                row[f"B_{metric}"] = None
            else:
                value = (float(xa) + float(xb)) / 2.0
                require(math.isfinite(value), "BLOCK_METRIC_NONFINITE")
                row[f"B_{metric}"] = value
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Scientific execution and artifacts
# ---------------------------------------------------------------------------
def _output_dir(root: Path, supplied: Path) -> Path:
    target = supplied.resolve()
    repo = root.resolve()
    require(target != repo and repo not in target.parents, "OUTPUT_DIRECTORY_MUST_BE_OUTSIDE_REPO")
    require(not target.exists(), "OUTPUT_DIRECTORY_ALREADY_EXISTS")
    return target


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _runtime_versions() -> dict[str, str]:
    import torch
    import transformers
    import tokenizers
    import huggingface_hub

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
    }


def execute_scientific(
    root: Path,
    model: Any,
    contracts: Sequence[Mapping[str, Any]],
    binding: CaptureBinding,
    output_dir: Path,
    runtime: Mapping[str, Any],
    hf: Mapping[str, Any],
    handoff: Mapping[str, Any],
    encoder_fp: Mapping[str, Any],
    preflight: Mapping[str, Any],
    feasibility: Mapping[str, Any],
) -> dict[str, Any]:
    layer_map = registered_mamba_layers(model)
    item_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    item_primary: list[dict[str, Any]] = []

    for i, contract in enumerate(contracts):
        branch_results: dict[str, dict[int, dict[str, Any]]] = {}
        for branch_name in ("matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl"):
            branch_results[branch_name] = run_native_branch(
                model,
                contract["branch_token_ids"][branch_name],
                int(contract["p"]),
                binding,
                layer_map,
            )

        layer_result_for_item: dict[int, dict[str, Any]] = {}
        for layer in range(N_LAYERS):
            branches = {name: branch_results[name][layer] for name in branch_results}
            result = recipient_layer_result(branches)
            layer_result_for_item[layer] = result
            layer_rows.append(
                {
                    "schema_version": LAYER_SCHEMA,
                    "item_index": i,
                    "block_index": contract["block_index"],
                    "block_id": contract["block_id"],
                    "stable_item_id": contract["stable_item_id"],
                    "donor_stable_item_id": contract["donor_stable_item_id"],
                    "layer_index": layer,
                    **result,
                }
            )

        primary = layer_result_for_item[PRIMARY_LAYER]
        item_rows.append(
            {
                "schema_version": ITEM_SCHEMA,
                **dict(contract),
                "primary_layer": PRIMARY_LAYER,
                "primary_layer_result": primary,
            }
        )
        item_primary.append(
            {
                "item_index": i,
                "block_index": contract["block_index"],
                "stable_item_id": contract["stable_item_id"],
                "X_R": primary["X_pair_specificity"]["R"],
                "X_D": primary["X_pair_specificity"]["D"],
                "X_P": primary["X_pair_specificity"]["P"],
            }
        )
        del branch_results, layer_result_for_item
        if (i + 1) % 10 == 0 or i == 0:
            print(f"K2S_PROGRESS items={i + 1}/{N_ITEMS}", flush=True)
        if (i + 1) % 25 == 0:
            gc.collect()

    blocks = block_rows_from_items(item_primary)
    primary = primary_statistics(blocks)

    output_dir.mkdir(parents=True, exist_ok=False)
    report = (
        "# K2S Pair-Specific Exact-Prefix Event Dynamics Result\n\n"
        f"Scientific verdict: `{primary['scientific_verdict']}`\n\n"
        f"Runtime HEAD: `{runtime['runtime_git_head']}`\n\n"
        "Primary inference uses 150 reciprocal blocks at layer 23 with R/D/P and Holm m=3.\n"
        "A0 task-head predictions were not used for population selection or promotion.\n"
        "This result is observational/event-specificity evidence only and does not establish a causal mechanism.\n"
    ).encode("utf-8")
    blobs: dict[str, bytes] = {
        "item_metrics.jsonl": canonical_jsonl(item_rows),
        "secondary_layer_metrics.jsonl": canonical_jsonl(layer_rows),
        "block_metrics.jsonl": canonical_jsonl(blocks),
        "primary_stats.json": canonical_json_line(primary),
        "report.md": report,
    }
    for name, data in blobs.items():
        _write(output_dir / name, data)

    artifact_hashes = {name: sha256_bytes(data) for name, data in blobs.items()}
    test_path = root / "tests" / "test_longterm_k2s_pair_specific_event_dynamics.py"
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "k2s_prereg_authority_commit": K2S_PREREG_AUTHORITY_COMMIT,
        "k2w_closure_commit": K2W_CLOSURE_COMMIT,
        "implementation_commit": runtime["runtime_git_head"],
        "runtime_git_head": runtime["runtime_git_head"],
        "runtime_branch": runtime["runtime_branch"],
        "runtime_dirty_contract": runtime["runtime_dirty_contract"],
        "script_sha256": file_sha256(Path(__file__)),
        "test_sha256": file_sha256(test_path) if test_path.is_file() else None,
        "population_git_commit": K2W_CLOSURE_COMMIT,
        "population_git_path": POPULATION_REL,
        "population_sha256": POPULATION_SHA256,
        "population_count": N_ITEMS,
        "original_source_git_commit": SOURCE_BLOB_COMMIT,
        "original_source_git_path": SOURCE_REL,
        "original_source_physical_sha256": SOURCE_PHYSICAL_SHA256,
        "original_source_semantic_sha256": SOURCE_SEMANTIC_SHA256,
        "reciprocal_mapping": [
            {
                "item_index": row["item_index"],
                "stable_item_id": row["stable_item_id"],
                "donor_index": row["donor_index"],
                "donor_stable_item_id": row["donor_stable_item_id"],
                "block_id": row["block_id"],
            }
            for row in contracts
        ],
        "scientific_input_binding": {
            "artifact": "item_metrics.jsonl",
            "sha256": artifact_hashes["item_metrics.jsonl"],
            "contains_exact_texts_and_token_arrays": True,
        },
        "feasibility": dict(feasibility),
        "event_anchor": "prefix_end_p",
        "window_W": W,
        "epsilon": EPSILON,
        "primary_layer": PRIMARY_LAYER,
        "primary_metrics": list(PRIMARY_ORDER),
        "hf": {key: value for key, value in hf.items() if key not in {"config", "tokenizer"}},
        "handoff": dict(handoff),
        "encoder_fingerprint": dict(encoder_fp),
        "instrumentation_preflight": dict(preflight),
        "instrumentation": {
            "capture_source_qualname": binding.qualname,
            "capture_source_path": str(binding.source_path),
            "capture_source_sha256": binding.source_sha256,
            "capture_source_bytes": binding.source_bytes,
            "recurrence_update_line": binding.recurrence_update_line,
            "capture_line": binding.capture_line,
            "capture_state_source": "native_selective_ssm_recurrent_state",
            "capture_state_timing": "post_consumption_s_t",
            "capture_method": "cpython_line_trace_local_ssm_state_clone",
            "raw_states_serialized": False,
            "all_24_layers_captured_and_reduced": True,
        },
        "artifact_sha256": artifact_hashes,
        "artifact_hash_binding_note": "all non-self scientific artifacts are bound here; manifest.json is bound by SHA256SUMS.txt",
        "required_artifacts": [
            "item_metrics.jsonl",
            "secondary_layer_metrics.jsonl",
            "block_metrics.jsonl",
            "primary_stats.json",
            "manifest.json",
            "report.md",
            "SHA256SUMS.txt",
        ],
        "primary_statistics": primary,
        "exact_command": sys.argv,
        **_runtime_versions(),
    }
    manifest_bytes = canonical_json_line(manifest)
    _write(output_dir / "manifest.json", manifest_bytes)

    sums_targets = [
        "item_metrics.jsonl",
        "secondary_layer_metrics.jsonl",
        "block_metrics.jsonl",
        "primary_stats.json",
        "manifest.json",
        "report.md",
    ]
    sums = "".join(
        f"{file_sha256(output_dir / name)}  {name}\n" for name in sums_targets
    ).encode("utf-8")
    _write(output_dir / "SHA256SUMS.txt", sums)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="K2S pair-specific native Mamba event dynamics")
    p.add_argument("--seed180-handoff", required=True)
    p.add_argument("--hf-revision", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--instrumentation-preflight", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    runtime = git_provenance(root, instrumentation_preflight=args.instrumentation_preflight)

    rows = load_frozen_population(root)
    blocks = reciprocal_blocks(rows)
    require(len(blocks) == N_BLOCKS, "RECIPROCAL_BLOCK_COUNT_MISMATCH")
    snapshot, hf = resolve_hf_snapshot(args.hf_revision)
    contracts, feasibility = build_input_contracts(rows, hf["tokenizer"])

    handoff = audit_handoff(Path(args.seed180_handoff))
    checkpoint = load_authenticated_checkpoint(handoff)
    encoder_fp = encoder_fingerprint(checkpoint["model_state_dict"])
    model = build_a0_model(root, snapshot, checkpoint)
    del checkpoint
    gc.collect()

    binding = resolve_capture_binding()
    preflight = run_instrumentation_preflight(model, hf["tokenizer"], binding)

    handoff_full = {
        **handoff,
        "encoder": encoder_fp,
        "strict_load": "PASS",
        "metadata_training_args": "PASS",
    }

    if args.instrumentation_preflight:
        print(
            canonical_json(
                {
                    "instrumentation_preflight": "PASS",
                    "feasibility": feasibility,
                    "handoff": handoff_full,
                    "instrumentation": preflight,
                    "runtime": runtime,
                }
            ).decode("utf-8")
        )
        return 0

    require(args.output_dir is not None, "SCIENTIFIC_OUTPUT_DIR_REQUIRED")
    target = _output_dir(root, Path(args.output_dir))
    manifest = execute_scientific(
        root=root,
        model=model,
        contracts=contracts,
        binding=binding,
        output_dir=target,
        runtime=runtime,
        hf=hf,
        handoff=handoff_full,
        encoder_fp=encoder_fp,
        preflight=preflight,
        feasibility=feasibility,
    )
    print(
        canonical_json(
            {
                "scientific_execution": "PASS",
                "output_dir": str(target),
                "scientific_verdict": manifest["primary_statistics"]["scientific_verdict"],
                "primary_statistics": manifest["primary_statistics"],
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
