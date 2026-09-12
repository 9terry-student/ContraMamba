"""K0-RVG-P1 raw native vector scientific execution runner.

Implementation authority permits:
- static authentication of frozen P0 artifacts,
- fabricated synthetic model/state validation,
- a future scientific path that is fail-closed unless a separately frozen,
  tracked execution-authority artifact explicitly authorizes the run.

This module must never execute the frozen 336-item scientific population
without that later authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Frozen authority / scope
# ---------------------------------------------------------------------------

AUTHORITY_COMMIT = "c2b1990b649701cbf5ec71a360f03b4b7ff27465"
P0_ARCHIVE_COMMIT = "265d060e96f98f73e4a40a9c54a88c3de482a312"
P0_IMPLEMENTATION_COMMIT = "421d79815045938ea45ddcbc37a870a14218d133"
PARENT_PREREG_COMMIT = "cdad87acf664cd61e48406f9d4568b6ab206da24"

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

RUNNER_REL = "scripts/longterm_k0_rvg_p1_raw_vector_execution.py"
TEST_REL = "tests/test_longterm_k0_rvg_p1_raw_vector_execution.py"

HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
IMPLEMENTATION_UNTRACKED = {RUNNER_REL, TEST_REL}


# ---------------------------------------------------------------------------
# Frozen P0 archive
# ---------------------------------------------------------------------------

P0_ARCHIVE_REL = "reports/longterm_k0_rvg_p0_state_blind_provisioning_421d798_v1"

P0_ARTIFACT_SHA256 = {
    "candidate_pool.jsonl": "743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246",
    "generated_source.jsonl": "8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf",
    "phase_pair_mapping.json": "c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda",
    "token_contracts.jsonl": "6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998",
    "provisioning_manifest.json": "feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52",
    "validation_report_candidate.md": "ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a",
}

P0_ITEM_COUNT = 336
P0_BLOCK_COUNT = 168
P0_SOURCE_ROW_COUNT = 4368
P0_CORRECTION_COUNTS = {"none": 168, "polarity_flip": 168}
P0_DIVERGENCE_HIST = {"1": 168, "2": 168}
P0_MIN_AVAILABILITY = 20


# ---------------------------------------------------------------------------
# Frozen observer / historical runtime bridge
# ---------------------------------------------------------------------------

OBSERVER_REL = "scripts/longterm_k0_rvg_raw_recurrence_observer.py"
OBSERVER_SHA256 = "12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25"
OBSERVER_GIT_BLOB = "f2dbdfe52661eca384897578ab272e602e36deac"
OBSERVER_COMMIT = "fcfe161c12f4ed8ef37aff435554cc0660e477af"

A0_COMMIT = "55debe94f0d19d16a334395e8561901fed6b52fa"
A0_MODEL_REL = "src/contramamba/modeling_v6b_minimal.py"
A0_HEADS_REL = "src/contramamba/heads"
A0_MODEL_BLOB_SHA = "f0ddc0eda64937de6fcd27943e30a296082c01d5"
A0_HEADS_TREE_SHA = "68d26855aa511fcd41d6f395ae5f87177a162678"

K2S_REL = "scripts/longterm_k2s_pair_specific_event_dynamics.py"
K2S_SHA256 = "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
K2S_GIT_BLOB = "3a651fb508669bdcf72441a4869b863d6eee6c1f"

EXPECTED_ZIP_SHA256 = "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
EXPECTED_CHECKPOINT_SHA256 = "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
COMMON_ENCODER_CANONICAL_SHA256 = "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
COMMON_ENCODER_RAW_CONCAT_SHA256 = "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
MAMBA_SOURCE_SHA256 = "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
MAMBA_UPSTREAM_BLOB = "87987e3e6646d8d0f9f0048bdd8a155d99c845db"

PRIMARY_LAYER = 23
WINDOW = 8
TARGET_COUNT = 9
EXPECTED_LAYER_COUNT = 24
EXPECTED_RECORDS_PER_BRANCH = EXPECTED_LAYER_COUNT * TARGET_COUNT
EXPECTED_STATE_SHAPE = (1, 1536, 16)
EXPECTED_DTYPE = "torch.float32"
EXPECTED_DEVICE = "cpu"
SUPPORT_MIN_BLOCKS = 160
ALPHA = 0.05

BRANCH_ORDER = (
    "matched_corr",
    "matched_ctrl",
    "swapped_corr",
    "swapped_ctrl",
)

RESULT_ARTIFACTS = (
    "item_metrics.jsonl",
    "block_metrics.jsonl",
    "endpoint_summary.json",
    "recurrence_audit.json",
    "state_hash_audit.jsonl",
    "execution_manifest.json",
)


class ContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(obj: Any, *, final_lf: bool = False) -> bytes:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    raw = b"".join(canonical_json_bytes(dict(row), final_lf=True) for row in rows)
    require(raw.endswith(b"\n"), "JSONL_FINAL_LF_MISSING")
    require(not raw.startswith(b"\xef\xbb\xbf"), "JSONL_BOM_FORBIDDEN")
    require(b"\r" not in raw, "JSONL_CR_FORBIDDEN")
    return raw


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_FAILURE:{' '.join(args)}") from exc


def _git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_SHOW_FAILURE:{spec}") from exc


def repo_contract(root: Path) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")

    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", AUTHORITY_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "P1_AUTHORITY_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()
    allowed = HISTORICAL_K1_UNTRACKED | IMPLEMENTATION_UNTRACKED
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(
            line[:2] == "??" and path in allowed,
            "GIT_DIRTY_CONTRACT_MISMATCH",
        )

    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "p1_authority_commit": AUTHORITY_COMMIT,
        "p1_authority_is_ancestor": True,
        "runtime_dirty_contract": status,
    }


def authenticate_code_dependencies(root: Path) -> dict[str, Any]:
    observer = root / OBSERVER_REL
    helper = root / K2S_REL
    require(observer.is_file(), "OBSERVER_MISSING")
    require(helper.is_file(), "K2S_HELPER_MISSING")
    require(file_sha256(observer) == OBSERVER_SHA256, "OBSERVER_SHA256_MISMATCH")
    require(file_sha256(helper) == K2S_SHA256, "K2S_SHA256_MISMATCH")
    require(
        _git(root, "rev-parse", f"HEAD:{OBSERVER_REL}") == OBSERVER_GIT_BLOB,
        "OBSERVER_GIT_BLOB_MISMATCH",
    )
    require(
        _git(root, "rev-parse", f"HEAD:{K2S_REL}") == K2S_GIT_BLOB,
        "K2S_GIT_BLOB_MISMATCH",
    )

    frozen_model_blob = _git(root, "rev-parse", f"{A0_COMMIT}:{A0_MODEL_REL}")
    current_model_blob = _git(root, "rev-parse", f"HEAD:{A0_MODEL_REL}")
    frozen_heads_tree = _git(root, "rev-parse", f"{A0_COMMIT}:{A0_HEADS_REL}")
    current_heads_tree = _git(root, "rev-parse", f"HEAD:{A0_HEADS_REL}")

    require(frozen_model_blob == A0_MODEL_BLOB_SHA, "A0_FROZEN_MODEL_BLOB_MISMATCH")
    require(current_model_blob == A0_MODEL_BLOB_SHA, "A0_MODEL_SOURCE_DRIFT")
    require(frozen_heads_tree == A0_HEADS_TREE_SHA, "A0_FROZEN_HEADS_TREE_MISMATCH")
    require(current_heads_tree == A0_HEADS_TREE_SHA, "A0_HEADS_SOURCE_DRIFT")

    return {
        "observer_sha256": OBSERVER_SHA256,
        "observer_git_blob": OBSERVER_GIT_BLOB,
        "observer_commit": OBSERVER_COMMIT,
        "k2s_helper_sha256": K2S_SHA256,
        "k2s_helper_git_blob": K2S_GIT_BLOB,
        "a0_commit": A0_COMMIT,
        "a0_model_blob_sha": A0_MODEL_BLOB_SHA,
        "a0_heads_tree_sha": A0_HEADS_TREE_SHA,
    }


def _import_repo_module(root: Path, module_rel: str, module_name: str):
    path = root / module_rel
    require(path.is_file(), f"MODULE_MISSING:{module_rel}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"MODULE_SPEC_FAILURE:{module_rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_observer_and_k2s(root: Path):
    observer = _import_repo_module(root, OBSERVER_REL, "k0_rvg_p1_observer")
    k2s = _import_repo_module(root, K2S_REL, "k0_rvg_p1_k2s")
    return observer, k2s


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"JSONL_PARSE_FAILURE:{path}:{lineno}") from exc
        require(isinstance(obj, dict), f"JSON_OBJECT_REQUIRED:{path}:{lineno}")
        rows.append(obj)
    return rows


@dataclass(frozen=True)
class P0Archive:
    candidates: tuple[dict[str, Any], ...]
    mapping: tuple[dict[str, Any], ...]
    token_contracts: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    hashes: dict[str, str]


def authenticate_p0_archive(root: Path) -> P0Archive:
    archive = root / P0_ARCHIVE_REL
    require(archive.is_dir(), "P0_ARCHIVE_DIR_MISSING")
    for name, wanted in P0_ARTIFACT_SHA256.items():
        path = archive / name
        require(path.is_file(), f"P0_ARTIFACT_MISSING:{name}")
        require(file_sha256(path) == wanted, f"P0_ARTIFACT_SHA_MISMATCH:{name}")

    candidates = load_jsonl(archive / "candidate_pool.jsonl")
    contracts = load_jsonl(archive / "token_contracts.jsonl")
    mapping_obj = json.loads((archive / "phase_pair_mapping.json").read_text("utf-8"))
    manifest = json.loads((archive / "provisioning_manifest.json").read_text("utf-8"))
    require(isinstance(mapping_obj, dict), "P0_MAPPING_OBJECT_REQUIRED")
    blocks = mapping_obj.get("blocks")
    require(isinstance(blocks, list), "P0_MAPPING_BLOCKS_REQUIRED")

    require(len(candidates) == P0_ITEM_COUNT, "P0_CANDIDATE_COUNT_MISMATCH")
    require(len(contracts) == P0_ITEM_COUNT, "P0_TOKEN_CONTRACT_COUNT_MISMATCH")
    require(len(blocks) == P0_BLOCK_COUNT, "P0_BLOCK_COUNT_MISMATCH")
    require(manifest.get("item_count") == P0_ITEM_COUNT, "P0_MANIFEST_ITEM_COUNT_MISMATCH")
    require(manifest.get("phase_block_count") == P0_BLOCK_COUNT, "P0_MANIFEST_BLOCK_COUNT_MISMATCH")
    require(manifest.get("source_row_count") == P0_SOURCE_ROW_COUNT, "P0_MANIFEST_SOURCE_COUNT_MISMATCH")
    require(manifest.get("correction_source_counts") == P0_CORRECTION_COUNTS, "P0_CORRECTION_BALANCE_MISMATCH")
    overlaps = manifest.get("prior_overlap_counts", {})
    require(isinstance(overlaps, dict) and len(overlaps) == 12, "P0_OVERLAP_CARDINALITY_MISMATCH")
    require(set(overlaps.values()) == {0}, "P0_NONZERO_PRIOR_OVERLAP")
    require(
        manifest.get("matched_divergence_offset_histogram") == P0_DIVERGENCE_HIST,
        "P0_MATCHED_DIVERGENCE_HIST_MISMATCH",
    )
    require(
        manifest.get("swapped_divergence_offset_histogram") == P0_DIVERGENCE_HIST,
        "P0_SWAPPED_DIVERGENCE_HIST_MISMATCH",
    )
    require(
        manifest.get("minimum_branch_post_divergence_availability") == P0_MIN_AVAILABILITY,
        "P0_MIN_AVAILABILITY_MISMATCH",
    )
    for key in (
        "model_loaded",
        "checkpoint_loaded",
        "model_forward_executed",
        "logits_read",
        "recurrent_state_read",
        "observer_imported",
        "scientific_endpoint_computed",
    ):
        require(manifest.get(key) is False, f"P0_STATE_BLIND_FLAG_MISMATCH:{key}")
    require(
        manifest.get("status") == "STATE_BLIND_INPUT_CONTRACT_VALID",
        "P0_MANIFEST_STATUS_MISMATCH",
    )

    by_local = {}
    contract_by_local = {}
    for row in candidates:
        i = int(row["local_template_index"])
        require(i not in by_local, "P0_DUPLICATE_CANDIDATE_LOCAL_INDEX")
        by_local[i] = row
    for row in contracts:
        i = int(row["local_template_index"])
        require(i not in contract_by_local, "P0_DUPLICATE_CONTRACT_LOCAL_INDEX")
        contract_by_local[i] = row
    require(sorted(by_local) == list(range(P0_ITEM_COUNT)), "P0_CANDIDATE_LOCAL_INDEX_SET_MISMATCH")
    require(sorted(contract_by_local) == list(range(P0_ITEM_COUNT)), "P0_CONTRACT_LOCAL_INDEX_SET_MISMATCH")

    ordered_candidates = tuple(by_local[i] for i in range(P0_ITEM_COUNT))
    ordered_contracts = tuple(contract_by_local[i] for i in range(P0_ITEM_COUNT))

    for i, (candidate, contract) in enumerate(zip(ordered_candidates, ordered_contracts)):
        require(candidate["pair_id"] == contract["pair_id"], f"P0_PAIR_ID_MISMATCH:{i}")
        require(candidate["stable_item_id"] == contract["stable_item_id"], f"P0_STABLE_ID_MISMATCH:{i}")
        require(int(contract["phase_block_index"]) == i % P0_BLOCK_COUNT, f"P0_PHASE_BLOCK_INDEX_MISMATCH:{i}")
        mate = i + P0_BLOCK_COUNT if i < P0_BLOCK_COUNT else i - P0_BLOCK_COUNT
        require(
            contract["phase_mate_stable_id"] == ordered_candidates[mate]["stable_item_id"],
            f"P0_PHASE_MATE_STABLE_ID_MISMATCH:{i}",
        )
        require(
            contract["phase_mate_pair_id"] == ordered_candidates[mate]["pair_id"],
            f"P0_PHASE_MATE_PAIR_ID_MISMATCH:{i}",
        )

    for p, block in enumerate(blocks):
        require(int(block["block_index"]) == p, f"P0_BLOCK_INDEX_MISMATCH:{p}")
        require(int(block["item_a_local_index"]) == p, f"P0_BLOCK_A_INDEX_MISMATCH:{p}")
        require(int(block["item_b_local_index"]) == p + P0_BLOCK_COUNT, f"P0_BLOCK_B_INDEX_MISMATCH:{p}")
        require(
            block["item_a_stable_id"] == ordered_candidates[p]["stable_item_id"],
            f"P0_BLOCK_A_STABLE_ID_MISMATCH:{p}",
        )
        require(
            block["item_b_stable_id"] == ordered_candidates[p + P0_BLOCK_COUNT]["stable_item_id"],
            f"P0_BLOCK_B_STABLE_ID_MISMATCH:{p}",
        )

    return P0Archive(
        candidates=ordered_candidates,
        mapping=tuple(dict(x) for x in blocks),
        token_contracts=ordered_contracts,
        manifest=manifest,
        hashes=dict(P0_ARTIFACT_SHA256),
    )


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    ids = encoded["input_ids"]
    require(isinstance(ids, list), "TOKEN_IDS_NOT_LIST")
    require(all(type(x) is int for x in ids), "TOKEN_IDS_NOT_INT")
    return ids


def token_id_sha256(ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(list(ids)))


def divergence_anchor(corr: Sequence[int], ctrl: Sequence[int], prefix_len: int) -> int:
    upper = min(len(corr), len(ctrl), prefix_len + 8)
    for k in range(prefix_len, upper):
        if corr[k] != ctrl[k]:
            return k
    raise ContractError("DIVERGENCE_NOT_WITHIN_FIRST_8_CONTINUATION_TOKENS")


@dataclass(frozen=True)
class BranchTexts:
    prefix: str
    matched_corr: str
    matched_ctrl: str
    swapped_corr: str
    swapped_ctrl: str


def reconstruct_branch_texts(
    candidates: Sequence[Mapping[str, Any]],
    local_index: int,
) -> BranchTexts:
    require(0 <= local_index < len(candidates), "LOCAL_INDEX_OUT_OF_RANGE")
    item = candidates[local_index]
    mate_index = (
        local_index + P0_BLOCK_COUNT
        if local_index < P0_BLOCK_COUNT
        else local_index - P0_BLOCK_COUNT
    )
    mate = candidates[mate_index]
    prefix = str(item["prefix_text"])
    return BranchTexts(
        prefix=prefix,
        matched_corr=prefix + str(item["correction_text"]),
        matched_ctrl=prefix + str(item["control_text"]),
        swapped_corr=prefix + str(mate["correction_text"]),
        swapped_ctrl=prefix + str(mate["control_text"]),
    )


@dataclass(frozen=True)
class RevalidatedTokenContract:
    prefix_ids: tuple[int, ...]
    matched_corr_ids: tuple[int, ...]
    matched_ctrl_ids: tuple[int, ...]
    swapped_corr_ids: tuple[int, ...]
    swapped_ctrl_ids: tuple[int, ...]
    matched_te: int
    swapped_te: int


def revalidate_token_contract(
    tokenizer: Any,
    branch_texts: BranchTexts,
    archived: Mapping[str, Any],
) -> RevalidatedTokenContract:
    prefix = token_ids(tokenizer, branch_texts.prefix)
    branches = {
        "matched_corr": token_ids(tokenizer, branch_texts.matched_corr),
        "matched_ctrl": token_ids(tokenizer, branch_texts.matched_ctrl),
        "swapped_corr": token_ids(tokenizer, branch_texts.swapped_corr),
        "swapped_ctrl": token_ids(tokenizer, branch_texts.swapped_ctrl),
    }
    for name, ids in branches.items():
        require(ids[: len(prefix)] == prefix, f"PREFIX_IDENTITY_FAILURE:{name}")

    require(len(prefix) == int(archived["prefix_token_count"]), "PREFIX_TOKEN_COUNT_MISMATCH")
    require(token_id_sha256(prefix) == archived["prefix_token_sha256"], "PREFIX_TOKEN_SHA_MISMATCH")

    matched_te = divergence_anchor(branches["matched_corr"], branches["matched_ctrl"], len(prefix))
    swapped_te = divergence_anchor(branches["swapped_corr"], branches["swapped_ctrl"], len(prefix))

    require(matched_te == int(archived["matched_divergence_anchor"]), "MATCHED_TE_MISMATCH")
    require(swapped_te == int(archived["swapped_divergence_anchor"]), "SWAPPED_TE_MISMATCH")
    require(matched_te - len(prefix) == int(archived["matched_divergence_offset_from_prefix"]), "MATCHED_OFFSET_MISMATCH")
    require(swapped_te - len(prefix) == int(archived["swapped_divergence_offset_from_prefix"]), "SWAPPED_OFFSET_MISMATCH")
    require(matched_te >= 1 and swapped_te >= 1, "INCOMING_VELOCITY_UNAVAILABLE")

    expected_counts = {
        "matched_corr": int(archived["matched_correction_token_count"]),
        "matched_ctrl": int(archived["matched_control_token_count"]),
        "swapped_corr": int(archived["swapped_correction_token_count"]),
        "swapped_ctrl": int(archived["swapped_control_token_count"]),
    }
    for name, ids in branches.items():
        require(len(ids) == expected_counts[name], f"TOKEN_COUNT_MISMATCH:{name}")

    require(
        min(len(branches["matched_corr"]) - matched_te, len(branches["matched_ctrl"]) - matched_te) >= WINDOW,
        "MATCHED_W8_UNAVAILABLE",
    )
    require(
        min(len(branches["swapped_corr"]) - swapped_te, len(branches["swapped_ctrl"]) - swapped_te) >= WINDOW,
        "SWAPPED_W8_UNAVAILABLE",
    )
    require(archived["matched_w8_available"] is True, "ARCHIVED_MATCHED_W8_FALSE")
    require(archived["swapped_w8_available"] is True, "ARCHIVED_SWAPPED_W8_FALSE")

    return RevalidatedTokenContract(
        prefix_ids=tuple(prefix),
        matched_corr_ids=tuple(branches["matched_corr"]),
        matched_ctrl_ids=tuple(branches["matched_ctrl"]),
        swapped_corr_ids=tuple(branches["swapped_corr"]),
        swapped_ctrl_ids=tuple(branches["swapped_ctrl"]),
        matched_te=matched_te,
        swapped_te=swapped_te,
    )


def target_indices(t_e: int) -> tuple[int, ...]:
    require(t_e >= 1, "TARGET_TE_LT_ONE")
    return tuple([t_e - 1] + list(range(t_e, t_e + WINDOW)))


def _to_float64(tensor: Any):
    import torch

    require(isinstance(tensor, torch.Tensor), "SCIENTIFIC_TENSOR_REQUIRED")
    return tensor.detach().to(device="cpu", dtype=torch.float64)


def frobenius_dot(x: Any, y: Any) -> float:
    import torch

    x64 = _to_float64(x)
    y64 = _to_float64(y)
    require(tuple(x64.shape) == tuple(y64.shape), "FROBENIUS_SHAPE_MISMATCH")
    return float(torch.sum(x64 * y64).item())


def frobenius_norm(x: Any) -> float:
    value = frobenius_dot(x, x)
    require(value >= 0.0, "FROBENIUS_NEGATIVE_SQUARED_NORM")
    return math.sqrt(value)


def frobenius_cosine(x: Any, y: Any) -> float | None:
    nx = frobenius_norm(x)
    ny = frobenius_norm(y)
    if nx == 0.0 or ny == 0.0:
        return None
    value = frobenius_dot(x, y) / (nx * ny)
    # Numerical roundoff may move by a few ulps beyond [-1, 1].
    if value > 1.0 and value <= 1.0 + 1e-12:
        value = 1.0
    if value < -1.0 and value >= -1.0 - 1e-12:
        value = -1.0
    require(-1.0 <= value <= 1.0, "FROBENIUS_COSINE_OUT_OF_RANGE")
    return float(value)


def velocity(record: Any):
    return record.s_post - record.s_prev


def carry_velocity(record: Any):
    return (record.g - 1.0) * record.s_prev


def write_velocity(record: Any):
    return record.w


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return sha256_bytes(array.tobytes())


def common_incoming_state(corr_record: Any, ctrl_record: Any) -> int:
    import torch

    comparisons = 0
    for role in ("s_prev", "g", "w", "s_post"):
        comparisons += 1
        require(
            torch.equal(getattr(corr_record, role), getattr(ctrl_record, role)),
            f"INCOMING_COMMON_STATE_MISMATCH:{role}",
        )
    return comparisons


def mean_defined(values: Sequence[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    require(bool(values), "EMPTY_MEAN")
    return float(sum(float(v) for v in values if v is not None) / len(values))


@dataclass(frozen=True)
class PairMetrics:
    valid_turning: bool
    valid_coherence: bool
    a_corr: float | None
    a_ctrl: float | None
    turning: float | None
    coherence: float | None
    carry_write_diagnostics: dict[str, float | None]


def compute_pair_metrics(
    corr_records: Mapping[int, Any],
    ctrl_records: Mapping[int, Any],
    t_e: int,
) -> PairMetrics:
    require(set(corr_records) == set(target_indices(t_e)), "CORR_TARGET_SET_MISMATCH")
    require(set(ctrl_records) == set(target_indices(t_e)), "CTRL_TARGET_SET_MISMATCH")
    common_incoming_state(corr_records[t_e - 1], ctrl_records[t_e - 1])
    incoming = velocity(corr_records[t_e - 1])

    corr_align: list[float | None] = []
    ctrl_align: list[float | None] = []
    response: list[Any] = []
    response_carry: list[Any] = []
    response_write: list[Any] = []

    for tau in range(WINDOW):
        idx = t_e + tau
        v_corr = velocity(corr_records[idx])
        v_ctrl = velocity(ctrl_records[idx])
        corr_align.append(frobenius_cosine(v_corr, incoming))
        ctrl_align.append(frobenius_cosine(v_ctrl, incoming))
        response.append(v_corr - v_ctrl)
        response_carry.append(carry_velocity(corr_records[idx]) - carry_velocity(ctrl_records[idx]))
        response_write.append(write_velocity(corr_records[idx]) - write_velocity(ctrl_records[idx]))

    a_corr = mean_defined(corr_align)
    a_ctrl = mean_defined(ctrl_align)
    turning = None if a_corr is None or a_ctrl is None else float(a_ctrl - a_corr)

    coherence_terms = [
        frobenius_cosine(response[tau], response[tau - 1])
        for tau in range(1, WINDOW)
    ]
    coherence = mean_defined(coherence_terms)

    total_norms = [frobenius_norm(x) for x in response]
    carry_norms = [frobenius_norm(x) for x in response_carry]
    write_norms = [frobenius_norm(x) for x in response_write]
    cw_total = [frobenius_cosine(w, r) for w, r in zip(response_write, response)]
    cc_total = [frobenius_cosine(c, r) for c, r in zip(response_carry, response)]
    cw_cc = [frobenius_cosine(w, c) for w, c in zip(response_write, response_carry)]

    diag = {
        "response_norm_mean": float(sum(total_norms) / WINDOW),
        "carry_response_norm_mean": float(sum(carry_norms) / WINDOW),
        "write_response_norm_mean": float(sum(write_norms) / WINDOW),
        "write_total_cosine_mean": mean_defined(cw_total),
        "carry_total_cosine_mean": mean_defined(cc_total),
        "write_carry_cosine_mean": mean_defined(cw_cc),
    }
    return PairMetrics(
        valid_turning=turning is not None,
        valid_coherence=coherence is not None,
        a_corr=a_corr,
        a_ctrl=a_ctrl,
        turning=turning,
        coherence=coherence,
        carry_write_diagnostics=diag,
    )


def item_endpoint_metrics(matched: PairMetrics, swapped: PairMetrics) -> dict[str, Any]:
    valid_turn = matched.valid_turning and swapped.valid_turning
    valid_coh = matched.valid_coherence and swapped.valid_coherence
    x_turn = (
        float(matched.turning - swapped.turning)
        if valid_turn and matched.turning is not None and swapped.turning is not None
        else None
    )
    x_coh = (
        float(matched.coherence - swapped.coherence)
        if valid_coh and matched.coherence is not None and swapped.coherence is not None
        else None
    )
    return {
        "turning_valid": valid_turn,
        "coherence_valid": valid_coh,
        "matched": {
            "A_corr": matched.a_corr,
            "A_ctrl": matched.a_ctrl,
            "T_M": matched.turning,
            "C_M": matched.coherence,
            "diagnostics": matched.carry_write_diagnostics,
        },
        "swapped": {
            "A_corr": swapped.a_corr,
            "A_ctrl": swapped.a_ctrl,
            "T_S": swapped.turning,
            "C_S": swapped.coherence,
            "diagnostics": swapped.carry_write_diagnostics,
        },
        "X_turn": x_turn,
        "X_coh": x_coh,
    }


def aggregate_blocks(
    item_rows: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require(len(item_rows) == P0_ITEM_COUNT, "BLOCK_ITEM_COUNT_MISMATCH")
    require(len(mapping) == P0_BLOCK_COUNT, "BLOCK_MAPPING_COUNT_MISMATCH")
    by_index = {int(row["local_template_index"]): row for row in item_rows}
    require(sorted(by_index) == list(range(P0_ITEM_COUNT)), "BLOCK_ITEM_INDEX_SET_MISMATCH")

    rows: list[dict[str, Any]] = []
    for p, block in enumerate(mapping):
        a = by_index[p]
        b = by_index[p + P0_BLOCK_COUNT]

        turn_valid = bool(a["turning_valid"]) and bool(b["turning_valid"])
        coh_valid = bool(a["coherence_valid"]) and bool(b["coherence_valid"])
        b_turn = (
            float((float(a["X_turn"]) + float(b["X_turn"])) / 2.0)
            if turn_valid
            else None
        )
        b_coh = (
            float((float(a["X_coh"]) + float(b["X_coh"])) / 2.0)
            if coh_valid
            else None
        )
        rows.append({
            "schema_version": "k0-rvg-p1-block-metrics-v1",
            "block_index": p,
            "phase_class": int(block["phase_class"]),
            "item_a_stable_id": a["stable_item_id"],
            "item_b_stable_id": b["stable_item_id"],
            "turning_valid": turn_valid,
            "B_turn": b_turn,
            "turning_sign": sign_classification(b_turn) if turn_valid else None,
            "coherence_valid": coh_valid,
            "B_coh": b_coh,
            "coherence_sign": sign_classification(b_coh) if coh_valid else None,
        })
    return rows


def sign_classification(value: float | None) -> str | None:
    if value is None:
        return None
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def exact_two_sided_sign_test(positive: int, negative: int) -> float | None:
    require(positive >= 0 and negative >= 0, "SIGN_COUNT_NEGATIVE")
    n = positive + negative
    if n == 0:
        return None
    k = min(positive, negative)
    numerator = 2 * sum(math.comb(n, r) for r in range(k + 1))
    denominator = 2 ** n
    return float(min(1.0, numerator / denominator))


def holm_m2(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    require(set(pvalues) == {"turning", "response_coherence"}, "HOLM_ENDPOINT_SET_MISMATCH")
    # Preserve the preregistered two-test family even if one endpoint is
    # non-evaluable. Treat a missing p-value as 1.0 for ordering/multiplicity,
    # but keep that endpoint's reported adjusted p-value as null.
    ordered = sorted(
        (
            1.0 if value is None else float(value),
            key,
            value is None,
        )
        for key, value in pvalues.items()
    )
    (p1, k1, missing1), (p2, k2, missing2) = ordered
    adj1 = min(1.0, 2.0 * p1)
    adj2 = min(1.0, max(adj1, p2))
    return {
        k1: None if missing1 else adj1,
        k2: None if missing2 else adj2,
    }


def endpoint_summary_one(name: str, values: Sequence[float | None], holm_p: float | None) -> dict[str, Any]:
    valid = [float(v) for v in values if v is not None]
    valid_count = len(valid)
    invalid_count = len(values) - valid_count
    if valid_count < SUPPORT_MIN_BLOCKS:
        return {
            "endpoint": name,
            "valid_block_count": valid_count,
            "invalid_block_count": invalid_count,
            "positive_count": None,
            "negative_count": None,
            "zero_count": None,
            "effective_n": None,
            "raw_sign_test_p": None,
            "holm_adjusted_p": None,
            "sign_effect": None,
            "endpoint_verdict": "NOT_EVALUABLE_DUE_TO_VECTOR_NORM_SUPPORT_FAILURE",
        }

    positive = sum(v > 0.0 for v in valid)
    negative = sum(v < 0.0 for v in valid)
    zero = sum(v == 0.0 for v in valid)
    n_eff = positive + negative
    raw_p = exact_two_sided_sign_test(positive, negative)
    sign_effect = None if n_eff == 0 else float((positive - negative) / n_eff)
    prefix = "TURNING" if name == "turning" else "RESPONSE_COHERENCE"
    if raw_p is None or holm_p is None or sign_effect is None:
        verdict = "NOT_EVALUABLE_DUE_TO_VECTOR_NORM_SUPPORT_FAILURE"
    elif holm_p < ALPHA and sign_effect > 0.0:
        verdict = f"{prefix}_POSITIVE_DIRECTIONAL_SIGNAL"
    elif holm_p < ALPHA and sign_effect < 0.0:
        verdict = f"{prefix}_REVERSED_DIRECTIONAL_SIGNAL"
    else:
        verdict = f"{prefix}_DIRECTIONAL_SIGNAL_NOT_ESTABLISHED"
    return {
        "endpoint": name,
        "valid_block_count": valid_count,
        "invalid_block_count": invalid_count,
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "effective_n": n_eff,
        "raw_sign_test_p": raw_p,
        "holm_adjusted_p": holm_p,
        "sign_effect": sign_effect,
        "endpoint_verdict": verdict,
    }


def overall_verdict(turning: Mapping[str, Any], coherence: Mapping[str, Any]) -> str:
    t = str(turning["endpoint_verdict"])
    c = str(coherence["endpoint_verdict"])
    if "NOT_EVALUABLE" in t or "NOT_EVALUABLE" in c:
        return "RAW_NATIVE_VECTOR_ORGANIZATION_NOT_EVALUABLE_DUE_TO_VECTOR_NORM_SUPPORT_FAILURE"

    t_pos = t == "TURNING_POSITIVE_DIRECTIONAL_SIGNAL"
    c_pos = c == "RESPONSE_COHERENCE_POSITIVE_DIRECTIONAL_SIGNAL"
    t_rev = t == "TURNING_REVERSED_DIRECTIONAL_SIGNAL"
    c_rev = c == "RESPONSE_COHERENCE_REVERSED_DIRECTIONAL_SIGNAL"

    if t_pos and c_pos:
        return "RAW_NATIVE_VECTOR_ORGANIZATION_CONVERGENT"
    if (t_pos and c_rev) or (c_pos and t_rev):
        return "RAW_NATIVE_VECTOR_ORGANIZATION_MIXED_NOT_PROMOTABLE"
    if t_pos ^ c_pos:
        return "RAW_NATIVE_VECTOR_ORGANIZATION_ENDPOINT_SPECIFIC"
    if t_rev or c_rev:
        return "RAW_NATIVE_VECTOR_ORGANIZATION_DIRECTIONALLY_CONTRADICTED_OR_MIXED"
    return "RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED"


def summarize_endpoints(block_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(block_rows) == P0_BLOCK_COUNT, "SUMMARY_BLOCK_COUNT_MISMATCH")
    turn_values = [row["B_turn"] if row["turning_valid"] else None for row in block_rows]
    coh_values = [row["B_coh"] if row["coherence_valid"] else None for row in block_rows]

    def raw_p(values):
        valid = [float(v) for v in values if v is not None]
        if len(valid) < SUPPORT_MIN_BLOCKS:
            return None
        pos = sum(v > 0.0 for v in valid)
        neg = sum(v < 0.0 for v in valid)
        return exact_two_sided_sign_test(pos, neg)

    raw = {
        "turning": raw_p(turn_values),
        "response_coherence": raw_p(coh_values),
    }
    adjusted = holm_m2(raw)
    turning = endpoint_summary_one("turning", turn_values, adjusted["turning"])
    coherence = endpoint_summary_one("response_coherence", coh_values, adjusted["response_coherence"])
    return {
        "schema_version": "k0-rvg-p1-endpoint-summary-v1",
        "alpha": ALPHA,
        "multiplicity_method": "HOLM_M2",
        "primary_endpoint_count": 2,
        "turning": turning,
        "response_coherence": coherence,
        "overall_verdict": overall_verdict(turning, coherence),
    }


def authority_markers(text: str) -> dict[str, str]:
    markers: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if " = " not in stripped:
            continue
        key, value = stripped.split(" = ", 1)
        markers[key.strip()] = value.strip()
    return markers


def validate_p1_implementation_commit(
    root: Path,
    implementation_commit: str,
) -> dict[str, str]:
    require(len(implementation_commit) == 40, "EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_INVALID")
    parent = _git(root, "rev-parse", f"{implementation_commit}^")
    require(parent == AUTHORITY_COMMIT, "P1_IMPLEMENTATION_PARENT_MISMATCH")

    changed = sorted(
        line
        for line in _git(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            implementation_commit,
        ).splitlines()
        if line
    )
    require(
        changed == sorted([RUNNER_REL, TEST_REL]),
        "P1_IMPLEMENTATION_SCOPE_MISMATCH",
    )

    return {
        "runner_blob": _git(root, "rev-parse", f"{implementation_commit}:{RUNNER_REL}"),
        "test_blob": _git(root, "rev-parse", f"{implementation_commit}:{TEST_REL}"),
    }


@dataclass(frozen=True)
class ExecutionAuthority:
    path: str
    commit: str
    implementation_commit: str
    git_blob: str


def authenticate_execution_authority(
    root: Path,
    authority_path: str,
) -> ExecutionAuthority:
    rel = Path(authority_path)
    require(not rel.is_absolute(), "EXECUTION_AUTHORITY_PATH_ABSOLUTE")
    require(".." not in rel.parts, "EXECUTION_AUTHORITY_PATH_TRAVERSAL")
    normalized = rel.as_posix()
    require(normalized.startswith("reports/"), "EXECUTION_AUTHORITY_NOT_REPORT")

    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "--", normalized],
            cwd=root,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError("EXECUTION_AUTHORITY_NOT_TRACKED") from exc

    path = root / normalized
    require(path.is_file(), "EXECUTION_AUTHORITY_WORKTREE_MISSING")
    worktree_bytes = path.read_bytes()
    head_bytes = _git_bytes(root, f"HEAD:{normalized}")
    require(worktree_bytes == head_bytes, "EXECUTION_AUTHORITY_WORKTREE_DRIFT")

    authority_blob = _git(root, "rev-parse", f"HEAD:{normalized}")
    require(len(authority_blob) == 40, "EXECUTION_AUTHORITY_GIT_BLOB_INVALID")

    text = worktree_bytes.decode("utf-8", "strict")
    markers = authority_markers(text)

    required_yes = {
        "SCIENTIFIC_EXECUTION_AUTHORIZED": "YES",
        "SCIENTIFIC_MODEL_FORWARD_AUTHORIZED": "YES",
        "SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED": "YES",
    }
    required_no = {
        "LOGITS_READ_AUTHORIZED": "NO",
        "CAUSAL_INTERVENTION_AUTHORIZED": "NO",
    }
    for key, value in required_yes.items():
        require(markers.get(key) == value, f"EXECUTION_AUTHORITY_MARKER_MISMATCH:{key}")
    for key, value in required_no.items():
        require(markers.get(key) == value, f"EXECUTION_AUTHORITY_MARKER_MISMATCH:{key}")

    implementation_commit = markers.get("P1_IMPLEMENTATION_COMMIT", "")
    implementation_blobs = validate_p1_implementation_commit(root, implementation_commit)

    head = _git(root, "rev-parse", "HEAD")
    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", implementation_commit, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "P1_IMPLEMENTATION_NOT_ANCESTOR")

    runner_sha = file_sha256(root / RUNNER_REL)
    test_sha = file_sha256(root / TEST_REL)
    require(markers.get("P1_RUNNER_SHA256") == runner_sha, "EXECUTION_AUTHORITY_RUNNER_SHA_MISMATCH")
    require(markers.get("P1_TEST_SHA256") == test_sha, "EXECUTION_AUTHORITY_TEST_SHA_MISMATCH")

    runner_blob_impl = implementation_blobs["runner_blob"]
    test_blob_impl = implementation_blobs["test_blob"]
    runner_blob_head = _git(root, "rev-parse", f"HEAD:{RUNNER_REL}")
    test_blob_head = _git(root, "rev-parse", f"HEAD:{TEST_REL}")
    require(runner_blob_impl == runner_blob_head, "P1_RUNNER_BLOB_DRIFT_AFTER_IMPLEMENTATION")
    require(test_blob_impl == test_blob_head, "P1_TEST_BLOB_DRIFT_AFTER_IMPLEMENTATION")

    expected_authority_hash_markers = {
        "P0_CANDIDATE_POOL_SHA256": P0_ARTIFACT_SHA256["candidate_pool.jsonl"],
        "P0_GENERATED_SOURCE_SHA256": P0_ARTIFACT_SHA256["generated_source.jsonl"],
        "P0_PHASE_PAIR_MAPPING_SHA256": P0_ARTIFACT_SHA256["phase_pair_mapping.json"],
        "P0_TOKEN_CONTRACTS_SHA256": P0_ARTIFACT_SHA256["token_contracts.jsonl"],
        "P0_PROVISIONING_MANIFEST_SHA256": P0_ARTIFACT_SHA256["provisioning_manifest.json"],
        "P0_VALIDATION_REPORT_SHA256": P0_ARTIFACT_SHA256["validation_report_candidate.md"],
        "OBSERVER_SHA256": OBSERVER_SHA256,
    }
    for key, value in expected_authority_hash_markers.items():
        require(markers.get(key) == value, f"EXECUTION_AUTHORITY_HASH_MISMATCH:{key}")

    authority_commit = _git(root, "log", "-1", "--format=%H", "--", normalized)
    require(len(authority_commit) == 40, "EXECUTION_AUTHORITY_COMMIT_UNRESOLVED")

    implementation_to_authority = subprocess.call(
        ["git", "merge-base", "--is-ancestor", implementation_commit, authority_commit],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        implementation_to_authority == 0,
        "P1_IMPLEMENTATION_NOT_ANCESTOR_OF_EXECUTION_AUTHORITY",
    )

    authority_ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", authority_commit, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(authority_ancestor == 0, "EXECUTION_AUTHORITY_COMMIT_NOT_ANCESTOR")

    return ExecutionAuthority(
        path=normalized,
        commit=authority_commit,
        implementation_commit=implementation_commit,
        git_blob=authority_blob,
    )


@dataclass
class AuditAccumulator:
    recurrence_count: int = 0
    max_abs_residual: float = 0.0
    max_relative_residual: float = 0.0
    max_scaled_tolerance_residual: float = 0.0
    incoming_common_state_comparisons: int = 0
    incoming_common_state_failures: int = 0

    def add_recurrence(self, result: Mapping[str, Any]) -> None:
        self.recurrence_count += 1
        self.max_abs_residual = max(self.max_abs_residual, float(result["max_abs_residual"]))
        self.max_relative_residual = max(self.max_relative_residual, float(result["max_relative_residual"]))
        self.max_scaled_tolerance_residual = max(
            self.max_scaled_tolerance_residual,
            float(result["max_scaled_tolerance_residual"]),
        )


def validate_layer23_record(observer: Any, record: Any, audit: AuditAccumulator) -> None:
    require(record.layer_index == PRIMARY_LAYER, "PRIMARY_LAYER_RECORD_MISMATCH")
    require(tuple(record.s_prev_meta.shape) == EXPECTED_STATE_SHAPE, "PRIMARY_STATE_SHAPE_MISMATCH")
    require(record.s_prev_meta.dtype == EXPECTED_DTYPE, "PRIMARY_STATE_DTYPE_MISMATCH")
    require(record.s_prev_meta.device == EXPECTED_DEVICE, "PRIMARY_STATE_DEVICE_MISMATCH")
    result = observer.validate_recurrence_record(record)
    require(result["recurrence_exact"] == "PASS_EXACT", "PRIMARY_RECURRENCE_NOT_EXACT")
    require(result["velocity_rearrangement"] == "PASS_TOLERANCE", "PRIMARY_VELOCITY_REARRANGEMENT_FAIL")
    audit.add_recurrence(result)


def capture_branch(
    observer: Any,
    k2s: Any,
    model: Any,
    layer_map: Mapping[int, int],
    binding: Any,
    bundle: Mapping[str, Sequence[int]],
    t_e: int,
    local_index: int,
    branch_role: str,
    audit: AuditAccumulator,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    targets = target_indices(t_e)
    collector = observer.RawRecurrenceCollector(binding, layer_map, targets)
    with collector.capture():
        k2s._full_model_forward(model, bundle)
    require(collector.records is not None, "BRANCH_RECORDS_MISSING")
    require(len(collector.records) == EXPECTED_RECORDS_PER_BRANCH, "BRANCH_CAPTURE_COMPLETENESS_FAILURE")
    expected_keys = {
        (layer, token)
        for layer in range(EXPECTED_LAYER_COUNT)
        for token in targets
    }
    require(set(collector.records) == expected_keys, "BRANCH_CAPTURE_COORDINATE_SET_MISMATCH")

    primary = {
        token: collector.records[(PRIMARY_LAYER, token)]
        for token in targets
    }
    hashes: list[dict[str, Any]] = []
    for token in targets:
        record = primary[token]
        validate_layer23_record(observer, record, audit)
        hashes.append({
            "schema_version": "k0-rvg-p1-state-hash-v1",
            "local_template_index": local_index,
            "branch_role": branch_role,
            "token_index": token,
            "S_prev_sha256": tensor_sha256(record.s_prev),
            "G_sha256": tensor_sha256(record.g),
            "W_sha256": tensor_sha256(record.w),
            "S_post_sha256": tensor_sha256(record.s_post),
        })
    return primary, hashes


def run_pair(
    observer: Any,
    k2s: Any,
    model: Any,
    layer_map: Mapping[int, int],
    binding: Any,
    tokenizer: Any,
    corr_text: str,
    ctrl_text: str,
    t_e: int,
    local_index: int,
    corr_role: str,
    ctrl_role: str,
    audit: AuditAccumulator,
) -> tuple[PairMetrics, list[dict[str, Any]]]:
    corr_bundle = k2s.task_mask_bundle(tokenizer, corr_text)
    ctrl_bundle = k2s.task_mask_bundle(tokenizer, ctrl_text)
    corr_records, corr_hashes = capture_branch(
        observer, k2s, model, layer_map, binding, corr_bundle,
        t_e, local_index, corr_role, audit,
    )
    ctrl_records, ctrl_hashes = capture_branch(
        observer, k2s, model, layer_map, binding, ctrl_bundle,
        t_e, local_index, ctrl_role, audit,
    )
    try:
        comparisons = common_incoming_state(corr_records[t_e - 1], ctrl_records[t_e - 1])
    except ContractError:
        audit.incoming_common_state_failures += 1
        raise
    audit.incoming_common_state_comparisons += comparisons
    metrics = compute_pair_metrics(corr_records, ctrl_records, t_e)
    return metrics, corr_hashes + ctrl_hashes


def validate_runtime_and_build_model(root: Path, handoff_path: Path):
    observer, k2s = import_observer_and_k2s(root)
    require(file_sha256(root / OBSERVER_REL) == OBSERVER_SHA256, "OBSERVER_RUNTIME_SHA_MISMATCH")
    require(file_sha256(root / K2S_REL) == K2S_SHA256, "K2S_RUNTIME_SHA_MISMATCH")

    handoff = k2s.audit_handoff(handoff_path)
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    encoder = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    require(handoff["zip_sha256"] == EXPECTED_ZIP_SHA256, "HANDOFF_ZIP_SHA_MISMATCH")
    require(handoff["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "CHECKPOINT_SHA_MISMATCH")
    require(encoder["canonical_digest"] == COMMON_ENCODER_CANONICAL_SHA256, "ENCODER_CANONICAL_MISMATCH")
    require(encoder["raw_concat_digest"] == COMMON_ENCODER_RAW_CONCAT_SHA256, "ENCODER_RAW_MISMATCH")

    snapshot, hf = k2s.resolve_hf_snapshot(HF_REVISION)
    require(hf["hf_model_id"] == HF_MODEL, "HF_MODEL_MISMATCH")
    require(hf["resolved_hf_revision"] == HF_REVISION, "HF_REVISION_MISMATCH")
    require(hf["transformers_version"] == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")
    tokenizer = hf["tokenizer"]
    model = k2s.build_a0_model(root, snapshot, checkpoint)
    model.eval()

    binding = observer.resolve_source_binding()
    require(binding.source_sha256 == MAMBA_SOURCE_SHA256, "MAMBA_SOURCE_SHA_MISMATCH")
    layer_map = observer.registered_mamba_layers(model)
    require(layer_map == k2s.registered_mamba_layers(model), "MAMBA_LAYER_MAP_DRIFT")

    return observer, k2s, model, tokenizer, binding, layer_map, encoder, handoff, hf


def _synthetic_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        {
            "local_template_index": 0,
            "pair_id": "synthetic_pair_a",
            "stable_item_id": "synthetic-stable-a",
            "prefix_text": "Claim: synthetic blorp alpha\nEvidence: synthetic snarp alpha\nAdditional evidence:\n",
            "correction_text": " synthetic corrective amber cedar quartz river summit valley willow ember frost cobalt jade.",
            "control_text": " synthetic control bronze maple crystal meadow harbor ridge birch slate pearl topaz ivory.",
        },
        {
            "local_template_index": 1,
            "pair_id": "synthetic_pair_b",
            "stable_item_id": "synthetic-stable-b",
            "prefix_text": "Claim: synthetic blorp beta\nEvidence: synthetic snarp beta\nAdditional evidence:\n",
            "correction_text": " synthetic corrective lunar solar comet planet orbit galaxy nebula meteor aurora zenith nadir.",
            "control_text": " synthetic control delta gamma theta kappa lambda sigma omega epsilon zeta eta iota.",
        },
    ]
    mapping = [{
        "block_index": 0,
        "phase_class": 0,
        "item_a_local_index": 0,
        "item_b_local_index": 1,
        "item_a_stable_id": "synthetic-stable-a",
        "item_b_stable_id": "synthetic-stable-b",
    }]
    return candidates, mapping


def synthetic_contract(tokenizer: Any, candidates: Sequence[Mapping[str, Any]], i: int) -> tuple[BranchTexts, dict[str, Any]]:
    item = candidates[i]
    mate = candidates[1 - i]
    prefix = str(item["prefix_text"])
    texts = BranchTexts(
        prefix=prefix,
        matched_corr=prefix + str(item["correction_text"]),
        matched_ctrl=prefix + str(item["control_text"]),
        swapped_corr=prefix + str(mate["correction_text"]),
        swapped_ctrl=prefix + str(mate["control_text"]),
    )
    prefix_ids = token_ids(tokenizer, prefix)
    branch_ids = {
        "matched_corr": token_ids(tokenizer, texts.matched_corr),
        "matched_ctrl": token_ids(tokenizer, texts.matched_ctrl),
        "swapped_corr": token_ids(tokenizer, texts.swapped_corr),
        "swapped_ctrl": token_ids(tokenizer, texts.swapped_ctrl),
    }
    for name, ids in branch_ids.items():
        require(ids[:len(prefix_ids)] == prefix_ids, f"SYNTHETIC_PREFIX_IDENTITY_FAILURE:{name}")
    m_te = divergence_anchor(branch_ids["matched_corr"], branch_ids["matched_ctrl"], len(prefix_ids))
    s_te = divergence_anchor(branch_ids["swapped_corr"], branch_ids["swapped_ctrl"], len(prefix_ids))
    require(m_te >= 1 and s_te >= 1, "SYNTHETIC_TE_LT_ONE")
    require(min(len(branch_ids["matched_corr"]) - m_te, len(branch_ids["matched_ctrl"]) - m_te) >= WINDOW, "SYNTHETIC_MATCHED_W8_FAIL")
    require(min(len(branch_ids["swapped_corr"]) - s_te, len(branch_ids["swapped_ctrl"]) - s_te) >= WINDOW, "SYNTHETIC_SWAPPED_W8_FAIL")
    contract = {
        "prefix_token_count": len(prefix_ids),
        "prefix_token_sha256": token_id_sha256(prefix_ids),
        "matched_correction_token_count": len(branch_ids["matched_corr"]),
        "matched_control_token_count": len(branch_ids["matched_ctrl"]),
        "matched_divergence_anchor": m_te,
        "matched_divergence_offset_from_prefix": m_te - len(prefix_ids),
        "matched_w8_available": True,
        "swapped_correction_token_count": len(branch_ids["swapped_corr"]),
        "swapped_control_token_count": len(branch_ids["swapped_ctrl"]),
        "swapped_divergence_anchor": s_te,
        "swapped_divergence_offset_from_prefix": s_te - len(prefix_ids),
        "swapped_w8_available": True,
    }
    return texts, contract


def run_synthetic_preflight(root: Path, handoff_path: Path) -> dict[str, Any]:
    repo = repo_contract(root)
    deps = authenticate_code_dependencies(root)
    p0 = authenticate_p0_archive(root)  # static/hash read only; no P0 text is forwarded.
    observer, k2s, model, tokenizer, binding, layer_map, encoder, handoff, hf = (
        validate_runtime_and_build_model(root, handoff_path)
    )

    candidates, _ = _synthetic_candidates()
    audit = AuditAccumulator()
    item_rows = []
    state_hash_rows = []
    deterministic_snapshots = []

    for i in range(2):
        texts, archived_like = synthetic_contract(tokenizer, candidates, i)
        token_contract = revalidate_token_contract(tokenizer, texts, archived_like)
        matched, hashes_m = run_pair(
            observer, k2s, model, layer_map, binding, tokenizer,
            texts.matched_corr, texts.matched_ctrl, token_contract.matched_te,
            i, "matched_corr", "matched_ctrl", audit,
        )
        swapped, hashes_s = run_pair(
            observer, k2s, model, layer_map, binding, tokenizer,
            texts.swapped_corr, texts.swapped_ctrl, token_contract.swapped_te,
            i, "swapped_corr", "swapped_ctrl", audit,
        )
        endpoint = item_endpoint_metrics(matched, swapped)
        row = {
            "local_template_index": i,
            "pair_id": candidates[i]["pair_id"],
            "stable_item_id": candidates[i]["stable_item_id"],
            **endpoint,
        }
        item_rows.append(row)
        hashes = hashes_m + hashes_s
        state_hash_rows.extend(hashes)
        deterministic_snapshots.append({
            "metrics": row,
            "hashes": hashes,
        })

    # Repeat the first synthetic item end-to-end with fresh collectors.
    texts0, archived0 = synthetic_contract(tokenizer, candidates, 0)
    token0 = revalidate_token_contract(tokenizer, texts0, archived0)
    audit_repeat = AuditAccumulator()
    m0, hm0 = run_pair(
        observer, k2s, model, layer_map, binding, tokenizer,
        texts0.matched_corr, texts0.matched_ctrl, token0.matched_te,
        0, "matched_corr", "matched_ctrl", audit_repeat,
    )
    s0, hs0 = run_pair(
        observer, k2s, model, layer_map, binding, tokenizer,
        texts0.swapped_corr, texts0.swapped_ctrl, token0.swapped_te,
        0, "swapped_corr", "swapped_ctrl", audit_repeat,
    )
    repeat_snapshot = {
        "metrics": {
            "local_template_index": 0,
            "pair_id": candidates[0]["pair_id"],
            "stable_item_id": candidates[0]["stable_item_id"],
            **item_endpoint_metrics(m0, s0),
        },
        "hashes": hm0 + hs0,
    }
    require(
        canonical_json_bytes(repeat_snapshot) == canonical_json_bytes(deterministic_snapshots[0]),
        "SYNTHETIC_REPEAT_IDENTITY_FAILURE",
    )

    require(audit.incoming_common_state_failures == 0, "SYNTHETIC_INCOMING_COMMON_STATE_FAILURE")
    require(audit.recurrence_count == 2 * 4 * TARGET_COUNT, "SYNTHETIC_RECURRENCE_COUNT_MISMATCH")
    require(len(state_hash_rows) == 2 * 4 * TARGET_COUNT, "SYNTHETIC_STATE_HASH_COUNT_MISMATCH")

    # Pure-math fixtures.
    import torch
    z = torch.zeros(EXPECTED_STATE_SHAPE, dtype=torch.float32)
    require(frobenius_cosine(z, z) is None, "SYNTHETIC_ZERO_VECTOR_POLICY_FAILURE")
    require(exact_two_sided_sign_test(10, 0) == 2 / (2 ** 10), "SYNTHETIC_SIGN_TEST_FIXTURE_FAILURE")
    holm = holm_m2({"turning": 0.01, "response_coherence": 0.04})
    require(holm == {"turning": 0.02, "response_coherence": 0.04}, "SYNTHETIC_HOLM_FIXTURE_FAILURE")

    return {
        "schema_version": "k0-rvg-p1-synthetic-preflight-v1",
        "status": "PASS_SYNTHETIC_P1_RAW_VECTOR_RUNNER",
        "repo": repo,
        "dependencies": deps,
        "p0_static_artifact_sha256": p0.hashes,
        "handoff": {
            "zip_sha256": handoff["zip_sha256"],
            "checkpoint_sha256": handoff["checkpoint_sha256"],
        },
        "encoder": {
            "canonical_digest": encoder["canonical_digest"],
            "raw_concat_digest": encoder["raw_concat_digest"],
        },
        "hf": {
            "model_id": hf["hf_model_id"],
            "revision": hf["resolved_hf_revision"],
            "transformers_version": hf["transformers_version"],
        },
        "checks": {
            "synthetic_item_count": 2,
            "synthetic_branch_forward_count": 2 * 4 + 4,
            "layer23_recurrence_checks": audit.recurrence_count,
            "incoming_common_state_comparisons": audit.incoming_common_state_comparisons,
            "incoming_common_state_failures": audit.incoming_common_state_failures,
            "state_hash_rows": len(state_hash_rows),
            "repeat_first_item_identity": "PASS_EXACT",
            "zero_vector_policy": "PASS_UNDEFINED",
            "exact_sign_test_fixture": "PASS",
            "holm_m2_fixture": "PASS",
            "max_velocity_abs_residual": audit.max_abs_residual,
            "max_velocity_relative_frobenius_residual": audit.max_relative_residual,
            "max_velocity_scaled_tolerance_residual": audit.max_scaled_tolerance_residual,
        },
        "scientific_population_model_forward_executed": False,
        "scientific_population_recurrent_state_read": False,
        "scientific_endpoint_computed": False,
        "logits_read": False,
        "causal_intervention_executed": False,
    }


def _item_scientific_execution(
    observer: Any,
    k2s: Any,
    model: Any,
    tokenizer: Any,
    binding: Any,
    layer_map: Mapping[int, int],
    archive: P0Archive,
    local_index: int,
    audit: AuditAccumulator,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = archive.candidates[local_index]
    archived_contract = archive.token_contracts[local_index]
    texts = reconstruct_branch_texts(archive.candidates, local_index)
    contract = revalidate_token_contract(tokenizer, texts, archived_contract)

    matched, hashes_m = run_pair(
        observer, k2s, model, layer_map, binding, tokenizer,
        texts.matched_corr, texts.matched_ctrl, contract.matched_te,
        local_index, "matched_corr", "matched_ctrl", audit,
    )
    swapped, hashes_s = run_pair(
        observer, k2s, model, layer_map, binding, tokenizer,
        texts.swapped_corr, texts.swapped_ctrl, contract.swapped_te,
        local_index, "swapped_corr", "swapped_ctrl", audit,
    )
    endpoint = item_endpoint_metrics(matched, swapped)
    row = {
        "schema_version": "k0-rvg-p1-item-metrics-v1",
        "local_template_index": local_index,
        "pair_id": candidate["pair_id"],
        "stable_item_id": candidate["stable_item_id"],
        "phase_block_index": int(archived_contract["phase_block_index"]),
        "phase_mate_stable_id": archived_contract["phase_mate_stable_id"],
        "matched_divergence_anchor": contract.matched_te,
        "swapped_divergence_anchor": contract.swapped_te,
        **endpoint,
        "recurrence_coordinates_validated": 4 * TARGET_COUNT,
    }
    return row, hashes_m + hashes_s


def build_execution_manifest(
    root: Path,
    authority: ExecutionAuthority,
    archive: P0Archive,
    handoff: Mapping[str, Any],
    encoder: Mapping[str, Any],
    result_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "k0-rvg-p1-execution-manifest-v1",
        "runtime_git_head": _git(root, "rev-parse", "HEAD"),
        "p1_implementation_authority_commit": AUTHORITY_COMMIT,
        "scientific_execution_authority_path": authority.path,
        "scientific_execution_authority_commit": authority.commit,
        "scientific_execution_authority_git_blob": authority.git_blob,
        "p1_implementation_commit": authority.implementation_commit,
        "p1_runner_sha256": file_sha256(root / RUNNER_REL),
        "p1_test_sha256": file_sha256(root / TEST_REL),
        "validated_observer": {
            "commit": OBSERVER_COMMIT,
            "sha256": OBSERVER_SHA256,
            "git_blob": OBSERVER_GIT_BLOB,
        },
        "k2s_helper": {
            "sha256": K2S_SHA256,
            "git_blob": K2S_GIT_BLOB,
        },
        "a0_runtime_contract": {
            "commit": A0_COMMIT,
            "model_blob_sha": A0_MODEL_BLOB_SHA,
            "heads_tree_sha": A0_HEADS_TREE_SHA,
        },
        "p0_archive_sha256": archive.hashes,
        "handoff": {
            "zip_sha256": handoff["zip_sha256"],
            "checkpoint_sha256": handoff["checkpoint_sha256"],
        },
        "encoder": {
            "canonical_digest": encoder["canonical_digest"],
            "raw_concat_digest": encoder["raw_concat_digest"],
        },
        "hf": {
            "model_id": HF_MODEL,
            "revision": HF_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "mamba_source_sha256": MAMBA_SOURCE_SHA256,
            "mamba_upstream_git_blob": MAMBA_UPSTREAM_BLOB,
        },
        "primary_layer": PRIMARY_LAYER,
        "window": WINDOW,
        "item_count": P0_ITEM_COUNT,
        "block_count": P0_BLOCK_COUNT,
        "result_artifact_sha256": dict(result_hashes),
        "scientific_model_forward_executed": True,
        "scientific_recurrent_state_read": True,
        "logits_read": False,
        "causal_intervention_executed": False,
        "learned_geometry_used": False,
        "outcome_selected_subspace_used": False,
        "status": "COMPLETE_SCIENTIFIC_RAW_VECTOR_EXECUTION",
    }


def write_scientific_artifacts_atomic(
    output_dir: Path,
    artifacts_without_manifest: Mapping[str, bytes],
    manifest_builder,
) -> dict[str, str]:
    require(not output_dir.exists(), "SCIENTIFIC_OUTPUT_ALREADY_EXISTS")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=parent))
    try:
        expected_without_manifest = set(RESULT_ARTIFACTS) - {"execution_manifest.json"}
        require(set(artifacts_without_manifest) == expected_without_manifest, "SCIENTIFIC_ARTIFACT_SET_MISMATCH")
        hashes = {}
        for name in RESULT_ARTIFACTS:
            if name == "execution_manifest.json":
                continue
            raw = artifacts_without_manifest[name]
            (tmp / name).write_bytes(raw)
            hashes[name] = sha256_bytes(raw)
        manifest = manifest_builder(hashes)
        manifest_raw = canonical_json_bytes(manifest, final_lf=True)
        (tmp / "execution_manifest.json").write_bytes(manifest_raw)
        hashes["execution_manifest.json"] = sha256_bytes(manifest_raw)
        observed = {p.name for p in tmp.iterdir() if p.is_file()}
        require(observed == set(RESULT_ARTIFACTS), "SCIENTIFIC_TMP_ARTIFACT_SET_MISMATCH")
        tmp.rename(output_dir)
        return hashes
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def execute_scientific(
    root: Path,
    handoff_path: Path,
    authority_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    # Critical order: authority gate before scientific model construction.
    repo = repo_contract(root)
    authenticate_code_dependencies(root)
    archive = authenticate_p0_archive(root)
    authority = authenticate_execution_authority(root, authority_path)

    observer, k2s, model, tokenizer, binding, layer_map, encoder, handoff, hf = (
        validate_runtime_and_build_model(root, handoff_path)
    )

    # Revalidate every token contract before first scientific model forward.
    revalidated: list[tuple[BranchTexts, RevalidatedTokenContract]] = []
    for i in range(P0_ITEM_COUNT):
        texts = reconstruct_branch_texts(archive.candidates, i)
        contract = revalidate_token_contract(tokenizer, texts, archive.token_contracts[i])
        revalidated.append((texts, contract))
    require(len(revalidated) == P0_ITEM_COUNT, "SCIENTIFIC_TOKEN_REVALIDATION_COUNT_MISMATCH")

    audit = AuditAccumulator()
    item_rows: list[dict[str, Any]] = []
    state_hash_rows: list[dict[str, Any]] = []

    for i in range(P0_ITEM_COUNT):
        row, hashes = _item_scientific_execution(
            observer, k2s, model, tokenizer, binding, layer_map, archive, i, audit
        )
        item_rows.append(row)
        state_hash_rows.extend(hashes)

    require(len(item_rows) == P0_ITEM_COUNT, "SCIENTIFIC_ITEM_COUNT_MISMATCH")
    require(len(state_hash_rows) == P0_ITEM_COUNT * 4 * TARGET_COUNT, "SCIENTIFIC_STATE_HASH_COUNT_MISMATCH")
    require(audit.incoming_common_state_failures == 0, "SCIENTIFIC_INCOMING_COMMON_STATE_FAILURE")

    block_rows = aggregate_blocks(item_rows, archive.mapping)
    endpoint = summarize_endpoints(block_rows)

    recurrence_audit = {
        "schema_version": "k0-rvg-p1-recurrence-audit-v1",
        "layer": PRIMARY_LAYER,
        "tensor_shape": list(EXPECTED_STATE_SHAPE),
        "dtype": EXPECTED_DTYPE,
        "device": EXPECTED_DEVICE,
        "total_layer23_recurrence_records_validated": audit.recurrence_count,
        "exact_recurrence_pass_count": audit.recurrence_count,
        "max_velocity_abs_residual": audit.max_abs_residual,
        "max_velocity_relative_frobenius_residual": audit.max_relative_residual,
        "max_velocity_scaled_tolerance_residual": audit.max_scaled_tolerance_residual,
        "incoming_common_state_comparison_count": audit.incoming_common_state_comparisons,
        "incoming_common_state_comparison_failures": audit.incoming_common_state_failures,
    }

    artifacts = {
        "item_metrics.jsonl": canonical_jsonl_bytes(item_rows),
        "block_metrics.jsonl": canonical_jsonl_bytes(block_rows),
        "endpoint_summary.json": canonical_json_bytes(endpoint, final_lf=True),
        "recurrence_audit.json": canonical_json_bytes(recurrence_audit, final_lf=True),
        "state_hash_audit.jsonl": canonical_jsonl_bytes(state_hash_rows),
    }

    def manifest_builder(result_hashes):
        return build_execution_manifest(
            root, authority, archive, handoff, encoder, result_hashes
        )

    hashes = write_scientific_artifacts_atomic(output_dir, artifacts, manifest_builder)
    return {
        "status": "COMPLETE_SCIENTIFIC_RAW_VECTOR_EXECUTION",
        "repo": repo,
        "output_dir": str(output_dir),
        "artifact_sha256": hashes,
        "overall_verdict": endpoint["overall_verdict"],
        "scientific_model_forward_executed": True,
        "scientific_recurrent_state_read": True,
        "logits_read": False,
        "causal_intervention_executed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K0-RVG-P1 raw-vector runner.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-preflight", action="store_true")
    mode.add_argument("--execute-scientific", action="store_true")
    parser.add_argument("--seed180-handoff", type=Path)
    parser.add_argument("--execution-authority", type=str)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]

    if args.seed180_handoff is None:
        parser.error("--seed180-handoff is required")

    if args.synthetic_preflight:
        require(args.execution_authority is None, "SYNTHETIC_EXECUTION_AUTHORITY_FORBIDDEN")
        require(args.output_dir is None, "SYNTHETIC_OUTPUT_DIR_FORBIDDEN")
        result = run_synthetic_preflight(root, args.seed180_handoff.resolve())
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0

    require(args.execute_scientific, "UNKNOWN_MODE")
    if args.execution_authority is None:
        parser.error("--execution-authority is required for --execute-scientific")
    if args.output_dir is None:
        parser.error("--output-dir is required for --execute-scientific")
    result = execute_scientific(
        root,
        args.seed180_handoff.resolve(),
        args.execution_authority,
        args.output_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
