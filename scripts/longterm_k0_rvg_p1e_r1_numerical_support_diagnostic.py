"""K0-RVG-P1E-R1 bounded numerical-support diagnostic.

This module implements an instrument/root-cause diagnostic only.

Implementation-stage authority permits fabricated synthetic model forwards and
static authentication of the frozen P0 archive. A scientific diagnostic forward
is fail-closed unless a later tracked R1E execution authority explicitly
authorizes exactly local item 0 / matched_corr / one forward.

No P1 primary endpoint is implemented here.
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
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Frozen R1 protocol / implementation scope
# ---------------------------------------------------------------------------

R1_PROTOCOL_COMMIT = "02f184a186aa6b90fc98ece717723de0adafc89c"
R1_INCIDENT_COMMIT = "6fe6a80f5fa0125971ac83d436e6e400cb7ab6f1"
R1_PROTOCOL_REL = (
    "reports/longterm_k0_rvg_p1e_r1_"
    "numerical_support_recovery_diagnostic_authority_candidate.md"
)
R1_PROTOCOL_SHA256 = "d9a10812c96d9453460356658d09f282a6f4da022b299ce9ceab99cb098f2fb0"

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

RUNNER_REL = "scripts/longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py"
TEST_REL = "tests/test_longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py"

HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
IMPLEMENTATION_UNTRACKED = {RUNNER_REL, TEST_REL}

R1_LOCAL_TEMPLATE_INDEX = 0
R1_BRANCH_ROLE = "matched_corr"
PRIMARY_LAYER = 23
WINDOW = 8
TARGET_COUNT = 9
EXPECTED_LAYER_COUNT = 24
EXPECTED_RECORDS_PER_BRANCH = EXPECTED_LAYER_COUNT * TARGET_COUNT
EXPECTED_STATE_SHAPE = (1, 1536, 16)
EXPECTED_DTYPE = "torch.float32"
EXPECTED_DEVICE = "cpu"

VELOCITY_ATOL = 1e-6
VELOCITY_RTOL = 1e-5
NORM_FLOOR = 1e-12

DIAGNOSTIC_FILENAME = "numerical_support_diagnostic.json"


# ---------------------------------------------------------------------------
# Frozen P1 / P0 / observer / runtime identities
# ---------------------------------------------------------------------------

P1_IMPLEMENTATION_COMMIT = "2e6bb106d5d3081b7ae69ec4cde652e79d36070c"
P1_RUNNER_REL = "scripts/longterm_k0_rvg_p1_raw_vector_execution.py"
P1_TEST_REL = "tests/test_longterm_k0_rvg_p1_raw_vector_execution.py"
P1_RUNNER_SHA256 = "2bd59a25a6303dc86c36c9438296b42197cc67019e9d4e9ee2fbbe1d832ec6eb"
P1_TEST_SHA256 = "06c30c974029126fdc853a235071376738cc2769409369ccdb60f0878bac7706"

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

OBSERVER_REL = "scripts/longterm_k0_rvg_raw_recurrence_observer.py"
OBSERVER_SHA256 = "12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25"
OBSERVER_GIT_BLOB = "f2dbdfe52661eca384897578ab272e602e36deac"
OBSERVER_COMMIT = "fcfe161c12f4ed8ef37aff435554cc0660e477af"

K2S_REL = "scripts/longterm_k2s_pair_specific_event_dynamics.py"
K2S_SHA256 = "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
K2S_GIT_BLOB = "3a651fb508669bdcf72441a4869b863d6eee6c1f"

A0_COMMIT = "55debe94f0d19d16a334395e8561901fed6b52fa"
A0_MODEL_REL = "src/contramamba/modeling_v6b_minimal.py"
A0_HEADS_REL = "src/contramamba/heads"
A0_MODEL_BLOB_SHA = "f0ddc0eda64937de6fcd27943e30a296082c01d5"
A0_HEADS_TREE_SHA = "68d26855aa511fcd41d6f395ae5f87177a162678"

EXPECTED_ZIP_SHA256 = "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
EXPECTED_CHECKPOINT_SHA256 = "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
COMMON_ENCODER_CANONICAL_SHA256 = "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
COMMON_ENCODER_RAW_CONCAT_SHA256 = "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
MAMBA_SOURCE_SHA256 = "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"


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


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.call(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0


def implementation_repo_contract(root: Path) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")
    require(_is_ancestor(root, R1_PROTOCOL_COMMIT, head), "R1_PROTOCOL_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()
    allowed_untracked = set(HISTORICAL_K1_UNTRACKED)
    if head == R1_PROTOCOL_COMMIT:
        allowed_untracked |= IMPLEMENTATION_UNTRACKED
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(
            line[:2] == "??" and path in allowed_untracked,
            "R1_IMPLEMENTATION_DIRTY_CONTRACT_MISMATCH",
        )
    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "r1_protocol_is_ancestor": True,
        "runtime_dirty_contract": status,
    }


def authenticate_static_dependencies(root: Path) -> dict[str, Any]:
    exact_files = {
        R1_PROTOCOL_REL: R1_PROTOCOL_SHA256,
        P1_RUNNER_REL: P1_RUNNER_SHA256,
        P1_TEST_REL: P1_TEST_SHA256,
        OBSERVER_REL: OBSERVER_SHA256,
        K2S_REL: K2S_SHA256,
    }
    for rel, wanted in exact_files.items():
        path = root / rel
        require(path.is_file(), f"REQUIRED_FILE_MISSING:{rel}")
        require(file_sha256(path) == wanted, f"REQUIRED_FILE_SHA_MISMATCH:{rel}")

    require(
        _git(root, "rev-parse", f"HEAD:{OBSERVER_REL}") == OBSERVER_GIT_BLOB,
        "OBSERVER_GIT_BLOB_MISMATCH",
    )
    require(
        _git(root, "rev-parse", f"HEAD:{K2S_REL}") == K2S_GIT_BLOB,
        "K2S_GIT_BLOB_MISMATCH",
    )
    require(
        _git(root, "rev-parse", f"{A0_COMMIT}:{A0_MODEL_REL}") == A0_MODEL_BLOB_SHA,
        "A0_FROZEN_MODEL_BLOB_MISMATCH",
    )
    require(
        _git(root, "rev-parse", f"HEAD:{A0_MODEL_REL}") == A0_MODEL_BLOB_SHA,
        "A0_CURRENT_MODEL_BLOB_MISMATCH",
    )
    require(
        _git(root, "rev-parse", f"{A0_COMMIT}:{A0_HEADS_REL}") == A0_HEADS_TREE_SHA,
        "A0_FROZEN_HEADS_TREE_MISMATCH",
    )
    require(
        _git(root, "rev-parse", f"HEAD:{A0_HEADS_REL}") == A0_HEADS_TREE_SHA,
        "A0_CURRENT_HEADS_TREE_MISMATCH",
    )
    return {
        "r1_protocol_sha256": R1_PROTOCOL_SHA256,
        "p1_runner_sha256": P1_RUNNER_SHA256,
        "p1_test_sha256": P1_TEST_SHA256,
        "observer_sha256": OBSERVER_SHA256,
        "observer_git_blob": OBSERVER_GIT_BLOB,
        "k2s_sha256": K2S_SHA256,
        "k2s_git_blob": K2S_GIT_BLOB,
        "a0_model_blob_sha": A0_MODEL_BLOB_SHA,
        "a0_heads_tree_sha": A0_HEADS_TREE_SHA,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"JSONL_PARSE_FAILURE:{path}:{lineno}") from exc
        require(isinstance(row, dict), f"JSONL_OBJECT_REQUIRED:{path}:{lineno}")
        rows.append(row)
    return rows


@dataclass(frozen=True)
class P0Static:
    candidate: dict[str, Any]
    token_contract: dict[str, Any]
    hashes: dict[str, str]


def authenticate_p0_static(root: Path) -> P0Static:
    archive = root / P0_ARCHIVE_REL
    require(archive.is_dir(), "P0_ARCHIVE_DIR_MISSING")

    for name, wanted in P0_ARTIFACT_SHA256.items():
        path = archive / name
        require(path.is_file(), f"P0_ARTIFACT_MISSING:{name}")
        require(file_sha256(path) == wanted, f"P0_ARTIFACT_SHA_MISMATCH:{name}")

    candidates = load_jsonl(archive / "candidate_pool.jsonl")
    contracts = load_jsonl(archive / "token_contracts.jsonl")
    manifest = json.loads((archive / "provisioning_manifest.json").read_text("utf-8"))

    require(len(candidates) == P0_ITEM_COUNT, "P0_CANDIDATE_COUNT_MISMATCH")
    require(len(contracts) == P0_ITEM_COUNT, "P0_TOKEN_CONTRACT_COUNT_MISMATCH")
    require(manifest.get("item_count") == P0_ITEM_COUNT, "P0_MANIFEST_ITEM_COUNT_MISMATCH")
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

    by_local = {int(row["local_template_index"]): row for row in candidates}
    contract_by_local = {int(row["local_template_index"]): row for row in contracts}
    require(len(by_local) == P0_ITEM_COUNT, "P0_DUPLICATE_CANDIDATE_INDEX")
    require(len(contract_by_local) == P0_ITEM_COUNT, "P0_DUPLICATE_CONTRACT_INDEX")
    require(R1_LOCAL_TEMPLATE_INDEX in by_local, "P0_ITEM0_MISSING")
    require(R1_LOCAL_TEMPLATE_INDEX in contract_by_local, "P0_ITEM0_CONTRACT_MISSING")

    candidate = dict(by_local[R1_LOCAL_TEMPLATE_INDEX])
    contract = dict(contract_by_local[R1_LOCAL_TEMPLATE_INDEX])
    require(candidate["pair_id"] == contract["pair_id"], "P0_ITEM0_PAIR_ID_MISMATCH")
    require(candidate["stable_item_id"] == contract["stable_item_id"], "P0_ITEM0_STABLE_ID_MISMATCH")
    require(int(contract["local_template_index"]) == 0, "P0_ITEM0_LOCAL_INDEX_MISMATCH")
    return P0Static(
        candidate=candidate,
        token_contract=contract,
        hashes=dict(P0_ARTIFACT_SHA256),
    )


def _import_repo_module(root: Path, rel: str, name: str):
    path = root / rel
    require(path.is_file(), f"MODULE_MISSING:{rel}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"MODULE_SPEC_FAILURE:{rel}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_runtime_modules(root: Path):
    observer = _import_repo_module(root, OBSERVER_REL, "k0_rvg_r1_observer")
    k2s = _import_repo_module(root, K2S_REL, "k0_rvg_r1_k2s")
    return observer, k2s


def validate_runtime_and_build_model(root: Path, handoff_path: Path):
    authenticate_static_dependencies(root)
    observer, k2s = import_runtime_modules(root)

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

    model = k2s.build_a0_model(root, snapshot, checkpoint)
    model.eval()
    binding = observer.resolve_source_binding()
    require(binding.source_sha256 == MAMBA_SOURCE_SHA256, "MAMBA_SOURCE_SHA_MISMATCH")
    layer_map = observer.registered_mamba_layers(model)
    require(layer_map == k2s.registered_mamba_layers(model), "MAMBA_LAYER_MAP_DRIFT")

    return observer, k2s, model, hf["tokenizer"], binding, layer_map, handoff, encoder, hf


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


def target_indices(t_e: int) -> tuple[int, ...]:
    require(t_e >= 1, "TARGET_TE_LT_ONE")
    return tuple([t_e - 1] + list(range(t_e, t_e + WINDOW)))


@dataclass(frozen=True)
class FixedBranch:
    prefix_text: str
    branch_text: str
    matched_control_text: str
    t_e: int


def static_item0_fixed_branch(tokenizer: Any, p0: P0Static) -> FixedBranch:
    item = p0.candidate
    archived = p0.token_contract

    prefix = str(item["prefix_text"])
    corr = prefix + str(item["correction_text"])
    ctrl = prefix + str(item["control_text"])

    prefix_ids = token_ids(tokenizer, prefix)
    corr_ids = token_ids(tokenizer, corr)
    ctrl_ids = token_ids(tokenizer, ctrl)

    require(corr_ids[:len(prefix_ids)] == prefix_ids, "ITEM0_CORR_PREFIX_IDENTITY_FAILURE")
    require(ctrl_ids[:len(prefix_ids)] == prefix_ids, "ITEM0_CTRL_PREFIX_IDENTITY_FAILURE")
    require(len(prefix_ids) == int(archived["prefix_token_count"]), "ITEM0_PREFIX_COUNT_MISMATCH")
    require(token_id_sha256(prefix_ids) == archived["prefix_token_sha256"], "ITEM0_PREFIX_SHA_MISMATCH")

    t_e = divergence_anchor(corr_ids, ctrl_ids, len(prefix_ids))
    require(t_e == int(archived["matched_divergence_anchor"]), "ITEM0_MATCHED_TE_MISMATCH")
    require(
        t_e - len(prefix_ids) == int(archived["matched_divergence_offset_from_prefix"]),
        "ITEM0_MATCHED_OFFSET_MISMATCH",
    )
    require(len(corr_ids) == int(archived["matched_correction_token_count"]), "ITEM0_CORR_COUNT_MISMATCH")
    require(len(ctrl_ids) == int(archived["matched_control_token_count"]), "ITEM0_CTRL_COUNT_MISMATCH")
    require(len(corr_ids) - t_e >= WINDOW, "ITEM0_CORR_W8_UNAVAILABLE")
    require(len(ctrl_ids) - t_e >= WINDOW, "ITEM0_CTRL_W8_UNAVAILABLE")
    require(archived["matched_w8_available"] is True, "ITEM0_ARCHIVED_W8_FALSE")

    return FixedBranch(
        prefix_text=prefix,
        branch_text=corr,
        matched_control_text=ctrl,
        t_e=t_e,
    )


def synthetic_fixed_branch(tokenizer: Any) -> FixedBranch:
    prefix = (
        "Claim: fabricated r1 alpha\n"
        "Evidence: fabricated r1 beta\n"
        "Additional evidence:\n"
    )
    corr = prefix + (
        " fabricated correction amber cedar quartz river summit valley "
        "willow ember frost cobalt jade silver."
    )
    ctrl = prefix + (
        " fabricated control bronze maple crystal meadow harbor ridge "
        "birch slate pearl topaz ivory copper."
    )
    prefix_ids = token_ids(tokenizer, prefix)
    corr_ids = token_ids(tokenizer, corr)
    ctrl_ids = token_ids(tokenizer, ctrl)
    require(corr_ids[:len(prefix_ids)] == prefix_ids, "SYNTHETIC_CORR_PREFIX_IDENTITY_FAILURE")
    require(ctrl_ids[:len(prefix_ids)] == prefix_ids, "SYNTHETIC_CTRL_PREFIX_IDENTITY_FAILURE")
    t_e = divergence_anchor(corr_ids, ctrl_ids, len(prefix_ids))
    require(t_e >= 1, "SYNTHETIC_TE_LT_ONE")
    require(len(corr_ids) - t_e >= WINDOW, "SYNTHETIC_CORR_W8_UNAVAILABLE")
    require(len(ctrl_ids) - t_e >= WINDOW, "SYNTHETIC_CTRL_W8_UNAVAILABLE")
    return FixedBranch(prefix, corr, ctrl, t_e)


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return sha256_bytes(array.tobytes())


def _norm64(tensor: Any) -> float:
    import torch
    x = tensor.detach().to(device="cpu", dtype=torch.float64)
    return float(torch.linalg.vector_norm(x.reshape(-1), ord=2).item())


def _finite_scalar(value: float, name: str) -> float:
    require(math.isfinite(value), f"NONFINITE_DIAGNOSTIC:{name}")
    return float(value)


def diagnose_record(record: Any) -> dict[str, Any]:
    import torch

    require(record.layer_index == PRIMARY_LAYER, "R1_LAYER_MISMATCH")
    require(tuple(record.s_prev_meta.shape) == EXPECTED_STATE_SHAPE, "R1_STATE_SHAPE_MISMATCH")
    require(record.s_prev_meta.dtype == EXPECTED_DTYPE, "R1_STATE_DTYPE_MISMATCH")
    require(record.s_prev_meta.device == EXPECTED_DEVICE, "R1_STATE_DEVICE_MISMATCH")

    s_prev = record.s_prev.detach().cpu()
    g = record.g.detach().cpu()
    w = record.w.detach().cpu()
    s_post = record.s_post.detach().cpu()

    require(s_prev.dtype == torch.float32, "R1_S_PREV_NOT_FLOAT32")
    require(g.dtype == torch.float32, "R1_G_NOT_FLOAT32")
    require(w.dtype == torch.float32, "R1_W_NOT_FLOAT32")
    require(s_post.dtype == torch.float32, "R1_S_POST_NOT_FLOAT32")

    exact_reconstruction = g * s_prev + w
    exact_pass = bool(torch.equal(exact_reconstruction, s_post))

    v_raw32 = s_post - s_prev
    v_rearr32 = (g - 1.0) * s_prev + w
    d32 = v_raw32 - v_rearr32
    abs_d32 = torch.abs(d32)
    tol32 = VELOCITY_ATOL + VELOCITY_RTOL * torch.abs(v_rearr32)

    allclose_pass = bool(
        torch.allclose(
            v_raw32,
            v_rearr32,
            atol=VELOCITY_ATOL,
            rtol=VELOCITY_RTOL,
        )
    )

    flat_abs = abs_d32.reshape(-1)
    max_flat_index = int(torch.argmax(flat_abs).item())
    max_abs = float(flat_abs[max_flat_index].item())

    flat_tol = tol32.reshape(-1)
    scaled = abs_d32 / tol32
    flat_scaled = scaled.reshape(-1)
    max_scaled = float(torch.max(flat_scaled).item())

    exceeds = abs_d32 > tol32
    fail_count = int(torch.count_nonzero(exceeds).item())
    element_count = int(abs_d32.numel())
    fail_fraction = float(fail_count / element_count)

    d32_norm = _norm64(d32)
    v_raw32_norm = _norm64(v_raw32)
    v_rearr32_norm = _norm64(v_rearr32)
    rel32 = d32_norm / max(v_raw32_norm, NORM_FLOOR)

    s_prev_norm = _norm64(s_prev)
    s_post_norm = _norm64(s_post)
    w_norm = _norm64(w)
    carry32 = (g - 1.0) * s_prev
    carry_norm = _norm64(carry32)
    state_velocity_ratio = s_prev_norm / max(v_raw32_norm, NORM_FLOOR)

    s_prev64 = s_prev.to(torch.float64)
    g64 = g.to(torch.float64)
    w64 = w.to(torch.float64)
    s_post64 = s_post.to(torch.float64)
    v_raw64 = s_post64 - s_prev64
    v_rearr64 = (g64 - 1.0) * s_prev64 + w64
    d64 = v_raw64 - v_rearr64
    d64_norm = _norm64(d64)
    v_raw64_norm = _norm64(v_raw64)
    rel64 = d64_norm / max(v_raw64_norm, NORM_FLOOR)
    max_abs64 = float(torch.max(torch.abs(d64)).item())

    def flat_value(x):
        return float(x.reshape(-1)[max_flat_index].item())

    audit = {
        "flat_index": max_flat_index,
        "S_prev": flat_value(s_prev),
        "S_post": flat_value(s_post),
        "G": flat_value(g),
        "W": flat_value(w),
        "V_raw32": flat_value(v_raw32),
        "V_rearr32": flat_value(v_rearr32),
        "D32": flat_value(d32),
        "tolerance_scale": flat_value(tol32),
        "scaled_residual": float(flat_scaled[max_flat_index].item()),
    }

    result = {
        "schema_version": "k0-rvg-p1e-r1-coordinate-v1",
        "token_index": int(record.token_index),
        "exact_recurrence_pass": exact_pass,
        "frozen_allclose_pass": allclose_pass,
        "max_abs_residual": _finite_scalar(max_abs, "max_abs_residual"),
        "residual_frobenius_norm": _finite_scalar(d32_norm, "residual_frobenius_norm"),
        "v_raw32_frobenius_norm": _finite_scalar(v_raw32_norm, "v_raw32_frobenius_norm"),
        "v_rearr32_frobenius_norm": _finite_scalar(v_rearr32_norm, "v_rearr32_frobenius_norm"),
        "relative_frobenius_residual": _finite_scalar(rel32, "relative_frobenius_residual"),
        "tolerance_scale_min": _finite_scalar(
            float(torch.min(tol32).item()),
            "tolerance_scale_min",
        ),
        "tolerance_scale_max": _finite_scalar(
            float(torch.max(tol32).item()),
            "tolerance_scale_max",
        ),
        "max_scaled_tolerance_residual": _finite_scalar(max_scaled, "max_scaled_tolerance_residual"),
        "failing_element_count": fail_count,
        "element_count": element_count,
        "failing_element_fraction": _finite_scalar(fail_fraction, "failing_element_fraction"),
        "s_prev_frobenius_norm": _finite_scalar(s_prev_norm, "s_prev_frobenius_norm"),
        "s_post_frobenius_norm": _finite_scalar(s_post_norm, "s_post_frobenius_norm"),
        "w_frobenius_norm": _finite_scalar(w_norm, "w_frobenius_norm"),
        "carry_frobenius_norm": _finite_scalar(carry_norm, "carry_frobenius_norm"),
        "state_to_raw_velocity_norm_ratio": _finite_scalar(
            state_velocity_ratio,
            "state_to_raw_velocity_norm_ratio",
        ),
        "max_residual_element": audit,
        "float64_snapshot": {
            "max_abs_residual": _finite_scalar(max_abs64, "float64_max_abs_residual"),
            "residual_frobenius_norm": _finite_scalar(d64_norm, "float64_residual_norm"),
            "v_raw64_frobenius_norm": _finite_scalar(v_raw64_norm, "v_raw64_norm"),
            "relative_frobenius_residual": _finite_scalar(rel64, "float64_relative_residual"),
        },
        "state_hashes": {
            "S_prev_sha256": tensor_sha256(s_prev),
            "G_sha256": tensor_sha256(g),
            "W_sha256": tensor_sha256(w),
            "S_post_sha256": tensor_sha256(s_post),
        },
    }
    # Ensure the persisted object is JSON-finite now, not later.
    canonical_json_bytes(result)
    return result


def classify_coordinate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    require(len(rows) == TARGET_COUNT, "R1_COORDINATE_COUNT_MISMATCH")
    require(len({int(row["token_index"]) for row in rows}) == TARGET_COUNT, "R1_DUPLICATE_TOKEN_INDEX")

    exact_fail = [row for row in rows if not bool(row["exact_recurrence_pass"])]
    allclose_fail = [row for row in rows if not bool(row["frozen_allclose_pass"])]

    if exact_fail:
        classification = "EXACT_RECURRENCE_OR_CAPTURE_FAILURE"
    elif allclose_fail:
        classification = "EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE"
    else:
        classification = "ORIGINAL_FAILURE_NOT_REPRODUCED_IN_FIXED_SINGLE_BRANCH_DIAGNOSTIC"

    fail_count = len(allclose_fail)
    if fail_count == 0:
        profile = "NO_FAILURE_REPRODUCED_WITHIN_FIXED_BRANCH"
    elif fail_count == 1:
        profile = "SINGLE_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH"
    else:
        profile = "MULTI_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH"

    scaled = sorted(float(row["max_scaled_tolerance_residual"]) for row in rows)
    middle = len(scaled) // 2
    median_scaled = scaled[middle]

    first_fail = None
    if allclose_fail:
        first_fail = min(int(row["token_index"]) for row in allclose_fail)

    support = {
        "coordinate_count": TARGET_COUNT,
        "failing_coordinate_count": fail_count,
        "passing_coordinate_count": TARGET_COUNT - fail_count,
        "first_failing_token_index": first_fail,
        "maximum_scaled_tolerance_residual": max(scaled),
        "median_scaled_tolerance_residual": median_scaled,
    }
    return classification, profile, support


def retain_primary_records_and_discard_nonprimary(
    records: dict[tuple[int, int], Any],
    targets: Sequence[int],
) -> dict[int, Any]:
    expected = {
        (layer, token)
        for layer in range(EXPECTED_LAYER_COUNT)
        for token in targets
    }
    require(set(records) == expected, "R1_CAPTURE_COORDINATE_SET_MISMATCH")
    primary = {
        token: records[(PRIMARY_LAYER, token)]
        for token in targets
    }
    # Frozen R1 authority requires all non-layer-23 tensor records to be
    # discarded immediately after capture-completeness validation.
    records.clear()
    return primary


def capture_fixed_branch(
    observer: Any,
    k2s: Any,
    model: Any,
    layer_map: Mapping[int, int],
    binding: Any,
    tokenizer: Any,
    branch_text: str,
    t_e: int,
) -> list[dict[str, Any]]:
    targets = target_indices(t_e)
    bundle = k2s.task_mask_bundle(tokenizer, branch_text)
    collector = observer.RawRecurrenceCollector(binding, layer_map, targets)
    with collector.capture():
        k2s._full_model_forward(model, bundle)

    require(collector.records is not None, "R1_CAPTURE_RECORDS_MISSING")
    require(
        len(collector.records) == EXPECTED_RECORDS_PER_BRANCH,
        "R1_CAPTURE_COMPLETENESS_FAILURE",
    )
    primary = retain_primary_records_and_discard_nonprimary(
        collector.records,
        targets,
    )

    rows = [
        diagnose_record(primary[token])
        for token in targets
    ]
    require([row["token_index"] for row in rows] == list(targets), "R1_PRIMARY_ORDER_MISMATCH")
    return rows


def fabricated_fixture(pass_tolerance: bool):
    import torch

    class Meta:
        shape = EXPECTED_STATE_SHAPE
        dtype = EXPECTED_DTYPE
        device = EXPECTED_DEVICE

    class Record:
        layer_index = PRIMARY_LAYER
        token_index = 7
        s_prev_meta = Meta()

    record = Record()
    if pass_tolerance:
        s_prev = torch.ones(EXPECTED_STATE_SHAPE, dtype=torch.float32)
        g = torch.full(EXPECTED_STATE_SHAPE, 0.5, dtype=torch.float32)
        w = torch.full(EXPECTED_STATE_SHAPE, 0.25, dtype=torch.float32)
        s_post = g * s_prev + w
    else:
        # Exact float32 recurrence remains intact because +1 is rounded away at
        # 1e8, while algebraic rearrangement exposes the write term.
        s_prev = torch.full(EXPECTED_STATE_SHAPE, 1.0e8, dtype=torch.float32)
        g = torch.ones(EXPECTED_STATE_SHAPE, dtype=torch.float32)
        w = torch.ones(EXPECTED_STATE_SHAPE, dtype=torch.float32)
        s_post = g * s_prev + w

    record.s_prev = s_prev
    record.g = g
    record.w = w
    record.s_post = s_post
    return record


def run_synthetic_preflight(root: Path, handoff_path: Path) -> dict[str, Any]:
    repo = implementation_repo_contract(root)
    deps = authenticate_static_dependencies(root)
    p0 = authenticate_p0_static(root)  # static only; no P0 branch is forwarded.
    observer, k2s, model, tokenizer, binding, layer_map, handoff, encoder, hf = (
        validate_runtime_and_build_model(root, handoff_path)
    )

    fixed = synthetic_fixed_branch(tokenizer)
    rows_first = capture_fixed_branch(
        observer, k2s, model, layer_map, binding, tokenizer,
        fixed.branch_text, fixed.t_e,
    )
    rows_second = capture_fixed_branch(
        observer, k2s, model, layer_map, binding, tokenizer,
        fixed.branch_text, fixed.t_e,
    )
    require(
        canonical_json_bytes(rows_first) == canonical_json_bytes(rows_second),
        "R1_SYNTHETIC_REPEAT_IDENTITY_FAILURE",
    )

    classification, profile, support = classify_coordinate_rows(rows_first)

    pass_fixture = diagnose_record(fabricated_fixture(True))
    fail_fixture = diagnose_record(fabricated_fixture(False))
    require(pass_fixture["exact_recurrence_pass"] is True, "R1_PASS_FIXTURE_EXACT_FAIL")
    require(pass_fixture["frozen_allclose_pass"] is True, "R1_PASS_FIXTURE_TOLERANCE_FAIL")
    require(fail_fixture["exact_recurrence_pass"] is True, "R1_FAIL_FIXTURE_EXACT_FAIL")
    require(fail_fixture["frozen_allclose_pass"] is False, "R1_FAIL_FIXTURE_DID_NOT_FAIL_TOLERANCE")
    require(
        fail_fixture["max_scaled_tolerance_residual"] > 1.0,
        "R1_FAIL_FIXTURE_SCALED_RESIDUAL_NOT_GT_ONE",
    )

    return {
        "schema_version": "k0-rvg-p1e-r1-synthetic-preflight-v1",
        "status": "PASS_SYNTHETIC_R1_NUMERICAL_SUPPORT_DIAGNOSTIC",
        "repo": repo,
        "dependencies": deps,
        "p0_static_artifact_sha256": p0.hashes,
        "runtime": {
            "handoff_zip_sha256": handoff["zip_sha256"],
            "checkpoint_sha256": handoff["checkpoint_sha256"],
            "encoder_canonical_sha256": encoder["canonical_digest"],
            "encoder_raw_concat_sha256": encoder["raw_concat_digest"],
            "hf_model": hf["hf_model_id"],
            "hf_revision": hf["resolved_hf_revision"],
            "transformers_version": hf["transformers_version"],
        },
        "synthetic": {
            "model_forward_count": 2,
            "branch_role": R1_BRANCH_ROLE,
            "coordinate_count": len(rows_first),
            "classification": classification,
            "local_support_profile": profile,
            "support": support,
            "repeat_identity": "PASS_EXACT",
            "pass_fixture": {
                "exact_recurrence_pass": pass_fixture["exact_recurrence_pass"],
                "frozen_allclose_pass": pass_fixture["frozen_allclose_pass"],
            },
            "fail_fixture": {
                "exact_recurrence_pass": fail_fixture["exact_recurrence_pass"],
                "frozen_allclose_pass": fail_fixture["frozen_allclose_pass"],
                "max_scaled_tolerance_residual": fail_fixture[
                    "max_scaled_tolerance_residual"
                ],
            },
        },
        "scientific_population_model_forward_executed": False,
        "scientific_population_recurrent_state_read": False,
        "p1_endpoint_computed": False,
        "replacement_tolerance_selected": False,
        "logits_read": False,
        "causal_intervention_executed": False,
        "scientific_result_interpreted": False,
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


@dataclass(frozen=True)
class R1ExecutionAuthority:
    path: str
    commit: str
    git_blob: str
    implementation_commit: str


def validate_r1_implementation_commit(
    root: Path,
    implementation_commit: str,
) -> dict[str, str]:
    require(len(implementation_commit) == 40, "R1_IMPLEMENTATION_COMMIT_INVALID")
    parent = _git(root, "rev-parse", f"{implementation_commit}^")
    require(parent == R1_PROTOCOL_COMMIT, "R1_IMPLEMENTATION_PARENT_MISMATCH")

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
        "R1_IMPLEMENTATION_SCOPE_MISMATCH",
    )
    return {
        "runner_blob": _git(root, "rev-parse", f"{implementation_commit}:{RUNNER_REL}"),
        "test_blob": _git(root, "rev-parse", f"{implementation_commit}:{TEST_REL}"),
    }


def authenticate_r1_execution_authority(
    root: Path,
    authority_path: str,
) -> R1ExecutionAuthority:
    rel = Path(authority_path)
    require(not rel.is_absolute(), "R1E_AUTHORITY_PATH_ABSOLUTE")
    require(".." not in rel.parts, "R1E_AUTHORITY_PATH_TRAVERSAL")
    normalized = rel.as_posix()
    require(normalized.startswith("reports/"), "R1E_AUTHORITY_NOT_REPORT")

    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "--", normalized],
            cwd=root,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError("R1E_AUTHORITY_NOT_TRACKED") from exc

    path = root / normalized
    require(path.is_file(), "R1E_AUTHORITY_WORKTREE_MISSING")
    worktree = path.read_bytes()
    head_bytes = _git_bytes(root, f"HEAD:{normalized}")
    require(worktree == head_bytes, "R1E_AUTHORITY_WORKTREE_DRIFT")

    markers = authority_markers(worktree.decode("utf-8", "strict"))
    required = {
        "P1E_R1E_EXECUTION_AUTHORITY_FROZEN": "YES",
        "R1_SCIENTIFIC_MODEL_FORWARD_AUTHORIZED": "YES",
        "R1_SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED": "YES",
        "R1_SCIENTIFIC_ENDPOINT_COMPUTATION_AUTHORIZED": "NO",
        "R1_P1_ENDPOINT_COMPUTATION_AUTHORIZED": "NO",
        "R1_REPLACEMENT_TOLERANCE_SELECTION_AUTHORIZED": "NO",
        "R1_LOCAL_TEMPLATE_INDEX": "0",
        "R1_BRANCH_ROLE": "MATCHED_CORR",
        "LOGITS_READ_AUTHORIZED": "NO",
        "CAUSAL_INTERVENTION_AUTHORIZED": "NO",
    }
    for key, value in required.items():
        require(markers.get(key) == value, f"R1E_AUTHORITY_MARKER_MISMATCH:{key}")

    implementation_commit = markers.get("R1_IMPLEMENTATION_COMMIT", "")
    blobs = validate_r1_implementation_commit(root, implementation_commit)
    head = _git(root, "rev-parse", "HEAD")
    require(_is_ancestor(root, implementation_commit, head), "R1_IMPLEMENTATION_NOT_ANCESTOR")

    runner_sha = file_sha256(root / RUNNER_REL)
    test_sha = file_sha256(root / TEST_REL)
    require(markers.get("R1_RUNNER_SHA256") == runner_sha, "R1E_RUNNER_SHA_MISMATCH")
    require(markers.get("R1_TEST_SHA256") == test_sha, "R1E_TEST_SHA_MISMATCH")
    require(markers.get("R1_PROTOCOL_SHA256") == R1_PROTOCOL_SHA256, "R1E_PROTOCOL_SHA_MISMATCH")
    require(markers.get("OBSERVER_SHA256") == OBSERVER_SHA256, "R1E_OBSERVER_SHA_MISMATCH")
    for filename, wanted in P0_ARTIFACT_SHA256.items():
        marker = {
            "candidate_pool.jsonl": "P0_CANDIDATE_POOL_SHA256",
            "generated_source.jsonl": "P0_GENERATED_SOURCE_SHA256",
            "phase_pair_mapping.json": "P0_PHASE_PAIR_MAPPING_SHA256",
            "token_contracts.jsonl": "P0_TOKEN_CONTRACTS_SHA256",
            "provisioning_manifest.json": "P0_PROVISIONING_MANIFEST_SHA256",
            "validation_report_candidate.md": "P0_VALIDATION_REPORT_SHA256",
        }[filename]
        require(markers.get(marker) == wanted, f"R1E_P0_HASH_MISMATCH:{filename}")

    require(
        blobs["runner_blob"] == _git(root, "rev-parse", f"HEAD:{RUNNER_REL}"),
        "R1_RUNNER_BLOB_DRIFT",
    )
    require(
        blobs["test_blob"] == _git(root, "rev-parse", f"HEAD:{TEST_REL}"),
        "R1_TEST_BLOB_DRIFT",
    )

    authority_blob = _git(root, "rev-parse", f"HEAD:{normalized}")
    authority_commit = _git(root, "log", "-1", "--format=%H", "--", normalized)
    require(_is_ancestor(root, implementation_commit, authority_commit), "R1_IMPLEMENTATION_NOT_BEFORE_R1E")
    require(_is_ancestor(root, authority_commit, head), "R1E_AUTHORITY_NOT_ANCESTOR")

    return R1ExecutionAuthority(
        path=normalized,
        commit=authority_commit,
        git_blob=authority_blob,
        implementation_commit=implementation_commit,
    )


def build_diagnostic_output(
    *,
    root: Path,
    authority: R1ExecutionAuthority,
    p0: P0Static,
    fixed: FixedBranch,
    rows: Sequence[Mapping[str, Any]],
    handoff: Mapping[str, Any],
    encoder: Mapping[str, Any],
) -> dict[str, Any]:
    classification, profile, support = classify_coordinate_rows(rows)
    return {
        "schema_version": "k0-rvg-p1e-r1-numerical-support-diagnostic-v1",
        "runtime_git_head": _git(root, "rev-parse", "HEAD"),
        "r1_execution_authority": {
            "path": authority.path,
            "commit": authority.commit,
            "git_blob": authority.git_blob,
        },
        "r1_protocol": {
            "commit": R1_PROTOCOL_COMMIT,
            "sha256": R1_PROTOCOL_SHA256,
        },
        "r1_implementation_commit": authority.implementation_commit,
        "r1_runner_sha256": file_sha256(root / RUNNER_REL),
        "r1_test_sha256": file_sha256(root / TEST_REL),
        "p0_artifact_sha256": p0.hashes,
        "observer": {
            "commit": OBSERVER_COMMIT,
            "sha256": OBSERVER_SHA256,
            "git_blob": OBSERVER_GIT_BLOB,
        },
        "a0": {
            "commit": A0_COMMIT,
            "model_blob_sha": A0_MODEL_BLOB_SHA,
            "heads_tree_sha": A0_HEADS_TREE_SHA,
        },
        "handoff": {
            "zip_sha256": handoff["zip_sha256"],
            "checkpoint_sha256": handoff["checkpoint_sha256"],
            "encoder_canonical_sha256": encoder["canonical_digest"],
            "encoder_raw_concat_sha256": encoder["raw_concat_digest"],
        },
        "local_item": {
            "local_template_index": R1_LOCAL_TEMPLATE_INDEX,
            "pair_id": p0.candidate["pair_id"],
            "stable_item_id": p0.candidate["stable_item_id"],
            "branch_role": R1_BRANCH_ROLE,
            "matched_divergence_anchor": fixed.t_e,
            "target_token_indices": list(target_indices(fixed.t_e)),
        },
        "coordinates": list(rows),
        "classification": classification,
        "local_support_profile": profile,
        "support": support,
        "flags": {
            "p1_endpoints_computed": False,
            "logits_read": False,
            "causal_intervention": False,
            "tolerance_sweep": False,
            "replacement_tolerance_selected": False,
            "scientific_result_interpretation": False,
        },
    }


def write_diagnostic_atomic(output_dir: Path, result: Mapping[str, Any]) -> str:
    require(not output_dir.exists(), "R1_OUTPUT_DIR_ALREADY_EXISTS")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))
    try:
        raw = canonical_json_bytes(dict(result), final_lf=True)
        path = tmp / DIAGNOSTIC_FILENAME
        path.write_bytes(raw)
        require(
            {p.name for p in tmp.iterdir() if p.is_file()} == {DIAGNOSTIC_FILENAME},
            "R1_OUTPUT_ARTIFACT_SET_MISMATCH",
        )
        tmp.rename(output_dir)
        return sha256_bytes(raw)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def execute_diagnostic(
    root: Path,
    handoff_path: Path,
    authority_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    # Critical gate ordering: no model construction before R1E authentication.
    authenticate_static_dependencies(root)
    p0 = authenticate_p0_static(root)
    authority = authenticate_r1_execution_authority(root, authority_path)

    observer, k2s, model, tokenizer, binding, layer_map, handoff, encoder, hf = (
        validate_runtime_and_build_model(root, handoff_path)
    )
    fixed = static_item0_fixed_branch(tokenizer, p0)
    rows = capture_fixed_branch(
        observer,
        k2s,
        model,
        layer_map,
        binding,
        tokenizer,
        fixed.branch_text,
        fixed.t_e,
    )
    result = build_diagnostic_output(
        root=root,
        authority=authority,
        p0=p0,
        fixed=fixed,
        rows=rows,
        handoff=handoff,
        encoder=encoder,
    )
    artifact_sha = write_diagnostic_atomic(output_dir, result)
    return {
        "status": "COMPLETE_R1_FIXED_SINGLE_BRANCH_NUMERICAL_SUPPORT_DIAGNOSTIC",
        "output_dir": str(output_dir),
        "artifact": DIAGNOSTIC_FILENAME,
        "artifact_sha256": artifact_sha,
        "classification": result["classification"],
        "local_support_profile": result["local_support_profile"],
        "scientific_model_forward_count": 1,
        "p1_endpoint_computed": False,
        "replacement_tolerance_selected": False,
        "scientific_result_interpreted": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K0-RVG-P1E-R1 numerical-support diagnostic.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-preflight", action="store_true")
    mode.add_argument("--execute-diagnostic", action="store_true")
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
        require(args.execution_authority is None, "R1_SYNTHETIC_AUTHORITY_FORBIDDEN")
        require(args.output_dir is None, "R1_SYNTHETIC_OUTPUT_DIR_FORBIDDEN")
        result = run_synthetic_preflight(root, args.seed180_handoff.resolve())
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0

    require(args.execute_diagnostic, "R1_UNKNOWN_MODE")
    if args.execution_authority is None:
        parser.error("--execution-authority is required for --execute-diagnostic")
    if args.output_dir is None:
        parser.error("--output-dir is required for --execute-diagnostic")
    result = execute_diagnostic(
        root,
        args.seed180_handoff.resolve(),
        args.execution_authority,
        args.output_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
