#!/usr/bin/env python3
"""E2: read real local source bytes, not self-asserted runtime provenance.

CPU only. No model/tokenizer load, no encoding, no CUDA probe, no artifact
publication and no execution entrypoint. Source files and CPU bundles passed to
functions are *not* evidence that those bytes produced live model tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE = "9b6fcc8150c45e717ff4c0d8f1d7f39731566f9a"
BRANCH = "gen5-causal-role-state-ownership"
SCHEMA = "GEN5_V10_E2_LOCAL_SOURCE_BYTE_VERIFICATION_CANDIDATE_V1"
MODEL_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
CHECKPOINT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
DEV_ORDER_SHA256 = "b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25"
DEV_ENCODING_SHA256 = "e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51"
SNAPSHOT_FILES = {
    "config.json": (895, "784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a"),
    "tokenizer.json": (None, "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf"),
    "tokenizer_config.json": (None, "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb"),
    "special_tokens_map.json": (None, "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8"),
}
FROZEN_BLOBS = {
    "scripts/train_reason_router_gen5_phase3a_contention.py": "45e333128c4fa31f4f502c5fdcf324243c1084fd",
    "scripts/train_reason_router_gen5_phase2_state_update_ownership.py": "8f711776c6b8cab90fbdafcda166780f3295e4ae",
    "scripts/audit_native_downstream_functional_reconvergence_raw_write_transport.py": "984319fecb5ed2c103b92da0ac868ac7746770d8",
    "scripts/audit_native_recurrent_generator_anchor_pair_localization.py": "f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635",
    "scripts/gen5_v10_stage_e1_bounded_cpu_dispatch_candidate.py": "5cd4e67b94dcb5c4612613f8fdb9eda600be03fd",
}
INPUT_KEYS = frozenset(("input_ids", "attention_mask", "claim_mask", "evidence_mask", "final_labels"))
ENCODING_KEYS = frozenset(("model_inputs", "stressor_active", "target_indices", "row_ids", "pair_ids", "contrast_cell_ids"))
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
MAX_CHUNK_BYTES = 8 * 1024 * 1024
VALID_TOKEN_COUNT = 60094


class E2Blocked(ValueError):
    pass


def require(ok: bool, why: str) -> None:
    if not ok:
        raise E2Blocked("GEN5_V10_E2_SOURCE_BLOCKED:" + why)


def canonical_bytes(obj: Any) -> bytes:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise E2Blocked("NONCANONICAL_JSON") from exc


def _actual_file_sha256(path: Path, *, no_symlink: bool = False) -> tuple[str, int]:
    """Read a real local file with a bounded host buffer and mutation checks.

    A snapshot file may be a normal HuggingFace symlink into the blobs cache;
    its *resolved bytes* are checked against an independently pinned hash.
    Reject a checkpoint symlink to keep its input path strictly explicit.
    """
    require(isinstance(path, Path) and path.is_file(), "SOURCE_NOT_REGULAR_FILE")
    if no_symlink:
        require(not path.is_symlink(), "SOURCE_SYMLINK_FORBIDDEN")
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            before = os.fstat(f.fileno())
            require(os.path.isfile(path) and before.st_size > 0, "SOURCE_EMPTY_OR_UNSTABLE")
            size = 0
            while True:
                block = f.read(MAX_CHUNK_BYTES)
                if not block:
                    break
                size += len(block)
                h.update(block)
            after = os.fstat(f.fileno())
            require(size == before.st_size == after.st_size and
                    before.st_mtime_ns == after.st_mtime_ns and
                    before.st_ino == after.st_ino, "SOURCE_MUTATED_DURING_READ")
            require(path.stat().st_size == size, "SOURCE_PATH_MUTATED")
    except (OSError, OverflowError) as exc:
        raise E2Blocked("GEN5_V10_E2_SOURCE_BLOCKED:SOURCE_READ_ERROR") from exc
    return h.hexdigest(), size


def verify_checkpoint_file(path: Path) -> dict[str, Any]:
    sha, count = _actual_file_sha256(path, no_symlink=True)
    require(sha == CHECKPOINT_SHA256, "CHECKPOINT_REAL_BYTE_DIGEST_MISMATCH")
    return {"sha256": sha, "bytes_read": count, "observed": "LOCAL_FILE_BYTES_ONLY"}


def verify_snapshot_bytes(snapshot: Path) -> dict[str, Any]:
    require(isinstance(snapshot, Path) and snapshot.is_dir() and
            not snapshot.is_symlink() and snapshot.name == MODEL_REVISION,
            "MODEL_SNAPSHOT_REVISION_DIRECTORY")
    output = {}
    for name, (byte_count, pinned_hash) in SNAPSHOT_FILES.items():
        sha, actual_count = _actual_file_sha256(snapshot / name)
        require(sha == pinned_hash, "SNAPSHOT_REAL_BYTE_DIGEST:" + name)
        if byte_count is not None:
            require(actual_count == byte_count, "SNAPSHOT_SIZE:" + name)
        output[name] = {"sha256": sha, "bytes_read": actual_count}
    return {"snapshot_revision_dir": MODEL_REVISION, "verified_files": output,
            "model_constructed": False, "tokenizer_loaded": False}


def dev_order_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Byte-for-byte equivalent to frozen P3a.row_order_sha256, no imports."""
    require(isinstance(rows, (list, tuple)), "DEV_ROWS_SEQUENCE")
    value = []
    try:
        for row in rows:
            require(isinstance(row, Mapping), "DEV_ROW_MAPPING")
            value.append({
                "id": str(row["id"]), "pair_id": str(row["pair_id"]),
                "contrast_cell_id": str(row["contrast_cell_id"]),
                "final_label": str(row["final_label"]),
                "final_label_id": int(row["final_label_id"]),
                "stressor_domain": bool(row["stressor_domain"]),
            })
    except (KeyError, ValueError, TypeError) as exc:
        raise E2Blocked("GEN5_V10_E2_SOURCE_BLOCKED:DEV_ROW_FIELDS") from exc
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def verify_dev_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(isinstance(rows, (list, tuple)) and len(rows) == 840, "DEV_840_ROWS")
    require(all(isinstance(row, Mapping) and type(row.get("row_id")) is str
                and row["row_id"] for row in rows), "ROW_IDS_REQUIRED")
    ids = [row["row_id"] for row in rows]
    require(len(set(ids)) == 840, "DUPLICATE_DEV_ROW_IDS")
    digest = dev_order_digest(rows)
    require(digest == DEV_ORDER_SHA256, "DEV_ORDER_CANONICAL_DIGEST_MISMATCH")
    return {"row_count": 840, "canonical_order_sha256": digest,
            "bounded_row_ids": ids[800:832]}


def _cpu_tensor(value: Any, *, name: str, dtype: Any, shape: tuple[int, ...]) -> Any:
    import torch
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
            value.dtype == dtype and tuple(value.shape) == shape,
            "CPU_ENCODED_TENSOR:" + name)
    return value


def validate_encoded_cpu_bundle(bundle: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """Check actual supplied CPU tensors; never invoke any tokenizer encoder."""
    import torch
    require(type(bundle) is dict and set(bundle) == ENCODING_KEYS,
            "ENCODED_BUNDLE_EXACT_KEYS")
    require(isinstance(rows, (list, tuple)) and len(rows) == 840,
            "ENCODED_ROWS_REQUIRED")
    mi = bundle["model_inputs"]
    require(type(mi) is dict and set(mi) == INPUT_KEYS, "ENCODED_MODEL_INPUT_KEYS")
    _cpu_tensor(mi["input_ids"], name="input_ids", dtype=torch.int64, shape=(840,128))
    for name in ("attention_mask", "claim_mask", "evidence_mask"):
        _cpu_tensor(mi[name], name=name, dtype=torch.bool, shape=(840,128))
    _cpu_tensor(mi["final_labels"], name="final_labels", dtype=torch.int64, shape=(840,))
    act = _cpu_tensor(bundle["stressor_active"], name="stressor_active",
                      dtype=torch.bool, shape=(840,))
    target = _cpu_tensor(bundle["target_indices"], name="target_indices",
                         dtype=torch.int64, shape=(840,))
    for key in ("row_ids", "pair_ids", "contrast_cell_ids"):
        require(type(bundle[key]) is list and len(bundle[key]) == 840 and
                all(type(v) is str for v in bundle[key]), "ENCODED_ROW_KEY:" + key)
    require(len(set(bundle["row_ids"])) == 840, "ENCODED_DUPLICATE_IDS")
    require(all(bundle["row_ids"][i] == str(row["row_id"]) and
                bundle["pair_ids"][i] == str(row["source_pair_id"]) and
                bundle["contrast_cell_ids"][i] == str(row["contrast_cell_id"])
                for i,row in enumerate(rows)), "ENCODED_ROW_LINEAGE_INCONSISTENT")
    labels = mi["final_labels"].tolist()
    require(all(labels[i] == int(row["final_label_id"]) and
                bool(act[i]) == bool(row["stressor_domain"])
                for i,row in enumerate(rows)), "ENCODED_LABEL_OR_ACTIVE_DRIFT")
    att, claim, evidence = (mi[k] for k in ("attention_mask", "claim_mask", "evidence_mask"))
    require(bool(torch.all((claim & evidence) == False).item()) and
            bool(torch.all((claim | evidence) <= att).item()), "ENCODED_MASK_OVERLAP")
    eos = att & ~(claim | evidence)
    require(bool(torch.all(eos.sum(1) == 1).item()) and
            bool(torch.all(claim.sum(1) >= 1).item()) and
            bool(torch.all(evidence.sum(1) >= 1).item()), "ENCODED_EOS_CLAIM_EVIDENCE")
    # Frozen encoding is claim + EOS + evidence, then padded zeros.
    require(bool(torch.all(att[:,0]).item()) and
            bool(torch.all(att.to(torch.int64).diff(dim=1) <= 0).item()) and
            bool(torch.all(mi["input_ids"][eos] == 0).item()) and
            bool(torch.all(mi["input_ids"][~att] == 0).item()), "ENCODED_SERIALIZATION_LAYOUT")
    require(bool(torch.all((target[act] >= 0) & (target[act] < att.sum(1)[act])).item()) and
            bool(torch.all(target[~act] == -1).item()), "ENCODED_TARGET_INDEX")
    require(int(att.sum().item()) == VALID_TOKEN_COUNT, "ENCODED_VALID_TOKEN_COUNT")


def encoded_bundle_digest(bundle: Mapping[str, Any]) -> str:
    """Exact frozen P3a.encoded_bundle_sha256 format from CPU values."""
    h = hashlib.sha256()
    for key in ("row_ids", "pair_ids", "contrast_cell_ids"):
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(canonical_bytes(list(bundle[key])))
        h.update(b"\n")
    for key in ("stressor_active", "target_indices"):
        tensor = bundle[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(tensor.numpy().tobytes())
        h.update(b"\n")
    for key in sorted(bundle["model_inputs"]):
        tensor = bundle["model_inputs"][key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(tensor.numpy().tobytes())
        h.update(b"\n")
    return h.hexdigest()


def verify_encoded_cpu_bundle(bundle: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_encoded_cpu_bundle(bundle, rows)
    sha = encoded_bundle_digest(bundle)
    require(sha == DEV_ENCODING_SHA256, "ENCODED_CANONICAL_DIGEST_MISMATCH")
    return {"dev_encoding_sha256": sha, "bounded_row_ids": bundle["row_ids"][800:832],
            "checked_cpu_tensors": 7, "tokenizer_invocations": 0}


def inspect_pinned_git(repo: Path) -> dict[str, Any]:
    require(isinstance(repo, Path) and repo.is_dir() and not repo.is_symlink(), "GIT_REPO_DIR")
    def git(*parts: str) -> str:
        try:
            r = subprocess.run(["git", "-C", str(repo), *parts],
                               capture_output=True, text=True, check=False, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise E2Blocked("GEN5_V10_E2_SOURCE_BLOCKED:GIT_READ_ERROR") from exc
        require(r.returncode == 0, "GIT_QUERY:" + parts[0])
        return r.stdout.strip()
    require(git("rev-parse", "HEAD") == BASE, "GIT_HEAD")
    require(git("branch", "--show-current") == BRANCH, "GIT_BRANCH")
    require(git("diff", "--name-only") == "" and
            git("diff", "--cached", "--name-only") == "", "GIT_TRACKED_DRIFT")
    for path, sha in FROZEN_BLOBS.items():
        require(git("rev-parse", "HEAD:" + path) == sha,
                "GIT_BLOB_DRIFT:" + path)
    return {"head": BASE, "pinned_blobs": len(FROZEN_BLOBS),
            "loaded_runtime_modules_authenticated": False}


def audit_local_source_bytes(*, checkpoint: Path, snapshot: Path,
                             dev_rows: Sequence[Mapping[str, Any]],
                             dev_bundle: Any) -> dict[str, Any]:
    """Byte-hash real inputs; does not authenticate their later use in CUDA."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
            "GPU_MUST_BE_HIDDEN")
    cp = verify_checkpoint_file(checkpoint)
    snap = verify_snapshot_bytes(snapshot)
    dev = verify_dev_rows(dev_rows)
    enc = verify_encoded_cpu_bundle(dev_bundle, dev_rows)
    require(dev["bounded_row_ids"] == enc["bounded_row_ids"], "BOUNDED_ROW_ORDER_BINDING")
    return {"schema_version": SCHEMA, "frozen_base": BASE,
            "checkpoint_file": cp, "model_snapshot": snap,
            "dev_order_sha256": dev["canonical_order_sha256"],
            "dev_encoding_sha256": enc["dev_encoding_sha256"],
            "bounded_row_ids": dev["bounded_row_ids"],
            "local_source_bytes_match_pins": True,
            "torch_model_loaded": False, "tokenizer_loaded": False,
            "active_gpu_tensor_lineage_authenticated": False,
            "checkpoint_bytes_loaded_into_model_authenticated": False,
            "runtime_module_identity_authenticated": False,
            "physical_gpu_memory_measured": False,
            "GPU_EXECUTION": "NOT_AUTHORIZED", "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
            "E0": "NOT_EVALUATED", "E1": "NOT_EVALUATED",
            "E2": "NOT_EVALUATED", "E3": "NOT_EVALUATED",
            "PRODUCTION": "BLOCKED"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    flags = p.add_mutually_exclusive_group(required=True)
    flags.add_argument("--static-contract-check", action="store_true")
    flags.add_argument("--inspect-local-git", action="store_true")
    args = p.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
            "GPU_MUST_BE_HIDDEN")
    if args.inspect_local_git:
        r = inspect_pinned_git(Path(__file__).resolve().parents[1])
        print("GEN5_V10_STAGE_E2_PINNED_GIT_SOURCE_PASS")
        print("HEAD=" + r["head"] + ";BLOBS=" + str(r["pinned_blobs"]))
    else:
        require(CHECKPOINT_SHA256 and DEV_ORDER_SHA256 and DEV_ENCODING_SHA256,
                "MISSING_SOURCE_PINS")
        print("GEN5_V10_STAGE_E2_CPU_SOURCE_BYTE_STATIC_PASS")
        print("SCHEMA=" + SCHEMA + ";BASE=" + BASE)
    print("GPU=OFF;MODEL_LOAD=NO;TOKENIZER_INVOKED=NO;E0_E3=NOT_EVALUATED")
    print("READINESS=BLOCKED_PENDING_ACTUAL_RUNTIME_TENSOR_LINEAGE_AND_MEMORY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
