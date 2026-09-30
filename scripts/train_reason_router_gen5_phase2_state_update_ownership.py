"""Gen5 Phase 2 state-update ownership training runner.

Authority:
    5ab3174cc24731f867e512c9381b9abcc3263915

This runner has three explicit modes:

1. ``--static-preflight-only``:
   historical-tokenizer/data/split identity only; no model/checkpoint/CUDA.
2. ``--cuda-preflight-only``:
   qualified CUDA runtime + exact full-train forward graph feasibility only;
   no backward and no optimizer step.
3. ``--run-matrix``:
   the frozen 3-arm x 3-seed, 20-step training matrix.

The fresh XG1 ownership-assay population is outside this runner's authority.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from contramamba.gen5_phase2_state_update_ownership import (  # noqa: E402
    ARMS,
    EXPECTED_TRAINABLE_NUMEL,
    StateWriteCorrection,
    correction_optimizer_parameters,
    correction_parameter_audit,
    install_phase2_layer22_wrapper,
    load_frozen_owner_bases,
    parent_parameter_fingerprint,
    phase2_active_mask,
    phase2_final_three_way_ce,
)
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

TRAINING_EXECUTION_AUTHORITY_COMMIT = (
    "5ab3174cc24731f867e512c9381b9abcc3263915"
)
ACCELERATED_IMPLEMENTATION_COMMIT = (
    "a69a0dd47853e3ec68f7c5aa21bb8d692c89a86a"
)
IMPLEMENTATION_AUTHORITY_COMMIT = (
    "e2f8975d8271c0e95c92b9389dfc4717a221f7df"
)

AUTHORITY_PATH = (
    "reports/"
    "reason_router_gen5_phase2_training_execution_authority_spec_candidate.md"
)
AUTHORITY_SHA256 = (
    "b6996a97ca9a8a21f8c5d54935a37bcc7369cee12963224fd245062b63ce5446"
)

# Exact Git blob identities at the training-execution authority commit.
FROZEN_BLOBS = {
    AUTHORITY_PATH:
        "1ff3d83cf1123f75965d3a138a3feabe49650168",
    "src/contramamba/gen5_phase2_state_update_ownership.py":
        "a5c18f5b5d597d9830c3c44a9299af8233677f37",
    "scripts/verify_reason_router_gen5_phase2_state_update_ownership.py":
        "e2276324cad9c09c4f452db8d03aed93cc8d8283",
    "src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py":
        "e29d361d99579b0a86a626ebcdebd59ba465881c",
    "scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py":
        "b3b52c4116ca142fc1e31c4bdda57828c610d6e7",
    "scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py":
        "4a8326c442b883510ba452f049c88f3163677fc5",
    "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py":
        "00a81ce6ec4ada4d5c0bf36418347b222543d174",
    "scripts/reason_router_gen4_six_cell_tier2_scientific_inference.py":
        "1d0ea63b46a6147bb3ab45cc1043244c42f562d3",
    "scripts/build_controlled_v5.py":
        "baee23a9f71333125f4a8735c2c92d20cab7eb4f",
}

RUNNER_REL = "scripts/train_reason_router_gen5_phase2_state_update_ownership.py"
TEST_REL = "tests/test_reason_router_gen5_phase2_state_update_ownership.py"
AUTHORIZED_OPENING_PATHS = frozenset({RUNNER_REL, TEST_REL})

MODEL_NAME = "state-spaces/mamba-130m-hf"
MODEL_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
MODEL_CONFIG_SHA256 = (
    "784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a"
)
MODEL_CONFIG_BYTES = 895

TOKENIZER_FILE_SHA256 = {
    "tokenizer.json":
        "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
    "tokenizer_config.json":
        "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
    "special_tokens_map.json":
        "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
}

PARENT_ARM = "G3-GROUP-D-HALF"
PARENT_SEED = 180
PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)
NATIVE_BACKBONE_SIGNATURE_SHA256 = (
    "81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415"
)

DATASET_REL = (
    "reports/"
    "reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458/"
    "controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
)
DATASET_GIT_LF_SHA256 = (
    "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
)
DATASET_SEMANTIC_SHA256 = (
    "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
)

SIDECAR_REL = (
    "reports/"
    "reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_"
    "ff181f565cefa0a28280c084246862286daf1f2d_"
    "149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/"
    "p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl"
)
SIDECAR_GIT_LF_SHA256 = (
    "9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d"
)
SIDECAR_SEMANTIC_SHA256 = (
    "2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9"
)

SPLIT_SEED = 8192
DEV_RATIO = 0.2
TOTAL_ROWS = 3600
TRAIN_ROWS = 2880
DEV_ROWS = 720
TRAIN_PAIRS = 240
DEV_PAIRS = 60

# Historical static-preparation references. The original static-preparation
# serializer was not promoted into a reusable repository API, so execution
# authenticates the exact dataset bytes + semantic hash + deterministic split
# function + frozen dev-pair set, and additionally pins the runner's own ordered
# identity digests across local historical-tokenizer and Kaggle execution.
HISTORICAL_ORDERED_TRAIN_ROW_SHA256_REFERENCE = (
    "478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8"
)
HISTORICAL_ORDERED_DEV_ROW_SHA256_REFERENCE = (
    "7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4"
)

EXPECTED_DEV_PAIR_IDS = frozenset({
    "clinic_expansion",
    "forest_mapping",
    "garden_award",
    "generated_fact_034",
    "generated_fact_042",
    "generated_fact_045",
    "generated_fact_048",
    "generated_fact_051",
    "generated_fact_056",
    "generated_fact_062",
    "generated_fact_073",
    "generated_fact_076",
    "generated_fact_078",
    "generated_fact_085",
    "generated_fact_087",
    "generated_fact_089",
    "generated_fact_090",
    "generated_fact_091",
    "generated_fact_096",
    "generated_fact_102",
    "generated_fact_108",
    "generated_fact_118",
    "generated_fact_133",
    "generated_fact_136",
    "generated_fact_138",
    "generated_fact_139",
    "generated_fact_144",
    "generated_fact_152",
    "generated_fact_157",
    "generated_fact_165",
    "generated_fact_166",
    "generated_fact_167",
    "generated_fact_174",
    "generated_fact_179",
    "generated_fact_181",
    "generated_fact_192",
    "generated_fact_193",
    "generated_fact_195",
    "generated_fact_205",
    "generated_fact_225",
    "generated_fact_227",
    "generated_fact_240",
    "generated_fact_241",
    "generated_fact_242",
    "generated_fact_243",
    "generated_fact_248",
    "generated_fact_249",
    "generated_fact_257",
    "generated_fact_258",
    "generated_fact_259",
    "generated_fact_261",
    "generated_fact_272",
    "generated_fact_278",
    "generated_fact_282",
    "generated_fact_285",
    "generated_fact_286",
    "jazz_archive",
    "museum_purchase",
    "railway_restoration",
    "satellite_launch",
})

TRAINING_SEEDS = (5201, 5202, 5203)
TRAINING_ARMS = ("G5-C0", "G5-C1", "G5-M1")
TRAINING_MATRIX = tuple(
    (seed, arm)
    for seed in TRAINING_SEEDS
    for arm in TRAINING_ARMS
)

EPOCHS = 20
TOTAL_OPTIMIZER_STEPS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP_NORM = 5.0
MAX_LENGTH = 128

R22_SHA256 = (
    "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
)
C22_SHA256 = (
    "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"
)

HISTORICAL_TOKENIZER_RUNTIME = {
    "transformers": "4.45.0",
    "tokenizers": "0.20.3",
}
CUDA_RUNTIME_EXPECTED = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "torch": "2.10.0+cu128",
    "transformers": "5.0.0",
    "cuda": "12.8",
    "device_name": "Tesla T4",
    "capability": (7, 5),
    "default_dtype": "torch.float32",
    "autocast": False,
}

FROZEN_TRAINING_CONTRACT = {
    "train_rows": TRAIN_ROWS,
    "dev_rows": DEV_ROWS,
    "split_seed": SPLIT_SEED,
    "optimizer": "torch.optim.AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "scheduler": None,
    "epochs": EPOCHS,
    "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
    "gradient_clip_norm": GRADIENT_CLIP_NORM,
    "training_seeds": list(TRAINING_SEEDS),
    "arms": list(TRAINING_ARMS),
    "checkpoint_selection": "FINAL_FIXED_STEP_ONLY",
    "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
    "correction_backend": "CHECKPOINTED_STREAMING_REFERENCE_EQUIVALENT",
    "logical_batch": "EXACT_FULL_2880_ROW_TRAIN_SPLIT",
    "microbatching": False,
    "gradient_accumulation": False,
}

FORBIDDEN_FRESH_ASSAY_FRAGMENTS = (
    "reason_router_gen5_phase2_xg1_ownership_assay_v1",
    "xg1_fact_8701",
    "xg1_fact_9000",
)


class Phase2TrainingError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Phase2TrainingError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase2TrainingError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_lf_sha256(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    return sha256_bytes(normalized)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_ROW_NOT_OBJECT:{line_number}")
            rows.append(value)
    return rows


def semantic_jsonl_sha256(path: str | Path) -> str:
    return sha256_bytes(canonical_json_bytes(read_jsonl(path)))


def semantic_sidecar_sha256(path: str | Path) -> str:
    rows = read_jsonl(path)
    canonical = [
        {key: row[key] for key in sorted(row) if key != "created_at"}
        for row in rows
    ]
    return sha256_bytes(canonical_json_bytes(canonical))


def ordered_row_identity_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "id": str(row["id"]),
            "pair_id": str(row["pair_id"]),
            "intervention_type": str(row["intervention_type"]),
            "final_label": str(row["final_label"]),
        }
        for row in rows
    ]
    return sha256_bytes(canonical_json_bytes(identity))


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(tensor.numpy().tobytes())


def encoded_bundle_sha256(bundle: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for list_key in ("pair_ids", "intervention_types"):
        values = bundle[list_key]
        h.update(list_key.encode("utf-8"))
        h.update(b"\0")
        h.update(canonical_json_bytes(list(values)))
        h.update(b"\n")

    model_inputs = bundle["model_inputs"]
    for key in sorted(model_inputs):
        value = model_inputs[key]
        require(torch.is_tensor(value), f"ENCODED_NOT_TENSOR:{key}")
        tensor = value.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(tensor.numpy().tobytes())
        h.update(b"\n")
    return h.hexdigest()


def _status_paths() -> set[str]:
    # Do not route porcelain output through ``git()``: that helper applies
    # ``.strip()`` to the whole command output, which removes the leading
    # status-space from an unstaged first line (``" M path"``) and shifts
    # the fixed-width porcelain path column by one character.
    try:
        raw = subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase2TrainingError("GIT_FAILURE:status --porcelain=v1") from exc

    result: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        require(len(line) >= 4, f"MALFORMED_GIT_STATUS_LINE:{line!r}")
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        result.add(path)
    return result


def authenticate_repo(
    expected_head: str,
    *,
    allow_opening_worktree: bool = False,
) -> None:
    branch = git("branch", "--show-current")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            TRAINING_EXECUTION_AUTHORITY_COMMIT,
            expected_head,
        ) == 0,
        "TRAINING_AUTHORITY_NOT_ANCESTOR",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            ACCELERATED_IMPLEMENTATION_COMMIT,
            expected_head,
        ) == 0,
        "ACCELERATED_IMPLEMENTATION_NOT_ANCESTOR",
    )

    status = _status_paths()
    if allow_opening_worktree:
        require(
            status <= AUTHORIZED_OPENING_PATHS,
            f"UNAUTHORIZED_WORKTREE_PATHS:{sorted(status)}",
        )
    else:
        require(not status, f"WORKTREE_NOT_CLEAN:{sorted(status)}")

    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(
            observed == expected_blob,
            f"FROZEN_BLOB_DRIFT:{path}:{observed}",
        )

    # Once runner-opening is committed, the only authority-descendant tracked
    # delta is the exact two-file opening scope.
    if expected_head != TRAINING_EXECUTION_AUTHORITY_COMMIT:
        changed = {
            line.strip().replace("\\", "/")
            for line in git(
                "diff",
                "--name-only",
                f"{TRAINING_EXECUTION_AUTHORITY_COMMIT}..{expected_head}",
            ).splitlines()
            if line.strip()
        }
        require(
            changed <= AUTHORIZED_OPENING_PATHS,
            f"POST_AUTHORITY_SCOPE_DRIFT:{sorted(changed)}",
        )


def validate_authority_file() -> None:
    path = ROOT / AUTHORITY_PATH
    require(path.is_file(), "AUTHORITY_FILE_MISSING")
    require(sha256_file(path) == AUTHORITY_SHA256, "AUTHORITY_FILE_SHA256")


def validate_no_fresh_assay_path(value: str | Path | None) -> None:
    if value is None:
        return
    lowered = str(value).replace("\\", "/").lower()
    for fragment in FORBIDDEN_FRESH_ASSAY_FRAGMENTS:
        require(
            fragment.lower() not in lowered,
            f"FRESH_ASSAY_PATH_FORBIDDEN:{fragment}",
        )


def resolve_exact_snapshot(path: str | Path | None) -> Path:
    if path is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise Phase2TrainingError("HUGGINGFACE_HUB_REQUIRED") from exc
        resolved = Path(
            snapshot_download(
                repo_id=MODEL_NAME,
                revision=MODEL_REVISION,
                allow_patterns=[
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
                local_files_only=False,
                token=False,
            )
        )
    else:
        resolved = Path(path)

    require(resolved.is_dir(), f"MODEL_SNAPSHOT_MISSING:{resolved}")
    require(
        resolved.name == MODEL_REVISION,
        f"MODEL_SNAPSHOT_REVISION:{resolved.name}",
    )

    config = resolved / "config.json"
    require(config.is_file(), "MODEL_CONFIG_MISSING")
    require(config.stat().st_size == MODEL_CONFIG_BYTES, "MODEL_CONFIG_BYTES")
    require(sha256_file(config) == MODEL_CONFIG_SHA256, "MODEL_CONFIG_SHA256")

    for filename, expected in TOKENIZER_FILE_SHA256.items():
        candidate = resolved / filename
        require(candidate.is_file(), f"TOKENIZER_FILE_MISSING:{filename}")
        require(
            sha256_file(candidate) == expected,
            f"TOKENIZER_FILE_SHA256:{filename}",
        )
    return resolved


def load_training_tokenizer(snapshot: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        require(tokenizer.eos_token_id is not None, "TOKENIZER_EOS_MISSING")
        tokenizer.pad_token = tokenizer.eos_token
    require(tokenizer.eos_token_id == 0, f"TOKENIZER_EOS_ID:{tokenizer.eos_token_id}")
    require(tokenizer.pad_token_id == 0, f"TOKENIZER_PAD_ID:{tokenizer.pad_token_id}")
    return tokenizer


def authenticate_static_inputs() -> dict[str, Any]:
    from scripts import build_controlled_v5

    dataset_path = ROOT / DATASET_REL
    sidecar_path = ROOT / SIDECAR_REL

    require(dataset_path.is_file(), "DATASET_MISSING")
    require(sidecar_path.is_file(), "SIDECAR_MISSING")

    dataset_git_lf = git_lf_sha256(dataset_path)
    dataset_semantic = semantic_jsonl_sha256(dataset_path)
    sidecar_git_lf = git_lf_sha256(sidecar_path)
    sidecar_semantic = semantic_sidecar_sha256(sidecar_path)

    require(dataset_git_lf == DATASET_GIT_LF_SHA256, "DATASET_GIT_LF_SHA256")
    require(dataset_semantic == DATASET_SEMANTIC_SHA256, "DATASET_SEMANTIC_SHA256")
    require(sidecar_git_lf == SIDECAR_GIT_LF_SHA256, "SIDECAR_GIT_LF_SHA256")
    require(sidecar_semantic == SIDECAR_SEMANTIC_SHA256, "SIDECAR_SEMANTIC_SHA256")

    records = build_controlled_v5.load_jsonl(dataset_path)
    require(len(records) == TOTAL_ROWS, f"TOTAL_ROWS:{len(records)}")

    train_records, dev_records = build_controlled_v5.split_by_pair_id(
        records,
        dev_ratio=DEV_RATIO,
        seed=SPLIT_SEED,
    )
    require(len(train_records) == TRAIN_ROWS, "TRAIN_ROW_COUNT")
    require(len(dev_records) == DEV_ROWS, "DEV_ROW_COUNT")

    train_pair_ids = {str(row["pair_id"]) for row in train_records}
    dev_pair_ids = {str(row["pair_id"]) for row in dev_records}
    require(len(train_pair_ids) == TRAIN_PAIRS, "TRAIN_PAIR_COUNT")
    require(len(dev_pair_ids) == DEV_PAIRS, "DEV_PAIR_COUNT")
    require(not (train_pair_ids & dev_pair_ids), "PAIR_SPLIT_OVERLAP")
    require(dev_pair_ids == EXPECTED_DEV_PAIR_IDS, "DEV_PAIR_IDENTITY")

    return {
        "dataset_path": dataset_path,
        "sidecar_path": sidecar_path,
        "records": records,
        "train_records": train_records,
        "dev_records": dev_records,
        "dataset_git_lf_sha256": dataset_git_lf,
        "dataset_semantic_sha256": dataset_semantic,
        "sidecar_git_lf_sha256": sidecar_git_lf,
        "sidecar_semantic_sha256": sidecar_semantic,
        "train_order_sha256_v2": ordered_row_identity_sha256(train_records),
        "dev_order_sha256_v2": ordered_row_identity_sha256(dev_records),
    }


def encode_splits(
    snapshot: Path,
    train_records: Sequence[dict[str, Any]],
    dev_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    from scripts import train_controlled_v5 as v5

    tokenizer = load_training_tokenizer(snapshot)
    train_bundle = v5.encode_mamba_records(
        list(train_records),
        tokenizer,
        max_length=MAX_LENGTH,
    )
    dev_bundle = v5.encode_mamba_records(
        list(dev_records),
        tokenizer,
        max_length=MAX_LENGTH,
    )
    return {
        "tokenizer": tokenizer,
        "train_bundle": train_bundle,
        "dev_bundle": dev_bundle,
        "train_encoding_sha256": encoded_bundle_sha256(train_bundle),
        "dev_encoding_sha256": encoded_bundle_sha256(dev_bundle),
    }


def runtime_versions() -> dict[str, Any]:
    import transformers

    try:
        tokenizers_version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError:
        tokenizers_version = None

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": (
            int(torch.cuda.device_count())
            if torch.cuda.is_available()
            else 0
        ),
        "gpu0_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "gpu0_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
        "default_dtype": str(torch.get_default_dtype()),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
    }


def validate_historical_tokenizer_runtime() -> dict[str, Any]:
    runtime = runtime_versions()
    require(
        runtime["transformers"] == HISTORICAL_TOKENIZER_RUNTIME["transformers"],
        f"HISTORICAL_TRANSFORMERS:{runtime['transformers']}",
    )
    require(
        runtime["tokenizers"] == HISTORICAL_TOKENIZER_RUNTIME["tokenizers"],
        f"HISTORICAL_TOKENIZERS:{runtime['tokenizers']}",
    )
    return runtime


def validate_cuda_runtime() -> tuple[dict[str, Any], Any, Any]:
    from scripts import (
        reason_router_gen4_generator_family_prevalence_kernel_compat
        as kernel_compat,
    )
    from scripts import (
        reason_router_gen4_k_fast_cuda_one_pair_equivalence
        as backend,
    )

    backend.runtime_gate()
    runtime = runtime_versions()
    require(runtime["python"] == CUDA_RUNTIME_EXPECTED["python"], "CUDA_PYTHON")
    require(runtime["torch"] == CUDA_RUNTIME_EXPECTED["torch"], "CUDA_TORCH")
    require(
        runtime["transformers"] == CUDA_RUNTIME_EXPECTED["transformers"],
        "CUDA_TRANSFORMERS",
    )
    require(runtime["cuda_runtime"] == CUDA_RUNTIME_EXPECTED["cuda"], "CUDA_RUNTIME")
    require(runtime["gpu0_name"] == CUDA_RUNTIME_EXPECTED["device_name"], "CUDA_DEVICE")
    require(
        tuple(runtime["gpu0_capability"]) == CUDA_RUNTIME_EXPECTED["capability"],
        "CUDA_CAPABILITY",
    )
    require(runtime["default_dtype"] == CUDA_RUNTIME_EXPECTED["default_dtype"], "CUDA_DTYPE")
    require(runtime["autocast_enabled"] is CUDA_RUNTIME_EXPECTED["autocast"], "CUDA_AUTOCAST")

    # Transformers 5.0.0 installs the module-level causal_conv1d and
    # mamba_ssm bindings during MambaMixer construction via lazy_load_kernel.
    # Their exact identity is therefore validated after backbone construction
    # in _load_parent_model(), under exact_transformers_kernel_loader().
    return runtime, kernel_compat, backend


def validate_expected_execution_identities(
    args: argparse.Namespace,
    static: Mapping[str, Any],
    encoded: Mapping[str, Any],
) -> None:
    required = {
        "expected_train_order_sha256_v2": static["train_order_sha256_v2"],
        "expected_dev_order_sha256_v2": static["dev_order_sha256_v2"],
        "expected_train_encoding_sha256": encoded["train_encoding_sha256"],
        "expected_dev_encoding_sha256": encoded["dev_encoding_sha256"],
    }
    for attr, observed in required.items():
        expected = getattr(args, attr)
        require(expected is not None, f"EXPECTED_IDENTITY_REQUIRED:{attr}")
        require(
            str(expected).lower() == str(observed).lower(),
            f"EXECUTION_IDENTITY_MISMATCH:{attr}:expected={expected}:observed={observed}",
        )


def correction_initialization_manifest(
    r22: torch.Tensor,
    c22: torch.Tensor,
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        rows: dict[str, Any] = {}
        a_hashes: set[str] = set()
        b_hashes: set[str] = set()
        for arm in TRAINING_ARMS:
            module = StateWriteCorrection(
                arm=arm,
                r22=r22,
                c22=c22,
                seed=seed,
            )
            a_sha = tensor_sha256(module.A_theta.weight)
            b_sha = tensor_sha256(module.B_theta.weight)
            require(
                int(torch.count_nonzero(module.B_theta.weight).item()) == 0,
                f"B_INIT_NONZERO:{seed}:{arm}",
            )
            rows[arm] = {
                "A_theta_sha256": a_sha,
                "B_theta_sha256": b_sha,
            }
            a_hashes.add(a_sha)
            b_hashes.add(b_sha)
        require(len(a_hashes) == 1, f"A_INIT_CROSS_ARM_MISMATCH:{seed}")
        require(len(b_hashes) == 1, f"B_INIT_CROSS_ARM_MISMATCH:{seed}")
        by_seed[str(seed)] = {
            "arms": rows,
            "cross_arm_identical": True,
        }
    return by_seed


def authenticate_checkpoint(path: str | Path) -> str:
    candidate = Path(path)
    require(candidate.is_file(), f"CHECKPOINT_MISSING:{candidate}")
    observed = sha256_file(candidate)
    require(observed == PARENT_CHECKPOINT_SHA256, f"CHECKPOINT_SHA256:{observed}")
    return observed


def _load_parent_model(
    *,
    snapshot: Path,
    checkpoint: Path,
    device: torch.device,
    kernel_compat: Any,
    kernels: Mapping[str, Any],
) -> tuple[torch.nn.Module, dict[str, int]]:
    from scripts import (
        reason_router_gen4_six_cell_tier2_inference_adapter
        as adapter,
    )
    from scripts import (
        reason_router_gen4_six_cell_tier2_scientific_inference
        as r5,
    )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    with kernel_compat.exact_transformers_kernel_loader(kernels) as constructor_calls:
        backbone = r5.build_local_mamba_backbone(snapshot)

    counts = Counter(constructor_calls)
    require(set(counts) == {"causal-conv1d", "mamba-ssm"}, f"KERNEL_NAMES:{dict(counts)}")
    require(
        counts["causal-conv1d"] > 0
        and counts["causal-conv1d"] == counts["mamba-ssm"],
        f"KERNEL_CONSTRUCTOR_COUNTS:{dict(counts)}",
    )
    kernel_compat.validate_transformers_kernel_bindings(kernels)

    model = adapter.build_historical_model_from_backbone(
        backbone=backbone,
        arm=PARENT_ARM,
    )
    result = adapter.strict_load_state_dict(model, payload)
    require(
        len(result.missing_keys) == 0
        and len(result.unexpected_keys) == 0,
        "STRICT_PARENT_LOAD",
    )
    del payload

    model.to(device)
    model.mamba.config.use_cache = False
    require(model.mamba.config.use_cache is False, "USE_CACHE_FALSE")
    return model, {key: int(value) for key, value in counts.items()}


@contextlib.contextmanager
def count_layer22_fast_calls(native_mixer: torch.nn.Module) -> Iterator[dict[str, int]]:
    prior = native_mixer.__dict__.get("cuda_kernels_forward", None)
    original = native_mixer.cuda_kernels_forward
    counter = {"calls": 0}

    def wrapped(self, *args: Any, **kwargs: Any):
        counter["calls"] += 1
        return original(*args, **kwargs)

    native_mixer.cuda_kernels_forward = types.MethodType(wrapped, native_mixer)
    try:
        yield counter
    finally:
        if prior is None:
            del native_mixer.__dict__["cuda_kernels_forward"]
        else:
            native_mixer.cuda_kernels_forward = prior


def _feature_batch_to_device(
    bundle: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = bundle["model_inputs"]
    features = {
        key: inputs[key].to(device)
        for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask")
    }
    labels = inputs["final_labels"].to(device)
    require(tuple(labels.shape) == (TRAIN_ROWS,), "TRAIN_LABEL_SHAPE")
    require(
        tuple(features["input_ids"].shape) == (TRAIN_ROWS, MAX_LENGTH),
        "TRAIN_INPUT_SHAPE",
    )
    return features, labels


def _historical_forward(
    model: torch.nn.Module,
    features: Mapping[str, torch.Tensor],
) -> Mapping[str, Any]:
    from scripts import (
        reason_router_gen4_six_cell_tier2_inference_adapter
        as adapter,
    )

    return adapter.historical_forward(
        model,
        features,
        arm=PARENT_ARM,
    )


def _prepare_cuda_model(
    *,
    arm: str,
    seed: int,
    snapshot: Path,
    checkpoint: Path,
    device: torch.device,
    kernel_compat: Any,
    kernels: Mapping[str, Any],
    r22: torch.Tensor,
    c22: torch.Tensor,
) -> tuple[torch.nn.Module, Any, dict[str, int], str]:
    require(arm in TRAINING_ARMS, f"ARM:{arm}")
    require(seed in TRAINING_SEEDS, f"SEED:{seed}")

    # Parent constructor stochasticity must be matched across arms.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model, constructor_counts = _load_parent_model(
        snapshot=snapshot,
        checkpoint=checkpoint,
        device=device,
        kernel_compat=kernel_compat,
        kernels=kernels,
    )
    parent_before = parent_parameter_fingerprint(model)

    wrapper = install_phase2_layer22_wrapper(
        model,
        arm=arm,
        r22=r22,
        c22=c22,
        seed=seed,
    )
    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PARENT_CHANGED_DURING_WRAPPER_INSTALL",
    )
    audit = correction_parameter_audit(model)
    require(audit["trainable_tensor_count"] == 2, "TRAINABLE_TENSOR_COUNT")
    require(audit["trainable_numel"] == EXPECTED_TRAINABLE_NUMEL, "TRAINABLE_NUMEL")
    require(audit["all_trainable_are_correction"], "TRAINABLE_OWNERSHIP")
    require(
        int(torch.count_nonzero(wrapper.correction.B_theta.weight).item()) == 0,
        "B_NOT_ZERO_AT_CELL_START",
    )
    return model, wrapper, constructor_counts, parent_before


def run_cuda_preflight(
    *,
    args: argparse.Namespace,
    snapshot: Path,
    checkpoint: Path,
    static: Mapping[str, Any],
    encoded: Mapping[str, Any],
) -> dict[str, Any]:
    runtime, kernel_compat, backend = validate_cuda_runtime()
    del backend

    r22, c22, basis_geometry = load_frozen_owner_bases(ROOT)
    init_manifest = correction_initialization_manifest(r22, c22)

    device = torch.device("cuda:0")
    model, wrapper, constructor_counts, parent_before = _prepare_cuda_model(
        arm="G5-M1",
        seed=5201,
        snapshot=snapshot,
        checkpoint=checkpoint,
        device=device,
        kernel_compat=kernel_compat,
        kernels=kernel_compat.load_exact_fast_kernels(),
        r22=r22,
        c22=c22,
    )

    features, labels = _feature_batch_to_device(
        encoded["train_bundle"],
        device,
    )

    # Make the forward graph match training mode, but do not backward or step.
    model.train()
    model.mamba.config.use_cache = False
    torch.manual_seed(5201)
    torch.cuda.manual_seed_all(5201)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    output = None
    loss = None
    fast_calls = 0
    try:
        with count_layer22_fast_calls(wrapper.native_mixer) as fast_counter:
            with phase2_active_mask(model, features["attention_mask"]):
                output = _historical_forward(model, features)
            logits = output["logits"]
            require(tuple(logits.shape) == (TRAIN_ROWS, 3), "PREFLIGHT_LOGITS_SHAPE")
            require(bool(torch.isfinite(logits).all().item()), "PREFLIGHT_LOGITS_NONFINITE")
            loss = phase2_final_three_way_ce(logits, labels)
            require(bool(torch.isfinite(loss).item()), "PREFLIGHT_LOSS_NONFINITE")
            fast_calls = int(fast_counter["calls"])
        require(fast_calls == 1, f"LAYER22_FAST_PATH_CALLS:{fast_calls}")
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        report = {
            "schema_version": "GEN5_PHASE2_CUDA_PREFLIGHT_V1",
            "result": "GEN5_PHASE2_FULL_BATCH_EXECUTION_FEASIBILITY_BLOCKED",
            "execution_head": args.expected_head,
            "authority_commit": TRAINING_EXECUTION_AUTHORITY_COMMIT,
            "runtime": runtime,
            "full_batch_rows": TRAIN_ROWS,
            "sequence_length": MAX_LENGTH,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "training_executed": False,
            "fresh_xg1_loaded": False,
            "scientific_p_value_count": 0,
            "oom_type": type(exc).__name__,
        }
        _write_preflight_report(Path(args.preflight_output), report)
        print("RESULT=GEN5_PHASE2_FULL_BATCH_EXECUTION_FEASIBILITY_BLOCKED")
        print("TRAINING_EXECUTED=False")
        raise Phase2TrainingError(
            "GEN5_PHASE2_FULL_BATCH_EXECUTION_FEASIBILITY_BLOCKED"
        ) from exc
    finally:
        del output, loss

    require(
        parent_parameter_fingerprint(model) == parent_before,
        "PREFLIGHT_PARENT_PARAMETER_MUTATION",
    )
    require(
        wrapper.correction.R22.grad is None
        and wrapper.correction.C22.grad is None,
        "PREFLIGHT_BASIS_GRAD",
    )
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "PREFLIGHT_GRADIENT_PRESENT",
    )

    report = {
        "schema_version": "GEN5_PHASE2_CUDA_PREFLIGHT_V1",
        "result": "PASS_GEN5_PHASE2_FULL_BATCH_CUDA_PREFLIGHT",
        "execution_head": args.expected_head,
        "authority_commit": TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "runtime": runtime,
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "r22_sha256": R22_SHA256,
        "c22_sha256": C22_SHA256,
        "basis_geometry": basis_geometry,
        "constructor_kernel_calls": constructor_counts,
        "layer22_fast_path_calls": fast_calls,
        "full_batch_rows": TRAIN_ROWS,
        "sequence_length": MAX_LENGTH,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "train_order_sha256_v2": static["train_order_sha256_v2"],
        "dev_order_sha256_v2": static["dev_order_sha256_v2"],
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "dev_encoding_sha256": encoded["dev_encoding_sha256"],
        "cross_arm_initialization": init_manifest,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "training_executed": False,
        "fresh_xg1_loaded": False,
        "scientific_p_value_count": 0,
        "task_evaluation_executed": False,
        "next_stage": "GEN5_PHASE2_TRAINING_MATRIX_EXECUTION",
    }
    _write_preflight_report(Path(args.preflight_output), report)

    del model, wrapper, features, labels
    torch.cuda.empty_cache()

    print("RESULT=PASS_GEN5_PHASE2_FULL_BATCH_CUDA_PREFLIGHT")
    print(f"PEAK_MEMORY_ALLOCATED_BYTES={peak_allocated}")
    print(f"PEAK_MEMORY_RESERVED_BYTES={peak_reserved}")
    print("LAYER22_FAST_PATH_CALLS=1")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_STEP_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("FRESH_XG1_LOADED=False")
    return report


def _write_preflight_report(path: Path, report: Mapping[str, Any]) -> None:
    require(not path.exists(), f"PREFLIGHT_OUTPUT_COLLISION:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(report)) + b"\n")


def _correction_checkpoint_payload(
    *,
    wrapper: Any,
    args: argparse.Namespace,
    seed: int,
    arm: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    a_cpu = wrapper.correction.A_theta.weight.detach().cpu().contiguous()
    b_cpu = wrapper.correction.B_theta.weight.detach().cpu().contiguous()
    tensor_hashes = {
        "A_theta.weight": tensor_sha256(a_cpu),
        "B_theta.weight": tensor_sha256(b_cpu),
    }
    payload = {
        "schema_version": "GEN5_PHASE2_FINAL_CORRECTION_V1",
        "execution_commit": args.expected_head,
        "authority_commit": TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "r22_sha256": R22_SHA256,
        "c22_sha256": C22_SHA256,
        "seed": seed,
        "arm": arm,
        "state_dict": {
            "A_theta.weight": a_cpu,
            "B_theta.weight": b_cpu,
        },
        "tensor_sha256": tensor_hashes,
    }
    return payload, tensor_hashes


def run_training_cell(
    *,
    args: argparse.Namespace,
    seed: int,
    arm: str,
    snapshot: Path,
    checkpoint: Path,
    static: Mapping[str, Any],
    encoded: Mapping[str, Any],
    runtime: Mapping[str, Any],
    kernel_compat: Any,
    kernels: Mapping[str, Any],
    r22: torch.Tensor,
    c22: torch.Tensor,
) -> dict[str, Any]:
    cell_dir = Path(args.output_root) / f"seed{seed}" / arm
    require(not cell_dir.exists(), f"CELL_OUTPUT_COLLISION:{cell_dir}")
    cell_dir.parent.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir()

    device = torch.device("cuda:0")
    model, wrapper, constructor_counts, parent_before = _prepare_cuda_model(
        arm=arm,
        seed=seed,
        snapshot=snapshot,
        checkpoint=checkpoint,
        device=device,
        kernel_compat=kernel_compat,
        kernels=kernels,
        r22=r22,
        c22=c22,
    )

    features, labels = _feature_batch_to_device(
        encoded["train_bundle"],
        device,
    )

    basis_r_before = tensor_sha256(wrapper.correction.R22)
    basis_c_before = tensor_sha256(wrapper.correction.C22)
    require(basis_r_before == R22_SHA256, "R22_RUNTIME_SHA256")
    require(basis_c_before == C22_SHA256, "C22_RUNTIME_SHA256")

    optimizer_parameters = correction_optimizer_parameters(model)
    require(len(optimizer_parameters) == 2, "OPTIMIZER_PARAMETER_COUNT")
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    provenance = {
        "schema_version": "GEN5_PHASE2_TRAINING_RUN_PROVENANCE_V1",
        "execution_commit": args.expected_head,
        "authority_commit": TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "accelerated_implementation_commit": ACCELERATED_IMPLEMENTATION_COMMIT,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "native_backbone_signature_sha256": NATIVE_BACKBONE_SIGNATURE_SHA256,
        "r22_sha256": R22_SHA256,
        "c22_sha256": C22_SHA256,
        "seed": seed,
        "arm": arm,
        "runtime": dict(runtime),
        "constructor_kernel_calls": constructor_counts,
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "split_seed": SPLIT_SEED,
        "train_order_sha256_v2": static["train_order_sha256_v2"],
        "dev_order_sha256_v2": static["dev_order_sha256_v2"],
        "historical_ordered_train_row_sha256_reference":
            HISTORICAL_ORDERED_TRAIN_ROW_SHA256_REFERENCE,
        "historical_ordered_dev_row_sha256_reference":
            HISTORICAL_ORDERED_DEV_ROW_SHA256_REFERENCE,
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "dev_encoding_sha256": encoded["dev_encoding_sha256"],
        "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "epochs": EPOCHS,
        "planned_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "logical_batch": "EXACT_FULL_2880_ROW_TRAIN_SPLIT",
        "microbatching": False,
        "gradient_accumulation": False,
        "task_evaluation_executed": False,
        "fresh_xg1_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
        "status": "RUNNING",
    }
    (cell_dir / "run_provenance.json").write_bytes(
        canonical_json_bytes(provenance) + b"\n"
    )

    model.train()
    model.mamba.config.use_cache = False

    # Reset immediately before scientific training so matched arms consume the
    # same global RNG stream for frozen-parent dropout.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    losses: list[float] = []
    grad_norms: list[float] = []
    a_grad_norms: list[float] = []
    b_grad_norms: list[float] = []

    for epoch_index in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)

        with phase2_active_mask(model, features["attention_mask"]):
            output = _historical_forward(model, features)

        logits = output["logits"]
        require(tuple(logits.shape) == (TRAIN_ROWS, 3), "TRAIN_LOGITS_SHAPE")
        require(bool(torch.isfinite(logits).all().item()), "TRAIN_LOGITS_NONFINITE")

        loss = phase2_final_three_way_ce(logits, labels)
        require(bool(torch.isfinite(loss).item()), f"LOSS_NONFINITE:{epoch_index + 1}")

        loss.backward()

        a_grad = wrapper.correction.A_theta.weight.grad
        b_grad = wrapper.correction.B_theta.weight.grad
        require(a_grad is not None, f"A_GRAD_MISSING:{epoch_index + 1}")
        require(b_grad is not None, f"B_GRAD_MISSING:{epoch_index + 1}")
        require(bool(torch.isfinite(a_grad).all().item()), "A_GRAD_NONFINITE")
        require(bool(torch.isfinite(b_grad).all().item()), "B_GRAD_NONFINITE")

        parent_grads = [
            name
            for name, parameter in model.named_parameters()
            if ".correction." not in name and parameter.grad is not None
        ]
        require(not parent_grads, f"PARENT_GRADIENT_PRESENT:{parent_grads[:5]}")
        require(wrapper.correction.R22.grad is None, "R22_GRAD_PRESENT")
        require(wrapper.correction.C22.grad is None, "C22_GRAD_PRESENT")

        clipped = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters,
            GRADIENT_CLIP_NORM,
        )
        require(bool(torch.isfinite(clipped).item()), "CLIPPED_GRAD_NONFINITE")

        optimizer.step()

        require(
            bool(torch.isfinite(wrapper.correction.A_theta.weight).all().item()),
            "A_PARAMETER_NONFINITE",
        )
        require(
            bool(torch.isfinite(wrapper.correction.B_theta.weight).all().item()),
            "B_PARAMETER_NONFINITE",
        )

        losses.append(float(loss.detach().cpu().item()))
        grad_norms.append(float(clipped.detach().cpu().item()))
        a_grad_norms.append(float(torch.linalg.vector_norm(a_grad).detach().cpu().item()))
        b_grad_norms.append(float(torch.linalg.vector_norm(b_grad).detach().cpu().item()))

        del output, logits, loss
        torch.cuda.synchronize()

    require(len(losses) == EPOCHS, "LOSS_HISTORY_COUNT")
    require(len(grad_norms) == TOTAL_OPTIMIZER_STEPS, "OPTIMIZER_STEP_COUNT")

    parent_after = parent_parameter_fingerprint(model)
    require(parent_after == parent_before, "PARENT_PARAMETER_MUTATION")
    require(tensor_sha256(wrapper.correction.R22) == basis_r_before, "R22_MUTATION")
    require(tensor_sha256(wrapper.correction.C22) == basis_c_before, "C22_MUTATION")

    payload, final_tensor_hashes = _correction_checkpoint_payload(
        wrapper=wrapper,
        args=args,
        seed=seed,
        arm=arm,
    )
    correction_path = cell_dir / "final_correction.pt"
    torch.save(payload, correction_path)
    correction_file_sha = sha256_file(correction_path)

    audit = correction_parameter_audit(model)
    report = {
        "schema_version": "GEN5_PHASE2_TRAINING_REPORT_V1",
        "result": "PASS_GEN5_PHASE2_TRAINING_CELL",
        "execution_commit": args.expected_head,
        "authority_commit": TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "r22_sha256": R22_SHA256,
        "c22_sha256": C22_SHA256,
        "seed": seed,
        "arm": arm,
        "runtime": dict(runtime),
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "split_seed": SPLIT_SEED,
        "train_order_sha256_v2": static["train_order_sha256_v2"],
        "dev_order_sha256_v2": static["dev_order_sha256_v2"],
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "dev_encoding_sha256": encoded["dev_encoding_sha256"],
        "training_losses": losses,
        "gradient_norms_before_clip": grad_norms,
        "A_gradient_norms_after_backward": a_grad_norms,
        "B_gradient_norms_after_backward": b_grad_norms,
        "optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "checkpoint_selection": "FINAL_FIXED_STEP_ONLY",
        "objective": "FINAL_3WAY_CROSS_ENTROPY_ONLY",
        "trainable_tensor_names": audit["trainable_names"],
        "trainable_tensor_count": audit["trainable_tensor_count"],
        "trainable_numel": audit["trainable_numel"],
        "parent_signature_before": parent_before,
        "parent_signature_after": parent_after,
        "r22_runtime_sha256_before": basis_r_before,
        "r22_runtime_sha256_after": tensor_sha256(wrapper.correction.R22),
        "c22_runtime_sha256_before": basis_c_before,
        "c22_runtime_sha256_after": tensor_sha256(wrapper.correction.C22),
        "final_correction_tensor_sha256": final_tensor_hashes,
        "final_correction_file_sha256": correction_file_sha,
        "task_evaluation_executed": False,
        "fresh_xg1_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
        "training_success": True,
    }
    (cell_dir / "training_report.json").write_bytes(
        canonical_json_bytes(report) + b"\n"
    )

    provenance["status"] = "PASS"
    provenance["actual_optimizer_steps"] = TOTAL_OPTIMIZER_STEPS
    provenance["training_report_sha256"] = sha256_file(cell_dir / "training_report.json")
    provenance["final_correction_file_sha256"] = correction_file_sha
    (cell_dir / "run_provenance.json").write_bytes(
        canonical_json_bytes(provenance) + b"\n"
    )

    del model, wrapper, optimizer, optimizer_parameters, features, labels
    torch.cuda.empty_cache()

    print(
        f"CELL_PASS seed={seed} arm={arm} "
        f"final_loss={losses[-1]:.9g} correction_sha256={correction_file_sha}"
    )
    return report


def run_matrix(
    *,
    args: argparse.Namespace,
    snapshot: Path,
    checkpoint: Path,
    static: Mapping[str, Any],
    encoded: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = Path(args.output_root)
    require(not output_root.exists(), f"OUTPUT_ROOT_COLLISION:{output_root}")
    output_root.mkdir(parents=True)

    runtime, kernel_compat, backend = validate_cuda_runtime()
    del backend
    kernels = kernel_compat.load_exact_fast_kernels()

    r22, c22, basis_geometry = load_frozen_owner_bases(ROOT)
    init_manifest = correction_initialization_manifest(r22, c22)

    top_provenance = {
        "schema_version": "GEN5_PHASE2_TRAINING_MATRIX_PROVENANCE_V1",
        "execution_commit": args.expected_head,
        "authority_commit": TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "runtime": runtime,
        "matrix": [
            {"seed": seed, "arm": arm}
            for seed, arm in TRAINING_MATRIX
        ],
        "cross_arm_initialization": init_manifest,
        "basis_geometry": basis_geometry,
        "train_order_sha256_v2": static["train_order_sha256_v2"],
        "dev_order_sha256_v2": static["dev_order_sha256_v2"],
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "dev_encoding_sha256": encoded["dev_encoding_sha256"],
        "fresh_xg1_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
        "status": "RUNNING",
    }
    (output_root / "matrix_provenance.json").write_bytes(
        canonical_json_bytes(top_provenance) + b"\n"
    )

    reports: list[dict[str, Any]] = []
    for seed, arm in TRAINING_MATRIX:
        reports.append(
            run_training_cell(
                args=args,
                seed=seed,
                arm=arm,
                snapshot=snapshot,
                checkpoint=checkpoint,
                static=static,
                encoded=encoded,
                runtime=runtime,
                kernel_compat=kernel_compat,
                kernels=kernels,
                r22=r22,
                c22=c22,
            )
        )

    require(len(reports) == 9, "TRAINING_MATRIX_CELL_COUNT")
    require(
        all(row.get("result") == "PASS_GEN5_PHASE2_TRAINING_CELL" for row in reports),
        "TRAINING_MATRIX_CELL_FAILURE",
    )

    manifest = {
        "schema_version": "GEN5_PHASE2_TRAINING_MATRIX_MANIFEST_V1",
        "result": "PASS_GEN5_PHASE2_TRAINING_MATRIX",
        "execution_commit": args.expected_head,
        "authority_commit": TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "cell_count": len(reports),
        "cells": [
            {
                "seed": int(row["seed"]),
                "arm": str(row["arm"]),
                "training_report_sha256": sha256_file(
                    output_root
                    / f"seed{row['seed']}"
                    / str(row["arm"])
                    / "training_report.json"
                ),
                "run_provenance_sha256": sha256_file(
                    output_root
                    / f"seed{row['seed']}"
                    / str(row["arm"])
                    / "run_provenance.json"
                ),
                "final_correction_sha256": str(
                    row["final_correction_file_sha256"]
                ),
            }
            for row in reports
        ],
        "task_evaluation_executed": False,
        "fresh_xg1_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
        "next_stage": "COLLECT_IMPORT_THEN_GEN5_PHASE2_FRESH_OWNERSHIP_ASSAY_EXECUTION",
    }
    (output_root / "matrix_manifest.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )

    checksums: list[str] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(
                f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}"
            )
    (output_root / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    top_provenance["status"] = "PASS"
    top_provenance["matrix_manifest_sha256"] = sha256_file(
        output_root / "matrix_manifest.json"
    )
    (output_root / "matrix_provenance.json").write_bytes(
        canonical_json_bytes(top_provenance) + b"\n"
    )

    print("RESULT=PASS_GEN5_PHASE2_TRAINING_MATRIX")
    print("TRAINING_CELL_COUNT=9")
    print("OPTIMIZER_STEPS_PER_CELL=20")
    print("TOTAL_OPTIMIZER_STEPS=180")
    print("TASK_EVALUATION_EXECUTED=False")
    print("FRESH_XG1_LOADED=False")
    print("SCIENTIFIC_P_VALUE_COUNT=0")
    print("SCIENTIFIC_CONCLUSION=None")
    return manifest


def static_preflight(
    args: argparse.Namespace,
) -> dict[str, Any]:
    authenticate_repo(
        args.expected_head,
        allow_opening_worktree=bool(args.allow_opening_worktree),
    )
    validate_authority_file()
    runtime = validate_historical_tokenizer_runtime()
    snapshot = resolve_exact_snapshot(args.model_snapshot)
    static = authenticate_static_inputs()
    encoded = encode_splits(
        snapshot,
        static["train_records"],
        static["dev_records"],
    )
    r22, c22, basis_geometry = load_frozen_owner_bases(ROOT)
    init_manifest = correction_initialization_manifest(r22, c22)

    report = {
        "schema_version": "GEN5_PHASE2_STATIC_EXECUTION_PREFLIGHT_V1",
        "result": "PASS_GEN5_PHASE2_STATIC_EXECUTION_PREFLIGHT",
        "head": args.expected_head,
        "authority_commit": TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "historical_tokenizer_runtime": runtime,
        "dataset_git_lf_sha256": static["dataset_git_lf_sha256"],
        "dataset_semantic_sha256": static["dataset_semantic_sha256"],
        "sidecar_git_lf_sha256": static["sidecar_git_lf_sha256"],
        "sidecar_semantic_sha256": static["sidecar_semantic_sha256"],
        "train_rows": TRAIN_ROWS,
        "dev_rows": DEV_ROWS,
        "train_pairs": TRAIN_PAIRS,
        "dev_pairs": DEV_PAIRS,
        "train_order_sha256_v2": static["train_order_sha256_v2"],
        "dev_order_sha256_v2": static["dev_order_sha256_v2"],
        "historical_ordered_train_row_sha256_reference":
            HISTORICAL_ORDERED_TRAIN_ROW_SHA256_REFERENCE,
        "historical_ordered_dev_row_sha256_reference":
            HISTORICAL_ORDERED_DEV_ROW_SHA256_REFERENCE,
        "train_encoding_sha256": encoded["train_encoding_sha256"],
        "dev_encoding_sha256": encoded["dev_encoding_sha256"],
        "basis_geometry": basis_geometry,
        "cross_arm_initialization": init_manifest,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "cuda_executed": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "training_executed": False,
        "fresh_xg1_loaded": False,
        "scientific_p_value_count": 0,
        "next_stage": "COMMIT_RUNNER_OPENING_THEN_CUDA_PREFLIGHT",
    }
    print(json.dumps(report, sort_keys=True))
    print("RESULT=PASS_GEN5_PHASE2_STATIC_EXECUTION_PREFLIGHT")
    print("TRAIN_ORDER_SHA256_V2=" + static["train_order_sha256_v2"])
    print("DEV_ORDER_SHA256_V2=" + static["dev_order_sha256_v2"])
    print("TRAIN_ENCODING_SHA256=" + encoded["train_encoding_sha256"])
    print("DEV_ENCODING_SHA256=" + encoded["dev_encoding_sha256"])
    print("MODEL_LOADED=False")
    print("CHECKPOINT_LOADED=False")
    print("TRAINING_EXECUTED=False")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gen5 Phase 2 state-update ownership execution runner."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-preflight-only", action="store_true")
    modes.add_argument("--cuda-preflight-only", action="store_true")
    modes.add_argument("--run-matrix", action="store_true")

    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--execution-authority-commit",
        default=TRAINING_EXECUTION_AUTHORITY_COMMIT,
    )
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-train-order-sha256-v2")
    parser.add_argument("--expected-dev-order-sha256-v2")
    parser.add_argument("--expected-train-encoding-sha256")
    parser.add_argument("--expected-dev-encoding-sha256")
    parser.add_argument(
        "--allow-opening-worktree",
        action="store_true",
        help="Static implementation validation only; dirty paths must be the exact two-file opening scope.",
    )
    parser.add_argument(
        "--print-frozen-contract",
        action="store_true",
    )
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    require(
        args.execution_authority_commit == TRAINING_EXECUTION_AUTHORITY_COMMIT,
        "TRAINING_AUTHORITY_COMMIT_MISMATCH",
    )
    validate_no_fresh_assay_path(args.model_snapshot)
    validate_no_fresh_assay_path(args.checkpoint)
    validate_no_fresh_assay_path(args.preflight_output)
    validate_no_fresh_assay_path(args.output_root)

    if args.static_preflight_only:
        require(args.checkpoint is None, "STATIC_PREFLIGHT_CHECKPOINT_FORBIDDEN")
        require(args.preflight_output is None, "STATIC_PREFLIGHT_OUTPUT_FILE_FORBIDDEN")
        require(args.output_root is None, "STATIC_PREFLIGHT_OUTPUT_ROOT_FORBIDDEN")
        return

    require(not args.allow_opening_worktree, "EXECUTION_DIRTY_WORKTREE_FORBIDDEN")
    require(args.checkpoint is not None, "CHECKPOINT_REQUIRED")

    for attr in (
        "expected_train_order_sha256_v2",
        "expected_dev_order_sha256_v2",
        "expected_train_encoding_sha256",
        "expected_dev_encoding_sha256",
    ):
        require(getattr(args, attr) is not None, f"{attr.upper()}_REQUIRED")

    if args.cuda_preflight_only:
        require(args.preflight_output is not None, "PREFLIGHT_OUTPUT_REQUIRED")
        require(args.output_root is None, "CUDA_PREFLIGHT_OUTPUT_ROOT_FORBIDDEN")

    if args.run_matrix:
        require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")
        require(args.preflight_output is None, "RUN_MATRIX_PREFLIGHT_OUTPUT_FORBIDDEN")


def prepare_execution_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    authenticate_repo(args.expected_head, allow_opening_worktree=False)
    validate_authority_file()
    snapshot = resolve_exact_snapshot(args.model_snapshot)
    checkpoint = Path(args.checkpoint)
    authenticate_checkpoint(checkpoint)
    static = authenticate_static_inputs()
    encoded = encode_splits(
        snapshot,
        static["train_records"],
        static["dev_records"],
    )
    validate_expected_execution_identities(args, static, encoded)
    return snapshot, checkpoint, static, encoded


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_mode_args(args)

    if args.print_frozen_contract:
        print(json.dumps(FROZEN_TRAINING_CONTRACT, sort_keys=True, indent=2))

    if args.static_preflight_only:
        static_preflight(args)
        return

    snapshot, checkpoint, static, encoded = prepare_execution_inputs(args)

    if args.cuda_preflight_only:
        run_cuda_preflight(
            args=args,
            snapshot=snapshot,
            checkpoint=checkpoint,
            static=static,
            encoded=encoded,
        )
        return

    if args.run_matrix:
        run_matrix(
            args=args,
            snapshot=snapshot,
            checkpoint=checkpoint,
            static=static,
            encoded=encoded,
        )
        return

    raise Phase2TrainingError("UNREACHABLE_MODE")


if __name__ == "__main__":
    main()
