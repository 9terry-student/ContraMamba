#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import tempfile
import traceback
import types
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"

# Cross-backbone protocol constants.
ORIGINAL_LAYER_COUNT = 24
ORIGINAL_INTERVENTION_LAYER = 17
NEW_LAYER_COUNT = 48
LOCAL_LAYER_OFFSETS = (-2, -1, 0)
INTERVENTION_LAYER = round(
    (ORIGINAL_INTERVENTION_LAYER / (ORIGINAL_LAYER_COUNT - 1))
    * (NEW_LAYER_COUNT - 1)
)
SOURCE_BLOCK = INTERVENTION_LAYER + LOCAL_LAYER_OFFSETS[0]
TARGET_RESIDUAL_LAYER = INTERVENTION_LAYER + LOCAL_LAYER_OFFSETS[1]
TARGET_OFFSET = 2

HIDDEN_SIZE = 1536
INTERMEDIATE_SIZE = 3072
STATE_SIZE = 16
LAYER_COUNT = 48
CONV_KERNEL = 4
LAG0_KERNEL_INDEX = 3

K = 5
FAMILIES = ("xg2", "xg4")
SOURCE_PAIR_COUNT = 300
FORWARDS_PER_PAIR = 4
FORWARDS_PER_FAMILY = SOURCE_PAIR_COUNT * FORWARDS_PER_PAIR
TOTAL_FORWARD_BUDGET = len(FAMILIES) * FORWARDS_PER_FAMILY
GPU_COUNT = 2

TARGET_PLUS_CELL = "C2_NAME"
TARGET_MINUS_CELL = "C0_SHAM"
REFERENCE_PLUS_CELL = "C5_TITLE_NAME"
REFERENCE_MINUS_CELL = "C1_TITLE"
ANCHOR_NAME = "A_IDENTITY"

HF_REPO = "state-spaces/mamba-790m-hf"
HF_REVISION = "9822dd4b76af2bd9099b6ce2f19efd8329189a7e"
HF_FILES: dict[str, tuple[int, str]] = {
    "config.json": (
        878,
        "f55bce4f993e4eeba025d57977eb48c4815416a352742c74e412e2c1e8136ea1",
    ),
    "generation_config.json": (
        137,
        "248fa733db101c19a8e3c2f311a180fdd7c769323ec2873edf7060cec9f5ee32",
    ),
    "model.safetensors": (
        3172869936,
        "3fbcf74674d28034e728b911315e4749cc4fc34ebfab36099f46d92bb657a617",
    ),
    "tokenizer.json": (
        2113738,
        "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
    ),
    "tokenizer_config.json": (
        4793,
        "3ba257483d22a5a84aab5465aa427e59bdaeb55f09fb14349e2d571ff67e8020",
    ),
}

COMPACT_CHECKPOINT_REL = Path(
    "reports/reason_router_gen4_mamba790m_training_runs/"
    "g4k-mamba790m-train-g3d-seed181-dualt4-cache-2f10382-retry2/"
    "selected_downstream_checkpoint.pt"
)
COMPACT_CHECKPOINT_SHA256 = (
    "af5582df61ed2dc2f7c77c0154c10e40b7aa6646a2141a6b3b30c52a45fc9072"
)
SOURCE_FULL_CHECKPOINT_SHA256 = (
    "b0bd62d0581c572fcca2789042522d09ebd75c6516c6a0396e2de131d91c77e8"
)
PRETRAINED_BACKBONE_STATE_STREAM_SHA256 = (
    "c7720da316a49b7dc4ff56b5f56e4bba4d089f2bad53aeb3cebee9a345c3f5c0"
)
DOWNSTREAM_STATE_CANONICAL_SHA256 = (
    "f446224dc6912d0b92d7e965cf3decbffdff337aaa68523d871d77d27de2e398"
)
FULL_STATE_CANONICAL_SHA256 = (
    "1904e06e9645ab1406d8b7934f93666219911a5e1586aac6a6878efd7ba8e8a6"
)
TRAINING_EXECUTION_COMMIT = (
    "2f10382e786ecce0547e2681540351a6d17411d5"
)
TRAINING_SEED = 181
SPLIT_SEED = 8192
SELECTED_EPOCH = 20
ARM = "G3-GROUP-D-HALF"

COHORT_IDENTITIES = {
    "xg2": {
        "source_sha256":
            "c02d2fea5a7f3c8b5243598ad5505bba3f276c099be7b8c428c06eb39ffc7141",
        "rows_sha256":
            "1ca4f1c79caf5719e8970bf889c75b0e59f8711e5c78a36f7578727020e23670",
        "manifest_sha256":
            "7df94b0788a55c6a4927926f6b92ef7e90d66346c5342608f5716ddd517e47f9",
    },
    "xg4": {
        "source_sha256":
            "0666c9345505f993bce70d66b5b1f9b784edca782eecb13a750ecf827f14ba3a",
        "rows_sha256":
            "b407613cdcee15d193847130f64e6aa674e666f72d6d08a30b573005e5e9de7d",
        "manifest_sha256":
            "ee8d9bc4ae0ca135c335cf9eaa6bd9025cd5354304d435f1e3bb703a8b11e80d",
    },
}

TOKENIZER_FILE_SHA256 = {
    "tokenizer.json": HF_FILES["tokenizer.json"][1],
    "tokenizer_config.json": HF_FILES["tokenizer_config.json"][1],
}

EXPECTED_TOKENIZERS_VERSION = "0.22.2"
EOS_TOKEN_ID = 0
EFFECTIVE_PAD_TOKEN_ID = 0

BASIS_EIGENGAP_TOL = 1.0e-10
ORTHONORMALITY_TOL = 1.0e-10
PLANE_TOL = 2.0e-10

MANIFEST_FILE = "artifact_manifest.json"
SUMMARY_FILE = "geometry_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
STRONG_INDEX_FILE = "strong_indices.json"
PROJECTOR_CONTRAST_FILE = "projector_contrast.f64le"

RESULT_PASS = "PASS_MAMBA790M_GEOMETRY_PREPARATION"


class GeometryPreparationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GeometryPreparationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def raw_f64le(value: torch.Tensor) -> bytes:
    arr = np.asarray(
        value.detach().cpu().to(torch.float64).contiguous().numpy(),
        dtype=np.dtype("<f8"),
    )
    return arr.tobytes(order="C")


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GeometryPreparationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def validate_checkout_identity(
    *,
    branch: str,
    head: str,
    status: str,
    expected_head: str,
) -> None:
    # Local development may be on the named branch, while the Kaggle
    # bootstrap intentionally checks out the exact execution commit detached.
    # Exact commit identity and a clean worktree are the scientific contract.
    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, "HEAD_MISMATCH")
    require(status == "", "WORKTREE_NOT_CLEAN")


def authenticate_repo(expected_head: str) -> None:
    validate_checkout_identity(
        branch=git("branch", "--show-current"),
        head=git("rev-parse", "HEAD"),
        status=git("status", "--porcelain"),
        expected_head=expected_head,
    )


def validate_protocol_constants() -> None:
    require(INTERVENTION_LAYER == 35, "INTERVENTION_LAYER_MAPPING")
    require(
        (SOURCE_BLOCK, TARGET_RESIDUAL_LAYER, INTERVENTION_LAYER)
        == (33, 34, 35),
        "LOCAL_LAYER_OFFSETS",
    )
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(K == 5, "K")
    require(TOTAL_FORWARD_BUDGET == 2400, "FORWARD_BUDGET")


def strong_index_sha256(indices: Sequence[int]) -> str:
    raw = ",".join(str(int(value)) for value in indices).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def strong_partition(conv_weight: torch.Tensor) -> dict[str, Any]:
    require(torch.is_tensor(conv_weight), "CONV_WEIGHT_NOT_TENSOR")
    weight = conv_weight.detach().cpu().to(torch.float64).contiguous()
    require(
        tuple(weight.shape) == (INTERMEDIATE_SIZE, 1, CONV_KERNEL),
        f"CONV_WEIGHT_SHAPE:{tuple(weight.shape)}",
    )
    require(
        bool(torch.isfinite(weight).all().item()),
        "NONFINITE_KERNEL_STATISTIC",
    )

    k = weight[:, 0, LAG0_KERNEL_INDEX]
    k2 = k.square()
    require(bool(torch.isfinite(k2).all().item()), "NONFINITE_KERNEL_STATISTIC")

    mu = float(k2.mean().item())
    require(math.isfinite(mu), "NONFINITE_KERNEL_STATISTIC")

    strong = torch.nonzero(k2 > mu, as_tuple=False).flatten().to(torch.long)
    weak = torch.nonzero(k2 < mu, as_tuple=False).flatten().to(torch.long)
    equal = torch.nonzero(k2 == mu, as_tuple=False).flatten().to(torch.long)

    count = int(strong.numel())
    require(count > 0, "ZERO_SELECTED_CHANNELS")
    require(count < INTERMEDIATE_SIZE, "ALL_CHANNELS_SELECTED")

    indices = [int(v) for v in strong.tolist()]
    digest = strong_index_sha256(indices)

    reconstructed = torch.zeros(INTERMEDIATE_SIZE, dtype=torch.bool)
    reconstructed[strong] = True
    roundtrip = torch.nonzero(reconstructed, as_tuple=False).flatten().tolist()
    require(
        strong_index_sha256(roundtrip) == digest,
        "MASK_RECONSTRUCTION_HASH_MISMATCH",
    )

    require(
        int(strong.numel() + weak.numel() + equal.numel()) == INTERMEDIATE_SIZE,
        "MASK_PARTITION_CARDINALITY",
    )

    return {
        "mu_k2": mu,
        "strong_indices": indices,
        "strong_count": count,
        "weak_count": int(weak.numel()),
        "equal_count": int(equal.numel()),
        "strong_index_sha256": digest,
        "mask": reconstructed,
    }


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.numpy().tobytes(order="C")


def canonical_state_sha256(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        require(torch.is_tensor(value), f"NON_TENSOR_STATE:{key}")
        tensor = value.detach().cpu().contiguous()

        header = json.dumps(
            {
                "name": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)

        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)

    return h.hexdigest()


def validate_snapshot(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    require(snapshot.name == HF_REVISION, "HF_REVISION_PATH")

    observed: dict[str, Any] = {}
    for name, (expected_size, expected_sha) in HF_FILES.items():
        path = snapshot / name
        require(path.is_file(), f"HF_FILE_MISSING:{name}")
        size = path.stat().st_size
        digest = sha256_file(path)
        require(size == expected_size, f"HF_FILE_SIZE:{name}:{size}")
        require(digest == expected_sha, f"HF_FILE_SHA256:{name}:{digest}")
        observed[name] = {
            "bytes": size,
            "sha256": digest,
        }
    return observed


def _token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def load_tokenizer(snapshot: Path) -> tuple[Any, dict[str, Any]]:
    validate_snapshot(snapshot)

    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise GeometryPreparationError("TOKENIZERS_PACKAGE_REQUIRED") from exc

    require(
        tokenizers.__version__ == EXPECTED_TOKENIZERS_VERSION,
        (
            "TOKENIZERS_VERSION_MISMATCH:"
            f"{tokenizers.__version__}"
        ),
    )

    for name, digest in TOKENIZER_FILE_SHA256.items():
        require(
            sha256_file(snapshot / name) == digest,
            f"TOKENIZER_SHA256:{name}",
        )

    config = json.loads(
        (snapshot / "tokenizer_config.json").read_text(
            encoding="utf-8-sig"
        )
    )

    eos_token = _token_content(config.get("eos_token"))
    require(
        eos_token is not None,
        "TOKENIZER_EOS_DECLARATION",
    )

    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()

    eos_id = tokenizer.token_to_id(eos_token)
    require(eos_id == EOS_TOKEN_ID, f"EOS_TOKEN_ID:{eos_id}")

    pad_token = _token_content(config.get("pad_token"))
    pad_id = (
        tokenizer.token_to_id(pad_token)
        if pad_token is not None
        else eos_id
    )
    require(
        pad_id == EFFECTIVE_PAD_TOKEN_ID,
        f"EFFECTIVE_PAD_TOKEN_ID:{pad_id}",
    )

    return tokenizer, {
        "repo": HF_REPO,
        "revision": HF_REVISION,
        "tokenizers_version": tokenizers.__version__,
        "vocab_size": tokenizer.get_vocab_size(),
        "file_sha256": dict(TOKENIZER_FILE_SHA256),
    }


def validate_fast_runtime_for_device(gpu_id: int) -> None:
    from scripts import (
        reason_router_gen4_generator_family_prevalence_kernel_compat
        as kernel_compat,
    )
    from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend
    import transformers

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    require(
        observed == backend.EXPECTED_RUNTIME,
        f"RUNTIME_MISMATCH:{observed}",
    )
    require(
        kernel_compat._kernel_package_version() == backend.KERNELS_VERSION,
        "KERNELS_VERSION",
    )
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")
    require(torch.version.cuda == backend.EXPECTED_CUDA_RUNTIME, "CUDA_RUNTIME")
    require(0 <= gpu_id < torch.cuda.device_count(), "GPU_ID_RANGE")

    torch.cuda.set_device(gpu_id)
    require(
        torch.cuda.get_device_name(gpu_id) == backend.EXPECTED_DEVICE_NAME,
        f"CUDA_DEVICE_NAME:{gpu_id}",
    )
    require(
        tuple(torch.cuda.get_device_capability(gpu_id))
        == backend.EXPECTED_CAPABILITY,
        f"CUDA_CAPABILITY:{gpu_id}",
    )

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def reconstruct_model(
    *,
    snapshot: Path,
    compact_checkpoint: Path,
    gpu_id: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from transformers import MambaConfig, MambaModel
    from scripts import (
        reason_router_gen4_generator_family_prevalence_kernel_compat
        as kernel_compat,
    )
    from scripts import (
        reason_router_gen4_six_cell_tier2_inference_adapter
        as adapter,
    )

    snapshot_identity = validate_snapshot(snapshot)

    require(
        compact_checkpoint.is_file(),
        "COMPACT_CHECKPOINT_MISSING",
    )
    require(
        sha256_file(compact_checkpoint) == COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA256",
    )

    try:
        compact = torch.load(
            compact_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        compact = torch.load(
            compact_checkpoint,
            map_location="cpu",
        )

    require(
        compact.get("schema_version")
        == "contramamba_mamba790m_compact_checkpoint_v1",
        "COMPACT_SCHEMA",
    )

    pretrained = compact.get("pretrained_backbone")
    source_full = compact.get("source_full_checkpoint")
    downstream_meta = compact.get("downstream")
    training_identity = compact.get("training_identity")
    reconstruction = compact.get("reconstruction_verification")

    require(isinstance(pretrained, Mapping), "PRETRAINED_BACKBONE_META")
    require(isinstance(source_full, Mapping), "SOURCE_FULL_META")
    require(isinstance(downstream_meta, Mapping), "DOWNSTREAM_META")
    require(isinstance(training_identity, Mapping), "TRAINING_IDENTITY_META")
    require(isinstance(reconstruction, Mapping), "RECONSTRUCTION_META")

    require(
        pretrained.get("repo") == HF_REPO
        and pretrained.get("revision") == HF_REVISION,
        "COMPACT_HF_IDENTITY",
    )
    require(
        pretrained.get("state_stream_sha256")
        == PRETRAINED_BACKBONE_STATE_STREAM_SHA256,
        "PRETRAINED_STATE_STREAM_SHA256",
    )
    expected_file_identities = {
        name: {
            "bytes": size,
            "sha256": digest,
        }
        for name, (size, digest) in HF_FILES.items()
    }
    require(
        pretrained.get("file_identities") == expected_file_identities,
        "PRETRAINED_FILE_IDENTITIES",
    )
    require(
        int(pretrained.get("state_key_count", -1)) == 482,
        "PRETRAINED_STATE_KEY_COUNT",
    )

    require(
        source_full.get("sha256")
        == SOURCE_FULL_CHECKPOINT_SHA256,
        "SOURCE_FULL_CHECKPOINT_SHA256",
    )
    require(
        source_full.get("full_state_canonical_sha256")
        == FULL_STATE_CANONICAL_SHA256,
        "SOURCE_FULL_STATE_CANONICAL_SHA256",
    )
    require(
        int(source_full.get("state_key_count", -1)) == 518,
        "SOURCE_FULL_STATE_KEY_COUNT",
    )

    require(
        downstream_meta.get("state_canonical_sha256")
        == DOWNSTREAM_STATE_CANONICAL_SHA256,
        "DOWNSTREAM_META_CANONICAL_SHA256",
    )
    require(
        int(downstream_meta.get("state_key_count", -1)) == 36,
        "DOWNSTREAM_META_KEY_COUNT",
    )

    require(
        training_identity.get("execution_commit")
        == TRAINING_EXECUTION_COMMIT,
        "TRAINING_EXECUTION_COMMIT",
    )
    require(
        training_identity.get("selected_epoch") == SELECTED_EPOCH,
        "SELECTED_EPOCH",
    )
    require(
        training_identity.get("training_seed") == TRAINING_SEED,
        "TRAINING_SEED",
    )
    require(
        training_identity.get("split_seed") == SPLIT_SEED,
        "SPLIT_SEED",
    )
    require(training_identity.get("arm") == ARM, "ARM")

    require(
        reconstruction.get("key_set_exact") is True
        and reconstruction.get("tensor_equal_all_backbone_keys") is True
        and reconstruction.get(
            "downstream_extracted_exactly_from_full_checkpoint"
        ) is True,
        "RECONSTRUCTION_VERIFICATION",
    )

    downstream = compact.get("downstream_state_dict")
    require(
        isinstance(downstream, Mapping) and len(downstream) == 36,
        "DOWNSTREAM_STATE",
    )
    require(
        canonical_state_sha256(downstream)
        == DOWNSTREAM_STATE_CANONICAL_SHA256,
        "DOWNSTREAM_CANONICAL_SHA256",
    )

    cfg = MambaConfig.from_pretrained(
        snapshot,
        local_files_only=True,
    )
    require(int(cfg.hidden_size) == HIDDEN_SIZE, "CONFIG_HIDDEN_SIZE")
    require(
        int(cfg.intermediate_size) == INTERMEDIATE_SIZE,
        "CONFIG_INTERMEDIATE_SIZE",
    )
    require(int(cfg.num_hidden_layers) == LAYER_COUNT, "CONFIG_LAYER_COUNT")
    require(int(cfg.state_size) == STATE_SIZE, "CONFIG_STATE_SIZE")
    require(int(cfg.conv_kernel) == CONV_KERNEL, "CONFIG_CONV_KERNEL")
    cfg.use_mamba_kernels = True

    kernels = kernel_compat.load_exact_fast_kernels()

    with kernel_compat.exact_transformers_kernel_loader(kernels) as calls:
        backbone = MambaModel.from_pretrained(
            snapshot,
            config=cfg,
            local_files_only=True,
        )

    counts = Counter(calls)
    require(
        set(counts) == {"causal-conv1d", "mamba-ssm"}
        and counts["causal-conv1d"] > 0
        and counts["causal-conv1d"] == counts["mamba-ssm"],
        f"KERNEL_CONSTRUCTOR:{dict(counts)}",
    )
    kernel_compat.validate_transformers_kernel_bindings(kernels)

    require(len(backbone.layers) == LAYER_COUNT, "BACKBONE_LAYER_COUNT")

    model = adapter.build_historical_model_from_backbone(
        backbone=backbone,
        arm=ARM,
        hidden_size=HIDDEN_SIZE,
    )

    require(
        len(model.mamba.state_dict()) == 482,
        "MAMBA_STATE_KEY_COUNT",
    )

    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    require(len(state) == 518, "FULL_STATE_KEY_COUNT_BEFORE")
    require(
        set(downstream).issubset(state),
        "DOWNSTREAM_KEY_SET",
    )

    for key, value in downstream.items():
        require(
            tuple(state[key].shape) == tuple(value.shape),
            f"DOWNSTREAM_SHAPE:{key}",
        )
        state[key] = value.detach().cpu().clone()

    require(
        canonical_state_sha256(state) == FULL_STATE_CANONICAL_SHA256,
        "FULL_STATE_CANONICAL_SHA256",
    )

    loaded = model.load_state_dict(state, strict=True)
    require(
        not loaded.missing_keys and not loaded.unexpected_keys,
        "STRICT_LOAD_KEYS",
    )

    for parameter in model.mamba.parameters():
        parameter.requires_grad = False
    require(
        not any(p.requires_grad for p in model.mamba.parameters()),
        "BACKBONE_NOT_FROZEN",
    )

    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    model.eval()

    require(
        all(
            parameter.device == device
            for parameter in model.mamba.parameters()
        ),
        "MAMBA_DEVICE",
    )

    return model, kernels, {
        "snapshot": snapshot_identity,
        "kernel_constructor_calls": dict(counts),
        "compact_checkpoint_sha256": COMPACT_CHECKPOINT_SHA256,
        "source_full_checkpoint_sha256": SOURCE_FULL_CHECKPOINT_SHA256,
        "pretrained_backbone_state_stream_sha256": PRETRAINED_BACKBONE_STATE_STREAM_SHA256,
        "downstream_state_canonical_sha256": DOWNSTREAM_STATE_CANONICAL_SHA256,
        "full_state_canonical_sha256": FULL_STATE_CANONICAL_SHA256,
    }


def runtime_components(model: Any) -> dict[str, Any]:
    backbone = getattr(model, "mamba", None)
    require(backbone is not None, "MAMBA_BACKBONE_MISSING")
    layers = getattr(backbone, "layers", None)
    require(layers is not None and len(layers) == LAYER_COUNT, "LAYER_COUNT")

    source = layers[SOURCE_BLOCK]
    intervention = layers[INTERVENTION_LAYER]
    source_mixer = getattr(source, "mixer", None)
    mixer = getattr(intervention, "mixer", None)
    norm = getattr(intervention, "norm", None)
    require(source_mixer is not None, "SOURCE_MIXER")
    require(mixer is not None, "INTERVENTION_MIXER")
    require(norm is not None, "INTERVENTION_NORM")

    in_proj = getattr(mixer, "in_proj", None)
    conv = getattr(mixer, "conv1d", None)
    require(in_proj is not None, "IN_PROJ")
    require(conv is not None, "CONV1D")
    require(
        tuple(in_proj.weight.shape)
        == (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        f"IN_PROJ_SHAPE:{tuple(in_proj.weight.shape)}",
    )

    partition = strong_partition(conv.weight)
    mask = partition["mask"]

    w_hidden = (
        in_proj.weight[:INTERMEDIATE_SIZE]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    gamma = (
        norm.weight.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    require(tuple(gamma.shape) == (HIDDEN_SIZE,), "GAMMA_SHAPE")

    return {
        "source_layer": source,
        "source_mixer": source_mixer,
        "intervention_layer": intervention,
        "intervention_mixer": mixer,
        "intervention_norm": norm,
        "w_hidden": w_hidden,
        "gamma": gamma,
        "strong_mask": mask,
        "partition": {
            key: value
            for key, value in partition.items()
            if key != "mask"
        },
    }


def _finite64(value: Any, label: str) -> torch.Tensor:
    require(torch.is_tensor(value), f"{label}_NOT_TENSOR")
    out = value.detach().cpu().to(torch.float64).contiguous()
    require(bool(torch.isfinite(out).all().item()), f"{label}_NONFINITE")
    return out


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    xx = _finite64(x, "COS_X")
    yy = _finite64(y, "COS_Y")
    require(xx.shape == yy.shape and xx.numel() > 0, "COS_SHAPE")
    a = float(torch.linalg.vector_norm(xx).item())
    b = float(torch.linalg.vector_norm(yy).item())
    require(a > 0.0 and b > 0.0, "COS_ZERO_NORM")
    value = float(torch.dot(xx, yy).item() / (a * b))
    require(math.isfinite(value), "COS_NONFINITE")
    require(-1.0 - 1e-12 <= value <= 1.0 + 1e-12, "COS_RANGE")
    return min(1.0, max(-1.0, value))


def reconstruct_pair_geometry(
    plus: Mapping[str, Any],
    minus: Mapping[str, Any],
    *,
    gamma: torch.Tensor,
    w_hidden: torch.Tensor,
    strong_mask: torch.Tensor,
) -> dict[str, Any]:
    rp = _finite64(plus["R"], "R_PLUS")
    rm = _finite64(minus["R"], "R_MINUS")
    yp = _finite64(plus["Y"], "Y_PLUS")
    ym = _finite64(minus["Y"], "Y_MINUS")
    xp = _finite64(plus["X"], "X_PLUS")
    xm = _finite64(minus["X"], "X_MINUS")
    gamma64 = _finite64(gamma, "GAMMA")
    w = _finite64(w_hidden, "W_HIDDEN")
    mask = strong_mask.detach().cpu().bool().contiguous()

    require(tuple(rp.shape) == (HIDDEN_SIZE,), "R_SHAPE")
    require(tuple(yp.shape) == (HIDDEN_SIZE,), "Y_SHAPE")
    require(tuple(xp.shape) == (HIDDEN_SIZE,), "X_SHAPE")
    require(tuple(gamma64.shape) == (HIDDEN_SIZE,), "GAMMA_SHAPE")
    require(
        tuple(w.shape) == (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "W_HIDDEN_SHAPE",
    )
    require(mask.numel() == INTERMEDIATE_SIZE, "MASK_WIDTH")
    require(0 < int(mask.sum().item()) < INTERMEDIATE_SIZE, "MASK_COUNT")

    sp = float(plus["rms_scale"])
    sm = float(minus["rms_scale"])
    require(math.isfinite(sp) and math.isfinite(sm), "RMS_SCALE")
    sbar = 0.5 * (sp + sm)

    dr = rp - rm
    dy = yp - ym
    dx = xp - xm

    d = float(torch.linalg.vector_norm(dx).item())
    require(math.isfinite(d) and d > 0.0, "DELTA_X_ZERO")

    q_r = gamma64 * (sbar * dr)
    q_y = gamma64 * (sbar * dy)

    h_r = torch.mv(w, q_r)
    h_y = torch.mv(w, q_y)

    x = (h_r[mask] / d).contiguous()
    y = (h_y[mask] / d).contiguous()

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ZERO_GEOMETRY_NORM")

    c = cosine(x, y)
    interaction = 2.0 * float(torch.dot(x, y).item())
    require(
        abs(interaction - 2.0 * a * b * c) <= 5e-11,
        "INTERACTION_CLOSURE",
    )

    return {
        "x": x,
        "y": y,
        "d": d,
        "A": a,
        "B": b,
        "C": c,
        "I": interaction,
    }


def alignment_delta(
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    target_cosine: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = _finite64(x_target, "ALIGN_X")
    y = _finite64(y_target, "ALIGN_Y")
    require(x.shape == y.shape and x.numel() > 0, "ALIGN_SHAPE")

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ALIGN_ZERO_NORM")
    require(math.isfinite(float(target_cosine)), "TARGET_C_NONFINITE")
    require(-1.0 <= float(target_cosine) <= 1.0, "TARGET_C_RANGE")

    c = cosine(x, y)
    u = x / a
    raw = y / b - c * u
    n = float(torch.linalg.vector_norm(raw).item())
    require(math.isfinite(n) and n > 1e-12, "DEGENERATE_ORTHOGONAL_COMPONENT")
    v = raw / n

    ct = float(target_cosine)
    y_star = b * (
        ct * u
        + math.sqrt(max(0.0, 1.0 - ct * ct)) * v
    )
    realized = cosine(x, y_star)
    require(abs(realized - ct) <= 5e-11, "ALIGN_TARGET_COSINE")
    require(
        abs(float(torch.linalg.vector_norm(y_star).item()) - b) <= 5e-11,
        "ALIGN_B_PRESERVATION",
    )

    return (y_star - y).contiguous(), {
        "baseline_A": a,
        "baseline_B": b,
        "baseline_C": c,
        "target_C": ct,
        "realized_C": realized,
    }


def _take_token(full: Any, target_abs: int, label: str) -> torch.Tensor:
    require(torch.is_tensor(full), f"{label}_NOT_TENSOR")
    value = full.detach()
    require(value.ndim == 3 and value.shape[0] == 1, f"{label}_SHAPE")
    require(0 <= target_abs < value.shape[1], f"{label}_TARGET_RANGE")
    result = value[0, target_abs, :].detach().cpu().contiguous().clone()
    require(bool(torch.isfinite(result).all().item()), f"{label}_NONFINITE")
    return result


def capture_geometry_branch(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    input_ids: torch.Tensor,
    anchor: int,
    device: torch.device,
) -> dict[str, Any]:
    target_abs = int(anchor) + TARGET_OFFSET
    require(tuple(input_ids.shape) == (1, 128), "INPUT_SHAPE")

    source_layer = runtime_ctx["source_layer"]
    source_mixer = runtime_ctx["source_mixer"]
    norm = runtime_ctx["intervention_norm"]
    mixer = runtime_ctx["intervention_mixer"]

    holders: dict[str, torch.Tensor] = {}
    counts = {"r": 0, "y": 0, "r_int": 0, "x": 0}

    def source_pre(_module, args):
        counts["r"] += 1
        require(counts["r"] == 1 and len(args) >= 1, "SOURCE_PRE_HOOK")
        holders["R"] = _take_token(args[0], target_abs, "R_FULL")

    def source_post(_module, _args, output):
        counts["y"] += 1
        require(counts["y"] == 1, "SOURCE_POST_HOOK")
        holders["Y"] = _take_token(output, target_abs, "Y_FULL")

    def norm_pre(_module, args):
        counts["r_int"] += 1
        require(counts["r_int"] == 1 and len(args) == 1, "NORM_PRE_HOOK")
        holders["R_INT"] = _take_token(args[0], target_abs, "R_INT_FULL")

    def norm_post(_module, _args, output):
        counts["x"] += 1
        require(counts["x"] == 1, "NORM_POST_HOOK")
        holders["X"] = _take_token(output, target_abs, "X_FULL")

    handles = [
        source_layer.register_forward_pre_hook(source_pre),
        source_mixer.register_forward_hook(source_post),
        norm.register_forward_pre_hook(norm_pre),
        norm.register_forward_hook(norm_post),
    ]

    original_cuda = mixer.cuda_kernels_forward
    fast_calls = {"count": 0}

    def cuda_wrapper(
        _self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        fast_calls["count"] += 1
        return original_cuda(
            hidden_states,
            cache_params,
            cache_position,
            attention_mask,
        )

    mixer.cuda_kernels_forward = types.MethodType(cuda_wrapper, mixer)

    try:
        model.mamba.eval()
        with torch.inference_mode():
            _ = model.mamba(
                input_ids=input_ids.detach().to(device).contiguous()
            )
        torch.cuda.synchronize(device)
    finally:
        for handle in reversed(handles):
            handle.remove()
        if "cuda_kernels_forward" in mixer.__dict__:
            del mixer.__dict__["cuda_kernels_forward"]

    require(
        counts == {"r": 1, "y": 1, "r_int": 1, "x": 1},
        f"HOOK_COUNTS:{counts}",
    )
    require(set(holders) == {"R", "Y", "R_INT", "X"}, "HOOK_CAPTURE")
    require(fast_calls["count"] == 1, f"CUDA_FAST_PATH_CALLS:{fast_calls['count']}")

    r_int = holders["R_INT"].to(torch.float64)
    eps = float(norm.variance_epsilon)
    scale = float(torch.rsqrt(r_int.pow(2).mean() + eps).item())
    require(math.isfinite(scale) and scale > 0.0, "RMS_SCALE")

    return {
        "R": holders["R"],
        "Y": holders["Y"],
        "X": holders["X"],
        "rms_scale": scale,
        "target_abs": target_abs,
        "fast_path_calls": 1,
    }


def _pair_order(family: str, rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    expected = tuple(
        f"{family}_fact_{index:03d}"
        for index in range(301, 601)
    )
    observed: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pair = str(row["source_pair_id"])
        if pair not in seen:
            seen.add(pair)
            observed.append(pair)
    require(tuple(observed) == expected, f"PAIR_ORDER:{family}")
    return expected


def _row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in out, f"DUPLICATE_ROW_KEY:{key}")
        out[key] = index
    return out


def compute_anchor_events(
    family: str,
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    from scripts import (
        reason_router_gen4_xg2_xg4_fresh_response_tokenizer_anchor_eligibility
        as eligibility,
    )

    inherited = eligibility.inherited
    failures = inherited.topology_failures(family, facts)
    require(not failures, f"ANCHOR_TOPOLOGY:{family}:{failures}")

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }
    require(len(facts_by_id) == SOURCE_PAIR_COUNT, "FACT_ID_COUNT")

    events: list[dict[str, Any]] = []
    for row in rows:
        pair = str(row["source_pair_id"])
        require(pair in facts_by_id, f"MISSING_FACT:{pair}")
        events.extend(
            dict(value)
            for value in inherited.analyze_row(
                family,
                row,
                facts_by_id[pair],
                tokenizer,
            )
        )

    require(len(events) == 1800, f"ANCHOR_EVENT_COUNT:{family}:{len(events)}")
    require(
        all(bool(row["post4_eligible"]) for row in events),
        f"ANCHOR_INELIGIBLE:{family}",
    )

    lookup = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        ): row
        for row in events
    }
    require(len(lookup) == 1800, f"ANCHOR_LOOKUP_COUNT:{family}")

    for pair in facts_by_id:
        for cell in ("C0_SHAM", "C2_NAME"):
            identity = lookup[(pair, cell, "A_IDENTITY")]
            name = lookup[(pair, cell, "A_NAME")]
            require(
                int(identity["absolute_anchor_token_index"])
                == int(name["absolute_anchor_token_index"]),
                f"IDENTITY_NAME_ANCHOR_MISMATCH:{family}:{pair}:{cell}",
            )

    return events, lookup


def load_family_inputs(
    family: str,
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Any],
]:
    from scripts import (
        reason_router_gen4_xg2_xg4_fresh_response_tokenizer_anchor_eligibility
        as eligibility,
    )
    from scripts import (
        reason_router_gen4_six_cell_tier2_inference_adapter
        as adapter,
    )

    facts, rows, manifest = eligibility.load_family(family, ROOT)
    frozen = COHORT_IDENTITIES[family]
    require(
        manifest["source_file_sha256"] == frozen["source_sha256"],
        f"COHORT_SOURCE:{family}",
    )
    require(
        manifest["row_file_sha256"] == frozen["rows_sha256"],
        f"COHORT_ROWS:{family}",
    )

    tokenizer, tokenizer_provenance = load_tokenizer(snapshot)
    encoded = adapter.encode_gen4_rows(rows, tokenizer)

    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in rows],
        f"ENCODED_PAIR_ORDER:{family}",
    )
    require(
        list(encoded["contrast_cell_id"])
        == [str(row["contrast_cell_id"]) for row in rows],
        f"ENCODED_CELL_ORDER:{family}",
    )
    require(
        tuple(encoded["input_ids"].shape) == (1800, 128),
        f"ENCODED_INPUT_SHAPE:{family}",
    )

    _events, lookup = compute_anchor_events(
        family,
        facts,
        rows,
        tokenizer,
    )

    _pair_order(family, rows)

    return rows, encoded, lookup, tokenizer_provenance


def run_family(
    *,
    family: str,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    snapshot: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows, encoded, events, _tokenizer_provenance = load_family_inputs(
        family,
        snapshot,
    )
    pairs = _pair_order(family, rows)
    row_index = _row_index(rows)

    items: list[dict[str, Any]] = []
    plans: list[torch.Tensor] = []
    forward_count = 0

    cells = {
        "tp": TARGET_PLUS_CELL,
        "tm": TARGET_MINUS_CELL,
        "rp": REFERENCE_PLUS_CELL,
        "rm": REFERENCE_MINUS_CELL,
    }

    for pair in pairs:
        branches: dict[str, dict[str, Any]] = {}
        anchors: dict[str, int] = {}

        for role, cell in cells.items():
            event = events[(pair, cell, ANCHOR_NAME)]
            anchor = int(event["absolute_anchor_token_index"])
            anchors[role] = anchor

            index = row_index[(pair, cell)]
            input_ids = encoded["input_ids"][index].unsqueeze(0)
            branches[role] = capture_geometry_branch(
                model=model,
                runtime_ctx=runtime_ctx,
                input_ids=input_ids,
                anchor=anchor,
                device=device,
            )
            forward_count += 1

        target = reconstruct_pair_geometry(
            branches["tp"],
            branches["tm"],
            gamma=runtime_ctx["gamma"],
            w_hidden=runtime_ctx["w_hidden"],
            strong_mask=runtime_ctx["strong_mask"],
        )
        reference = reconstruct_pair_geometry(
            branches["rp"],
            branches["rm"],
            gamma=runtime_ctx["gamma"],
            w_hidden=runtime_ctx["w_hidden"],
            strong_mask=runtime_ctx["strong_mask"],
        )

        delta_norm, alignment = alignment_delta(
            target["x"],
            target["y"],
            float(reference["C"]),
        )
        delta_h = (
            float(target["d"]) * delta_norm
        ).detach().cpu().to(torch.float64).contiguous()

        require(
            int(delta_h.numel()) == int(runtime_ctx["strong_mask"].sum().item()),
            f"PLAN_WIDTH:{family}:{pair}",
        )
        require(
            bool(torch.isfinite(delta_h).all().item()),
            f"PLAN_NONFINITE:{family}:{pair}",
        )

        plans.append(delta_h)
        items.append({
            "schema_version": "gen4-mamba790m-geometry-family-item-v1",
            "family_key": family,
            "source_pair_id": pair,
            "source_block": SOURCE_BLOCK,
            "target_residual_layer": TARGET_RESIDUAL_LAYER,
            "intervention_layer": INTERVENTION_LAYER,
            "relative_coordinate": TARGET_OFFSET,
            "target_plus_anchor": anchors["tp"],
            "target_minus_anchor": anchors["tm"],
            "reference_plus_anchor": anchors["rp"],
            "reference_minus_anchor": anchors["rm"],
            "target_A": float(target["A"]),
            "target_B": float(target["B"]),
            "target_C": float(target["C"]),
            "reference_A": float(reference["A"]),
            "reference_B": float(reference["B"]),
            "reference_C": float(reference["C"]),
            "target_d": float(target["d"]),
            "alignment_target_C": float(alignment["target_C"]),
            "alignment_realized_C": float(alignment["realized_C"]),
            "alignment_delta_h_l2": float(
                torch.linalg.vector_norm(delta_h).item()
            ),
            "model_forward_count": FORWARDS_PER_PAIR,
            "cuda_fast_path_required": True,
            "response_observed": False,
            "xg1_accessed": False,
        })

    require(forward_count == FORWARDS_PER_FAMILY, f"FORWARD_COUNT:{family}")
    plan = torch.stack(plans, dim=0).to(torch.float64).contiguous()
    require(
        tuple(plan.shape)
        == (
            SOURCE_PAIR_COUNT,
            int(runtime_ctx["strong_mask"].sum().item()),
        ),
        f"PLAN_SHAPE:{family}:{tuple(plan.shape)}",
    )
    return items, plan


def canonicalize_eigenvector_signs(
    basis: torch.Tensor,
) -> torch.Tensor:
    out = basis.detach().cpu().to(torch.float64).contiguous().clone()
    require(out.ndim == 2, "BASIS_SHAPE")
    for column in range(int(out.shape[1])):
        vector = out[:, column]
        pivot = int(torch.argmax(torch.abs(vector)).item())
        value = float(vector[pivot].item())
        require(math.isfinite(value) and value != 0.0, "BASIS_SIGN_PIVOT")
        if value < 0.0:
            out[:, column].mul_(-1.0)
    return out.contiguous()


def reconstruct_family_basis(
    family: str,
    plans: torch.Tensor,
) -> dict[str, Any]:
    require(
        plans.ndim == 2
        and int(plans.shape[0]) == SOURCE_PAIR_COUNT
        and plans.dtype == torch.float64,
        f"PLAN_INPUT:{family}:{tuple(plans.shape)}:{plans.dtype}",
    )
    x = plans.detach().cpu().to(torch.float64).contiguous()
    require(bool(torch.isfinite(x).all().item()), f"PLAN_NONFINITE:{family}")

    norms = torch.linalg.vector_norm(x, ord=2, dim=1)
    require(bool(torch.all(norms > 0.0).item()), f"PLAN_ZERO_NORM:{family}")
    unit = x / norms[:, None]

    second = (unit.T @ unit) / float(SOURCE_PAIR_COUNT)
    second = second.to(torch.float64).contiguous()
    symmetry = float(torch.max(torch.abs(second - second.T)).item())
    require(symmetry <= 1e-12, f"SECOND_MOMENT_SYMMETRY:{family}:{symmetry}")

    values_asc, vectors_asc = torch.linalg.eigh(second)
    require(
        bool(torch.isfinite(values_asc).all().item())
        and bool(torch.isfinite(vectors_asc).all().item()),
        f"BASIS_EIGH_NONFINITE:{family}",
    )

    order = torch.arange(
        int(values_asc.numel()) - 1,
        -1,
        -1,
        dtype=torch.long,
    )
    values = values_asc[order].contiguous()
    vectors = vectors_asc[:, order].contiguous()
    require(int(values.numel()) >= K + 1, f"BASIS_EIGENVALUE_COUNT:{family}")

    selected = values[:K].clone()
    gaps = values[:K] - values[1:K + 1]
    require(bool(torch.isfinite(gaps).all().item()), f"BASIS_GAP_NONFINITE:{family}")
    min_gap = float(torch.min(gaps).item())
    require(
        min_gap > BASIS_EIGENGAP_TOL,
        f"EIGENGAP_NOT_STRICT:{family}:{min_gap}",
    )

    basis = canonicalize_eigenvector_signs(vectors[:, :K])
    gram = basis.T @ basis
    identity = torch.eye(K, dtype=torch.float64)
    residual = float(torch.max(torch.abs(gram - identity)).item())
    require(
        residual <= ORTHONORMALITY_TOL,
        f"BASIS_ORTHONORMALITY:{family}:{residual}",
    )

    return {
        "basis": basis,
        "top5_eigenvalues": [float(v) for v in selected.tolist()],
        "top5_to_6_eigengaps": [float(v) for v in gaps.tolist()],
        "minimum_selected_eigengap": min_gap,
        "orthonormality_max_abs_residual": residual,
    }


def principal_pair(
    b2: torch.Tensor,
    b4: torch.Tensor,
    zero_based_index: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    u, singular, vh = torch.linalg.svd(
        b2.T @ b4,
        full_matrices=False,
    )
    a = (b2 @ u[:, zero_based_index]).contiguous()
    b = (b4 @ vh.T[:, zero_based_index]).contiguous()
    c = float(singular[zero_based_index].item())
    require(0.0 <= c <= 1.0 + 1e-12, f"PRINCIPAL_C:{zero_based_index}:{c}")
    c = min(1.0, max(0.0, c))
    s = math.sqrt(max(0.0, 1.0 - c * c))
    require(s > 0.0, f"DEGENERATE_PRINCIPAL_PLANE:{zero_based_index}")
    return a, b, c, s


def span_2d_projector_contrast_eigenvectors(
    a: torch.Tensor,
    b: torch.Tensor,
    c: float,
    s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    e1 = a
    e2 = (b - c * a) / s

    matrix = torch.tensor(
        [
            [s * s, -c * s],
            [-c * s, -s * s],
        ],
        dtype=torch.float64,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    require(
        bool(torch.isfinite(eigenvalues).all().item())
        and bool(torch.isfinite(eigenvectors).all().item()),
        "PLANE_EIGH_NONFINITE",
    )

    minus = (
        e1 * eigenvectors[0, 0]
        + e2 * eigenvectors[1, 0]
    ).to(torch.float64).contiguous()
    plus = (
        e1 * eigenvectors[0, 1]
        + e2 * eigenvectors[1, 1]
    ).to(torch.float64).contiguous()

    return plus, minus


def build_projector_geometry(
    b2: torch.Tensor,
    b4: torch.Tensor,
) -> dict[str, Any]:
    require(
        b2.ndim == 2
        and b4.ndim == 2
        and b2.shape == b4.shape
        and int(b2.shape[1]) == K,
        "FAMILY_BASIS_SHAPE",
    )
    width = int(b2.shape[0])

    p2 = b2 @ b2.T
    p4 = b4 @ b4.T
    contrast = (p2 - p4).to(torch.float64).contiguous()
    symmetry = float(
        torch.max(torch.abs(contrast - contrast.T)).item()
    )
    require(symmetry <= PLANE_TOL, f"PROJECTOR_CONTRAST_SYMMETRY:{symmetry}")

    planes: dict[str, dict[str, Any]] = {}
    columns: list[torch.Tensor] = []
    lambdas: list[float] = []

    for zero_index in range(K):
        plane = zero_index + 1
        a, b, c, s = principal_pair(b2, b4, zero_index)
        plus, minus = span_2d_projector_contrast_eigenvectors(
            a, b, c, s
        )

        plus_residual = float(
            torch.linalg.vector_norm(contrast @ plus - s * plus).item()
        )
        minus_residual = float(
            torch.linalg.vector_norm(contrast @ minus + s * minus).item()
        )
        require(
            plus_residual <= PLANE_TOL,
            f"PLUS_EIGENVECTOR_RESIDUAL:P{plane}:{plus_residual}",
        )
        require(
            minus_residual <= PLANE_TOL,
            f"MINUS_EIGENVECTOR_RESIDUAL:P{plane}:{minus_residual}",
        )

        plus_norm = float(torch.linalg.vector_norm(plus).item())
        minus_norm = float(torch.linalg.vector_norm(minus).item())
        dot = float(torch.dot(plus, minus).item())
        require(abs(plus_norm - 1.0) <= PLANE_TOL, f"PLUS_NORM:P{plane}")
        require(abs(minus_norm - 1.0) <= PLANE_TOL, f"MINUS_NORM:P{plane}")
        require(abs(dot) <= PLANE_TOL, f"PLANE_DOT:P{plane}")

        plus_raw = raw_f64le(plus)
        minus_raw = raw_f64le(minus)

        planes[f"P{plane}"] = {
            "scientific_principal_pair_number": plane,
            "zero_based_principal_pair_index": zero_index,
            "principal_cosine": c,
            "lambda_plus": s,
            "lambda_minus": -s,
            "plus_sha256": sha256_bytes(plus_raw),
            "minus_sha256": sha256_bytes(minus_raw),
            "plus_eigenvector_residual_l2": plus_residual,
            "minus_eigenvector_residual_l2": minus_residual,
            "plus": plus,
            "minus": minus,
            "plus_raw": plus_raw,
            "minus_raw": minus_raw,
        }
        columns.extend([plus, minus])
        lambdas.append(s)

    stacked = torch.stack(columns, dim=1)
    gram = stacked.T @ stacked
    identity = torch.eye(2 * K, dtype=torch.float64)
    gram_residual = float(torch.max(torch.abs(gram - identity)).item())
    require(
        gram_residual <= 5e-10,
        f"PRINCIPAL_10_VECTOR_GRAM:{gram_residual}",
    )

    # Also authenticate the 10 nonzero eigenvalues of the rebuilt projector
    # contrast without using them to reorder the prospectively defined planes.
    eigenvalues = torch.linalg.eigvalsh(contrast)
    nonzero = eigenvalues[torch.abs(eigenvalues) > 1e-10]
    require(int(nonzero.numel()) == 2 * K, f"PROJECTOR_NONZERO_RANK:{nonzero.numel()}")

    return {
        "width": width,
        "projector_contrast": contrast,
        "projector_contrast_symmetry_max_abs_residual": symmetry,
        "planes": planes,
        "lambda_plus_by_plane": lambdas,
        "full_principal_10_vector_gram_max_abs_residual": gram_residual,
        "projector_contrast_nonzero_eigenvalues_ascending": [
            float(v) for v in nonzero.tolist()
        ],
    }


def _worker_payload_paths(temp_dir: Path, gpu_id: int) -> dict[str, Path]:
    return {
        "meta": temp_dir / f"worker_{gpu_id}_meta.json",
        "items": temp_dir / f"worker_{gpu_id}_items.jsonl",
        "plan": temp_dir / f"worker_{gpu_id}_plan.pt",
        "error": temp_dir / f"worker_{gpu_id}_error.txt",
    }


def worker_run(
    *,
    gpu_id: int,
    family: str,
    expected_head: str,
    snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
) -> None:
    paths = _worker_payload_paths(Path(temp_dir), gpu_id)
    try:
        require(family == FAMILIES[gpu_id], "GPU_FAMILY_ASSIGNMENT")
        validate_protocol_constants()
        authenticate_repo(expected_head)
        validate_fast_runtime_for_device(gpu_id)

        device = torch.device(f"cuda:{gpu_id}")
        snapshot_path = Path(snapshot)
        checkpoint_path = Path(compact_checkpoint)

        model, kernels, model_provenance = reconstruct_model(
            snapshot=snapshot_path,
            compact_checkpoint=checkpoint_path,
            gpu_id=gpu_id,
        )

        from scripts import (
            reason_router_gen4_generator_family_prevalence_kernel_compat
            as kernel_compat,
        )
        kernel_compat.validate_transformers_kernel_bindings(kernels)

        ctx = runtime_components(model)

        items, plan = run_family(
            family=family,
            model=model,
            runtime_ctx=ctx,
            snapshot=snapshot_path,
            device=device,
        )
        require(len(items) == SOURCE_PAIR_COUNT, "WORKER_ITEM_COUNT")

        torch.save(plan, paths["plan"])
        paths["items"].write_bytes(jsonl_bytes(items))

        tokenizer, tokenizer_provenance = load_tokenizer(snapshot_path)
        del tokenizer

        meta = {
            "schema_version": "gen4-mamba790m-geometry-worker-v1",
            "gpu_id": gpu_id,
            "family_key": family,
            "device_name": torch.cuda.get_device_name(gpu_id),
            "pair_first": items[0]["source_pair_id"],
            "pair_last": items[-1]["source_pair_id"],
            "pair_count": len(items),
            "model_forward_count": FORWARDS_PER_FAMILY,
            "cuda_fast_path_required": True,
            "partition": ctx["partition"],
            "plan_shape": list(plan.shape),
            "plan_sha256": sha256_file(paths["plan"]),
            "items_sha256": sha256_file(paths["items"]),
            "model_provenance": model_provenance,
            "tokenizer_provenance": tokenizer_provenance,
            "xg1_accessed": False,
            "response_observed": False,
        }
        paths["meta"].write_bytes(pretty_json_bytes(meta))
    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        output.append(value)
    return output


def write_output_bundle(
    *,
    output_dir: Path,
    expected_head: str,
    snapshot: Path,
    worker_payloads: Sequence[Mapping[str, Any]],
    plans: Mapping[str, torch.Tensor],
    items: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(len(worker_payloads) == GPU_COUNT, "WORKER_PAYLOAD_COUNT")

    partitions = [payload["partition"] for payload in worker_payloads]
    first_partition = partitions[0]
    for other in partitions[1:]:
        require(
            other["strong_count"] == first_partition["strong_count"],
            "STRONG_COUNT_SHARD_MISMATCH",
        )
        require(
            other["strong_index_sha256"]
            == first_partition["strong_index_sha256"],
            "STRONG_MASK_HASH_SHARD_MISMATCH",
        )
        require(
            other["strong_indices"] == first_partition["strong_indices"],
            "STRONG_INDICES_SHARD_MISMATCH",
        )
        require(
            math.isclose(
                float(other["mu_k2"]),
                float(first_partition["mu_k2"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "STRONG_MU_SHARD_MISMATCH",
        )

    tokenizer_provenances = []
    for payload in worker_payloads:
        require(payload["model_forward_count"] == FORWARDS_PER_FAMILY, "WORKER_BUDGET")
        require(payload["cuda_fast_path_required"] is True, "FAST_PATH_FLAG")
        require(payload["xg1_accessed"] is False, "XG1_ACCESS")
        require(payload["response_observed"] is False, "RESPONSE_ACCESS")
        require(
            payload["model_provenance"]["full_state_canonical_sha256"]
            == FULL_STATE_CANONICAL_SHA256,
            "FULL_STATE_SHARD_MISMATCH",
        )
        tokenizer_provenances.append(payload["tokenizer_provenance"])

    require(
        tokenizer_provenances[0] == tokenizer_provenances[1],
        "TOKENIZER_PROVENANCE_SHARD_MISMATCH",
    )

    b2_info = reconstruct_family_basis("xg2", plans["xg2"])
    b4_info = reconstruct_family_basis("xg4", plans["xg4"])
    b2 = b2_info["basis"]
    b4 = b4_info["basis"]
    geometry = build_projector_geometry(b2, b4)

    strong_count = int(first_partition["strong_count"])
    require(int(b2.shape[0]) == strong_count, "XG2_BASIS_WIDTH")
    require(int(b4.shape[0]) == strong_count, "XG4_BASIS_WIDTH")

    output_dir.mkdir(parents=True, exist_ok=False)

    file_hashes: dict[str, str] = {}

    for family in FAMILIES:
        plan_name = f"{family}_alignment_delta_h.pt"
        torch.save(plans[family], output_dir / plan_name)
        file_hashes[plan_name] = sha256_file(output_dir / plan_name)

        items_name = f"{family}_geometry_items.jsonl"
        (output_dir / items_name).write_bytes(jsonl_bytes(items[family]))
        file_hashes[items_name] = sha256_file(output_dir / items_name)

    for family, basis in (("xg2", b2), ("xg4", b4)):
        name = f"{family}_basis.f64le"
        (output_dir / name).write_bytes(raw_f64le(basis))
        file_hashes[name] = sha256_file(output_dir / name)

    (output_dir / PROJECTOR_CONTRAST_FILE).write_bytes(
        raw_f64le(geometry["projector_contrast"])
    )
    file_hashes[PROJECTOR_CONTRAST_FILE] = sha256_file(
        output_dir / PROJECTOR_CONTRAST_FILE
    )

    plane_manifest: dict[str, Any] = {}
    for plane_name, entry in geometry["planes"].items():
        for sign in ("plus", "minus"):
            name = f"{plane_name.lower()}_{sign}.f64le"
            (output_dir / name).write_bytes(entry[f"{sign}_raw"])
            file_hashes[name] = sha256_file(output_dir / name)

        plane_manifest[plane_name] = {
            key: value
            for key, value in entry.items()
            if key not in {"plus", "minus", "plus_raw", "minus_raw"}
        }

    strong_indices = {
        "schema_version": "gen4-mamba790m-strong-mask-v1",
        "rule": "k^2 > mean(k^2) within intervention layer conv1d lag0 channel",
        "intervention_layer": INTERVENTION_LAYER,
        "lag0_kernel_index": LAG0_KERNEL_INDEX,
        **first_partition,
    }
    (output_dir / STRONG_INDEX_FILE).write_bytes(
        pretty_json_bytes(strong_indices)
    )
    file_hashes[STRONG_INDEX_FILE] = sha256_file(
        output_dir / STRONG_INDEX_FILE
    )

    summary = {
        "schema_version": "gen4-mamba790m-geometry-preparation-summary-v1",
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "phase": "cross_backbone_geometry_preparation",
        "claim_boundary": (
            "Independent 1.4B geometry reconstruction only; "
            "no XG1 discovery/confirmation response observed."
        ),
        "layer_mapping": {
            "formula": "round((17/(24-1))*(48-1))",
            "homologous_intervention_layer": INTERVENTION_LAYER,
            "preserved_local_offsets": list(LOCAL_LAYER_OFFSETS),
            "source_block": SOURCE_BLOCK,
            "target_residual_layer": TARGET_RESIDUAL_LAYER,
            "intervention_layer": INTERVENTION_LAYER,
        },
        "model": {
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "snapshot_file_identities": {
                name: {
                    "bytes": size,
                    "sha256": digest,
                }
                for name, (size, digest) in HF_FILES.items()
            },
            "compact_checkpoint_sha256": COMPACT_CHECKPOINT_SHA256,
            "source_full_checkpoint_sha256": SOURCE_FULL_CHECKPOINT_SHA256,
            "tokenizer": tokenizer_provenances[0],
            "training_execution_commit": TRAINING_EXECUTION_COMMIT,
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "selected_epoch": SELECTED_EPOCH,
            "arm": ARM,
        },
        "strong_mask": {
            key: value
            for key, value in first_partition.items()
            if key != "strong_indices"
        },
        "strong_index_file": STRONG_INDEX_FILE,
        "families": {
            "xg2": {
                "cohort_identity": COHORT_IDENTITIES["xg2"],
                "pair_first": "xg2_fact_301",
                "pair_last": "xg2_fact_600",
                "source_pair_count": SOURCE_PAIR_COUNT,
                "model_forward_count": FORWARDS_PER_FAMILY,
                "plan_shape": list(plans["xg2"].shape),
                "plan_file": "xg2_alignment_delta_h.pt",
                "plan_sha256": file_hashes["xg2_alignment_delta_h.pt"],
                "basis_shape": list(b2.shape),
                "basis_file": "xg2_basis.f64le",
                "basis_sha256": file_hashes["xg2_basis.f64le"],
                "basis_reconstruction": {
                    key: value
                    for key, value in b2_info.items()
                    if key != "basis"
                },
            },
            "xg4": {
                "cohort_identity": COHORT_IDENTITIES["xg4"],
                "pair_first": "xg4_fact_301",
                "pair_last": "xg4_fact_600",
                "source_pair_count": SOURCE_PAIR_COUNT,
                "model_forward_count": FORWARDS_PER_FAMILY,
                "plan_shape": list(plans["xg4"].shape),
                "plan_file": "xg4_alignment_delta_h.pt",
                "plan_sha256": file_hashes["xg4_alignment_delta_h.pt"],
                "basis_shape": list(b4.shape),
                "basis_file": "xg4_basis.f64le",
                "basis_sha256": file_hashes["xg4_basis.f64le"],
                "basis_reconstruction": {
                    key: value
                    for key, value in b4_info.items()
                    if key != "basis"
                },
            },
        },
        "projector_contrast": {
            "file": PROJECTOR_CONTRAST_FILE,
            "sha256": file_hashes[PROJECTOR_CONTRAST_FILE],
            "shape": [strong_count, strong_count],
            "symmetry_max_abs_residual":
                geometry["projector_contrast_symmetry_max_abs_residual"],
            "nonzero_eigenvalues_ascending":
                geometry["projector_contrast_nonzero_eigenvalues_ascending"],
        },
        "principal_planes": plane_manifest,
        "lambda_plus_by_plane": geometry["lambda_plus_by_plane"],
        "full_principal_10_vector_gram_max_abs_residual":
            geometry["full_principal_10_vector_gram_max_abs_residual"],
        "parallelization": {
            "gpu_count": GPU_COUNT,
            "gpu0_family": "xg2",
            "gpu1_family": "xg4",
            "independent_processes": True,
            "cuda_fast_path_mandatory": True,
            "shards": list(worker_payloads),
        },
        "scientific_model_forward_count_this_run": TOTAL_FORWARD_BUDGET,
        "xg1_model_forward_count": 0,
        "xg1_discovery_accessed": False,
        "xg1_confirmation_accessed": False,
        "causal_response_observed": False,
        "plane_selection_performed": False,
        "control_selection_performed": False,
        "statistical_testing_performed": False,
        "training_executed": False,
        "backward_executed": False,
        "rescue_performed": False,
    }
    (output_dir / SUMMARY_FILE).write_bytes(pretty_json_bytes(summary))
    file_hashes[SUMMARY_FILE] = sha256_file(output_dir / SUMMARY_FILE)

    manifest = {
        "schema_version": "gen4-mamba790m-geometry-preparation-manifest-v1",
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "output_file_sha256": dict(sorted(file_hashes.items())),
        "scientific_model_forward_count": TOTAL_FORWARD_BUDGET,
        "xg1_accessed": False,
        "response_observed": False,
        "training_executed": False,
        "backward_executed": False,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))
    file_hashes[MANIFEST_FILE] = sha256_file(output_dir / MANIFEST_FILE)

    checksum_lines = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(file_hashes.items())
    )
    (output_dir / CHECKSUM_FILE).write_text(
        checksum_lines,
        encoding="utf-8",
        newline="\n",
    )

    return summary


def run_geometry_preparation(
    *,
    expected_head: str,
    snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol_constants()
    authenticate_repo(expected_head)
    validate_snapshot(snapshot)

    require(
        compact_checkpoint.resolve()
        == (ROOT / COMPACT_CHECKPOINT_REL).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        sha256_file(compact_checkpoint)
        == COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba790m_geometry_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for gpu_id, family in enumerate(FAMILIES):
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "gpu_id": gpu_id,
                    "family": family,
                    "expected_head": expected_head,
                    "snapshot": str(snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                },
                name=f"mamba790m-geometry-{family}-gpu{gpu_id}",
            )
            process.start()
            processes.append(process)

        for gpu_id, process in enumerate(processes):
            process.join()
            if process.exitcode != 0:
                paths = _worker_payload_paths(temp_dir, gpu_id)
                detail = (
                    paths["error"].read_text(encoding="utf-8")
                    if paths["error"].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise GeometryPreparationError(
                    f"WORKER_{gpu_id}_FAILED:\n{detail}"
                )

        worker_payloads: list[dict[str, Any]] = []
        plans: dict[str, torch.Tensor] = {}
        items: dict[str, list[dict[str, Any]]] = {}

        for gpu_id, family in enumerate(FAMILIES):
            paths = _worker_payload_paths(temp_dir, gpu_id)
            require(paths["meta"].is_file(), f"WORKER_META_MISSING:{gpu_id}")
            require(paths["plan"].is_file(), f"WORKER_PLAN_MISSING:{gpu_id}")
            require(paths["items"].is_file(), f"WORKER_ITEMS_MISSING:{gpu_id}")

            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            require(meta["gpu_id"] == gpu_id, "WORKER_META_GPU")
            require(meta["family_key"] == family, "WORKER_META_FAMILY")
            require(meta["plan_sha256"] == sha256_file(paths["plan"]), "WORKER_PLAN_SHA")
            require(meta["items_sha256"] == sha256_file(paths["items"]), "WORKER_ITEMS_SHA")
            worker_payloads.append(meta)

            try:
                plan = torch.load(
                    paths["plan"],
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                plan = torch.load(paths["plan"], map_location="cpu")
            require(torch.is_tensor(plan), f"PLAN_NOT_TENSOR:{family}")
            plans[family] = plan.to(torch.float64).contiguous()
            items[family] = read_jsonl(paths["items"])

        return write_output_bundle(
            output_dir=output_dir,
            expected_head=expected_head,
            snapshot=snapshot,
            worker_payloads=worker_payloads,
            plans=plans,
            items=items,
        )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent Mamba-790M XG2/XG4 geometry reconstruction. "
            "Uses two GPUs and requires the frozen CUDA fast path. "
            "No XG1 response is accessed."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=ROOT / COMPACT_CHECKPOINT_REL,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    summary = run_geometry_preparation(
        expected_head=args.expected_head,
        snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )

    print("RESULT=" + str(summary["result"]))
    print(
        "LAYER_MAPPING="
        f"{SOURCE_BLOCK},{TARGET_RESIDUAL_LAYER},{INTERVENTION_LAYER}"
    )
    print(
        "STRONG_COUNT="
        + str(summary["strong_mask"]["strong_count"])
    )
    print(
        "STRONG_INDEX_SHA256="
        + str(summary["strong_mask"]["strong_index_sha256"])
    )
    print(
        "XG2_BASIS_SHA256="
        + str(summary["families"]["xg2"]["basis_sha256"])
    )
    print(
        "XG4_BASIS_SHA256="
        + str(summary["families"]["xg4"]["basis_sha256"])
    )
    for plane in range(1, K + 1):
        item = summary["principal_planes"][f"P{plane}"]
        print(
            f"P{plane}_LAMBDA_PLUS="
            + format(float(item["lambda_plus"]), ".17g")
        )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("XG1_DISCOVERY_ACCESSED=False")
    print("XG1_CONFIRMATION_ACCESSED=False")
    print("CAUSAL_RESPONSE_OBSERVED=False")
    print("PLANE_SELECTION_PERFORMED=False")
    print("CUDA_FAST_PATH_MANDATORY=True")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


if __name__ == "__main__":
    main()
