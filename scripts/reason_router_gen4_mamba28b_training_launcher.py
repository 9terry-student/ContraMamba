from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

MODEL_REPO = "state-spaces/mamba-2.8b-hf"
MODEL_REVISION = "96c48e0292b63f5346b6d30061af2551f7101e26"
EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"

GEN3_GROUPED_SOURCE_COMMIT = (
    "3e0e9a435068c552abf20f3a74e0c3eccca344a3"
)
GEN3_GROUPED_TRAINER_GIT_BLOB = (
    "ae902c06bcef92a20c012b9c35ef0ee8c8478b9f"
)
GEN3_GROUPED_MODEL_GIT_BLOB = (
    "e29d361d99579b0a86a626ebcdebd59ba465881c"
)
GEN3_GROUPED_TRAINER_SNAPSHOT = (
    ROOT
    / "scripts"
    / "train_controlled_v6b_minimal_gen3_grouped_snapshot.py"
)
GEN3_GROUPED_MODEL_SNAPSHOT = (
    ROOT
    / "src"
    / "contramamba"
    / "modeling_v6b_minimal_gen3_grouped_snapshot.py"
)
GEN3_GROUPED_MODEL_MODULE = (
    "contramamba.modeling_v6b_minimal_gen3_grouped_snapshot"
)
GEN3_GROUPED_TRAINER_MODULE = (
    "scripts.train_controlled_v6b_minimal_gen3_grouped_snapshot"
)

SNAPSHOT_FILES = {
    "config.json": {
        "bytes": 843,
        "sha256": "ec7f74c5f322716a5a3470632ff03572182e7a5027cdcb452520818c5276c356",
    },
    "generation_config.json": {
        "bytes": 137,
        "sha256": "248fa733db101c19a8e3c2f311a180fdd7c769323ec2873edf7060cec9f5ee32",
    },
    "model-00001-of-00003.safetensors": {
        "bytes": 4969727736,
        "sha256": "8d0022682157b0684b40659c4cba75394c59494cdb4cb79b4890070e81dd7756",
    },
    "model-00002-of-00003.safetensors": {
        "bytes": 4949332368,
        "sha256": "c51a5400ee9560f5611b5c64011b554f5b37c3437061cd0fa20d387ff1ae9986",
    },
    "model-00003-of-00003.safetensors": {
        "bytes": 1154393144,
        "sha256": "2691db90d7e98237e1e834a30d6630e89150576f50514c6e929b8d1db4cc1a43",
    },
    "model.safetensors.index.json": {
        "bytes": 50864,
        "sha256": "95f3794b7eece8798b37eba26ea7d7105f034b062ff2ebd545a3e91a9314deed",
    },
    "tokenizer.json": {
        "bytes": 2113738,
        "sha256": "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
    },
    "tokenizer_config.json": {
        "bytes": 4793,
        "sha256": "027cbbe0813aefd46e0037bf828acf22b9b591ce845153ddbf409e99770a3a79",
    },
}

DATA_PATH = Path(
    "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_"
    "4122078ab7962042e3d6bf89f8b4eb5cec463458/"
    "controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
)
SIDECAR_PATH = Path(
    "reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_"
    "integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_"
    "149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/"
    "p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl"
)
SIDECAR_SEMANTIC_SHA256 = (
    "2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9"
)

ARM = "G3-GROUP-D-HALF"
EDGE_LAMBDAS = {
    "F_TO_P": 1.0,
    "F_TO_S": 1.0,
    "P_TO_S": 1.0,
    "F_TO_Q": 1.0,
    "P_TO_Q": 1.0,
    "S_TO_Q": 1.0,
    "F_TO_D": 0.5,
    "P_TO_D": 0.5,
    "S_TO_D": 0.5,
    "Q_TO_D": 0.5,
}

TRAINING_SEED = 181
SPLIT_SEED = 8192
EPOCHS = 20
LEARNING_RATE = 0.001
MAX_LENGTH = 128
FORWARD_MICROBATCH_SIZE = 32
EVAL_MICROBATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
TRAIN_ROW_COUNT = 2880
DEV_ROW_COUNT = 720
DUAL_GPU_CACHE_BATCH_SIZE = 64
DUAL_GPU_CACHE_DEVICE_IDS = (0, 1)

EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "torch": "2.10.0+cu128",
    "transformers": "5.0.0",
    "kernels": "0.10.2",
    "cuda_runtime": "12.8",
}


class TrainingLauncherError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingLauncherError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_repo(expected_head: str) -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        text=True,
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    ).strip()

    require(head == expected_head, f"HEAD:{head}")
    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH:{branch}",
    )
    require(not dirty, "DIRTY_WORKTREE")


def validate_runtime() -> dict[str, Any]:
    import transformers

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "kernels": importlib.metadata.version("kernels"),
        "cuda_runtime": torch.version.cuda,
    }
    require(observed == EXPECTED_RUNTIME, f"RUNTIME:{observed}")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= 2, f"CUDA_DEVICE_COUNT:{torch.cuda.device_count()}")
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    capabilities = [tuple(torch.cuda.get_device_capability(i)) for i in range(2)]
    require(names == ["Tesla T4", "Tesla T4"], f"GPU_NAMES:{names}")
    require(capabilities == [(7, 5), (7, 5)], f"GPU_CAPABILITIES:{capabilities}")
    return {
        **observed,
        "gpu_count_visible": torch.cuda.device_count(),
        "gpu_names_first_two": names,
        "gpu_capabilities_first_two": [list(v) for v in capabilities],
        "training_device": "cuda:0",
        "dual_gpu_frozen_encoder_cache": True,
        "dual_gpu_cache_device_ids": list(DUAL_GPU_CACHE_DEVICE_IDS),
        "dual_gpu_cache_batch_size": DUAL_GPU_CACHE_BATCH_SIZE,
        "second_gpu_role": "frozen_mamba_encoder_cache_forward",
        "second_gpu_used_for_gradient_aggregation": False,
    }


def dual_gpu_cache_slices(
    n_rows: int,
    batch_size: int = DUAL_GPU_CACHE_BATCH_SIZE,
) -> list[tuple[slice, slice]]:
    require(n_rows >= 0, f"CACHE_N_ROWS:{n_rows}")
    require(batch_size >= 2, f"CACHE_BATCH_SIZE:{batch_size}")
    result: list[tuple[slice, slice]] = []
    for start in range(0, n_rows, batch_size):
        end = min(n_rows, start + batch_size)
        mid = start + (end - start + 1) // 2
        result.append((slice(start, mid), slice(mid, end)))
    return result


def install_dual_gpu_frozen_encoder_cache(
    *,
    trainer: Any,
    snapshot: Path,
) -> tuple[dict[str, Any], Any]:
    from transformers import MambaConfig, MambaModel

    stats: dict[str, Any] = {
        "enabled": True,
        "device_ids": list(DUAL_GPU_CACHE_DEVICE_IDS),
        "global_cache_batch_size": DUAL_GPU_CACHE_BATCH_SIZE,
        "datasets_cached": 0,
        "gpu0_rows": 0,
        "gpu1_rows": 0,
        "gpu0_forward_calls": 0,
        "gpu1_forward_calls": 0,
        "cross_gpu_equivalence_checked": False,
        "cross_gpu_equivalence_exact": False,
        "secondary_model_loaded_from_exact_snapshot": False,
        "secondary_model_released_after_cache": False,
    }
    secondary_holder: dict[str, Any] = {"model": None}

    def load_secondary() -> Any:
        secondary = secondary_holder["model"]
        if secondary is not None:
            return secondary
        config = MambaConfig.from_pretrained(
            str(snapshot),
            local_files_only=True,
        )
        config.use_mamba_kernels = True
        with torch.cuda.device(1):
            secondary = MambaModel.from_pretrained(
                str(snapshot),
                config=config,
                local_files_only=True,
            ).to(torch.device("cuda:1"))
        for parameter in secondary.parameters():
            parameter.requires_grad_(False)
        secondary.eval()
        secondary_holder["model"] = secondary
        stats["secondary_model_loaded_from_exact_snapshot"] = True
        return secondary

    def verify_cross_gpu_exact(
        primary: Any,
        secondary: Any,
        probe_ids: torch.Tensor,
    ) -> None:
        if stats["cross_gpu_equivalence_checked"]:
            return
        n = min(4, int(probe_ids.shape[0]))
        require(n > 0, "CACHE_EQUIVALENCE_EMPTY_PROBE")
        ids0 = probe_ids[:n].to(torch.device("cuda:0"))
        ids1 = probe_ids[:n].to(torch.device("cuda:1"))
        primary.eval()
        secondary.eval()
        with torch.no_grad():
            out0 = primary(input_ids=ids0).last_hidden_state.detach().cpu()
            out1 = secondary(input_ids=ids1).last_hidden_state.detach().cpu()
        stats["cross_gpu_equivalence_checked"] = True
        stats["cross_gpu_equivalence_exact"] = bool(torch.equal(out0, out1))
        require(
            stats["cross_gpu_equivalence_exact"],
            "DUAL_GPU_CACHE_CROSS_GPU_TENSOR_MISMATCH",
        )

    def run_primary(primary: Any, ids: torch.Tensor) -> torch.Tensor:
        with torch.cuda.device(0), torch.no_grad():
            result = primary(
                input_ids=ids.to(torch.device("cuda:0"), non_blocking=True)
            ).last_hidden_state
            torch.cuda.synchronize(0)
            return result

    def run_secondary(secondary: Any, ids: torch.Tensor) -> torch.Tensor:
        with torch.cuda.device(1), torch.no_grad():
            result = secondary(
                input_ids=ids.to(torch.device("cuda:1"), non_blocking=True)
            ).last_hidden_state
            result0 = result.to(torch.device("cuda:0"), non_blocking=True)
            torch.cuda.synchronize(1)
            return result0

    def dual_gpu_cache(
        model: Any,
        inputs: dict[str, torch.Tensor],
        batch_size: int = 8,
    ) -> None:
        del batch_size
        require(
            not any(parameter.requires_grad for parameter in model.mamba.parameters()),
            "DUAL_GPU_CACHE_REQUIRES_FROZEN_ENCODER",
        )
        require(torch.cuda.device_count() >= 2, "DUAL_GPU_CACHE_REQUIRES_TWO_GPUS")

        primary = model.mamba
        primary.eval()
        secondary = load_secondary()
        verify_cross_gpu_exact(primary, secondary, inputs["input_ids"])

        chunks: list[torch.Tensor] = []
        plans = dual_gpu_cache_slices(
            int(inputs["input_ids"].shape[0]),
            DUAL_GPU_CACHE_BATCH_SIZE,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            for left, right in plans:
                ids_left = inputs["input_ids"][left]
                ids_right = inputs["input_ids"][right]

                future0 = (
                    pool.submit(run_primary, primary, ids_left)
                    if int(ids_left.shape[0]) > 0
                    else None
                )
                future1 = (
                    pool.submit(run_secondary, secondary, ids_right)
                    if int(ids_right.shape[0]) > 0
                    else None
                )

                out0 = future0.result() if future0 is not None else None
                out1 = future1.result() if future1 is not None else None

                if out0 is not None:
                    chunks.append(out0)
                    stats["gpu0_rows"] += int(ids_left.shape[0])
                    stats["gpu0_forward_calls"] += 1
                if out1 is not None:
                    chunks.append(out1)
                    stats["gpu1_rows"] += int(ids_right.shape[0])
                    stats["gpu1_forward_calls"] += 1

        require(chunks, "DUAL_GPU_CACHE_NO_CHUNKS")
        inputs["encoder_hidden_states"] = torch.cat(chunks, dim=0)
        require(
            int(inputs["encoder_hidden_states"].shape[0])
            == int(inputs["input_ids"].shape[0]),
            "DUAL_GPU_CACHE_ROW_COUNT_MISMATCH",
        )
        stats["datasets_cached"] += 1

        if stats["datasets_cached"] == 2:
            secondary_holder["model"] = None
            del secondary
            torch.cuda.synchronize(1)
            torch.cuda.empty_cache()
            stats["secondary_model_released_after_cache"] = True

    original = trainer.v5.cache_frozen_encoder_states
    trainer.v5.cache_frozen_encoder_states = dual_gpu_cache
    return stats, original


def restore_encoder_cache_function(*, trainer: Any, original: Any) -> None:
    trainer.v5.cache_frozen_encoder_states = original



def git_blob_identity(path: Path) -> str:
    require(path.is_file(), f"GIT_BLOB_FILE_MISSING:{path}")
    return subprocess.check_output(
        ["git", "hash-object", str(path)],
        cwd=ROOT,
        text=True,
    ).strip()


def load_exact_gen3_grouped_trainer() -> Any:
    require(
        git_blob_identity(GEN3_GROUPED_TRAINER_SNAPSHOT)
        == GEN3_GROUPED_TRAINER_GIT_BLOB,
        "GEN3_GROUPED_TRAINER_BLOB_MISMATCH",
    )
    require(
        git_blob_identity(GEN3_GROUPED_MODEL_SNAPSHOT)
        == GEN3_GROUPED_MODEL_GIT_BLOB,
        "GEN3_GROUPED_MODEL_BLOB_MISMATCH",
    )

    model_snapshot = importlib.import_module(
        GEN3_GROUPED_MODEL_MODULE
    )
    package = importlib.import_module("contramamba")

    canonical_name = "contramamba.modeling_v6b_minimal"
    sentinel = object()

    previous_module = sys.modules.get(canonical_name, sentinel)
    previous_attribute = getattr(
        package,
        "modeling_v6b_minimal",
        sentinel,
    )

    # Force a fresh import of the historical trainer under the exact
    # historical model binding.
    sys.modules.pop(GEN3_GROUPED_TRAINER_MODULE, None)
    sys.modules[canonical_name] = model_snapshot
    setattr(
        package,
        "modeling_v6b_minimal",
        model_snapshot,
    )

    try:
        trainer = importlib.import_module(
            GEN3_GROUPED_TRAINER_MODULE
        )
    finally:
        if previous_module is sentinel:
            sys.modules.pop(canonical_name, None)
        else:
            sys.modules[canonical_name] = previous_module

        if previous_attribute is sentinel:
            try:
                delattr(package, "modeling_v6b_minimal")
            except AttributeError:
                pass
        else:
            setattr(
                package,
                "modeling_v6b_minimal",
                previous_attribute,
            )

    require(
        getattr(trainer, "ContraMambaV6BMinimal", None)
        is model_snapshot.ContraMambaV6BMinimal,
        "GEN3_GROUPED_MODEL_CLASS_BINDING_MISMATCH",
    )
    require(
        ARM in getattr(trainer, "G3_GROUPED_ARM_IDS", ()),
        "GEN3_GROUPED_ARM_MISSING",
    )
    require(
        trainer.P2_ARM_CONTRACTS.get(ARM)
        == ("explicit_product", "edge_specific"),
        "GEN3_GROUPED_ARM_CONTRACT_MISMATCH",
    )
    require(
        tuple(
            trainer.G3_GROUPED_ARM_EDGE_SETS.get(
                ARM,
                (),
            )
        )
        == (
            "F_TO_D",
            "P_TO_D",
            "S_TO_D",
            "Q_TO_D",
        ),
        "GEN3_GROUPED_D_EDGE_SET_MISMATCH",
    )

    return trainer

def resolve_exact_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            allow_patterns=sorted(SNAPSHOT_FILES),
        )
    ).resolve()
    require(snapshot.name == MODEL_REVISION, f"SNAPSHOT_REVISION_PATH:{snapshot.name}")

    observed_names = {
        path.name for path in snapshot.iterdir()
        if path.is_file() and path.name in SNAPSHOT_FILES
    }
    require(observed_names == set(SNAPSHOT_FILES), f"SNAPSHOT_FILE_SET:{sorted(observed_names)}")

    for name, expected in SNAPSHOT_FILES.items():
        path = snapshot / name
        require(path.is_file(), f"SNAPSHOT_FILE_MISSING:{name}")
        require(path.stat().st_size == int(expected["bytes"]), f"SNAPSHOT_BYTES:{name}")
        require(sha256_file(path) == expected["sha256"], f"SNAPSHOT_SHA256:{name}")

    return snapshot


def trainer_argv(snapshot: Path, run_dir: Path) -> list[str]:
    edge_json = json.dumps(EDGE_LAMBDAS, separators=(",", ":"), sort_keys=False)
    return [
        "--data", DATA_PATH.as_posix(),
        "--architecture", "v6b_minimal",
        "--backbone", "mamba",
        "--model-name", str(snapshot),
        "--freeze-encoder", "true",
        "--frame-downstream-gradient-mode", "joint",
        "--epochs", str(EPOCHS),
        "--max-length", str(MAX_LENGTH),
        "--dev-ratio", "0.2",
        "--seed", str(TRAINING_SEED),
        "--split-seed", str(SPLIT_SEED),
        "--device", "cuda",
        "--flag-source", "controlled_heuristic",
        "--select-metric", "final_macro_f1",
        "--ranking-weight", "0.0",
        "--class-weighting", "none",
        "--stage174c-clean-pairwise-mode", "off",
        "--stage174c-clean-pairwise-weight", "0.0",
        "--stage174c-clean-polarity-preservation-weight", "0.0",
        "--stage175b-support-anchor-mode", "off",
        "--stage175b-support-anchor-weight", "0.0",
        "--stage177c-frame-pairwise-mode", "off",
        "--stage177c-frame-pairwise-weight", "0.0",
        "--compatible-positive-margin-logit", "0.0",
        "--compatible-positive-margin-weight", "0.0",
        "--lr", str(LEARNING_RATE),
        "--controlled-integrity-sidecar-path", SIDECAR_PATH.as_posix(),
        "--expected-integrity-sidecar-semantic-sha256", SIDECAR_SEMANTIC_SHA256,
        "--save-selected-checkpoint",
        "--selected-checkpoint-filename", "selected_checkpoint.pt",
        "--reason-router-arm", ARM,
        "--reason-router-mode", "explicit_product",
        "--gradient-ownership-mode", "edge_specific",
        "--edge-gradient-lambdas", edge_json,
        "--reason-loss-weight", "0.0",
        "--train-batch-size", str(FORWARD_MICROBATCH_SIZE),
        "--eval-batch-size", str(EVAL_MICROBATCH_SIZE),
        "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--output-json", str(run_dir / "training_report.json"),
        "--output-predictions-json", str(run_dir / "clean_dev_predictions.json"),
    ]


def _update_tensor_state_digest(
    digest: "hashlib._Hash",
    name: str,
    tensor: torch.Tensor,
) -> None:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def tensor_state_sha256(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        _update_tensor_state_digest(digest, name, tensor)
    return digest.hexdigest()


def iter_pinned_backbone_tensors(
    snapshot: Path,
) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    index_name = "model.safetensors.index.json"
    index_path = snapshot / index_name

    require(
        index_path.is_file(),
        "SAFETENSORS_INDEX_MISSING",
    )

    expected_index = SNAPSHOT_FILES[index_name]

    require(
        index_path.stat().st_size
        == int(expected_index["bytes"]),
        "SAFETENSORS_INDEX_BYTES",
    )
    require(
        sha256_file(index_path)
        == expected_index["sha256"],
        "SAFETENSORS_INDEX_SHA256",
    )

    index_payload = json.loads(
        index_path.read_text(encoding="utf-8")
    )
    require(
        isinstance(index_payload, dict),
        "SAFETENSORS_INDEX_PAYLOAD",
    )

    weight_map = index_payload.get("weight_map")
    require(
        isinstance(weight_map, dict) and bool(weight_map),
        "SAFETENSORS_INDEX_WEIGHT_MAP",
    )
    require(
        all(
            isinstance(key, str)
            and bool(key)
            and isinstance(shard, str)
            and bool(shard)
            for key, shard in weight_map.items()
        ),
        "SAFETENSORS_INDEX_WEIGHT_MAP_TYPES",
    )

    shard_names = sorted(set(weight_map.values()))
    expected_shards = sorted(
        name
        for name in SNAPSHOT_FILES
        if name.startswith("model-")
        and name.endswith(".safetensors")
    )

    require(
        shard_names == expected_shards,
        f"SAFETENSORS_SHARD_SET:{shard_names}",
    )

    observed_keys: set[str] = set()

    # Canonical pretrained stream order is deliberately:
    # safetensors filename, then tensor key.
    for shard_name in shard_names:
        shard_path = snapshot / shard_name
        expected = SNAPSHOT_FILES[shard_name]

        require(
            shard_path.is_file(),
            f"SAFETENSORS_SHARD_MISSING:{shard_name}",
        )
        require(
            shard_path.stat().st_size
            == int(expected["bytes"]),
            f"SAFETENSORS_SHARD_BYTES:{shard_name}",
        )
        require(
            sha256_file(shard_path)
            == expected["sha256"],
            f"SAFETENSORS_SHARD_SHA256:{shard_name}",
        )

        expected_keys = sorted(
            key
            for key, mapped_shard in weight_map.items()
            if mapped_shard == shard_name
        )
        require(
            bool(expected_keys),
            f"SAFETENSORS_SHARD_INDEX_EMPTY:{shard_name}",
        )

        with safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as handle:
            actual_keys = sorted(handle.keys())

            require(
                actual_keys == expected_keys,
                f"SAFETENSORS_SHARD_KEY_SET:{shard_name}",
            )

            for key in actual_keys:
                require(
                    key not in observed_keys,
                    f"SAFETENSORS_DUPLICATE_KEY:{key}",
                )
                observed_keys.add(key)
                yield key, handle.get_tensor(key)

    require(
        observed_keys == set(weight_map),
        "SAFETENSORS_INDEX_COVERAGE",
    )


def compact_selected_checkpoint(
    *,
    full_checkpoint: Path,
    snapshot: Path,
    compact_checkpoint: Path,
    manifest_path: Path,
    expected_head: str,
) -> dict[str, Any]:
    require(full_checkpoint.is_file(), f"FULL_CHECKPOINT_MISSING:{full_checkpoint}")
    full_sha = sha256_file(full_checkpoint)
    full_bytes = full_checkpoint.stat().st_size

    payload = torch.load(full_checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "FULL_CHECKPOINT_PAYLOAD")
    require(payload.get("schema_version") == "stage176a0_selected_checkpoint_v1", "FULL_CHECKPOINT_SCHEMA")
    full_state = payload.get("model_state_dict")
    metadata = payload.get("metadata")
    require(isinstance(full_state, dict) and full_state, "FULL_STATE_DICT")
    require(isinstance(metadata, dict), "FULL_METADATA")
    require(metadata.get("selected_epoch") is not None, "SELECTED_EPOCH_MISSING")

    full_backbone_keys = {key for key in full_state if key.startswith("mamba.")}
    observed_backbone_keys: set[str] = set()
    pretrained_stream_digest = hashlib.sha256()

    for key, reference in iter_pinned_backbone_tensors(snapshot):
        full_key = "mamba." + key.removeprefix("backbone.")
        require(full_key in full_state, f"FULL_BACKBONE_KEY_MISSING:{full_key}")
        observed = full_state[full_key]
        require(torch.is_tensor(observed), f"FULL_BACKBONE_NOT_TENSOR:{full_key}")
        require(torch.equal(observed.detach().cpu(), reference.detach().cpu()), f"BACKBONE_TENSOR_MISMATCH:{full_key}")
        observed_backbone_keys.add(full_key)
        _update_tensor_state_digest(pretrained_stream_digest, key, reference)

    require(full_backbone_keys == observed_backbone_keys, "BACKBONE_KEY_SET_MISMATCH")

    downstream = {
        key: value.detach().cpu().clone()
        for key, value in full_state.items()
        if not key.startswith("mamba.")
    }
    require(downstream, "DOWNSTREAM_STATE_EMPTY")

    full_state_hash = tensor_state_sha256(
        (key, value) for key, value in full_state.items() if torch.is_tensor(value)
    )
    downstream_hash = tensor_state_sha256(downstream.items())
    pretrained_stream_hash = pretrained_stream_digest.hexdigest()

    compact_payload = {
        "schema_version": "contramamba_mamba28b_compact_checkpoint_v1",
        "downstream_state_dict": downstream,
        "metadata": metadata,
        "pretrained_backbone": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "file_identities": SNAPSHOT_FILES,
            "state_stream_sha256": pretrained_stream_hash,
            "state_stream_order": "safetensors_filename_then_tensor_key",
            "state_key_count": len(observed_backbone_keys),
        },
        "source_full_checkpoint": {
            "bytes": full_bytes,
            "sha256": full_sha,
            "full_state_canonical_sha256": full_state_hash,
            "state_key_count": len(full_state),
        },
        "downstream": {
            "state_canonical_sha256": downstream_hash,
            "state_key_count": len(downstream),
            "tensor_bytes": sum(
                int(value.numel() * value.element_size()) for value in downstream.values()
            ),
        },
        "training_identity": {
            "arm": ARM,
            "execution_commit": expected_head,
            "selected_epoch": int(metadata["selected_epoch"]),
            "split_seed": SPLIT_SEED,
            "training_seed": TRAINING_SEED,
            "forward_microbatch_size": FORWARD_MICROBATCH_SIZE,
            "logical_full_batch_rows": TRAIN_ROW_COUNT,
            "optimizer_steps_per_epoch": 1,
        },
        "reconstruction_verification": {
            "key_set_exact": True,
            "tensor_equal_all_backbone_keys": True,
            "downstream_extracted_exactly_from_full_checkpoint": True,
            "scientific_forward_executed": False,
            "training_executed_during_compaction": False,
        },
    }

    compact_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temp = compact_checkpoint.with_suffix(compact_checkpoint.suffix + ".tmp")
    torch.save(compact_payload, temp)
    os.replace(temp, compact_checkpoint)

    reloaded = torch.load(compact_checkpoint, map_location="cpu", weights_only=False)
    require(reloaded.get("schema_version") == compact_payload["schema_version"], "COMPACT_RELOAD_SCHEMA")
    reloaded_downstream = reloaded.get("downstream_state_dict")
    require(isinstance(reloaded_downstream, dict), "COMPACT_RELOAD_DOWNSTREAM")
    require(set(reloaded_downstream) == set(downstream), "COMPACT_RELOAD_KEY_SET")
    for key in downstream:
        require(torch.equal(reloaded_downstream[key], downstream[key]), f"COMPACT_RELOAD_TENSOR:{key}")

    compact_sha = sha256_file(compact_checkpoint)
    manifest = {
        "schema_version": "contramamba_mamba28b_compact_checkpoint_manifest_v1",
        **{key: value for key, value in compact_payload.items() if key != "downstream_state_dict"},
        "compact_checkpoint": {
            "path": (
                compact_checkpoint.relative_to(ROOT).as_posix()
                if compact_checkpoint.is_relative_to(ROOT)
                else compact_checkpoint.as_posix()
            ),
            "bytes": compact_checkpoint.stat().st_size,
            "sha256": compact_sha,
        },
        "full_checkpoint_removed_after_verified_compaction": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    full_checkpoint.unlink()
    require(not full_checkpoint.exists(), "FULL_CHECKPOINT_DELETE_FAILED")
    return manifest


def validate_training_outputs(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "training_report.json"
    predictions_path = run_dir / "clean_dev_predictions.json"
    provenance_path = run_dir / "run_provenance.json"
    full_checkpoint = run_dir / "selected_checkpoint.pt"

    for path in (report_path, predictions_path, provenance_path, full_checkpoint):
        require(path.is_file(), f"TRAINING_OUTPUT_MISSING:{path.name}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = report.get("runs") or {}
    require(isinstance(runs, dict), f"RUNS_SCHEMA:{type(runs).__name__}")
    single = runs.get("single") or {}
    require(
        isinstance(single, dict),
        f"SINGLE_RUN_SCHEMA:{type(single).__name__}",
    )

    best_epoch = report.get("best_epoch")
    if best_epoch is None:
        best_epoch = single.get("best_epoch")
    require(best_epoch is not None, "BEST_EPOCH_MISSING")

    select_metric = report.get("select_metric")
    if select_metric is None:
        select_metric = single.get("select_metric")
    require(
        select_metric == "final_macro_f1",
        f"SELECT_METRIC:{select_metric}",
    )

    p2 = (
        report.get("reason_router_p2")
        or single.get("reason_router_p2")
        or {}
    )
    require(
        isinstance(p2, dict),
        f"REASON_ROUTER_P2_SCHEMA:{type(p2).__name__}",
    )
    contract = p2.get("contract") or {}
    require(
        isinstance(contract, dict),
        f"ARM_CONTRACT_SCHEMA:{type(contract).__name__}",
    )
    require(
        contract.get("arm") == ARM
        or contract.get("reason_router_arm") == ARM,
        f"ARM_CONTRACT:{contract}",
    )

    best_dev_metrics = report.get("best_dev_metrics")
    if best_dev_metrics is None:
        best_dev_metrics = single.get("best_dev_metrics")

    return {
        "best_epoch": int(best_epoch),
        "best_dev_metrics": best_dev_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authenticate_repo(args.expected_head)
    runtime = validate_runtime()

    run_dir = ROOT / "reports" / "reason_router_gen4_mamba28b_training_runs" / args.run_name
    require(not run_dir.exists(), f"RUN_DIR_EXISTS:{run_dir}")
    run_dir.mkdir(parents=True)

    snapshot = resolve_exact_snapshot()

    from scripts import reason_router_gen4_generator_family_prevalence_kernel_compat as kernel_compat
    trainer = load_exact_gen3_grouped_trainer()

    kernels = kernel_compat.load_exact_fast_kernels()
    require(
        kernels.get("transport_identity_status") == "EXACT_FROZEN_BINARY_SHA256_MATCH",
        "KERNEL_TRANSPORT_IDENTITY",
    )

    argv = trainer_argv(snapshot, run_dir)

    launcher_manifest = {
        "schema_version": "contramamba_mamba28b_training_launcher_v1",
        "expected_head": args.expected_head,
        "run_name": args.run_name,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "snapshot_files": SNAPSHOT_FILES,
        "gen3_grouped_training_source": {
            "historical_execution_commit": GEN3_GROUPED_SOURCE_COMMIT,
            "trainer_snapshot_path": (
                GEN3_GROUPED_TRAINER_SNAPSHOT
                .relative_to(ROOT)
                .as_posix()
            ),
            "trainer_git_blob": GEN3_GROUPED_TRAINER_GIT_BLOB,
            "model_snapshot_path": (
                GEN3_GROUPED_MODEL_SNAPSHOT
                .relative_to(ROOT)
                .as_posix()
            ),
            "model_git_blob": GEN3_GROUPED_MODEL_GIT_BLOB,
            "arm": ARM,
            "router_mode": "explicit_product",
            "gradient_ownership_mode": "edge_specific",
            "edge_gradient_lambdas": EDGE_LAMBDAS,
        },
        "runtime": runtime,
        "training_contract": {
            "arm": ARM,
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "class_weighting": "none",
            "forward_microbatch_size": FORWARD_MICROBATCH_SIZE,
            "eval_microbatch_size": EVAL_MICROBATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "logical_full_batch_rows": TRAIN_ROW_COUNT,
            "optimizer_steps_per_epoch": 1,
            "selection_metric": "final_macro_f1",
            "dual_gpu_frozen_encoder_cache": True,
            "dual_gpu_cache_batch_size": DUAL_GPU_CACHE_BATCH_SIZE,
            "dual_gpu_cache_device_ids": list(DUAL_GPU_CACHE_DEVICE_IDS),
            "dual_gpu_cache_expected_total_rows": TRAIN_ROW_COUNT + DEV_ROW_COUNT,
            "dual_gpu_cache_expected_rows_per_gpu": (
                (TRAIN_ROW_COUNT + DEV_ROW_COUNT) // 2
            ),
        },
        "trainer_argv": argv,
        "scientific_scope": "training_and_internal_clean_dev_checkpoint_selection_only",
        "external_evaluation_executed": False,
    }
    (run_dir / "training_launcher_manifest.json").write_text(
        json.dumps(launcher_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    cache_stats, original_cache_function = install_dual_gpu_frozen_encoder_cache(
        trainer=trainer,
        snapshot=snapshot,
    )
    try:
        with kernel_compat.exact_transformers_kernel_loader(kernels):
            rc = trainer.main(argv)
    finally:
        restore_encoder_cache_function(
            trainer=trainer,
            original=original_cache_function,
        )
    require(rc in (0, None), f"TRAINER_RETURN_CODE:{rc}")
    require(cache_stats["datasets_cached"] == 2, f"CACHE_DATASETS:{cache_stats}")
    require(cache_stats["cross_gpu_equivalence_exact"] is True, f"CACHE_EQUIVALENCE:{cache_stats}")
    require(
        cache_stats["gpu0_rows"] + cache_stats["gpu1_rows"]
        == TRAIN_ROW_COUNT + DEV_ROW_COUNT,
        f"CACHE_TOTAL_ROWS:{cache_stats}",
    )
    require(
        cache_stats["gpu0_rows"] == cache_stats["gpu1_rows"]
        == (TRAIN_ROW_COUNT + DEV_ROW_COUNT) // 2,
        f"CACHE_GPU_ROW_BALANCE:{cache_stats}",
    )
    require(cache_stats["gpu0_forward_calls"] > 0, f"CACHE_GPU0_UNUSED:{cache_stats}")
    require(cache_stats["gpu1_forward_calls"] > 0, f"CACHE_GPU1_UNUSED:{cache_stats}")
    require(
        cache_stats["secondary_model_released_after_cache"] is True,
        f"CACHE_SECONDARY_NOT_RELEASED:{cache_stats}",
    )
    (run_dir / "dual_gpu_encoder_cache_stats.json").write_text(
        json.dumps(cache_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    training_summary = validate_training_outputs(run_dir)

    compact_manifest = compact_selected_checkpoint(
        full_checkpoint=run_dir / "selected_checkpoint.pt",
        snapshot=snapshot,
        compact_checkpoint=run_dir / "selected_downstream_checkpoint.pt",
        manifest_path=run_dir / "compact_checkpoint_manifest.json",
        expected_head=args.expected_head,
    )

    print("RESULT=PASS_MAMBA28B_TRAINING_AND_COMPACTION")
    print("MODEL_REVISION=" + MODEL_REVISION)
    print("ARM=" + ARM)
    print("TRAINING_SEED=" + str(TRAINING_SEED))
    print("SPLIT_SEED=" + str(SPLIT_SEED))
    print("BEST_EPOCH=" + str(training_summary["best_epoch"]))
    print("DUAL_GPU_ENCODER_CACHE=PASS")
    print("GPU0_CACHE_ROWS=" + str(cache_stats["gpu0_rows"]))
    print("GPU1_CACHE_ROWS=" + str(cache_stats["gpu1_rows"]))
    print("GPU0_CACHE_FORWARD_CALLS=" + str(cache_stats["gpu0_forward_calls"]))
    print("GPU1_CACHE_FORWARD_CALLS=" + str(cache_stats["gpu1_forward_calls"]))
    print("CROSS_GPU_CACHE_EQUIVALENCE_EXACT=" + str(cache_stats["cross_gpu_equivalence_exact"]))
    print("COMPACT_CHECKPOINT_SHA256=" + compact_manifest["compact_checkpoint"]["sha256"])
    print("FULL_CHECKPOINT_PRESENT=" + str((run_dir / "selected_checkpoint.pt").exists()))
    print("EXTERNAL_EVALUATION_EXECUTED=False")


if __name__ == "__main__":
    main()
