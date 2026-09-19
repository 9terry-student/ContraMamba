from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

MODEL_REPO = "state-spaces/mamba-1.4b-hf"
MODEL_REVISION = "6e46eae61c27280517feef46f536d16b91076f08"
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

SNAPSHOT_FILES = {
    "config.json": {
        "bytes": 879,
        "sha256": "275174e4771bb65f70cb1ea03389c9711510438d7944885e8f02f525bec03a9d",
    },
    "generation_config.json": {
        "bytes": 137,
        "sha256": "248fa733db101c19a8e3c2f311a180fdd7c769323ec2873edf7060cec9f5ee32",
    },
    "model-00001-of-00002.safetensors": {
        "bytes": 4960287720,
        "sha256": "3a0351b2c6a044c5f19f0f8992362e57c0d6f71b0e0702d46f43f3d8a7120c84",
    },
    "model-00002-of-00002.safetensors": {
        "bytes": 528479144,
        "sha256": "6cd1ac80d0acd00e6b6afebad9690115c5d932009ba644a6c8af1e0aed8414de",
    },
    "model.safetensors.index.json": {
        "bytes": 38175,
        "sha256": "a4739638258ef26762ff808249c3888fdb8bd4a15404a0db4811feb5cf2444b5",
    },
    "tokenizer.json": {
        "bytes": 2113738,
        "sha256": "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
    },
    "tokenizer_config.json": {
        "bytes": 4793,
        "sha256": "3ba257483d22a5a84aab5465aa427e59bdaeb55f09fb14349e2d571ff67e8020",
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

    require(branch == EXPECTED_BRANCH, f"BRANCH:{branch}")
    require(head == expected_head, f"HEAD:{head}")
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
        "second_gpu_used_for_gradient_aggregation": False,
    }


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
    raw = value.view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def tensor_state_sha256(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        _update_tensor_state_digest(digest, name, tensor)
    return digest.hexdigest()


def iter_pinned_backbone_tensors(snapshot: Path) -> Iterator[tuple[str, torch.Tensor]]:
    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "SAFETENSORS_WEIGHT_MAP")

    by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(str(filename), []).append(str(key))

    for filename in sorted(by_file):
        with safe_open(snapshot / filename, framework="pt", device="cpu") as handle:
            for key in sorted(by_file[filename]):
                yield key, handle.get_tensor(key)


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
        full_key = f"mamba.{key}"
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
        "schema_version": "contramamba_mamba14b_compact_checkpoint_v1",
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
        "schema_version": "contramamba_mamba14b_compact_checkpoint_manifest_v1",
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
    require(report.get("best_epoch") is not None, "BEST_EPOCH_MISSING")
    require(report.get("select_metric") == "final_macro_f1", "SELECT_METRIC")
    p2 = report.get("reason_router_p2") or {}
    contract = p2.get("contract") or {}
    require(contract.get("arm") == ARM or contract.get("reason_router_arm") == ARM, f"ARM_CONTRACT:{contract}")
    return {
        "best_epoch": int(report["best_epoch"]),
        "best_dev_metrics": report.get("best_dev_metrics"),
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

    run_dir = ROOT / "reports" / "reason_router_gen4_mamba14b_training_runs" / args.run_name
    require(not run_dir.exists(), f"RUN_DIR_EXISTS:{run_dir}")
    run_dir.mkdir(parents=True)

    snapshot = resolve_exact_snapshot()

    from scripts import reason_router_gen4_generator_family_prevalence_kernel_compat as kernel_compat
    from scripts import train_controlled_v6b_minimal as trainer

    kernels = kernel_compat.load_exact_fast_kernels()
    require(
        kernels.get("transport_identity_status") == "EXACT_FROZEN_BINARY_SHA256_MATCH",
        "KERNEL_TRANSPORT_IDENTITY",
    )

    argv = trainer_argv(snapshot, run_dir)

    launcher_manifest = {
        "schema_version": "contramamba_mamba14b_training_launcher_v1",
        "expected_head": args.expected_head,
        "run_name": args.run_name,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "snapshot_files": SNAPSHOT_FILES,
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

    with kernel_compat.exact_transformers_kernel_loader(kernels):
        rc = trainer.main(argv)
    require(rc in (0, None), f"TRAINER_RETURN_CODE:{rc}")

    training_summary = validate_training_outputs(run_dir)

    compact_manifest = compact_selected_checkpoint(
        full_checkpoint=run_dir / "selected_checkpoint.pt",
        snapshot=snapshot,
        compact_checkpoint=run_dir / "selected_downstream_checkpoint.pt",
        manifest_path=run_dir / "compact_checkpoint_manifest.json",
        expected_head=args.expected_head,
    )

    print("RESULT=PASS_MAMBA14B_TRAINING_AND_COMPACTION")
    print("MODEL_REVISION=" + MODEL_REVISION)
    print("ARM=" + ARM)
    print("TRAINING_SEED=" + str(TRAINING_SEED))
    print("SPLIT_SEED=" + str(SPLIT_SEED))
    print("BEST_EPOCH=" + str(training_summary["best_epoch"]))
    print("COMPACT_CHECKPOINT_SHA256=" + compact_manifest["compact_checkpoint"]["sha256"])
    print("FULL_CHECKPOINT_PRESENT=" + str((run_dir / "selected_checkpoint.pt").exists()))
    print("EXTERNAL_EVALUATION_EXECUTED=False")


if __name__ == "__main__":
    main()
