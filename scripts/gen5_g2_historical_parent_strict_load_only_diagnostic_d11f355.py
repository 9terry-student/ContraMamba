#!/usr/bin/env python3
"""Gen5 G2 isolated historical-parent STRICT-LOAD-ONLY diagnostic.

Two explicitly separated modes:
  --mode preflight    : stdlib/git only; no torch import or checkpoint opening.
  --mode strict-load  : future explicit GPU authorization required; constructs
                        and strict-loads historical parent, NEVER calls forward.

Does not attach Gen5 correction, tokenize, train, evaluate, or write to repo.
Legacy Phase2/Phase3A source files remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

BASE_HEAD = "d11f355fe9d33aee36da4a6515dd80d87aec0010"
BRANCH = "gen5-causal-role-state-ownership"
MODEL_NAME = "state-spaces/mamba-130m-hf"
MODEL_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
PARENT_SHA = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
PARENT_SIGNATURE = "b0b62100691686acf2597159add2a9e252a757f63f61ef52115136d9cf33a3c9"
STATE_SCHEMA_SHA = "c09ca5426a95ac9b0538e75cf287d27f346683b78c8fefe65b4e16e7c88367e3"
SOURCE_BLOBS = {
    "scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py": "b3b52c4116ca142fc1e31c4bdda57828c610d6e7",
    "scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py": "4a8326c442b883510ba452f049c88f3163677fc5",
    "scripts/reason_router_gen4_six_cell_tier2_scientific_inference.py": "1d0ea63b46a6147bb3ab45cc1043244c42f562d3",
    "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py": "00a81ce6ec4ada4d5c0bf36418347b222543d174",
    "scripts/train_reason_router_gen5_phase2_state_update_ownership.py": "8f711776c6b8cab90fbdafcda166780f3295e4ae",
    "src/contramamba/modeling_v6b_minimal_gen3_grouped_snapshot.py": "e29d361d99579b0a86a626ebcdebd59ba465881c",
    "src/contramamba/gen5_phase2_state_update_ownership.py": "a5c18f5b5d597d9830c3c44a9299af8233677f37",
    "src/contramamba/gen5_shared_bypass_functional_core_diagnostic.py": "cc979b3d07c299420dfc00110c968068f56379e0",
    "src/contramamba/gen5_shared_bypass_g1_installation_diagnostic.py": "f9536e247e260b7751ca45e641acee643cb339e4",
    "src/contramamba/gen5_shared_bypass_g2_loaded_parent_bridge.py": "18d83dd187ba404f651013f79b6fce5654ed9be3",
}


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError("BLOCKED:" + why)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_repository(repo: Path, expected_head: str) -> None:
    require(git(repo, "rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git(repo, "branch", "--show-current") == BRANCH, "BRANCH_MISMATCH")
    require(git(repo, "status", "--porcelain") == "", "WORKTREE_DIRTY")
    ancestor = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE_HEAD, expected_head],
        capture_output=True, check=False,
    )
    require(ancestor.returncode == 0, "BASE_HEAD_NOT_ANCESTOR")
    for name, expected_blob in SOURCE_BLOBS.items():
        observed = git(repo, "rev-parse", f"HEAD:{name}")
        require(observed == expected_blob, f"FROZEN_SOURCE_BLOB_DRIFT:{name}")


def check_loader_source_ast(repo: Path) -> None:
    """Narrow structural source checks; not a runtime strict-load proof."""
    expectations = {
        "scripts/train_reason_router_gen5_phase2_state_update_ownership.py": {
            "_load_parent_model", "validate_cuda_runtime", "authenticate_checkpoint",
            "resolve_exact_snapshot",
        },
        "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py": {
            "strict_load_state_dict", "build_historical_model_from_backbone",
        },
        "scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py": {
            "load_exact_fast_kernels", "exact_transformers_kernel_loader",
            "validate_transformers_kernel_bindings",
        },
    }
    for name, expected in expectations.items():
        text = (repo / name).read_text(encoding="utf-8")
        module = ast.parse(text, filename=name)
        functions = {node.name for node in module.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        require(expected <= functions, f"SOURCE_FUNCTION_CONTRACT:{name}:{sorted(expected-functions)}")


def preflight(repo: Path, expected_head: str) -> None:
    check_repository(repo, expected_head)
    check_loader_source_ast(repo)
    print("GEN5_G2_STRICT_LOAD_ONLY_STATIC_PREFLIGHT_PASS")
    print(f"HEAD={expected_head}")
    print(f"FROZEN_BLOBS={len(SOURCE_BLOBS)}; LOADER_FUNCTION_CONTRACTS=PASS")
    print("STRICT_LOAD=NOT_EXECUTED; MODEL_CONSTRUCTED=NO; CHECKPOINT_OPENED=NO")
    print("FORWARD=NO; TASK_EVAL=NO; TRAINING=NO; GPU=NO; NETWORK=NO")


def strict_load_only(repo: Path, expected_head: str, snapshot: Path,
                     checkpoint: Path, out: Path, authorized: bool) -> None:
    require(authorized, "EXPLICIT_GPU_STRICT_LOAD_AUTHORIZATION_REQUIRED")
    check_repository(repo, expected_head)
    check_loader_source_ast(repo)
    require(not out.exists(), "OUTPUT_COLLISION")
    require(not out.resolve().is_relative_to(repo.resolve()), "OUTPUT_MUST_BE_OFF_REPO")
    require(snapshot.is_dir() and snapshot.name == MODEL_REVISION, "SNAPSHOT_REVISION")
    require(checkpoint.is_file(), "PARENT_CHECKPOINT_MISSING")
    require(sha256_file(checkpoint) == PARENT_SHA, "PARENT_SHA256")

    # Set offline mode before importing any hub-integrated libraries.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))

    import torch
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from contramamba.gen5_phase2_state_update_ownership import (
        parent_parameter_fingerprint, validate_runtime_mixer_dimensions,
    )
    from contramamba.gen5_shared_bypass_g1_installation_diagnostic import _native_buffer_sha256

    require(p2train.PARENT_CHECKPOINT_SHA256 == PARENT_SHA, "PARENT_CONSTANT_DRIFT")
    require(p2train.MODEL_REVISION == MODEL_REVISION, "MODEL_REVISION_DRIFT")
    require(p2train.MODEL_NAME == MODEL_NAME, "MODEL_ID_DRIFT")
    # Explicit paths: the existing helper then makes no snapshot network call.
    verified_snapshot = p2train.resolve_exact_snapshot(snapshot)
    p2train.authenticate_checkpoint(checkpoint)

    # The historical runtime gate is intentionally not bypassed.
    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()
    # This historical kernel resolver otherwise has a download fallback.
    # Replace that transport only; cached, SHA256-authenticated kernels remain usable.
    def forbid_download(**_kwargs: Any) -> str:
        raise RuntimeError("BLOCKED:OFFLINE_KERNEL_DOWNLOAD_FORBIDDEN")
    kernel_compat._hub_snapshot_download = forbid_download
    kernels = kernel_compat.load_exact_fast_kernels()
    require(kernels["transport_identity_status"] == "EXACT_FROZEN_BINARY_SHA256_MATCH",
            "KERNEL_BINARY_IDENTITY")

    with torch.no_grad():
        model, constructor_counts = p2train._load_parent_model(
            snapshot=verified_snapshot,
            checkpoint=checkpoint,
            device=torch.device("cuda:0"),
            kernel_compat=kernel_compat,
            kernels=kernels,
        )
        model.eval()
        model.requires_grad_(False)
        require(not model.training and not model.mamba.training, "PARENT_NOT_EVAL")
        require(not any(p.requires_grad for p in model.parameters()), "PARENT_TRAINABLE")
        require(all(p.device.type == "cuda" for p in model.parameters()), "PARENT_NOT_CUDA")
        require(model.mamba.config.use_cache is False, "CACHE_NOT_DISABLED")
        require(len(model.mamba.layers) == 24, "PARENT_LAYER_COUNT")
        for layer_idx in range(24):
            native = model.mamba.layers[layer_idx].mixer
            require(int(getattr(native, "layer_idx", -1)) == layer_idx,
                    f"NATIVE_LAYER_IDX:{layer_idx}")
            validate_runtime_mixer_dimensions(native)
        state = model.state_dict()
        require(len(state) == 278, "PARENT_STATE_TENSOR_COUNT")
        ordered_schema = sorted(
            (name, list(value.shape), str(value.dtype))
            for name, value in state.items()
        )
        observed_schema = hashlib.sha256(
            json.dumps(ordered_schema, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        require(observed_schema == STATE_SCHEMA_SHA, "PARENT_STATE_SCHEMA_IDENTITY")
        observed_signature = parent_parameter_fingerprint(model)
        require(observed_signature == PARENT_SIGNATURE, "PARENT_PARAMETER_SIGNATURE")
        buffer_signature = _native_buffer_sha256(model)

    # No wrapper installation, input batch, model.forward(), or task artifact.
    record = {
        "schema": "GEN5_G2_HISTORICAL_PARENT_STRICT_LOAD_ONLY_V1",
        "status": "PASS_REAL_PARENT_STRICT_LOAD_NO_FORWARD",
        "repo_head": expected_head,
        "parent_checkpoint_sha256": PARENT_SHA,
        "parent_parameter_signature": observed_signature,
        "parent_buffer_signature": buffer_signature,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "layer_count": 24,
        "state_tensor_count": 278,
        "state_schema_sha256": observed_schema,
        "kernel_constructor_calls": constructor_counts,
        "kernel_transport_status": kernels["transport_identity_status"],
        "runtime": {k: v for k, v in runtime.items()
                    if k not in {"gpu0_name"} or isinstance(v, str)},
        "correction_installed": False,
        "model_forward": False,
        "task_evaluation": False,
        "training": False,
        "network": False,
        "gpu_for_strict_load": True,
        "limitations": [
            "Strict state loading is not a pretrained numerical forward-equivalence test",
            "No layer22 intervention or correction has been installed",
            "No task-functional or causal-core claim can be inferred",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    print("GEN5_G2_HISTORICAL_PARENT_STRICT_LOAD_ONLY_PASS")
    print(f"HEAD={expected_head}; PARENT_SHA256=PASS; PARAMETER_SIGNATURE=PASS")
    print("LAYERS=24; STRICT_LOAD=PASS; FORWARD=NO; G2_BRIDGE_INSTALLED=NO")
    print("TRAINING=NO; TASK_EVAL=NO; NETWORK=NO; GPU=STRICT_LOAD_ONLY")
    print(f"RESULT_SHA256={sha256_file(out)}")
    print(f"RESULT_JSON={out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "strict-load"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--authorize-gpu-strict-load", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    if args.mode == "preflight":
        require(not args.authorize_gpu_strict_load, "PREFLIGHT_GPU_AUTHORIZATION_FORBIDDEN")
        require(args.snapshot is None and args.checkpoint is None and args.out is None,
                "PREFLIGHT_EXTRA_ARGUMENTS_FORBIDDEN")
        preflight(repo, args.expected_head)
    else:
        require(args.snapshot is not None and args.checkpoint is not None
                and args.out is not None, "STRICT_LOAD_INPUT_REQUIRED")
        strict_load_only(repo, args.expected_head, args.snapshot,
                         args.checkpoint, args.out, args.authorize_gpu_strict_load)


if __name__ == "__main__":
    main()
