#!/usr/bin/env python3
"""Gen5 G2 historical-pretrained-parent install/restore-only diagnostic.

Modes:
  preflight: stdlib/git/JSON; no torch import, model, checkpoint or GPU.
  install: explicitly authorized, pinned historical CUDA strict-load then
           exactly ONE 6201/FULL G2 installation and restoration. NO forward.

Do not use this file to claim pretrained numerical or task-level equivalence.
Do not modify legacy Phase2/Phase3A source code or frozen checkpoints.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

BASE_HEAD = "d1174fce8770c77d4e9ef62f08e0bfa1cbd863e2"
MODEL_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
PARENT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
PARENT_SIGNATURE = "b0b62100691686acf2597159add2a9e252a757f63f61ef52115136d9cf33a3c9"
STATE_SCHEMA_SHA256 = "c09ca5426a95ac9b0538e75cf287d27f346683b78c8fefe65b4e16e7c88367e3"
PARENT_BUFFER_SHA256 = hashlib.sha256(b"").hexdigest()
STRICT_EVIDENCE_REL = (
    "reports/reason_router_gen5_g2_strict_load_only_runs/"
    "gen5-g2-parent-strictload-fda31c0-r1/strict_load_identity.json"
)
STRICT_EVIDENCE_BLOB = "012e66e1040dbe7d405d0889f7781c55780a0650"
G1_LOADER_REL = "scripts/gen5_g1_frozen_checkpoint_installation_cpu_7e19ecb.py"
G1_LOADER_BLOB = "2a47b6278d966e23d4920e51396c91f7f4b184ee"
STRICT_LOADER_REL = "scripts/gen5_g2_historical_parent_strict_load_only_diagnostic_d11f355.py"
STRICT_LOADER_BLOB = "0171cf23883d832ffab3b7d8d54b441e5a2079a8"
OUTPUT_PARENT = "reports/reason_router_gen5_g2_parent_installation_only_runs"
OUTPUT_BASENAME = "install_restore_identity.json"
HELDOUT_SEED = 6201
ARM = "FULL"


def require(cond: bool, reason: str) -> None:
    if not cond:
        raise RuntimeError("BLOCKED:" + reason)


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       check=True, text=True)
    return p.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_sources_static(repo: Path, expected_head: str) -> None:
    """Safe in preflight: never import torch or the G1 runner."""
    require(git(repo, "rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git(repo, "branch", "--show-current") in
            ("", "gen5-causal-role-state-ownership"), "BRANCH_MISMATCH")
    require(git(repo, "status", "--porcelain") == "", "WORKTREE_DIRTY")
    p = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                        BASE_HEAD, expected_head], capture_output=True)
    require(p.returncode == 0, "BASE_HEAD_NOT_ANCESTOR")
    for path, expected_blob in (
        (STRICT_LOADER_REL, STRICT_LOADER_BLOB),
        (G1_LOADER_REL, G1_LOADER_BLOB),
        (STRICT_EVIDENCE_REL, STRICT_EVIDENCE_BLOB),
    ):
        require(git(repo, "rev-parse", f"HEAD:{path}") == expected_blob,
                f"FROZEN_BLOB_DRIFT:{path}")
    for path, names in (
        (G1_LOADER_REL, {"load_real_sources", "sha_tensor", "sha_file"}),
        ("src/contramamba/gen5_shared_bypass_g2_loaded_parent_bridge.py",
         {"install_g2_loaded_parent", "remove_g2_loaded_parent", "installed_g2_loaded_parent"}),
    ):
        tree = ast.parse((repo / path).read_text(encoding="utf-8"), filename=path)
        found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        require(names <= found, f"FUNCTION_CONTRACT:{path}")
    # Additional frozen 10-blob / loader AST checks, still stdlib-only.
    sys.path.insert(0, str(repo))
    from scripts import gen5_g2_historical_parent_strict_load_only_diagnostic_d11f355 as strict
    strict.check_repository(repo, expected_head)
    strict.check_loader_source_ast(repo)
    read_strict_evidence(repo)


def read_strict_evidence(repo: Path) -> dict:
    record = json.loads((repo / STRICT_EVIDENCE_REL).read_text(encoding="utf-8"))
    for key, value in (
        ("schema", "GEN5_G2_HISTORICAL_PARENT_STRICT_LOAD_ONLY_V1"),
        ("status", "PASS_REAL_PARENT_STRICT_LOAD_NO_FORWARD"),
        ("repo_head", "fda31c03751d9195851fd2f5a66e08c518703ed7"),
        ("parent_checkpoint_sha256", PARENT_SHA256),
        ("parent_parameter_signature", PARENT_SIGNATURE),
        ("parent_buffer_signature", PARENT_BUFFER_SHA256),
        ("state_schema_sha256", STATE_SCHEMA_SHA256),
        ("kernel_transport_status", "EXACT_FROZEN_BINARY_SHA256_MATCH"),
        ("correction_installed", False),
        ("model_forward", False),
        ("task_evaluation", False),
        ("training", False),
    ):
        require(record.get(key) == value, f"STRICT_EVIDENCE:{key}")
    require(record.get("layer_count") == 24 and record.get("state_tensor_count") == 278,
            "STRICT_EVIDENCE:MODEL_SHAPE")
    return record


def check_output_path(repo: Path, out: Path) -> str:
    require(not out.exists(), "OUTPUT_COLLISION")
    rel = out.resolve().relative_to(repo.resolve()) if out.resolve().is_relative_to(repo.resolve()) else None
    require(rel is not None, "OUTPUT_NOT_REPO_LOCAL")
    parts = rel.parts
    require(len(parts) == 4 and parts[:2] == ("reports", "reason_router_gen5_g2_parent_installation_only_runs")
            and re.fullmatch(r"[a-z0-9][a-z0-9-]*", parts[2]) is not None
            and parts[3] == OUTPUT_BASENAME, "OUTPUT_SCOPE_MISMATCH")
    relative_root = rel.parent.as_posix()
    require(os.environ.get("CONTRAMAMBA_RUN_ARTIFACT_ROOT", "").replace("\\", "/") == relative_root,
            "OUTPUT_RUN_ROOT_ENV_MISMATCH")
    return relative_root


def preflight(repo: Path, expected_head: str) -> None:
    check_sources_static(repo, expected_head)
    print("GEN5_G2_REAL_PARENT_INSTALL_STATIC_PREFLIGHT_PASS")
    print(f"HEAD={expected_head}; FROZEN_PARENT_PROVENANCE=PASS")
    print("STRICT_LOAD=NOT_EXECUTED; CORRECTION_INSTALLED=NO; GPU=NO")
    print("FORWARD=NO; TRAINING=NO; TASK_EVAL=NO; NETWORK=NO")


def installation_only(repo: Path, expected_head: str, snapshot: Path,
                      checkpoint: Path, out: Path, authorized: bool) -> None:
    require(authorized, "EXPLICIT_GPU_INSTALL_AUTHORIZATION_REQUIRED")
    check_sources_static(repo, expected_head)
    check_output_path(repo, out)
    require(snapshot.is_dir() and snapshot.name == MODEL_REVISION, "MODEL_REVISION")
    require(checkpoint.is_file() and sha256_file(checkpoint) == PARENT_SHA256,
            "PARENT_CHECKPOINT_SHA256")

    # Prevent model/kernel-network fallback; user already provisioned exact bytes.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo))
    import torch
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from scripts.gen5_g1_frozen_checkpoint_installation_cpu_7e19ecb import (
        load_real_sources, CKPT_SHA, CHECKPOINT_ROOT,
    )
    from contramamba.gen5_phase2_state_update_ownership import (
        load_frozen_owner_bases, parent_parameter_fingerprint,
        validate_runtime_mixer_dimensions,
    )
    from contramamba.gen5_shared_bypass_functional_core_diagnostic import (
        validate_frozen_sources, EXPECTED_SOURCE_TENSOR_SHA256,
    )
    from contramamba.gen5_shared_bypass_g1_installation_diagnostic import _native_buffer_sha256
    from contramamba.gen5_shared_bypass_g2_loaded_parent_bridge import (
        installed_g2_loaded_parent,
    )

    require(p2train.PARENT_CHECKPOINT_SHA256 == PARENT_SHA256, "PARENT_CONSTANT")
    verified_snapshot = p2train.resolve_exact_snapshot(snapshot)
    p2train.authenticate_checkpoint(checkpoint)
    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()

    def forbid_download(**_kwargs: Any) -> str:
        raise RuntimeError("BLOCKED:OFFLINE_KERNEL_DOWNLOAD_FORBIDDEN")
    kernel_compat._hub_snapshot_download = forbid_download
    kernels = kernel_compat.load_exact_fast_kernels()
    require(kernels["transport_identity_status"] == "EXACT_FROZEN_BINARY_SHA256_MATCH",
            "KERNEL_IDENTITY")
    with torch.no_grad():
        model, constructor_counts = p2train._load_parent_model(
            snapshot=verified_snapshot, checkpoint=checkpoint,
            device=torch.device("cuda:0"), kernel_compat=kernel_compat, kernels=kernels,
        )
        model.eval().requires_grad_(False)
        require(not model.training and not model.mamba.training, "PARENT_NOT_EVAL")
        require(model.mamba.config.use_cache is False, "PARENT_CACHE_ENABLED")
        require(len(model.mamba.layers) == 24, "LAYER_COUNT")
        require(all(p.device.type == "cuda" and not p.requires_grad for p in model.parameters()),
                "PARENT_PARAMETERS_NOT_FROZEN_CUDA")
        for idx, layer in enumerate(model.mamba.layers):
            require(getattr(layer.mixer, "layer_idx", None) == idx, f"LAYER_IDX:{idx}")
            validate_runtime_mixer_dimensions(layer.mixer)
        schema = sorted((k, list(v.shape), str(v.dtype)) for k, v in model.state_dict().items())
        schema_sha = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
        require(len(schema) == 278 and schema_sha == STATE_SCHEMA_SHA256, "STATE_SCHEMA")
        parent_signature = parent_parameter_fingerprint(model)
        parent_buffer = _native_buffer_sha256(model)
        require(parent_signature == PARENT_SIGNATURE, "PARENT_SIGNATURE")
        require(parent_buffer == PARENT_BUFFER_SHA256, "PARENT_BUFFER_SIGNATURE")

        r22, c22, _geometry = load_frozen_owner_bases(repo)
        sources = load_real_sources(repo, r22, c22)
        require(set(sources) == set(CKPT_SHA) == {6201, 6202, 6203}, "SOURCE_SEEDS")
        device = torch.device("cuda:0")
        for seed, source in sources.items():
            source.to(device=device).eval().requires_grad_(False)
        validate_frozen_sources(sources)
        layer_ids_before = tuple(id(x.mixer) for x in model.mamba.layers)
        original_native = model.mamba.layers[22].mixer
        with installed_g2_loaded_parent(
            model, source_corrections=sources,
            heldout_seed=HELDOUT_SEED, arm=ARM,
        ) as wrapper:
            require(model.mamba.layers[22].mixer is wrapper, "INSTALL_LAYER22_WRAPPER")
            require(wrapper.native_mixer is original_native, "INSTALL_NATIVE_IDENTITY")
            require(not wrapper.training and not any(p.requires_grad for p in model.parameters()),
                    "INSTALL_TRAINABLE_OR_TRAINING")
            require(wrapper._phase2_active_mask is None, "INSTALL_MASK_LEAK")
            # No model, backbone, native-mixer, or correction forward calls.
        require(model.mamba.layers[22].mixer is original_native,
                "RESTORE_NATIVE_IDENTITY")
        require(tuple(id(x.mixer) for x in model.mamba.layers) == layer_ids_before,
                "RESTORE_ALL_LAYER_IDENTITIES")
        require(parent_parameter_fingerprint(model) == parent_signature,
                "RESTORE_PARENT_PARAMETERS")
        require(_native_buffer_sha256(model) == parent_buffer,
                "RESTORE_PARENT_BUFFERS")
        validate_frozen_sources(sources)

    record = {
        "schema": "GEN5_G2_REAL_PARENT_INSTALL_RESTORE_ONLY_V1",
        "status": "PASS_PRETRAINED_PARENT_INSTALL_RESTORE_NO_FORWARD",
        "repo_head": expected_head,
        "parent_checkpoint_sha256": PARENT_SHA256,
        "parent_parameter_signature": parent_signature,
        "parent_buffer_signature": parent_buffer,
        "parent_state_schema_sha256": schema_sha,
        "g2_bridge_heldout_seed": HELDOUT_SEED,
        "g2_bridge_arm": ARM,
        "source_checkpoints_sha256": {str(s): CKPT_SHA[s] for s in sorted(CKPT_SHA)},
        "source_factor_sha256": {str(s): EXPECTED_SOURCE_TENSOR_SHA256[s]
                                for s in sorted(CKPT_SHA)},
        "kernel_transport_status": kernels["transport_identity_status"],
        "kernel_constructor_calls": constructor_counts,
        "runtime": runtime,
        "install_success": True,
        "restoration_success": True,
        "model_forward": False,
        "task_evaluation": False,
        "training": False,
        "network": False,
        "limitations": [
            "One 6201/FULL installation and restoration only, not all arm/seed conditions",
            "No pretrained numerical forward or task-equivalence evidence",
            "No correction functional preservation or causal mechanism evidence",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print("GEN5_G2_REAL_PARENT_INSTALL_RESTORE_ONLY_PASS")
    print(f"HEAD={expected_head}; PARENT_SIGNATURE=PASS; SOURCE_FACTOR_SHA256=PASS")
    print(f"HELDOUT_SEED={HELDOUT_SEED}; ARM={ARM}; INSTALL=PASS; RESTORE=PASS")
    print("FORWARD=NO; TRAINING=NO; TASK_EVAL=NO; NETWORK=NO")
    print(f"RESULT_SHA256={sha256_file(out)}")
    print(f"RESULT_JSON={out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "install"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--authorize-gpu-install", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    if args.mode == "preflight":
        require(not args.authorize_gpu_install, "PREFLIGHT_GPU_FLAG")
        require(args.snapshot is None and args.checkpoint is None and args.out is None,
                "PREFLIGHT_EXTRA_ARGS")
        preflight(repo, args.expected_head)
    else:
        require(args.snapshot is not None and args.checkpoint is not None and args.out is not None,
                "INSTALL_ARGS_REQUIRED")
        installation_only(repo, args.expected_head, args.snapshot,
                          args.checkpoint, args.out, args.authorize_gpu_install)


if __name__ == "__main__":
    main()
