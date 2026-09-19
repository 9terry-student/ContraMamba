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
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

MODEL_REPO = "state-spaces/mamba-1.4b-hf"
DISCOVERY_REVISION = "main"

EXPECTED_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 4096,
    "num_hidden_layers": 48,
    "state_size": 16,
    "conv_kernel": 4,
    "vocab_size": 50280,
}

SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

GPU_COUNT = 2
SEQ_LEN = 128
PROBE_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
ARM = "G3-GROUP-D-HALF"

RESULT_PASS = "PASS_MAMBA14B_RUNTIME_FEASIBILITY_RAW_OBSERVATION"
REPORT_SCHEMA = "gen4-mamba14b-runtime-feasibility-v1"


class FeasibilityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FeasibilityError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FeasibilityError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(status == "", "WORKTREE_NOT_CLEAN")


def validate_config_dict(config: Mapping[str, Any]) -> None:
    for key, expected in EXPECTED_CONFIG.items():
        observed = int(config[key])
        require(
            observed == expected,
            f"CONFIG_MISMATCH:{key}:{observed}:{expected}",
        )


def snapshot_identity(snapshot: Path) -> dict[str, Any]:
    require(snapshot.is_dir(), "SNAPSHOT_DIR_MISSING")

    config_path = snapshot / "config.json"
    require(config_path.is_file(), "CONFIG_MISSING")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config_dict(config)

    files = {}
    for name in SNAPSHOT_FILES:
        path = snapshot / name
        require(path.is_file(), f"SNAPSHOT_FILE_MISSING:{name}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    resolved_revision = snapshot.name
    require(
        len(resolved_revision) == 40
        and all(c in "0123456789abcdef" for c in resolved_revision.lower()),
        f"RESOLVED_REVISION_NOT_SHA:{resolved_revision}",
    )

    return {
        "repo": MODEL_REPO,
        "requested_revision": DISCOVERY_REVISION,
        "resolved_revision": resolved_revision,
        "config": {
            key: int(config[key])
            for key in EXPECTED_CONFIG
        },
        "files": files,
        "total_snapshot_bytes": sum(v["bytes"] for v in files.values()),
    }


def resolve_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=DISCOVERY_REVISION,
            allow_patterns=list(SNAPSHOT_FILES),
            local_files_only=False,
        )
    ).resolve()

    snapshot_identity(path)
    return path


def runtime_identity() -> dict[str, Any]:
    import importlib.metadata
    import transformers

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "kernels": importlib.metadata.version("kernels"),
        "cuda_runtime": torch.version.cuda,
    }


def _worker_paths(temp_dir: Path, worker_id: int) -> dict[str, Path]:
    return {
        "report": temp_dir / f"worker_{worker_id}.json",
        "error": temp_dir / f"worker_{worker_id}.error.txt",
    }


def _bytes_of_parameters(parameters: Sequence[torch.nn.Parameter]) -> int:
    return sum(
        int(p.numel()) * int(p.element_size())
        for p in parameters
    )


def _synthetic_inputs(
    batch_size: int,
    *,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    base = torch.arange(
        batch_size * SEQ_LEN,
        dtype=torch.long,
        device=device,
    ).reshape(batch_size, SEQ_LEN)
    input_ids = (base % max(vocab_size - 1, 1)) + 1

    attention_mask = torch.ones(
        (batch_size, SEQ_LEN),
        dtype=torch.long,
        device=device,
    )
    claim_mask = torch.zeros(
        (batch_size, SEQ_LEN),
        dtype=torch.bool,
        device=device,
    )
    evidence_mask = torch.zeros_like(claim_mask)

    split = SEQ_LEN // 2
    claim_mask[:, :split] = True
    evidence_mask[:, split:] = True

    flags = torch.zeros(
        batch_size,
        dtype=torch.float32,
        device=device,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
        "temporal_mismatch_flags": flags,
        "predicate_mismatch_flags": flags.clone(),
    }


def _measure_batch(
    *,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    vocab_size: int,
    probe_mixer: Any,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    allocated_before = int(torch.cuda.memory_allocated(device))
    reserved_before = int(torch.cuda.memory_reserved(device))

    fast_calls = {"count": 0}
    original_cuda = probe_mixer.cuda_kernels_forward

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

    probe_mixer.cuda_kernels_forward = types.MethodType(
        cuda_wrapper,
        probe_mixer,
    )

    inputs = None
    output = None
    loss = None
    status = "PASS"
    error = None

    try:
        model.zero_grad(set_to_none=True)
        model.train()
        inputs = _synthetic_inputs(
            batch_size,
            vocab_size=vocab_size,
            device=device,
        )

        output = model(
            **inputs,
            gradient_ownership_mode="edge_specific",
        )
        require("logits" in output, "LOGITS_MISSING")
        logits = output["logits"]
        require(
            tuple(logits.shape) == (batch_size, 3),
            f"LOGITS_SHAPE:{tuple(logits.shape)}",
        )
        require(
            bool(torch.isfinite(logits).all().item()),
            "LOGITS_NONFINITE",
        )

        loss = logits.float().sum()
        loss.backward()
        torch.cuda.synchronize(device)

        trainable_with_grad = sum(
            int(p.numel())
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        )
        frozen_with_grad = sum(
            int(p.numel())
            for p in model.mamba.parameters()
            if p.grad is not None
        )
        require(trainable_with_grad > 0, "NO_TRAINABLE_GRADIENT")
        require(frozen_with_grad == 0, "FROZEN_BACKBONE_GRADIENT")

    except torch.cuda.OutOfMemoryError as exc:
        status = "OOM"
        error = str(exc)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    finally:
        if "cuda_kernels_forward" in probe_mixer.__dict__:
            del probe_mixer.__dict__["cuda_kernels_forward"]

    torch.cuda.synchronize(device)

    result = {
        "batch_size": batch_size,
        "sequence_length": SEQ_LEN,
        "status": status,
        "error": error,
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "fast_path_calls_at_probe_layer": int(fast_calls["count"]),
        "synthetic_forward_executed": status == "PASS",
        "synthetic_backward_executed": status == "PASS",
        "optimizer_step_executed": False,
        "scientific_data_accessed": False,
    }

    if status == "PASS":
        require(
            fast_calls["count"] == 1,
            f"FAST_PATH_CALL_COUNT:{batch_size}:{fast_calls['count']}",
        )

    del inputs, output, loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    return result


def worker_run(
    *,
    worker_id: int,
    physical_device: int,
    expected_head: str,
    snapshot: str,
    temp_dir: str,
) -> None:
    paths = _worker_paths(Path(temp_dir), worker_id)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)

        authenticate_repo(expected_head)

        import transformers
        from transformers import MambaConfig, MambaModel
        from scripts import (
            reason_router_gen4_generator_family_prevalence_kernel_compat
            as kernel_compat,
        )
        from scripts import (
            reason_router_gen4_six_cell_tier2_inference_adapter
            as adapter,
        )

        require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
        require(
            torch.cuda.device_count() == 1,
            f"VISIBLE_GPU_COUNT:{torch.cuda.device_count()}",
        )

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        device_name = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))

        require(device_name == "Tesla T4", f"DEVICE_NAME:{device_name}")
        require(capability == (7, 5), f"DEVICE_CAPABILITY:{capability}")

        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        snapshot_path = Path(snapshot)
        identity = snapshot_identity(snapshot_path)

        cfg = MambaConfig.from_pretrained(
            snapshot_path,
            local_files_only=True,
        )
        validate_config_dict(
            {
                "hidden_size": cfg.hidden_size,
                "intermediate_size": cfg.intermediate_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "state_size": cfg.state_size,
                "conv_kernel": cfg.conv_kernel,
                "vocab_size": cfg.vocab_size,
            }
        )
        cfg.use_mamba_kernels = True

        kernels = kernel_compat.load_exact_fast_kernels()

        with kernel_compat.exact_transformers_kernel_loader(
            kernels
        ) as calls:
            backbone = MambaModel.from_pretrained(
                snapshot_path,
                config=cfg,
                local_files_only=True,
            )

        constructor_counts = Counter(calls)
        require(
            set(constructor_counts)
            == {"causal-conv1d", "mamba-ssm"},
            f"KERNEL_CONSTRUCTOR_KEYS:{dict(constructor_counts)}",
        )
        require(
            constructor_counts["causal-conv1d"]
            == constructor_counts["mamba-ssm"]
            and constructor_counts["mamba-ssm"] > 0,
            f"KERNEL_CONSTRUCTOR_COUNTS:{dict(constructor_counts)}",
        )

        kernel_compat.validate_transformers_kernel_bindings(kernels)

        require(
            len(backbone.layers) == EXPECTED_CONFIG["num_hidden_layers"],
            "LAYER_COUNT",
        )

        model = adapter.build_historical_model_from_backbone(
            backbone=backbone,
            arm=ARM,
            hidden_size=EXPECTED_CONFIG["hidden_size"],
        )

        for parameter in model.mamba.parameters():
            parameter.requires_grad = False

        backbone_parameters = list(model.mamba.parameters())
        trainable_parameters = [
            p for p in model.parameters() if p.requires_grad
        ]
        frozen_parameters = [
            p for p in model.parameters() if not p.requires_grad
        ]

        require(
            sum(int(p.numel()) for p in backbone_parameters) > 0,
            "BACKBONE_PARAM_COUNT",
        )
        require(
            all(not p.requires_grad for p in backbone_parameters),
            "BACKBONE_NOT_FROZEN",
        )
        require(len(trainable_parameters) > 0, "NO_TRAINABLE_PARAMS")

        model.to(device)
        torch.cuda.synchronize(device)

        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))

        probe_layer = 35
        probe_mixer = model.mamba.layers[probe_layer].mixer
        require(
            hasattr(probe_mixer, "cuda_kernels_forward"),
            "CUDA_FAST_METHOD_MISSING",
        )

        batches = []
        saw_oom = False
        for batch_size in PROBE_BATCH_SIZES:
            if saw_oom:
                batches.append({
                    "batch_size": batch_size,
                    "sequence_length": SEQ_LEN,
                    "status": "SKIPPED_AFTER_OOM",
                    "error": None,
                    "synthetic_forward_executed": False,
                    "synthetic_backward_executed": False,
                    "optimizer_step_executed": False,
                    "scientific_data_accessed": False,
                })
                continue

            result = _measure_batch(
                model=model,
                batch_size=batch_size,
                device=device,
                vocab_size=EXPECTED_CONFIG["vocab_size"],
                probe_mixer=probe_mixer,
            )
            batches.append(result)
            if result["status"] == "OOM":
                saw_oom = True

        passing = [
            int(row["batch_size"])
            for row in batches
            if row["status"] == "PASS"
        ]
        require(passing, "NO_PASSING_BATCH")

        report = {
            "schema_version": "gen4-mamba14b-runtime-feasibility-worker-v1",
            "worker_id": worker_id,
            "physical_device": physical_device,
            "logical_device": 0,
            "device_name": device_name,
            "device_capability": list(capability),
            "runtime": runtime_identity(),
            "snapshot": identity,
            "kernel_constructor_counts":
                dict(sorted(constructor_counts.items())),
            "kernel_bindings_validated": True,
            "arm": ARM,
            "backbone_parameter_numel": sum(
                int(p.numel()) for p in backbone_parameters
            ),
            "backbone_parameter_bytes": _bytes_of_parameters(
                backbone_parameters
            ),
            "backbone_trainable_numel": sum(
                int(p.numel())
                for p in backbone_parameters
                if p.requires_grad
            ),
            "total_parameter_numel": sum(
                int(p.numel()) for p in model.parameters()
            ),
            "trainable_parameter_numel": sum(
                int(p.numel()) for p in trainable_parameters
            ),
            "trainable_parameter_bytes": _bytes_of_parameters(
                trainable_parameters
            ),
            "frozen_parameter_numel": sum(
                int(p.numel()) for p in frozen_parameters
            ),
            "baseline_cuda_allocated_bytes": baseline_allocated,
            "baseline_cuda_reserved_bytes": baseline_reserved,
            "probe_layer": probe_layer,
            "sequence_length": SEQ_LEN,
            "batch_measurements": batches,
            "max_passing_batch_size": max(passing),
            "scientific_model_forward_count": 0,
            "scientific_data_accessed": False,
            "training_executed": False,
            "evaluation_executed": False,
            "synthetic_runtime_forward_backward_only": True,
            "optimizer_step_executed": False,
        }

        paths["report"].write_text(
            json.dumps(
                report,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
            newline="\n",
        )
        raise


def merge_workers(
    *,
    temp_dir: Path,
    snapshot: Path,
    expected_head: str,
) -> dict[str, Any]:
    workers = []

    for worker_id in range(GPU_COUNT):
        paths = _worker_paths(temp_dir, worker_id)
        require(
            paths["report"].is_file(),
            f"WORKER_REPORT_MISSING:{worker_id}",
        )
        workers.append(
            json.loads(paths["report"].read_text(encoding="utf-8"))
        )

    resolved = {
        w["snapshot"]["resolved_revision"]
        for w in workers
    }
    require(len(resolved) == 1, f"REVISION_DISAGREEMENT:{resolved}")

    snapshot_hash_sets = {
        json.dumps(
            w["snapshot"]["files"],
            sort_keys=True,
        )
        for w in workers
    }
    require(
        len(snapshot_hash_sets) == 1,
        "SNAPSHOT_IDENTITY_DISAGREEMENT",
    )

    trainable_counts = {
        int(w["trainable_parameter_numel"])
        for w in workers
    }
    require(
        len(trainable_counts) == 1,
        f"TRAINABLE_COUNT_DISAGREEMENT:{trainable_counts}",
    )

    max_batches = [
        int(w["max_passing_batch_size"])
        for w in workers
    ]

    return {
        "schema_version": REPORT_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "model_repo": MODEL_REPO,
        "requested_revision": DISCOVERY_REVISION,
        "resolved_revision": workers[0]["snapshot"]["resolved_revision"],
        "snapshot": snapshot_identity(snapshot),
        "gpu_count": GPU_COUNT,
        "workers": workers,
        "minimum_max_passing_batch_size": min(max_batches),
        "maximum_max_passing_batch_size": max(max_batches),
        "trainable_parameter_numel":
            next(iter(trainable_counts)),
        "sequence_length": SEQ_LEN,
        "probe_batch_sizes": list(PROBE_BATCH_SIZES),
        "hard_gates": {
            "two_t4_workers": True,
            "exact_snapshot_same_on_both_workers": True,
            "expected_config": True,
            "fast_kernel_bindings": True,
            "frozen_backbone": all(
                int(w["backbone_trainable_numel"]) == 0
                for w in workers
            ),
            "synthetic_forward_backward_batch1": all(
                any(
                    int(row["batch_size"]) == 1
                    and row["status"] == "PASS"
                    for row in w["batch_measurements"]
                )
                for w in workers
            ),
        },
        "scientific_model_forward_count": 0,
        "scientific_data_accessed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "p_value_count": 0,
        "selection_performed": False,
        "scientific_conclusion_established": False,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-scientific Mamba-1.4B runtime feasibility probe. "
            "Resolves an immutable HF snapshot, validates exact fast CUDA "
            "kernel binding, freezes the Mamba backbone, and measures "
            "synthetic seq-128 forward/backward memory on both T4 GPUs. "
            "No scientific data, evaluation, optimizer step, or training."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    authenticate_repo(args.expected_head)
    require(
        not args.output_json.exists(),
        f"OUTPUT_EXISTS:{args.output_json}",
    )

    snapshot = resolve_snapshot()
    identity = snapshot_identity(snapshot)

    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(
        torch.cuda.device_count() >= GPU_COUNT,
        f"CUDA_DEVICE_COUNT:{torch.cuda.device_count()}",
    )

    with tempfile.TemporaryDirectory(
        prefix="mamba14b_feasibility_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes = []

        for worker_id in range(GPU_COUNT):
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "worker_id": worker_id,
                    "physical_device": worker_id,
                    "expected_head": args.expected_head,
                    "snapshot": str(snapshot),
                    "temp_dir": str(temp_dir),
                },
                name=f"mamba14b-feasibility-{worker_id}",
            )
            process.start()
            processes.append(process)

        for worker_id, process in enumerate(processes):
            process.join()
            if process.exitcode != 0:
                paths = _worker_paths(temp_dir, worker_id)
                detail = (
                    paths["error"].read_text(encoding="utf-8")
                    if paths["error"].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise FeasibilityError(
                    f"WORKER_FAILED:{worker_id}:\n{detail}"
                )

        report = merge_workers(
            temp_dir=temp_dir,
            snapshot=snapshot,
            expected_head=args.expected_head,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("RESULT=" + report["result"])
    print("MODEL_REPO=" + report["model_repo"])
    print("RESOLVED_REVISION=" + report["resolved_revision"])
    print(
        "TOTAL_SNAPSHOT_BYTES="
        + str(report["snapshot"]["total_snapshot_bytes"])
    )
    print(
        "TRAINABLE_PARAMETER_NUMEL="
        + str(report["trainable_parameter_numel"])
    )
    print(
        "MINIMUM_MAX_PASSING_BATCH_SIZE="
        + str(report["minimum_max_passing_batch_size"])
    )

    for worker in report["workers"]:
        print(
            f"GPU{worker['physical_device']}_BASELINE_ALLOCATED_BYTES="
            + str(worker["baseline_cuda_allocated_bytes"])
        )
        print(
            f"GPU{worker['physical_device']}_MAX_PASSING_BATCH_SIZE="
            + str(worker["max_passing_batch_size"])
        )
        for row in worker["batch_measurements"]:
            if row["status"] == "PASS":
                print(
                    f"GPU{worker['physical_device']}_B"
                    f"{row['batch_size']}_PEAK_ALLOCATED_BYTES="
                    + str(row["peak_allocated_bytes"])
                )
            elif row["status"] == "OOM":
                print(
                    f"GPU{worker['physical_device']}_B"
                    f"{row['batch_size']}_STATUS=OOM"
                )

    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("SCIENTIFIC_DATA_ACCESSED=False")
    print("TRAINING_EXECUTED=False")
    print("EVALUATION_EXECUTED=False")
    print("OPTIMIZER_STEP_EXECUTED=False")
    print("P_VALUE_COUNT=0")
    print("SELECTION_PERFORMED=False")
    print("SCIENTIFIC_CONCLUSION_ESTABLISHED=False")


if __name__ == "__main__":
    main()
