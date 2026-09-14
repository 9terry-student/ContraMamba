#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_k_directional_alignment_transport_core as core
from scripts import reason_router_gen4_k_directional_alignment_transport_runner as parent
from scripts import reason_router_gen4_k_directional_alignment_transport_runtime as transport_runtime
from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as gate
from scripts import reason_router_gen4_native_mamba_state_extraction as extraction
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement


ROOT = Path(__file__).resolve().parents[1]

ONE_PAIR_GATE_COMMIT = "d6e1521c3f2c92f16aa88bbaf3f6b6c332a85ae6"
ONE_PAIR_GATE_REL = (
    "scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py"
)
ONE_PAIR_GATE_BLOB = "4a8326c442b883510ba452f049c88f3163677fc5"

CPU_REFERENCE_HEAD = "8496ece911e0d461f0abbdf1a0fa619f8a2f22ab"
CPU_REFERENCE_ZIP_SHA256 = (
    "25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf"
)

SOURCE_PAIR_COUNT = parent.SOURCE_PAIR_COUNT
FORWARDS_PER_PAIR = parent.FORWARDS_PER_PAIR
FULL_FORWARD_BUDGET = parent.FULL_FORWARD_BUDGET

CUDA_MANIFEST_SCHEMA = "gen4-k-fast-cuda-full-manifest-v1"
CUDA_SUMMARY_SCHEMA = "gen4-k-fast-cuda-full-summary-v1"

MANIFEST_FILE = parent.MANIFEST_FILE
ITEM_FILE = parent.ITEM_FILE
SUMMARY_FILE = parent.SUMMARY_FILE
CHECKSUM_FILE = parent.CHECKSUM_FILE

EXPECTED_OUTPUT_FILES = frozenset(
    {
        MANIFEST_FILE,
        ITEM_FILE,
        SUMMARY_FILE,
        CHECKSUM_FILE,
    }
)


class FastCudaFullError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FastCudaFullError(message)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FastCudaFullError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_full_repo(expected_head: str) -> None:
    gate.authenticate_repo(expected_head)

    require(
        _git(
            "rev-parse",
            f"{ONE_PAIR_GATE_COMMIT}:{ONE_PAIR_GATE_REL}",
        )
        == ONE_PAIR_GATE_BLOB,
        "ONE_PAIR_GATE_BLOB_MISMATCH",
    )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            ONE_PAIR_GATE_COMMIT,
            expected_head,
            "--",
            ONE_PAIR_GATE_REL,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "ONE_PAIR_GATE_CHANGED_AFTER_PASS",
    )


def _finite_item_rows(items: Sequence[Mapping[str, Any]]) -> None:
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "ITEM_COUNT",
    )

    seen: set[str] = set()

    for index, row in enumerate(items):
        pair = str(row["source_pair_id"])
        require(
            pair not in seen,
            f"DUPLICATE_PAIR:{pair}",
        )
        seen.add(pair)

        for key, value in row.items():
            if isinstance(value, float):
                require(
                    math.isfinite(value),
                    f"NONFINITE:{index}:{key}",
                )


def build_cuda_manifest(
    *,
    head: str,
    checkpoint_sha: str,
    forward_count: int,
) -> dict[str, Any]:
    require(
        forward_count == FULL_FORWARD_BUDGET,
        "MANIFEST_FORWARD_COUNT",
    )

    return {
        "schema_version": CUDA_MANIFEST_SCHEMA,
        "mode": "full_backend_equivalence",
        "runtime_branch": gate.EXPECTED_BRANCH,
        "runtime_git_head": head,
        "one_pair_gate_commit": ONE_PAIR_GATE_COMMIT,
        "one_pair_gate_blob": ONE_PAIR_GATE_BLOB,
        "cpu_reference_head": CPU_REFERENCE_HEAD,
        "cpu_reference_zip_sha256": CPU_REFERENCE_ZIP_SHA256,
        "source_pair_count": SOURCE_PAIR_COUNT,
        "model_forward_count": forward_count,
        "forwards_per_pair": FORWARDS_PER_PAIR,
        "device": "cuda:0",
        "execution_backend": "transformers_fast_cuda_selective_scan",
        "kernels_version": gate.KERNELS_VERSION,
        "mamba_revision": gate.MAMBA_REV,
        "mamba_binary_sha256": gate.MAMBA_BINARY_SHA256,
        "causal_conv_revision": gate.CONV_REV,
        "causal_conv_binary_sha256": gate.CONV_BINARY_SHA256,
        "build_variant": gate.BUILD_VARIANT,
        "python_version": gate.EXPECTED_RUNTIME["python"],
        "numpy_version": gate.EXPECTED_RUNTIME["numpy"],
        "torch_version": gate.EXPECTED_RUNTIME["torch"],
        "transformers_version": gate.EXPECTED_RUNTIME["transformers"],
        "cuda_runtime": gate.EXPECTED_CUDA_RUNTIME,
        "cuda_device": gate.EXPECTED_DEVICE_NAME,
        "cuda_capability": list(gate.EXPECTED_CAPABILITY),
        "checkpoint_sha256": checkpoint_sha,
        "frozen_endpoint_rel": core.FROZEN_LAYER17_ENDPOINT_REL,
        "frozen_endpoint_sha256": core.FROZEN_LAYER17_ENDPOINT_SHA256,
        "event_manifest_rel": extraction.EVENT_MANIFEST.as_posix(),
        "event_manifest_sha256": extraction.EVENT_MANIFEST_SHA256,
        "mamba_source_sha256": measurement.MAMBA_SHA256,
        "source_block": core.SOURCE_BLOCK,
        "target_residual_layer": core.TARGET_RESIDUAL_LAYER,
        "intervention_layer": core.INTERVENTION_LAYER,
        "relative_coordinate": core.TARGET_OFFSET,
        "target_pair": [
            core.TARGET_PLUS_CELL,
            core.TARGET_MINUS_CELL,
        ],
        "reference_pair": [
            core.REFERENCE_PLUS_CELL,
            core.REFERENCE_MINUS_CELL,
        ],
        "anchor_name": core.ANCHOR_NAME,
        "strong_count": core.EXPECTED_STRONG_COUNT,
        "weak_count": core.EXPECTED_WEAK_COUNT,
        "equal_count": core.EXPECTED_EQUAL_COUNT,
        "strong_index_sha256": core.EXPECTED_STRONG_INDEX_SHA256,
        "backend_equivalence_only": True,
        "scientific_authority": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "raw_vectors_persisted": False,
        "tokenizer_invoked": True,
        "causal_intervention_executed": True,
        "statistical_testing": True,
        "cpu_reference_required_for_interpretation": True,
    }


def build_cuda_summary(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    # parent_runtime_rebind() fixes BASELINE_REPRO_TOL to the
    # pre-registered cross-backend PE tolerance. All other frozen
    # intervention/manipulation tolerances remain unchanged.
    base = dict(parent.build_full_summary(items))
    base["schema_version"] = CUDA_SUMMARY_SCHEMA
    base["backend"] = "fast_cuda"
    base["backend_equivalence_only"] = True
    base["scientific_authority"] = False
    base["cpu_reference_zip_sha256"] = CPU_REFERENCE_ZIP_SHA256
    base["baseline_reproduction_tolerance"] = gate.PE_ATOL
    base["cpu_reference_required_for_interpretation"] = True
    return base


def run_full_cuda(
    *,
    expected_head: str,
    output_dir: Path,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    authenticate_full_repo(expected_head)
    gate.runtime_gate()
    require(
        not output_dir.exists(),
        "OUTPUT_COLLISION",
    )

    with gate.parent_runtime_rebind():
        rows, encoded, event_rows = (
            extraction.load_scientific_inputs(
                tokenizer_snapshot
            )
        )

        pairs = list(
            extraction.canonical_pair_order(rows)
        )
        require(
            len(pairs) == SOURCE_PAIR_COUNT,
            "PAIR_COUNT",
        )

        events = parent.event_lookup(event_rows)
        parent.validate_transport_event_plan(
            pairs,
            events,
        )

        row_index = parent.build_row_index(rows)
        frozen_endpoint = (
            parent.load_frozen_layer17_endpoints()
        )

        trace_code, trace_line = (
            measurement._resolve_and_validate_runtime_binding()
        )

        model, checkpoint_sha = (
            parent.load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
        )

        require(
            checkpoint_sha
            == extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
            "CHECKPOINT_IDENTITY",
        )

        runtime_ctx = (
            transport_runtime.validate_runtime_components(
                model
            )
        )

        kernels = gate.load_exact_fast_kernels()

        model.to(torch.device("cuda:0"))
        model.eval()

        require(
            all(
                parameter.device.type == "cuda"
                for parameter
                in model.mamba.parameters()
            ),
            "MODEL_NOT_FULLY_CUDA",
        )

        fast_capture = gate._make_fast_capture(
            kernels
        )

        original_capture = parent.capture_branch
        budget = parent.ForwardBudget(
            FULL_FORWARD_BUDGET
        )
        items: list[dict[str, Any]] = []

        parent.capture_branch = fast_capture

        try:
            for pair in pairs:
                items.append(
                    parent.run_pair(
                        pair,
                        model=model,
                        runtime_ctx=runtime_ctx,
                        trace_code=trace_code,
                        trace_line=trace_line,
                        encoded=encoded,
                        row_index=row_index,
                        events=events,
                        frozen_endpoint=frozen_endpoint,
                        budget=budget,
                    )
                )
        finally:
            parent.capture_branch = original_capture

        budget.assert_exact()
        torch.cuda.synchronize()

        _finite_item_rows(items)

        summary = build_cuda_summary(items)

        require(
            summary[
                "all_mandatory_manipulation_checks_pass"
            ]
            is True,
            "CUDA_MANIPULATION_CHECKS_FAILED",
        )

        manifest = build_cuda_manifest(
            head=expected_head,
            checkpoint_sha=checkpoint_sha,
            forward_count=budget.used,
        )

        parent.publish_bundle(
            output_dir,
            manifest=manifest,
            items=items,
            summary=summary,
            preflight=None,
        )

    observed = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }
    require(
        observed == EXPECTED_OUTPUT_FILES,
        f"OUTPUT_FILE_SET:{sorted(observed)}",
    )

    require(
        sum(
            1
            for line in (
                output_dir / ITEM_FILE
            ).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
        == SOURCE_PAIR_COUNT,
        "OUTPUT_ITEM_COUNT",
    )

    return {
        "result": "PASS_FAST_CUDA_FULL_EXECUTION",
        "model_forward_count": FULL_FORWARD_BUDGET,
        "population_size": SOURCE_PAIR_COUNT,
        "backend_outcome": summary["outcome"],
        "scientific_conclusion": None,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    result = run_full_cuda(
        expected_head=args.expected_head,
        output_dir=Path(args.output_dir),
        model_snapshot=Path(args.model_snapshot),
        tokenizer_snapshot=Path(
            args.tokenizer_snapshot
        ),
        checkpoint_path=Path(args.checkpoint),
    )

    print("RESULT =", result["result"])
    print(
        "MODEL_FORWARD_COUNT =",
        result["model_forward_count"],
    )
    print(
        "POPULATION_SIZE =",
        result["population_size"],
    )
    print(
        "BACKEND_OUTCOME =",
        result["backend_outcome"],
    )
    print("SCIENTIFIC_CONCLUSION = NONE")


if __name__ == "__main__":
    main()
