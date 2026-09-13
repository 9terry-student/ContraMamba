"""Bounded Gen4 NAME Q1/Q3 scientific-extraction execution driver.

Implementation-only orchestration. Scientific execution remains unauthorized
until a later execution-authority commit is the exact execution HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import reason_router_gen4_native_mamba_state_extraction as phase_d  # noqa: E402
from scripts import reason_router_gen4_native_mamba_state_q1_q3_extraction as extraction  # noqa: E402
from scripts import reason_router_gen4_native_mamba_state_q1_q3_measurement as measurement  # noqa: E402
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter  # noqa: E402
from scripts import reason_router_gen4_six_cell_tier2_scientific_inference as r5  # noqa: E402


DRIVER_IMPLEMENTATION_CORRECTION_AUTHORITY = (
    "e8f163885f1c4d1cd9180521ee0b8026c5206181"
)
IMPLEMENTATION_VALIDATION_FREEZE_COMMIT = (
    "b8f2f22820f8f383dd47c349e63e0327d38b9f56"
)
Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT = (
    "d5317b2c5be09464c9196325429479c6bff25efd"
)

DRIVER_PATH = Path(
    "scripts/reason_router_gen4_native_mamba_state_q1_q3_execution.py"
)
EXECUTION_AUTHORITY_PATH = Path(
    "reports/reason_router_gen4_native_mamba_state_bridge_"
    "name_q1_q3_scientific_extraction_execution_authority_spec_candidate.md"
)
EXECUTION_AUTHORITY_ID = (
    "GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_"
    "SCIENTIFIC_EXTRACTION_EXECUTION_AUTHORITY"
)

Q1_Q3_MEASUREMENT_PATH = Path(
    "scripts/reason_router_gen4_native_mamba_state_q1_q3_measurement.py"
)
Q1_Q3_EXTRACTION_PATH = Path(
    "scripts/reason_router_gen4_native_mamba_state_q1_q3_extraction.py"
)
PRIMARY_MEASUREMENT_PATH = Path(
    "scripts/reason_router_gen4_native_mamba_state_measurement.py"
)
PRIMARY_EXTRACTION_PATH = Path(
    "scripts/reason_router_gen4_native_mamba_state_extraction.py"
)

Q1_Q3_MEASUREMENT_SHA256 = (
    "2236d19a46416e5042057ec84c565dbe523ef791f7266c30b26b033d260a510b"
)
Q1_Q3_EXTRACTION_SHA256 = (
    "9e9ca05aaee43970c8d0aa101c7f0c0b86ace9166ae798f70c800fcb61651c4f"
)
PRIMARY_MEASUREMENT_SHA256 = (
    "7729424f03058b86b4f120dc0e6da573d6c996b0877858f2d6d38aa94dac268c"
)
PRIMARY_EXTRACTION_SHA256 = (
    "653f96713d8bdc776cdf03733ae230784e413f960f58a9c120f00cb8c6a6d3eb"
)

EXPECTED_LAYERS = (5, 17)
EXPECTED_ROWS = 600
EXPECTED_FORWARD_COUNT = 600
EXPECTED_SUPPORT_ROWS = 7200
EXPECTED_ENDPOINT_ROWS = 1200

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ExecutionContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionContractError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def git_show_bytes(revision: str, path: str | Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{Path(path).as_posix()}"],
        cwd=ROOT,
    )


def git_commit_exists(revision: str) -> bool:
    if _HEX40.fullmatch(str(revision)) is None:
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def git_status_porcelain() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    )


def canonical_git_sha256(revision: str, path: str | Path) -> str:
    return sha256_bytes(git_show_bytes(revision, path))


def canonical_worktree_bytes(path: str | Path) -> bytes:
    raw = (ROOT / Path(path)).read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    require(b"\r" not in normalized, f"line-ending ambiguity: {path}")
    return normalized


def validate_frozen_dependency_identities(revision: str) -> None:
    expected = {
        Q1_Q3_MEASUREMENT_PATH: Q1_Q3_MEASUREMENT_SHA256,
        Q1_Q3_EXTRACTION_PATH: Q1_Q3_EXTRACTION_SHA256,
        PRIMARY_MEASUREMENT_PATH: PRIMARY_MEASUREMENT_SHA256,
        PRIMARY_EXTRACTION_PATH: PRIMARY_EXTRACTION_SHA256,
    }
    for path, digest in expected.items():
        observed = canonical_git_sha256(revision, path)
        require(
            observed == digest,
            f"frozen dependency identity mismatch: {path}",
        )

    # Also preserve the exact adapter/R5/historical-model identities already
    # proven and enforced by the canonical Phase-D orchestration.
    try:
        phase_d.validate_frozen_dependency_identities(revision)
    except Exception as exc:
        raise ExecutionContractError(
            "primary orchestration dependency identity mismatch"
        ) from exc


def validate_execution_authority_document(
    execution_authority_commit: str,
    *,
    driver_implementation_commit: str,
    expected_driver_sha256: str,
) -> None:
    try:
        raw = git_show_bytes(
            execution_authority_commit,
            EXECUTION_AUTHORITY_PATH,
        )
    except subprocess.CalledProcessError as exc:
        raise ExecutionContractError(
            "BLOCKED_Q1_Q3_EXECUTION_AUTHORITY"
        ) from exc

    text = raw.decode("utf-8")
    required = (
        f"AUTHORITY_ID =\n{EXECUTION_AUTHORITY_ID}",
        "SCIENTIFIC_EXECUTION_ALLOWED =\nYES_BOUNDED",
        "CANONICAL_TOKENIZER_EXECUTION_ALLOWED =\nYES_BOUNDED",
        "CHECKPOINT_LOADING_ALLOWED =\nYES_BOUNDED",
        "MODEL_FORWARD_ALLOWED =\nYES_BOUNDED",
        "NATIVE_STATE_EXTRACTION_ALLOWED =\nYES_BOUNDED",
        "STATISTICAL_TESTING_ALLOWED =\nNO",
        "TRAINING_ALLOWED =\nNO",
        f"Q1_Q3_EXECUTION_DRIVER_IMPLEMENTATION_COMMIT =\n{driver_implementation_commit}",
        f"Q1_Q3_EXECUTION_DRIVER_SHA256 =\n{expected_driver_sha256}",
        f"Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT =\n{Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT}",
    )
    for clause in required:
        require(
            clause in text,
            "BLOCKED_Q1_Q3_EXECUTION_AUTHORITY",
        )


def validate_execution_binding(
    *,
    expected_execution_head: str,
    driver_implementation_commit: str,
    expected_driver_sha256: str,
    execution_authority_commit: str,
) -> str:
    observed_head = git_head()
    require(
        observed_head == expected_execution_head,
        "execution HEAD mismatch",
    )
    require(
        execution_authority_commit == expected_execution_head,
        "BLOCKED_Q1_Q3_EXECUTION_AUTHORITY",
    )
    require(
        _HEX40.fullmatch(driver_implementation_commit) is not None,
        "driver implementation commit format",
    )
    require(
        _HEX64.fullmatch(expected_driver_sha256) is not None,
        "driver SHA256 format",
    )
    require(
        git_commit_exists(DRIVER_IMPLEMENTATION_CORRECTION_AUTHORITY),
        "driver correction authority missing",
    )
    require(
        git_commit_exists(driver_implementation_commit),
        "driver implementation commit missing",
    )
    require(
        git_commit_exists(execution_authority_commit),
        "BLOCKED_Q1_Q3_EXECUTION_AUTHORITY",
    )
    require(
        git_is_ancestor(
            DRIVER_IMPLEMENTATION_CORRECTION_AUTHORITY,
            driver_implementation_commit,
        ),
        "driver implementation outside correction authority",
    )
    require(
        git_is_ancestor(
            driver_implementation_commit,
            execution_authority_commit,
        ),
        "BLOCKED_Q1_Q3_EXECUTION_AUTHORITY",
    )
    require(
        git_is_ancestor(
            IMPLEMENTATION_VALIDATION_FREEZE_COMMIT,
            execution_authority_commit,
        ),
        "Q1/Q3 primitive freeze not ancestral",
    )

    validate_execution_authority_document(
        execution_authority_commit,
        driver_implementation_commit=driver_implementation_commit,
        expected_driver_sha256=expected_driver_sha256,
    )

    require(
        git_status_porcelain() == "",
        "repository dirty before scientific execution",
    )

    driver_bytes = git_show_bytes(
        driver_implementation_commit,
        DRIVER_PATH,
    )
    require(
        sha256_bytes(driver_bytes) == expected_driver_sha256,
        "driver identity mismatch",
    )
    require(
        canonical_git_sha256(observed_head, DRIVER_PATH)
        == expected_driver_sha256,
        "driver changed after implementation freeze",
    )
    require(
        canonical_worktree_bytes(DRIVER_PATH) == driver_bytes,
        "driver worktree mismatch",
    )

    validate_frozen_dependency_identities(observed_head)
    return observed_head


def validate_output_destination(output_dir: str | Path) -> None:
    final_dir = Path(output_dir)
    staging = final_dir.with_name(final_dir.name + ".staging")
    require(not final_dir.exists(), "output collision")
    require(not staging.exists(), "staging collision")


def load_scientific_inputs(
    tokenizer_snapshot: str | Path | None = None,
) -> tuple[
    list[Mapping[str, Any]],
    Mapping[str, Any],
    dict[str, Mapping[str, Any]],
]:
    structural_path = ROOT / adapter.CANONICAL_GEN4_ARTIFACT
    rows = extraction.authenticate_structural_artifact(structural_path)

    event_path = ROOT / phase_d.EVENT_MANIFEST
    event_rows = extraction.authenticate_event_manifest(event_path)

    # Preserve the stronger full-manifest schema check already frozen in Phase D.
    try:
        phase_d.validate_event_manifest(event_rows)
    except Exception as exc:
        raise ExecutionContractError(
            "event manifest schema validation failed"
        ) from exc

    if tokenizer_snapshot is None:
        tokenizer_snapshot = adapter.canonical_tokenizer_snapshot_dir()

    extraction.authenticate_tokenizer_snapshot(tokenizer_snapshot)
    tokenizer = adapter.load_canonical_tokenizer(tokenizer_snapshot)

    # This is the required full-1800 coordinate gate. No subset/model work
    # occurs before the frozen aggregate identity passes.
    selected, encoded = extraction.reconstruct_then_select(
        rows,
        lambda: adapter.encode_gen4_rows(rows, tokenizer),
    )
    events = extraction.bind_events(selected, event_rows)

    require(len(selected) == EXPECTED_ROWS, "Q1/Q3 selected row count")
    require(
        set(extraction.LAYERS) == set(EXPECTED_LAYERS),
        "Q1/Q3 fixed layer policy",
    )
    return selected, encoded, events


def load_representative_model(
    model_snapshot: str | Path | None = None,
) -> tuple[Any, str]:
    if model_snapshot is None:
        model_snapshot = r5.canonical_model_snapshot_dir()

    model, checkpoint_sha = phase_d.load_representative_model(
        model_snapshot
    )
    require(
        checkpoint_sha == extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        "representative checkpoint identity",
    )
    return model, checkpoint_sha


def execute_scientific_plan(
    *,
    selected: Sequence[Mapping[str, Any]],
    events: Mapping[str, Mapping[str, Any]],
    encoded: Mapping[str, Any],
    model: Any,
    capture_row: Callable[[Any, Any], Mapping[int, Sequence[Any]]] = (
        measurement.capture_model_row
    ),
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], int]:
    def forward_once(input_ids: Any) -> Mapping[int, Sequence[Any]]:
        cpu_input = (
            input_ids.to("cpu")
            if hasattr(input_ids, "to")
            else input_ids
        )
        return capture_row(model, cpu_input)

    return extraction.execute_dual_layer_forward_plan(
        selected,
        events,
        encoded,
        forward_once,
    )


def runtime_versions() -> dict[str, str]:
    import torch
    import transformers

    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    require(
        versions == extraction.EXPECTED_RUNTIME_VERSIONS,
        "runtime version contract",
    )
    return versions


def run_scientific_extraction(
    *,
    output_dir: str | Path,
    expected_execution_head: str,
    driver_implementation_commit: str,
    expected_driver_sha256: str,
    execution_authority_commit: str,
    model_snapshot: str | Path | None = None,
    tokenizer_snapshot: str | Path | None = None,
    runtime_gate: Callable[[], None] | None = None,
    input_loader: Callable[
        [str | Path | None],
        tuple[
            list[Mapping[str, Any]],
            Mapping[str, Any],
            dict[str, Mapping[str, Any]],
        ],
    ] | None = None,
    model_loader: Callable[[str | Path | None], tuple[Any, str]] | None = None,
    plan_executor: Callable[..., tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], int]] | None = None,
    bundle_writer: Callable[..., None] | None = None,
    runtime_version_loader: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    head = validate_execution_binding(
        expected_execution_head=expected_execution_head,
        driver_implementation_commit=driver_implementation_commit,
        expected_driver_sha256=expected_driver_sha256,
        execution_authority_commit=execution_authority_commit,
    )

    validate_output_destination(output_dir)

    if runtime_gate is None:
        runtime_gate = measurement.primary.runtime_gate
    if input_loader is None:
        input_loader = load_scientific_inputs
    if model_loader is None:
        model_loader = load_representative_model
    if plan_executor is None:
        plan_executor = execute_scientific_plan
    if bundle_writer is None:
        bundle_writer = extraction.write_future_bundle
    if runtime_version_loader is None:
        runtime_version_loader = runtime_versions

    # Runtime/source-role gate must pass before tokenizer reconstruction,
    # checkpoint deserialization, model construction, or forward.
    runtime_gate()

    # Full 1800-coordinate hash gate is inside the input loader and must
    # complete before model_loader is reached.
    selected, encoded, events = input_loader(tokenizer_snapshot)

    model, checkpoint_sha = model_loader(model_snapshot)
    require(
        checkpoint_sha == extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        "representative checkpoint identity",
    )
    require(hasattr(model, "mamba"), "missing Mamba backbone")
    model.mamba.eval()

    parameters = list(model.mamba.parameters())
    require(
        all(
            str(getattr(parameter, "device", "cpu")) == "cpu"
            for parameter in parameters
        ),
        "Mamba device must be CPU",
    )
    require(
        not any(
            bool(getattr(parameter, "requires_grad", False))
            for parameter in parameters
        ),
        "Mamba unexpectedly trainable",
    )

    support_rows, states, endpoints, forward_count = plan_executor(
        selected=selected,
        events=events,
        encoded=encoded,
        model=model,
    )

    require(
        forward_count == EXPECTED_FORWARD_COUNT,
        "scientific forward count",
    )
    require(
        len(support_rows) == EXPECTED_SUPPORT_ROWS,
        "support-state row count",
    )
    require(
        len(endpoints) == EXPECTED_ENDPOINT_ROWS,
        "endpoint row count",
    )

    versions = runtime_version_loader()

    bundle_writer(
        output_dir,
        support_rows=support_rows,
        states=states,
        endpoints=endpoints,
        implementation_commit=Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT,
        measurement_implementation_sha256=Q1_Q3_MEASUREMENT_SHA256,
        extraction_runner_sha256=Q1_Q3_EXTRACTION_SHA256,
        runtime_versions=versions,
    )

    return {
        "execution_head": head,
        "execution_authority_commit": execution_authority_commit,
        "driver_implementation_commit": driver_implementation_commit,
        "driver_sha256": expected_driver_sha256,
        "q1_q3_primitive_implementation_commit":
            Q1_Q3_PRIMITIVE_IMPLEMENTATION_COMMIT,
        "representative_checkpoint_sha256": checkpoint_sha,
        "layer_set": list(EXPECTED_LAYERS),
        "model_input_row_count": EXPECTED_ROWS,
        "backbone_forward_count": forward_count,
        "support_state_row_count": len(support_rows),
        "endpoint_row_count": len(endpoints),
        "training": False,
        "backward": False,
        "statistical_testing": False,
        "scientific_conclusion": "NONE",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--driver-implementation-commit", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument("--execution-authority-commit", required=True)
    parser.add_argument("--model-snapshot", default=None)
    parser.add_argument("--tokenizer-snapshot", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_scientific_extraction(
        output_dir=args.output_dir,
        expected_execution_head=args.expected_execution_head,
        driver_implementation_commit=args.driver_implementation_commit,
        expected_driver_sha256=args.expected_driver_sha256,
        execution_authority_commit=args.execution_authority_commit,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
    )


if __name__ == "__main__":
    main()
